"""Independent read-only metric audit of downloaded notebook-07 results."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]
from main_study.config import MainStudyConfig
from main_study.evaluation import baseline_predictions

KEYS = ("choice", "state", "second_choice", "reward")
METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl", "signature_rms_error", "probe_kl", "pull")
HISTORIES = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).reshape(256, 2, 4)
CHOICES = HISTORIES[:, :, 0]
COMMON = CHOICES == HISTORIES[:, :, 1]
CELLS = np.array(
    [
        (HISTORIES[:, -1, 3] == reward) & (COMMON[:, -1] == common)
        for reward in (0, 1)
        for common in (False, True)
    ],
    dtype=float,
)
ENV_MASS = np.where(COMMON[:, 0], 0.7, 0.3) * np.where(COMMON[:, 1], 0.7, 0.3) / 16


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def file_hashes(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def load_json(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def nll(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.where(y == 1, -np.log(p), -np.log1p(-p))


def kl(p, q):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    return np.maximum(0, p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def true_probe(data):
    cache = {}
    rows = np.arange(256)
    result = []
    for i in range(len(data["rare"])):
        if data["is_cue"][i]:
            key = ("cue", float(data["cue_reliability"][i]))
            if key not in cache:
                p = np.full((256, 3), 0.5)
                desired = np.where(HISTORIES[:, :, 3] == 1, HISTORIES[:, :, 1], 1 - HISTORIES[:, :, 1])
                p[:, 1:] = np.where(desired == 1, key[1], 1 - key[1])
                cache[key] = p
        else:
            key = (
                "hybrid",
                float(data["model_based_weight"][i]),
                float(data["alpha"][i]),
                float(data["beta"][i]),
            )
            if key not in cache:
                _, w, a, b = key
                first = np.full((256, 2), 0.5)
                second = np.full((256, 2, 2), 0.5)
                p = np.empty((256, 3))
                for t in range(3):
                    v = second.max(2)
                    advantage = (1 - w) * (first[:, 1] - first[:, 0]) + 0.4 * w * (v[:, 1] - v[:, 0])
                    p[:, t] = 1 / (1 + np.exp(-b * advantage))
                    if t < 2:
                        action, state, action2, reward = HISTORIES[:, t].T
                        old = second[rows, state, action2].copy()
                        first[rows, action] += a * (old - first[rows, action] + 0.5 * (reward - old))
                        second[rows, state, action2] += a * (reward - old)
                cache[key] = p
        result.append(cache[key])
    return np.array(result)


def signatures(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (256, 3) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probe probabilities")
    if (
        np.ptp(p[:, :, 0], axis=1).max() > 5e-6
        or np.ptp(p[:, :, 1].reshape(len(p), 16, 16), axis=2).max() > 5e-6
    ):
        raise ValueError("Probe prediction depends on later events")
    likelihood0 = np.where(CHOICES[:, 0] == 1, p[:, :, 0], 1 - p[:, :, 0])
    likelihood1 = np.where(CHOICES[:, 1] == 1, p[:, :, 1], 1 - p[:, :, 1])
    mass = likelihood0 * likelihood1 * ENV_MASS
    if np.max(abs(mass.sum(1) - 1)) > 5e-6:
        raise ValueError("Probe path mass does not sum to one")
    repeated = np.where(CHOICES[:, 1] == 1, p[:, :, 2], 1 - p[:, :, 2])
    return ((mass * repeated) @ CELLS.T) / (mass @ CELLS.T)


class Audit:
    def __init__(self):
        self.check_count = 0
        self.maximum_errors = {}
        self.issues = []

    def check(self, ok, label):
        self.check_count += 1
        if not ok:
            self.issues.append(label)

    def match(self, actual, expected, label, category="metrics", tolerance=1e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
            if not isinstance(actual, dict):
                return
            for k, v in expected.items():
                if k in actual:
                    self.match(actual[k], v, label + "." + str(k), category, tolerance)
        elif isinstance(expected, (tuple, list)):
            self.check(len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.match(a, b, label + "." + str(i), category, tolerance)
        elif expected is None or isinstance(expected, (str, bool)):
            self.check(actual == expected, label)
        else:
            self.check_count += 1
            if actual is None:
                self.issues.append(label + " unexpectedly null")
                return
            error = abs(float(actual) - float(expected))
            self.maximum_errors[category] = max(error, self.maximum_errors.get(category, 0))
            if not np.isfinite(error) or error > tolerance:
                self.issues.append(f"{label}: {actual} vs {expected} (error {error})")

    def array_match(self, a, b, label, category, tolerance=1e-12):
        a, b = np.asarray(a), np.asarray(b)
        self.check(a.shape == b.shape, label + " shape")
        if a.shape != b.shape:
            return
        error = float(np.max(abs(a.astype(float) - b.astype(float))))
        self.maximum_errors[category] = max(error, self.maximum_errors.get(category, 0))
        self.check_count += a.size
        if not np.isfinite(error) or error > tolerance:
            self.issues.append(label + " maximum discrepancy " + str(error))


def mean_supported(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def aggregate(values):
    first = values[0]
    if isinstance(first, dict):
        return {k: aggregate([x[k] for x in values]) for k in first}
    if isinstance(first, list):
        return [aggregate([x[i] for x in values]) for i in range(len(first))]
    available = [x for x in values if x is not None]
    return {
        "mean": float(np.mean(available)) if available else None,
        "min": min(available) if available else None,
        "max": max(available) if available else None,
        "supported_seed_count": len(available),
        "planned_seed_count": len(values),
        "positive_seed_count": sum(x > 0 for x in available),
        "negative_seed_count": sum(x < 0 for x in available),
        "tied_seed_count": sum(x == 0 for x in available),
    }


def audit_population(audit, folder, config):
    data = arrays(folder / "data.npz")
    p = arrays(folder / "predictions.npz")
    probes = arrays(folder / "probe_predictions.npz")
    report = load_json(folder / "report.json")
    truth = data["true_prob"]
    y = data["choice"]
    test = data["mask_test"]
    rare = data["rare"]
    diagnostic = test & (abs(truth - data["reference_prob"]) > 0.2)
    tp = true_probe(data)
    true_sig = signatures(tp)
    majority = true_sig[~rare].mean(0)
    directions = majority - true_sig
    denominator = (directions**2).sum(1)
    audit.array_match(majority, report["true_majority_signature"], "true majority signature", "true_probe")
    n = len(y)
    subject_models = [{} for _ in range(n)]
    for name, probability in p.items():
        audit.check(name in probes, "probe model " + name)
        audit.check(
            probability.shape == y.shape
            and np.isfinite(probability).all()
            and ((probability >= 0) & (probability <= 1)).all(),
            "valid predictions " + name,
        )
        loss = nll(y, probability)
        divergence = kl(truth, probability)
        sig = signatures(probes[name])
        error = sig - true_sig
        fit_rms = np.sqrt(np.mean(error**2, 1))
        probe_kl = kl(tp[:, :, 2], probes[name][:, :, 2]).mean(1)
        for i in range(n):
            fitted = {
                "nll": float(loss[i, test[i]].mean()),
                "kl": float(divergence[i, test[i]].mean()),
                "diagnostic_nll": float(loss[i, diagnostic[i]].mean()) if diagnostic[i].any() else None,
                "diagnostic_kl": float(divergence[i, diagnostic[i]].mean()) if diagnostic[i].any() else None,
                "signature": sig[i].tolist(),
                "signature_rms_error": float(fit_rms[i]),
                "probe_kl": float(probe_kl[i]),
                "pull": float(np.dot(error[i], directions[i]) / denominator[i])
                if denominator[i] > 1e-12
                else None,
            }
            subject_models[i][name] = fitted
            audit.match(report["per_subject"][i]["models"][name], fitted, f"{folder.name}.{name}.person{i}")
    subject_primary = []
    for i in range(n):
        audit.match(
            {
                k: report["per_subject"][i][k]
                for k in ("subject_id", "rare", "test_count", "diagnostic_count", "true_signature")
            },
            {
                "subject_id": i,
                "rare": bool(rare[i]),
                "test_count": int(test[i].sum()),
                "diagnostic_count": int(diagnostic[i].sum()),
                "true_signature": true_sig[i].tolist(),
            },
            f"person{i}.support_truth",
        )
        primary = {}
        for mode in ("population", "individual"):
            left = subject_models[i]["solo_" + mode]
            right = subject_models[i]["distilled_" + mode]
            gain = left["nll"] - right["nll"]
            distortion = right["signature_rms_error"] - left["signature_rms_error"]
            primary[mode] = {
                "prediction_gain": gain,
                "behavioral_distortion": distortion,
                "quadrant_fraction": float(gain > 0 and distortion > 0),
            }
        audit.match(report["per_subject"][i]["primary"], primary, f"person{i}.primary")
        subject_primary.append(primary)
    groups = {}
    for name, selected in [("rare", rare), ("typical", ~rare)]:
        ids = np.flatnonzero(selected)
        models = {}
        for model in p:
            scores = {
                metric: mean_supported([subject_models[i][model][metric] for i in ids]) for metric in METRICS
            }
            scores.update(
                signature=np.mean([subject_models[i][model]["signature"] for i in ids], axis=0).tolist(),
                pull_subject_count=sum(subject_models[i][model]["pull"] is not None for i in ids),
                diagnostic_subject_count=int(diagnostic[selected].any(1).sum()),
            )
            models[model] = scores
        primary = {
            mode: {
                metric: float(np.mean([subject_primary[i][mode][metric] for i in ids]))
                for metric in subject_primary[0][mode]
            }
            for mode in ("population", "individual")
        }
        groups[name] = {
            "subject_count": len(ids),
            "test_count": int(test[selected].sum()),
            "diagnostic_count": int(diagnostic[selected].sum()),
            "diagnostic_subject_count": int(diagnostic[selected].any(1).sum()),
            "models": models,
            "primary": primary,
        }
    audit.match(report["groups"], groups, folder.name + ".groups")
    # Refit only from the four observable history arrays; labels/ground truth never enter.
    observable = {key: data[key] for key in KEYS}
    baseline, baseline_probes, metadata = baseline_predictions(config, observable, data["mask_train"])
    for name, b in baseline.items():
        audit.array_match(p[name], b, "refit." + name, "baseline_prediction")
        audit.array_match(probes[name], baseline_probes[name], "refit_probe." + name, "baseline_probe")
    audit.match(report["baseline_metadata"], metadata, "baseline metadata", "baseline_metadata")
    return {
        "seed": report["seed"],
        "kind": report["kind"],
        "groups": groups,
        "rare_people_positive_both": {
            mode: int(sum(subject_primary[i][mode]["quadrant_fraction"] for i in np.flatnonzero(rare)))
            for mode in ("population", "individual")
        },
        "family_posterior_summary": {
            "cue_posterior_mean_among_rare": float(
                np.asarray(metadata["family_posterior_hybrid_cue"])[rare, 1].mean()
            ),
            "rare_people_classified_cue": int(
                (np.asarray(metadata["family_posterior_hybrid_cue"])[rare, 1] > 0.5).sum()
            ),
            "estimated_family_prior": metadata["family_prior_hybrid_cue"],
        },
    }


def audit_folder(folder):
    before = file_hashes(folder)
    summary = load_json(folder / "summary.json")
    config = MainStudyConfig(mode=summary["config"]["mode"])
    audit = Audit()
    audit.check(summary["status"] == "complete", "complete status")
    populations = []
    for item in summary["planned_populations"]:
        path = folder / item["kind"] / ("seed-" + str(item["seed"]))
        populations.append(audit_population(audit, path, config))
        print("raw metrics checked", folder.name, item["seed"], flush=True)
    for kind in config.design["population"]["rare_kinds"]:
        rows = [p for p in populations if p["kind"] == kind]
        saved = summary["kinds"][kind]
        audit.match(saved["groups"], aggregate([p["groups"] for p in rows]), kind + ".aggregate", "summary")
        audit.match(
            saved["per_seed"],
            [{"seed": p["seed"], "groups": p["groups"]} for p in rows],
            kind + ".per_seed",
            "summary",
        )
    unchanged = file_hashes(folder) == before
    audit.check(unchanged, "all downloaded bytes unchanged")
    return {
        "folder": str(folder),
        "status": "passed" if not audit.issues else "failed",
        "check_count": audit.check_count,
        "maximum_errors": audit.maximum_errors,
        "issues": audit.issues,
        "all_input_bytes_unchanged": unchanged,
        "input_file_count": len(before),
        "per_population": populations,
        "file_sha256_before": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for p in args.folders:
        if args.output.resolve().is_relative_to(p.resolve()):
            raise ValueError("Output must be outside all inputs")
    result = {
        "scope": "Independent saved-array reconstruction, separate true-policy/assay implementation; training-only baseline refit using reviewed estimator.",
        "numpy_version": np.__version__,
        "verification_script_sha256": sha(Path(__file__)),
        "folders": [audit_folder(p) for p in args.folders],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "statuses": [x["status"] for x in result["folders"]],
                "checks": [x["check_count"] for x in result["folders"]],
                "issues": [x["issues"][:10] for x in result["folders"]],
            },
            indent=2,
        )
    )
    raise SystemExit(1 if any(x["issues"] for x in result["folders"]) else 0)


if __name__ == "__main__":
    main()
