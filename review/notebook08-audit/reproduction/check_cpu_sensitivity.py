"""Quantify raw-score sensitivity to CPU forward passes through verified weights."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from audit_checkpoints import STATES, read, sha


def measure(data, probability, probes, true_signature, metrics):
    mask = data["mask_test"]
    diagnostic = mask & (np.abs(data["true_prob"] - data["reference_prob"]) > 0.2)
    loss = metrics.nll(data["choice"], probability)
    divergence = metrics.kl(data["true_prob"], probability)
    assay_error = np.sqrt(np.mean((metrics.signatures(probes) - true_signature) ** 2, axis=1))
    counts = diagnostic.sum(axis=1)
    return {
        "nll": (loss * mask).sum(axis=1) / mask.sum(axis=1),
        "kl": (divergence * mask).sum(axis=1) / mask.sum(axis=1),
        "diagnostic_nll": np.divide((loss * diagnostic).sum(axis=1), counts,
                                    out=np.full(len(counts), np.nan), where=counts > 0),
        "diagnostic_kl": np.divide((divergence * diagnostic).sum(axis=1), counts,
                                   out=np.full(len(counts), np.nan), where=counts > 0),
        "assay_rms": assay_error,
    }


def summarize_groups(values, rare):
    result = {}
    for label, mask in (("rare", rare), ("typical", ~rare)):
        result[label] = {}
        for key, value in values.items():
            present = mask & np.isfinite(value)
            result[label][key] = float(value[present].mean()) if present.any() else None
    return result


def primary(metrics_by_model, rare, selection):
    result = {}
    for label, mask in (("rare", rare), ("typical", ~rare)):
        solo = metrics_by_model[f"solo_{selection}"][label]
        distilled = metrics_by_model[f"distilled_{selection}"][label]
        result[label] = [solo["nll"] - distilled["nll"],
                         distilled["assay_rms"] - solo["assay_rms"]]
    return result


def compare_population(row, folder, config, runner, metrics):
    # Every replay input must still match the checked population manifest.
    expected = read(folder / "manifest.json")["completed_file_sha256"]
    for name in ("data.npz", "predictions.npz", "probe_predictions.npz", *(f"{arm}.pt" for arm in STATES)):
        path = folder / name
        if path.is_symlink() or sha(path) != expected[name]:
            raise ValueError(f"Unverified replay input {path}")
    data = metrics.arrays(folder / "data.npz")
    p = metrics.arrays(folder / "predictions.npz")
    probes = metrics.arrays(folder / "probe_predictions.npz")
    true_signature = metrics.signatures(metrics.true_probe(data))
    rare = data["rare"]
    x = torch.from_numpy(data["x"])
    comparisons = {}
    saved_by_model = {}
    replay_by_model = {}
    for arm, states in STATES.items():
        model = runner._new_model(config, arm).eval()
        checkpoint = torch.load(folder / f"{arm}.pt", map_location="cpu", weights_only=True)
        for name, state, _ in states:
            model.load_state_dict(checkpoint[state])
            with torch.no_grad():
                p_cpu = torch.sigmoid(model(x)).numpy()
                probe_cpu = runner.probe_forward(model, config.n_subjects)
            saved = measure(data, p[name], probes[name], true_signature, metrics)
            replay = measure(data, p_cpu, probe_cpu, true_signature, metrics)
            saved_by_model[name] = summarize_groups(saved, rare)
            replay_by_model[name] = summarize_groups(replay, rare)
            comparisons[name] = {
                "saved": saved_by_model[name],
                "cpu": replay_by_model[name],
                "cpu_minus_saved": summarize_groups({k: replay[k] - saved[k] for k in saved}, rare),
            }
            difference = np.abs(p_cpu - p[name])
            if (difference > 5e-6).any():
                worst = tuple(int(x) for x in np.unravel_index(np.argmax(difference), difference.shape))
                excess = difference > 5e-6
                report = comparisons[name]["probability_threshold_exceedance"] = {
                    "worst_zero_based_person_and_trial": worst,
                    "worst_cpu_float32": float(p_cpu[worst]), "worst_saved": float(p[name][worst]),
                    "count_rare": int(excess[rare].sum()),
                    "count_typical": int(excess[~rare].sum()),
                    "count_by_split": {
                        k: int((excess & data[f"mask_{k}"]).sum()) for k in ("train", "val", "test")
                    },
                }
                # A second numerical path uses the same weights in float64.
                # It diagnoses rounding sensitivity and does not emulate CUDA.
                model.double()
                with torch.no_grad():
                    p64 = torch.sigmoid(model(x.double())).numpy()
                report["worst_cpu_float64"] = float(p64[worst])
                report["max_abs_cpu_float32_minus_float64"] = float(np.abs(p_cpu - p64).max())
                report["max_abs_saved_minus_cpu_float64"] = float(np.abs(p[name] - p64).max())
                model.float()
    results = {
        "category": row["category"], "cell": row["cell"], "seed": row["seed"],
        "kind": row["kind"], "model_scores": comparisons, "primary": {},
    }
    for selection in ("population", "individual"):
        saved = primary(saved_by_model, rare, selection)
        replay = primary(replay_by_model, rare, selection)
        results["primary"][selection] = {
            label: {
                "saved": saved[label], "cpu": replay[label],
                "cpu_minus_saved": (np.array(replay[label]) - np.array(saved[label])).tolist(),
                "signs_match": bool(np.array_equal(np.sign(replay[label]), np.sign(saved[label]))),
            }
            for label in saved
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--checkpoints-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / "src"), str(args.source / "scripts")]
    from main_study import runner
    from main_study.config import MainStudyConfig
    from grid_study.config import GridTrancheConfig, grid_source_fingerprint
    import verify_main_study_metrics as metrics
    torch.set_num_threads(1)
    source_audit = read(args.checkpoints_audit)
    if runner.main_source_fingerprint() != source_audit["core_sha256"] or grid_source_fingerprint() != source_audit["grid_sha256"]:
        raise ValueError("Pinned numerical source hash mismatch")
    rows = [row for row in source_audit["populations"]
            if row["category"] != "smoke" and not row["structural_issues"]]
    roots = (args.source, args.new, args.baseline)
    output_path = args.output.resolve()
    if any(output_path.is_relative_to(root.resolve()) for root in roots):
        parser.error("Output must be outside the source checkout and input trees")
    if output_path == args.checkpoints_audit.resolve():
        parser.error("Output must differ from the checkpoint audit")
    runner._reject_symlinks(args.new)
    runner._reject_symlinks(args.baseline)
    conf = args.source / "configs"
    baseline = MainStudyConfig.from_file(conf / "main_study_first_tranche.json")
    grid = {cell.cell_id: cell for cell in GridTrancheConfig.from_file(conf / "rarity_tranche.json").cell_configs}
    output = {
        "pytorch": str(torch.__version__), "numpy": np.__version__, "script_sha256": sha(Path(__file__)),
        "checkpoint_audit_sha256": sha(args.checkpoints_audit),
        "outcome_order": ["prediction_gain", "assay_error_change"],
        "populations": [],
    }
    for row in rows:
        if row["category"] == "baseline":
            config, cell_folder = baseline, args.baseline
        elif row["category"] == "new":
            config, cell_folder = grid[row["cell"]], args.new / "cells" / row["cell"]
        else:
            raise ValueError(f"Unexpected checkpoint audit category {row['category']}")
        item = {key: row[key] for key in ("kind", "seed")}
        if item not in config.plan:
            raise ValueError(f"Checkpoint audit population is outside the pinned plan: {item}")
        folder = cell_folder / row["kind"] / f"seed-{row['seed']}"
        output["populations"].append(compare_population(row, folder, config, runner, metrics))
        print(row["category"], row["seed"], "checked", flush=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
