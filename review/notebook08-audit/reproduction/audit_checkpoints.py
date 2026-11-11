"""Read-only Notebook 08 checkpoint and population audit using the pinned numerical source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback


CORE_HASH = "b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372"
GRID_HASH = "a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df"
TOLERANCE = {
    "true_or_reference_probability": 1e-12,
    "observed_history_probability": 5e-6,
    "probe_probability": 5e-6,
    "mean_validation_nll": 5e-7,
    "individual_validation_nll": 5e-6,
    "train_mean_nll": 5e-7,
    "teacher_training_logit_runner_resume": 5e-6,
    "saved_teacher_target_probability_consistency": 5e-7,
}
CONTROLS = {
    "individual_grid_ml", "pooled_grid_ml", "hierarchical_hybrid",
    "hierarchical_hybrid_cue_mixture", "coinflip", "training_frequency",
}
STATES = {
    "teacher": (("teacher", "best_state", "best_val"),),
    **{
        arm: (
            (f"{arm}_population", "population_best_state", "population_best_val"),
            (f"{arm}_individual", "best_state", "best_val"),
        )
        for arm in ("solo", "distilled", "oracle")
    },
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class PopulationAudit:
    def __init__(self, folder, item, cell, category):
        self.output = {
            "path": str(folder), **item, "cell": cell, "category": category,
            "checks": 0, "structural_issues": [], "cpu_tolerance_exceedances": [],
            "models": {}, "checkpoints": {},
        }

    def check(self, condition, label):
        self.output["checks"] += 1
        if not condition:
            self.output["structural_issues"].append(label)
        return bool(condition)

    def compare(self, actual, expected, tolerance, label):
        delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
        maximum = float(delta.max())
        count = int((delta > tolerance).sum())
        self.output["checks"] += 1
        if count:
            self.output["cpu_tolerance_exceedances"].append(
                {"check": label, "max_abs": maximum, "over_tolerance": count, "tolerance": tolerance}
            )
        return maximum, count


def verified(folder, cell_path, protocol_hash, item, audit):
    manifest = read(folder / "manifest.json")
    cell_manifest = read(cell_path / "run_manifest.json")
    rel = folder.relative_to(cell_path).as_posix()
    completed = manifest.get("completed_file_sha256", {})
    ready = audit.check(
        manifest.get("identity") == {**item, "protocol_sha256": protocol_hash}
        and manifest.get("status") == "complete"
        and cell_manifest.get("completed_file_sha256", {}).get(f"{rel}/manifest.json")
        == sha(folder / "manifest.json"),
        "complete population and manifest bindings",
    )
    for name in runner.POPULATION_FILES:
        path = folder / name
        ready &= audit.check(
            path.is_file() and not path.is_symlink() and completed.get(name) == sha(path),
            f"verify completed hash before loading {name}",
        )
    return ready


def masked_losses(model, x, y, train, validation):
    with torch.no_grad():
        logits = model(x)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return logits, (losses * train).sum() / train.sum(), (losses * validation).sum(1) / validation.sum(1)


def verify_data(audit, config, data, seed, kind):
    expected = simulate_population(config, kind, seed)
    masks = make_trial_masks(config.n_subjects, config.n_trials, seed + 1)
    expected.update({f"mask_{name}": value for name, value in masks.items()})
    audit.check(data.keys() == expected.keys(), "exact regenerated data keys")
    max_regeneration_error = {}
    for key, value in expected.items():
        actual = data[key]
        if not audit.check((actual.shape, actual.dtype) == (value.shape, value.dtype),
                           f"regenerated shape/dtype {key}"):
            continue
        if key in {"true_prob", "reference_prob"}:
            error = float(np.max(np.abs(actual - value)))
            max_regeneration_error[key] = error
            audit.check(np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
                        and error <= TOLERANCE["true_or_reference_probability"],
                        f"regenerated finite bounded probabilities {key}")
        else:
            audit.check(np.array_equal(actual, value), f"regenerated exact {key}")
    audit.output["regenerated_probability_max_abs"] = max_regeneration_error
    rare_count = int(data["rare"].sum())
    audit.output["rare_people"] = rare_count
    audit.check(rare_count == round(config.n_subjects * config.rare_fraction), "exact rare count")
    audit.check(np.array_equal(data["is_cue"], data["rare"] if kind == "reward_cue"
                               else np.zeros_like(data["rare"])), "exact family assignments")
    history = np.stack([data[k] for k in ("choice", "state", "second_choice", "reward")], axis=-1)
    audit.check(np.all((history == 0) | (history == 1)), "binary observed histories")
    audit.check(np.all(data["x"][:, 0] == 0), "initial inputs zero")
    audit.check(np.array_equal(data["x"][:, 1:], 2 * history[:, :-1] - 1), "only prior trial encoded")
    total = sum(data[f"mask_{s}"].astype(np.int8) for s in ("train", "val", "test"))
    audit.check(np.all(total == 1), "disjoint exhaustive masks")
    mask_count = round(config.n_trials * .2)
    for name, count in (("train", config.n_trials - 2 * mask_count), ("val", mask_count), ("test", mask_count)):
        audit.check(np.all(data[f"mask_{name}"].sum(axis=1) == count), f"fixed per-person {name} counts")


def verify_checkpoint(audit, config, folder, arm, data, p, probes, soft, training, tensors):
    x, y, train, validation = tensors
    seed = audit.output["seed"]
    path = folder / f"{arm}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    runner._finite(saved, str(path))
    limit = config.teacher_steps if arm == "teacher" else config.student_steps
    initialization = seed + (2 if arm == "teacher" else 3)
    torch.manual_seed(initialization)
    model = runner._new_model(config, arm).eval()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    initial_logits, initial_train, initial_val = masked_losses(model, *tensors)
    history = saved["history"]
    completed = saved["completed_steps"]
    selected = min(history, key=lambda row: row["validation_nll"])
    population_step = saved["best_step"] if arm == "teacher" else saved["population_best_step"]
    step_report = {
        "initialization_seed": initialization, "completed_steps": completed,
        "population_step": population_step, "early_stopped": saved["early_stopped"],
        "elapsed_seconds": saved["elapsed_seconds"],
    }
    audit.output["checkpoints"][arm] = step_report
    # The reviewed source verifies signatures, tensor structure, Adam settings, RNG
    # shapes, counters and CPU scores. Compare score errors independently below.
    targets = torch.as_tensor(soft[arm]) if arm in soft else None
    try:
        runner._checkpoint(config, arm, seed, path, model, *tensors, targets)
        step_report["reviewed_validator"] = "passed"
    except ValueError as error:
        message = str(error)
        step_report["reviewed_validator"] = message
        if "checkpoint predictions differ" not in message.lower() and "selected validation predictions differ" not in message.lower():
            audit.check(False, f"{arm} reviewed checkpoint validator: {message}")

    actual_training = training[arm].copy()
    original_path = actual_training.pop("checkpoint_path")
    audit.check(Path(original_path).parts[-3:] == path.parts[-3:], f"{arm} checkpoint provenance path")
    expected_training = runner._phase_report(saved, arm, config, path)
    expected_training.pop("checkpoint_path")
    audit.check(actual_training == expected_training, f"{arm} checkpoint matches all training metadata")
    audit.check(completed == limit or (arm == "teacher" and saved["early_stopped"]),
                f"{arm} fixed update budget or teacher patience reached")
    audit.check(all(row["step"].item() == completed for row in saved["optimizer"]["state"].values()),
                f"{arm} Adam counters")
    expected_steps = [0, *range(config.eval_every, completed + 1, config.eval_every)]
    audit.check([h["step"] for h in history] == expected_steps
                and all(math.isfinite(h[k]) for h in history for k in ("train_nll", "validation_nll")),
                f"{arm} complete finite validation schedule")
    audit.check(population_step == selected["step"], f"{arm} earliest recorded population minimum")
    if arm == "teacher":
        stale = len(history) - 1 - expected_steps.index(population_step)
        audit.check(saved["stale"] == stale and (
            (saved["early_stopped"] and stale == config.patience)
            or (not saved["early_stopped"] and stale < config.patience)
        ), "teacher stale counter and termination")
        # Teacher uses one global mask reduction for selection and the mean of
        # person means in its log. Equal validation counts make them equivalent
        # up to float32 summation. Allow that tolerance when checking earlier stops.
        best = history[0]["validation_nll"]
        previous_stale = 0
        for row in history[1:-1]:
            value = row["validation_nll"]
            previous_stale = 0 if value < best else previous_stale + 1
            best = min(best, value)
            audit.check(previous_stale < config.patience, "teacher continues only before patience reached")
    else:
        steps = saved["best_step"].numpy()
        audit.check(steps.dtype == np.int64 and steps.shape == (config.n_subjects,)
                    and np.isin(steps, expected_steps).all(), f"{arm} valid selected rows")
        audit.check(not saved["early_stopped"], f"{arm} no early stop")
        late = 0.9 * limit
        rare = data["rare"]
        step_report["individual_selected"] = {
            "min": int(steps.min()), "median": float(np.median(steps)), "max": int(steps.max()),
            "at_limit": int((steps == limit).sum()), "rare_at_limit": int((steps[rare] == limit).sum()),
            "at_or_above_90pct": int((steps >= late).sum()), "rare_at_or_above_90pct": int((steps[rare] >= late).sum()),
            "histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(steps, return_counts=True))},
        }
        for step, state in ((0, initial_state), (completed, saved["current_state"]),
                            (population_step, saved["population_best_state"])):
            matched_rows = steps == step
            audit.check(
                all(torch.equal(v[matched_rows], state[k][matched_rows]) for k, v in saved["best_state"].items()),
                f"{arm} individually selected rows equal available state at step {step}",
            )
        if population_step in {0, completed}:
            state = initial_state if population_step == 0 else saved["current_state"]
            audit.check(all(torch.equal(v, state[k]) for k, v in saved["population_best_state"].items()),
                        f"{arm} population-selected weights at available step {population_step}")

    for label, row, train_nll, val_nll in (
        ("initial", history[0], initial_train, initial_val),
        ("current", history[-1], *masked_losses_after_loading(model, saved["current_state"], tensors)),
    ):
        step_report[f"{label}_train_nll_cpu_max_abs"], _ = audit.compare(
            float(train_nll), row["train_nll"], TOLERANCE["train_mean_nll"], f"{arm} {label} train NLL")
        step_report[f"{label}_val_nll_cpu_max_abs"], _ = audit.compare(
            float(val_nll.mean()), row["validation_nll"], TOLERANCE["mean_validation_nll"],
            f"{arm} {label} validation NLL")

    for name, state_key, value_key in STATES[arm]:
        model.load_state_dict(saved[state_key])
        logits, _, val = masked_losses(model, *tensors)
        individual = name.endswith("_individual")
        row = {}
        audit.output["models"][name] = row
        for metric, actual, expected, tol in (
            ("observed_history_probability", torch.sigmoid(logits).numpy(), p[name],
             TOLERANCE["observed_history_probability"]),
            ("probe_probability", runner.probe_forward(model, config.n_subjects), probes[name],
             TOLERANCE["probe_probability"]),
            ("selected_validation_nll", val.numpy() if individual else float(val.mean()), saved[value_key].numpy(),
             TOLERANCE["individual_validation_nll" if individual else "mean_validation_nll"]),
        ):
            row[metric], row[f"{metric}_count_over_tolerance"] = audit.compare(actual, expected, tol, f"{name} {metric}")
        if arm == "teacher":
            target_logits = soft["distilled"][data["mask_train"]]
            row["teacher_logit_cpu_max_abs"], row["teacher_logit_count_over_runner_resume_tolerance"] = audit.compare(
                logits.numpy()[data["mask_train"]], target_logits, TOLERANCE["teacher_training_logit_runner_resume"],
                "teacher training logits against direct CPU replay",
            )
            delta = float(np.max(np.abs(torch.sigmoid(torch.from_numpy(target_logits)).numpy()
                                       - p["teacher"][data["mask_train"]])))
            row["teacher_saved_target_probability_max_abs"] = delta
            audit.check(delta <= TOLERANCE["saved_teacher_target_probability_consistency"],
                        "saved teacher logits and saved teacher probability agree on training support")
        elif individual:
            # The recorded individual minimum must be no higher than any
            # available common-step validation score for the same person.
            for label, state in (("initial", initial_state), ("current", saved["current_state"]),
                                 ("population", saved["population_best_state"])):
                model.load_state_dict(state)
                _, _, available = masked_losses(model, *tensors)
                delta = float((saved["best_val"] - available).max())
                row[f"maximum_excess_over_{label}_validation"] = delta
                audit.check(delta <= TOLERANCE["individual_validation_nll"],
                            f"{arm} selected person score no higher than {label} score")
        if arm in soft:
            row["masked_soft_target_bce"] = soft_target_bce(logits, targets, train)
    if arm in soft:
        model.load_state_dict(saved["current_state"])
        with torch.no_grad():
            step_report["current_masked_soft_target_bce"] = soft_target_bce(model(x), targets, train)


def masked_losses_after_loading(model, state, tensors):
    model.load_state_dict(state)
    _, train_nll, val_nll = masked_losses(model, *tensors)
    return train_nll, val_nll


def soft_target_bce(logits, targets, train):
    values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.sigmoid(targets), reduction="none"
    )
    return float((values * train).sum() / train.sum())


def audit_population(config, cell_path, item, category):
    folder = runner.population_path(cell_path, **item)
    audit = PopulationAudit(folder, item, cell_path.name, category)
    try:
        if not verified(folder, cell_path, sha(cell_path / "protocol.json"), item, audit):
            return audit.output
        data = review.arrays(folder / "data.npz")
        p = review.arrays(folder / "predictions.npz")
        probes = review.arrays(folder / "probe_predictions.npz")
        soft = review.arrays(folder / "soft_training_logits.npz")
        report = read(folder / "report.json")
        training = read(folder / "training.json")
        audit.check(report["phases"] == training and set(training) == set(STATES),
                    "report and training metadata match every phase")
        verify_data(audit, config, data, **item)
        shape = (config.n_subjects, config.n_trials)
        expected_models = {name for rows in STATES.values() for name, _, _ in rows} | CONTROLS
        for name, archive, expected_shape in (
            ("observed history", p, shape), ("probe histories", probes, (shape[0], 256, 3))
        ):
            audit.check(set(archive) == expected_models, f"exact {name} model coverage")
            for key, values in archive.items():
                audit.check(values.shape == expected_shape and values.dtype.kind == "f"
                            and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                            f"valid {name} probabilities {key}")
        audit.check(set(soft) == {"distilled", "oracle"}, "exact target coverage")
        train = data["mask_train"]
        for name, values in soft.items():
            audit.check(values.shape == shape and values.dtype == np.float32 and np.isfinite(values).all()
                        and np.count_nonzero(values[~train]) == 0, f"{name} finite zeroed off training support")
        audit.check(
            np.array_equal(soft["oracle"], oracle_training_logits(data["true_prob"], train).numpy()),
            "oracle training logits reproduce exactly",
        )
        tensors = (torch.from_numpy(data["x"]), torch.tensor(data["choice"], dtype=torch.float32),
                   torch.from_numpy(train), torch.from_numpy(data["mask_val"]))
        for arm in STATES:
            try:
                verify_checkpoint(audit, config, folder, arm, data, p, probes, soft, training, tensors)
            except Exception:
                audit.output["structural_issues"].append(f"{arm} error:\n{traceback.format_exc()}")
    except Exception:
        audit.output["structural_issues"].append(traceback.format_exc())
    return audit.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "new", "baseline", "smoke", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--only", choices=["new", "baseline", "smoke", "all"], default="all")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    for root in (args.source, args.new, args.baseline, args.smoke):
        if args.output.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Audit output must be outside {root}")
    sys.path[:0] = [str(args.source), str(args.source / "src"), str(args.source / "scripts")]
    global np, torch, review, runner, simulate_population, make_trial_masks, oracle_training_logits
    import numpy as np
    import torch
    import verify_main_study_integrity as review
    from main_study import runner
    from main_study.config import MainStudyConfig
    from main_study.simulation import simulate_population
    from grid_study.config import GridTrancheConfig, grid_source_fingerprint
    from distilled_rnns.evaluation import make_trial_masks
    from distilled_rnns.oracle_control import oracle_training_logits

    torch.set_num_threads(args.cpu_threads)
    if runner.main_source_fingerprint() != CORE_HASH or grid_source_fingerprint() != GRID_HASH:
        raise ValueError("Numerical source differs from pinned hashes.")
    categories = {
        "new": (args.new, "rarity_tranche.json"),
        "baseline": (args.baseline, "main_study_first_tranche.json"),
        "smoke": (args.smoke, "rarity_smoke.json"),
    }
    output = {
        "version": 1, "script_sha256": sha(Path(__file__)), "source": str(args.source),
        "core_sha256": CORE_HASH, "grid_sha256": GRID_HASH,
        "environment": {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": str(torch.__version__), "cpu_threads": args.cpu_threads,
            "reproduction_device": "cpu", "cuda_available": torch.cuda.is_available(),
        },
        "tolerances_abs": TOLERANCE, "populations": [],
        "notes": [
            "No training, resume or notebook execution. Checkpoints loaded on CPU with weights_only=True.",
            "A CPU threshold exceedance does not change the saved-array outcome calculations.",
            "The final checkpoint does not archive the full per-person validation trajectory.",
        ],
    }
    for category, (root, name) in categories.items():
        if args.only not in ("all", category):
            continue
        runner._reject_symlinks(root)
        config_path = args.source / "configs" / name
        if category == "baseline":
            cells = [(root, MainStudyConfig.from_file(config_path))]
        else:
            tranche = GridTrancheConfig.from_file(config_path)
            cells = [(root / "cells" / cfg.cell_id, cfg) for cfg in tranche.cell_configs]
        for cell_path, config in cells:
            planned = {f"{item['kind']}/seed-{item['seed']}" for item in config.plan}
            actual = {p.relative_to(cell_path).as_posix() for p in cell_path.glob("*/seed-*")}
            if actual != planned:
                print(f"Plan mismatch at {cell_path}: missing {sorted(planned-actual)}, extra {sorted(actual-planned)}", flush=True)
            for item in config.plan:
                row = audit_population(config, cell_path, item, category)
                output["populations"].append(row)
                print(category, cell_path.name, item["kind"], item["seed"],
                      len(row["structural_issues"]), "structural issues and",
                      len(row["cpu_tolerance_exceedances"]), "CPU tolerance exceedances", flush=True)
                args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return int(any(row["structural_issues"] or row["cpu_tolerance_exceedances"]
                   for row in output["populations"]))


if __name__ == "__main__":
    raise SystemExit(main())
