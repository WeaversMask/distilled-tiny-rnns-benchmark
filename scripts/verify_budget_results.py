"""Read-only audit of the frozen notebook-06 result folders.

No training or resume entry point is called. NumPy archives disallow pickle;
checkpoints use weights_only=True on CPU. The output must be outside all inputs.
Run with --kind production or --kind smoke and an explicit --output JSON path.
"""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from distilled_rnns import budget_check as budget
from distilled_rnns.config import ExperimentConfig
from distilled_rnns.experiment import source_fingerprint
from distilled_rnns.evaluation import make_trial_masks
from distilled_rnns.simulation import simulate_population
from verify_budget_notebook import verify_summary


def main(argv=None):
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(
        description="Read-only downloaded budget-results audit; writes only the explicitly chosen JSON file."
    )
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("--original-parent", type=pathlib.Path)
    parser.add_argument("--kind", choices=("production", "smoke"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    folder_root = args.folder.resolve()
    original = args.original_parent.resolve() if args.original_parent else None
    for source in (folder_root, original):
        if source is not None and args.output.resolve().is_relative_to(source):
            raise ValueError("Audit output must be outside every input folder")
    budget._reject_symlinks(folder_root)

    def read_json(path):
        return json.loads(path.read_text())

    before = budget._file_hashes(folder_root)
    protocol = read_json(folder_root / "protocol.json")
    summary = read_json(folder_root / "summary.json")
    snapshot = read_json(folder_root / "parent_snapshot/snapshot_manifest.json")
    config = protocol["config"]
    base_config = ExperimentConfig(**config["parent_config"]["base"])
    checks = []
    issues = []
    reconstructions = []
    rows = []
    count = 0
    record = {
        "folder": str(folder_root),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "source_fingerprint": source_fingerprint(),
        "checks": checks,
        "issues": issues,
        "reproduction": reconstructions,
        "per_seed": rows,
        "hashes_before": before,
    }

    def check(ok, label):
        nonlocal count
        count += 1
        if not ok:
            issues.append(label)

    def close(a, c, label, tol=1e-12):
        if a is None or c is None:
            check(a is None and c is None, label)
        else:
            check(abs(float(a) - float(c)) <= tol, label)

    settings = read_json(ROOT / "configs/colab_budget_source.json")
    expected = budget.BudgetCheckConfig.from_file(
        ROOT / "configs" / ("budget_check_smoke.json" if args.kind == "smoke" else "budget_check.json")
    )
    if args.kind == "smoke":
        expected = dataclasses.replace(expected, expected_parent_source_sha256=settings["source_sha256"])
    if config != expected.to_dict():
        raise ValueError("Result configuration differs from frozen specification")
    if source_fingerprint() != settings["source_sha256"]:
        raise ValueError("Use the reviewed runtime: local source fingerprint differs")
    check(source_fingerprint() == protocol["source_sha256"], "source fingerprint")
    parent_audit = budget.validate_parent(expected, folder_root / "parent_snapshot", device="cpu")
    check(
        all((parent_audit["file_sha256"].get(k) == v for k, v in snapshot["file_sha256"].items())),
        "read-only parent validator listed hashes",
    )
    for key in ("protocol_sha256", "checkpoint_elapsed_seconds", "parent_training_seconds"):
        check(parent_audit[key] == snapshot[key], "read-only parent validator " + key)
    checks.append(
        "Trusted config/source and read-only parent validation passed; local path and additional snapshot manifest compared explicitly."
    )
    check(
        budget.legacy_source_fingerprint() == protocol["original_numerical_source_sha256"],
        "original numerical source fingerprint",
    )
    check(snapshot == protocol["parent"], "snapshot identity")
    if set(parent_audit["file_sha256"]) != set(snapshot["file_sha256"]) | {"snapshot_manifest.json"}:
        raise ValueError("Parent snapshot manifest does not cover the exact original files")
    critical_files = {"distilled.pt", "oracle.pt", "predictions.npz", "report.json", "training.json"}
    for manifest_path in folder_root.glob("seed-*/manifest.json"):
        if set(read_json(manifest_path)["completed_file_sha256"]) != critical_files:
            raise ValueError(f"Unexpected completed file list in {manifest_path.name}")
    for root, items in [(folder_root / "parent_snapshot", snapshot["file_sha256"])] + [
        (f.parent, read_json(f)["completed_file_sha256"]) for f in folder_root.glob("seed-*/manifest.json")
    ]:
        for f, h in items.items():
            check(budget._sha(root / f) == h, "hash " + str(root / f))
    if original is not None:
        for f, h in snapshot["file_sha256"].items():
            check(budget._sha(original / f) == h, "original parent " + f)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    check(protocol_hash == summary["protocol_sha256"], "protocol hash")
    check(
        summary["status"] == "complete"
        and summary["pending_seeds"] == []
        and (
            summary["planned_seeds"]
            == summary["completed_seeds"]
            == config["parent_config"]["replication_seeds"]
        ),
        "completed planned seeds",
    )
    check(
        sorted((f.name for f in folder_root.glob("seed-*")))
        == sorted(("seed-" + str(x) for x in summary["planned_seeds"])),
        "exact child folders",
    )
    for seed in summary["planned_seeds"]:
        folder = folder_root / f"seed-{seed}"
        parent = folder_root / "parent_snapshot" / folder.name
        base = dataclasses.replace(base_config, seed=seed)
        report = read_json(folder / "report.json")
        parent_report = read_json(parent / "report.json")
        manifest = read_json(folder / "manifest.json")
        seed_record = {"seed": seed, "arms": {}, "sessions": manifest["sessions"]}
        rows.append(seed_record)
        check(
            report["seed"] == seed
            and report["status"] == "complete"
            and (report["role"] == "replication")
            and (report["selection"] == summary["selection"]),
            f"{seed} child report identity",
        )
        check(report["baseline"] == parent_report, f"{seed} exact embedded baseline")
        check(report["phases"] == read_json(folder / "training.json"), f"{seed} training==report")
        check(
            manifest["identity"] == {"protocol_sha256": protocol_hash, "seed": seed}
            and manifest["status"] == "complete"
            and (manifest["sessions"][-1]["status"] == "complete"),
            f"{seed} manifest identity/status",
        )
        for root in (folder, parent):
            run_manifest = read_json(root / "manifest.json")
            budget._validate_inventories(root, run_manifest["sessions"])
            check(
                budget._sha(root / "environment-freeze.txt")
                in [ss["package_inventory"]["sha256"] for ss in run_manifest["sessions"]],
                f"{seed} freeze identity",
            )
        data = budget._arrays(parent / "data.npz")
        child_predictions = budget._arrays(folder / "predictions.npz")
        parent_predictions = budget._arrays(parent / "predictions.npz")
        soft = budget._arrays(parent / "soft_training_logits.npz")
        x = torch.tensor(data["x"])
        y = torch.tensor(data["choice"], dtype=torch.float32)
        generated = simulate_population(base.n_subjects, base.n_trials, base.rare_fraction, seed)
        masks = make_trial_masks(base.n_subjects, base.n_trials, seed + 1)
        generated.update(mask_train=masks["train"], mask_checkpoint=masks["val"], mask_test=masks["test"])
        check(set(data) == set(generated), f"{seed} regenerated keys")
        for key, v in generated.items():
            check(
                data[key].dtype == v.dtype
                and data[key].shape == v.shape
                and (
                    np.allclose(data[key], v, rtol=0, atol=1e-12)
                    if v.dtype == np.float64
                    else np.array_equal(data[key], v)
                ),
                f"{seed} regenerated {key}",
            )
        for name in ("teacher", "solo", "solo_population", "solo_individual", "coinflip", "frequency"):
            check(
                np.array_equal(child_predictions[name], parent_predictions[name]),
                f"{seed} fixed prediction {name}",
            )
        for arm in ("solo", "distilled", "oracle"):
            check(
                np.array_equal(child_predictions[arm], child_predictions[arm + "_population"]),
                f"{seed} primary alias {arm}",
            )
        check(
            np.array_equal(
                soft["oracle"], budget.oracle_training_logits(data["true_prob"], data["mask_train"]).numpy()
            ),
            f"{seed} oracle targets",
        )
        for a, v in soft.items():
            check(
                np.isfinite(v).all() and (not np.count_nonzero(v[~data["mask_train"]])),
                f"{seed} train-only finite targets {a}",
            )
        for stage, pred, stage_report in [
            ("baseline", parent_predictions, parent_report),
            ("extended", child_predictions, report),
        ]:
            diagnostic = data["mask_test"] & (
                np.abs(data["true_prob"] - data["reference_prob"]) > base.diagnostic_threshold
            )
            for name, qraw in pred.items():
                q = np.clip(qraw.astype(np.float64), 1e-12, 1 - 1e-12)
                truth = np.clip(data["true_prob"], 1e-12, 1 - 1e-12)
                loss = np.where(data["choice"] == 1, -np.log(q), -np.log1p(-q))
                kl = np.maximum(
                    truth * (np.log(truth) - np.log(q)) + (1 - truth) * (np.log1p(-truth) - np.log1p(-q)), 0
                )
                metric_values = {key: [] for key in ["nll", "kl", "diagnostic_nll", "diagnostic_kl"]}
                for i in range(len(q)):
                    for key, values, mask in [
                        ("nll", loss, data["mask_test"]),
                        ("kl", kl, data["mask_test"]),
                        ("diagnostic_nll", loss, diagnostic),
                        ("diagnostic_kl", kl, diagnostic),
                    ]:
                        val = float(values[i, mask[i]].mean()) if mask[i].any() else None
                        metric_values[key].append(val)
                        close(
                            val,
                            stage_report["per_subject"][i]["models"][name][key],
                            f"{seed} {stage} {name} {key} subject {i}",
                        )
                for group, flag in [("rare", True), ("typical", False)]:
                    group_metrics = stage_report["groups"][group]
                    indices = np.flatnonzero(data["rare"] == flag)
                    for key, values in metric_values.items():
                        supported = [values[i] for i in indices if values[i] is not None]
                        val = float(np.mean(supported)) if supported else None
                        close(
                            val,
                            group_metrics["models"][name][key],
                            f"{seed} {stage} {name} {key} group {group}",
                        )
                    check(
                        group_metrics["subject_count"] == len(indices)
                        and group_metrics["test_count"] == int(data["mask_test"][indices].sum())
                        and (group_metrics["diagnostic_count"] == int(diagnostic[indices].sum())),
                        f"{seed} support {group}",
                    )
                    check(
                        group_metrics["models"][name]["diagnostic_subject_count"]
                        == int(np.any(diagnostic[indices], axis=1).sum()),
                        f"{seed} support subjects {group}",
                    )
                score = np.mean([loss[i, data["mask_checkpoint"][i]].mean() for i in range(len(q))])
                close(
                    score, stage_report["checkpoint_scores"][name], f"{seed} {stage} checkpoint score {name}"
                )
            if stage == "extended":
                seed_record["raw_groups"] = stage_report["groups"]
        incremental = 0
        parent_seconds = 0
        for arm in ("teacher", "solo", "distilled", "oracle"):
            inherited = torch.load(parent / f"{arm}.pt", map_location="cpu", weights_only=True)
            parent_seconds += inherited["elapsed_seconds"]
            versions = [("baseline", inherited, base)]
            if arm in ("distilled", "oracle"):
                child = torch.load(folder / f"{arm}.pt", map_location="cpu", weights_only=True)
                versions.append(
                    ("extended", child, dataclasses.replace(base, student_steps=config["student_steps"]))
                )
                incremental += child["elapsed_seconds"] - inherited["elapsed_seconds"]
                check(child["signature"] == inherited["signature"], f"{seed} {arm} inherited signature")
                check(
                    child["history"][: len(inherited["history"])] == inherited["history"],
                    f"{seed} {arm} history prefix",
                )
                check(
                    torch.all(child["best_val"] <= inherited["best_val"]).item(),
                    f"{seed} {arm} cumulative individual values",
                )
                check(
                    child["population_best_val"] <= inherited["population_best_val"],
                    f"{seed} {arm} cumulative population value",
                )
                seed_record["arms"][arm] = {
                    "selected_step_before": inherited["population_best_step"],
                    "selected_step_after": child["population_best_step"],
                    "individual_selected_after_baseline": int(
                        (child["best_step"] > base.student_steps).sum()
                    ),
                    "individual_selected_at_limit": int(
                        (child["best_step"] == config["student_steps"]).sum()
                    ),
                    "convergence": report["convergence"][arm],
                    "soft_objective": report["soft_objective"][arm],
                    "current_validation_before": inherited["history"][-1]["validation_nll"],
                    "current_validation_after": child["history"][-1]["validation_nll"],
                }
                check(
                    child["completed_steps"] == config["student_steps"] and child["early_stopped"] == False,
                    f"{seed} {arm} completed budget",
                )
                for field, key in [
                    ("history", "history"),
                    ("best_step", "best_step"),
                    ("completed_steps", "completed_steps"),
                    ("best_validation_nll", "best_val"),
                    ("population_best_step", "population_best_step"),
                    ("population_best_validation_nll", "population_best_val"),
                ]:
                    val = child[key].tolist() if isinstance(child[key], torch.Tensor) else child[key]
                    check(val == report["phases"][arm][field], f"{seed} {arm} report {field}")
            for stage, saved, stage_config in versions:
                budget._finite(saved, f"{seed} {arm} {stage}")
                model = budget._model(stage_config, x.shape[-1], arm)
                try:
                    budget._validate_checkpoint(
                        saved,
                        model,
                        x,
                        y,
                        {
                            "train": data["mask_train"],
                            "checkpoint": data["mask_checkpoint"],
                            "solo_l2": config["parent_config"]["solo_l2"],
                        },
                        torch.tensor(soft[arm]) if arm in soft else None,
                        stage_config,
                        arm,
                        device="cpu",
                    )
                    valid = True
                except Exception as exc:
                    valid = str(exc)
                    issues.append(f"{seed} {arm} {stage} CPU validation: {exc}")
                for mode, key in (
                    [("individual", "best_state")]
                    if arm == "teacher"
                    else [("individual", "best_state"), ("population", "population_best_state")]
                ):
                    model.load_state_dict(saved[key])
                    model.eval()
                    with torch.no_grad():
                        logits = model(x)
                        prob = torch.sigmoid(logits).numpy()
                    name = arm if arm == "teacher" else arm + "_" + mode
                    expected_probability = (parent_predictions if stage == "baseline" else child_predictions)[
                        name
                    ]
                    diff = np.abs(prob - expected_probability)
                    item = {
                        "seed": seed,
                        "stage": stage,
                        "arm": arm,
                        "mode": mode,
                        "structural_and_nll_validation": valid,
                        "max_abs_probability_error": float(diff.max()),
                        "values_outside_5e_6": int(np.sum(diff > 5e-06)),
                    }
                    reconstructions.append(item)
                    if np.any(diff > 5e-06):
                        issues.append(
                            f"{seed} {stage} {name} CPU probability discrepancy {diff.max():.10g} ({np.sum(diff > 5e-06)} values)"
                        )
                if arm in ("distilled", "oracle"):
                    objective = budget._objective_snapshots(
                        saved, model, x, y, torch.tensor(data["mask_train"]), torch.tensor(soft[arm])
                    )
                    for k, v in objective.items():
                        close(
                            v,
                            report["soft_objective"][arm][stage][k],
                            f"{seed} {stage} {arm} soft objective {k}",
                            5e-07,
                        )
            close(
                inherited["elapsed_seconds"],
                snapshot["checkpoint_elapsed_seconds"][str(seed)][arm],
                f"{seed} {arm} parent elapsed",
            )
        close(incremental, report["timing"]["incremental_training_seconds"], f"{seed} incremental time")
        close(parent_seconds, report["timing"]["parent_training_seconds"], f"{seed} parent time")
        seed_record["incremental_training_seconds"] = incremental
        seed_record["parent_training_seconds"] = parent_seconds
        for group_name in summary["groups"]:
            summary_row = next((v for v in summary["groups"][group_name]["per_seed"] if v["seed"] == seed))
            for label in ("subject_count", "test_count", "diagnostic_count"):
                check(
                    summary_row[label] == report["groups"][group_name][label],
                    f"{seed} summary support {group_name} {label}",
                )
            check(
                summary_row["diagnostic_subject_count"]
                == report["groups"][group_name]["models"]["teacher"]["diagnostic_subject_count"],
                f"{seed} summary diagnostic people {group_name}",
            )
            check(
                summary_row["baseline_models"] == parent_report["groups"][group_name]["models"]
                and summary_row["extended_models"] == report["groups"][group_name]["models"],
                f"{seed} summary original scores {group_name}",
            )
        expected_cost = budget.t4_training_cost_proxy(
            incremental, base.t4_units_per_hour, manifest["sessions"]
        )
        close(
            report["timing"]["t4_equivalent_units_at_user_rate"],
            expected_cost["t4_equivalent_units_at_user_rate"],
            f"{seed} t4 accounting",
        )
        print("completed", seed, flush=True)
    verify_summary(summary)
    close(
        sum((seed_record["incremental_training_seconds"] for seed_record in rows)),
        summary["timing"]["incremental_training_seconds"],
        "aggregate incremental time",
    )
    close(
        sum((seed_record["parent_training_seconds"] for seed_record in rows)),
        summary["timing"]["parent_training_seconds"],
        "aggregate parent time",
    )
    hashes_unchanged = budget._file_hashes(folder_root) == before
    check(hashes_unchanged, "all result bytes unchanged")
    per_seed_units = [
        read_json(folder_root / f"seed-{seed}" / "report.json")["timing"]["t4_equivalent_units_at_user_rate"]
        for seed in summary["planned_seeds"]
    ]
    total_units = sum(per_seed_units) if all((value is not None for value in per_seed_units)) else None
    close(total_units, summary["timing"]["t4_equivalent_units_at_user_rate"], "summary T4 units")
    record.update(
        check_count=count,
        hashes_unchanged=hashes_unchanged,
        summary=summary,
        completed=True,
        status="passed" if not issues else "failed",
        hash_counts={
            "parent_snapshot_listed": len(snapshot["file_sha256"]),
            "child_completed_files": sum(
                (len(read_json(f)["completed_file_sha256"]) for f in folder_root.glob("seed-*/manifest.json"))
            ),
            "original_parent_comparisons": len(snapshot["file_sha256"]) if original is not None else 0,
            "total_source_files": len(before),
        },
        maximum_cpu_probability_error=max(
            (seed_record["max_abs_probability_error"] for seed_record in reconstructions)
        ),
        raw_metric_tolerance=1e-12,
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    pathlib.Path(handle.name).replace(args.output)
    print(json.dumps({"checks": count, "issues": issues, "output": str(args.output)}, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
