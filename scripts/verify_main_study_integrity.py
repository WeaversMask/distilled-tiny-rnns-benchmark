"""Read-only integrity/trajectory audit of notebook07 downloads. Never runs fitting."""
from pathlib import Path
import argparse, hashlib, json, math, sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]
from main_study.config import MainStudyConfig
from main_study import runner
from main_study.simulation import simulate_population
from distilled_rnns.evaluation import make_trial_masks
from distilled_rnns.oracle_control import oracle_training_logits
from distilled_rnns.experiment import t4_training_cost_proxy

EXPECTED_SOURCE = "b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def fingerprints(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def audit(folder, mode):
    folder = folder.resolve()
    runner._reject_symlinks(folder)
    before = fingerprints(folder)
    result = {
        "folder": str(folder),
        "mode": mode,
        "audit_mode": "read-only; no training/resume runner called",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "checks": 0,
        "issues": [],
        "backend_reproduction_issues": [],
        "file_sha256_before": before,
        "populations": [],
    }

    def check(condition, label):
        result["checks"] += 1
        if not bool(condition):
            result["issues"].append(label)

    def close(actual, expected, label, tolerance=1e-10):
        if actual is None or expected is None:
            check(actual is None and expected is None, label)
        else:
            check(math.isfinite(actual) and abs(actual - expected) <= tolerance, label)

    config = MainStudyConfig.from_file(
        PROJECT
        / "configs"
        / ("main_study_first_tranche.json" if mode == "production" else "main_study_smoke.json")
    )
    summary = read(folder / "summary.json")
    manifest = read(folder / "run_manifest.json")
    protocol = read(folder / "protocol.json")
    identity = {
        "schema_version": 1,
        "study": "main_study_first_tranche",
        "config": config.to_dict(),
        "source_sha256": EXPECTED_SOURCE,
    }
    check(runner.main_source_fingerprint() == EXPECTED_SOURCE, "current source matches reviewed runtime")
    check(protocol == {**identity, "design": config.design}, "exact frozen protocol/design")
    check(manifest["identity"] == identity, "root identity")
    check(manifest["status"] == "complete" and summary["status"] == "complete", "root completion")
    check(summary["config"] == config.to_dict(), "summary exact config")
    check(
        summary["planned_populations"] == summary["completed_populations"] == config.plan
        and not summary["pending_populations"],
        "all and only planned populations complete",
    )
    expected_dirs = {f"{p['kind']}/seed-{p['seed']}" for p in config.plan}
    actual_dirs = {p.relative_to(folder).as_posix() for p in folder.glob("*/seed-*")}
    check(actual_dirs == expected_dirs, "exact kind/seed folders")
    root_hash_checks = 0
    for mapping in (manifest["immutable_file_sha256"], manifest["completed_file_sha256"]):
        for name, digest in mapping.items():
            rel = Path(name)
            check(not rel.is_absolute() and ".." not in rel.parts, "safe root hash path " + name)
            check(sha(folder / rel) == digest, "root listed hash " + name)
            root_hash_checks += 1
    protocol_hash = sha(folder / "protocol.json")
    check(
        manifest["immutable_file_sha256"] == {"protocol.json": protocol_hash}, "root immutable protocol list"
    )
    known = [
        entry["wall_seconds"] for entry in manifest["invocations"] if entry.get("wall_seconds") is not None
    ]
    coverage = {"recorded_invocations": len(known), "total_invocations": len(manifest["invocations"])}
    check(summary["timing"]["invocation_wall_time_coverage"] == coverage, "root wall coverage")
    close(summary["timing"]["known_runner_wall_seconds"], sum(known), "root known wall seconds")
    close(
        summary["timing"]["runner_wall_seconds"],
        sum(known) if len(known) == len(manifest["invocations"]) else None,
        "root wall seconds",
    )
    check(manifest["invocations"][-1]["status"] == "complete", "final invocation complete")
    checkpoint_seconds = 0.0
    unit_values = []
    max_probability = 0.0
    max_probe = 0.0
    max_validation = 0.0
    max_mean_validation = 0.0
    max_person_validation = 0.0
    package_count = 0
    for item in config.plan:
        child = runner.population_path(folder, **item)
        cm = read(child / "manifest.json")
        report = read(child / "report.json")
        training = read(child / "training.json")
        row = {
            "kind": item["kind"],
            "seed": item["seed"],
            "sessions": cm["sessions"],
            "arms": {},
            "neural_reproduction": [],
        }
        result["populations"].append(row)
        check(
            cm["status"] == "complete" and cm["identity"] == {**item, "protocol_sha256": protocol_hash},
            str(item) + " child identity/status",
        )
        check(
            set(cm["completed_file_sha256"]) == set(runner.POPULATION_FILES),
            str(item) + " complete-file coverage",
        )
        for mapping in (cm["immutable_file_sha256"], cm["completed_file_sha256"]):
            for name, digest in mapping.items():
                check(sha(child / name) == digest, str(item) + " child listed hash " + name)
        check(
            report["seed"] == item["seed"]
            and report["kind"] == item["kind"]
            and report["status"] == "complete",
            str(item) + " report identity/status",
        )
        check(report["phases"] == training, str(item) + " report/training equality")
        for session in cm["sessions"]:
            inventory = session["package_inventory"]
            check(sha(child / inventory["path"]) == inventory["sha256"], str(item) + " package inventory")
            check(
                session["environment"]["device"] == "cuda" and session["environment"]["gpu"] == "Tesla T4",
                str(item) + " recorded T4 session",
            )
            package_count += 1
        check(
            sha(child / "environment-freeze.txt") == cm["sessions"][-1]["package_inventory"]["sha256"],
            str(item) + " latest freeze",
        )
        check(cm["sessions"][-1]["status"] == "complete", str(item) + " last population session complete")
        session_times = [s["wall_seconds"] for s in cm["sessions"] if s.get("wall_seconds") is not None]
        close(
            report["timing"]["known_session_wall_seconds"],
            sum(session_times),
            str(item) + " known session wall",
        )
        close(
            report["timing"]["session_wall_seconds"],
            sum(session_times) if len(session_times) == len(cm["sessions"]) else None,
            str(item) + " session wall",
        )
        check(
            report["timing"]["session_wall_time_coverage"]
            == {"recorded_sessions": len(session_times), "total_sessions": len(cm["sessions"])},
            str(item) + " session wall coverage",
        )
        data = arrays(child / "data.npz")
        prediction = arrays(child / "predictions.npz")
        probes = arrays(child / "probe_predictions.npz")
        soft = arrays(child / "soft_training_logits.npz")
        generated = simulate_population(config, item["kind"], item["seed"])
        masks = make_trial_masks(config.n_subjects, config.n_trials, item["seed"] + 1)
        generated.update({f"mask_{name}": mask for name, mask in masks.items()})
        check(set(data) == set(generated), str(item) + " generated keys")
        for key, expected in generated.items():
            actual = data[key]
            check(
                actual.shape == expected.shape and actual.dtype == expected.dtype,
                str(item) + " regenerated shape/dtype " + key,
            )
            if key in ("true_prob", "reference_prob"):
                error = float(np.abs(actual - expected).max())
                check(error <= 1e-12, str(item) + " regenerated " + key)
            else:
                check(np.array_equal(actual, expected), str(item) + " regenerated exact " + key)
        check(
            np.isfinite(data["x"]).all() and np.all(data["x"][:, 0] == 0),
            str(item) + " finite/zero initial causal inputs",
        )
        for name in ("predictions", "probes"):
            values = prediction if name == "predictions" else probes
            for key, value in values.items():
                check(
                    np.isfinite(value).all() and np.logical_and(value >= 0, value <= 1).all(),
                    str(item) + f" finite bounded {name} " + key,
                )
        check(set(soft) == {"oracle", "distilled"}, str(item) + " exact soft-target arms")
        train = torch.as_tensor(data["mask_train"])
        validation = torch.as_tensor(data["mask_val"])
        x = torch.as_tensor(data["x"])
        y = torch.as_tensor(data["choice"], dtype=torch.float32)
        check(
            not np.any(data["mask_train"] & data["mask_val"])
            and not np.any(data["mask_train"] & data["mask_test"])
            and not np.any(data["mask_val"] & data["mask_test"]),
            str(item) + " disjoint masks",
        )
        check(
            np.all(data["mask_train"] | data["mask_val"] | data["mask_test"]), str(item) + " exhaustive masks"
        )
        check(
            np.array_equal(
                soft["oracle"], oracle_training_logits(data["true_prob"], data["mask_train"]).numpy()
            ),
            str(item) + " oracle targets",
        )
        for key, value in soft.items():
            check(
                np.isfinite(value).all() and not np.count_nonzero(value[~data["mask_train"]]),
                str(item) + " finite train-only " + key,
            )
        seconds = 0.0
        for arm in runner.ARMS:
            saved = torch.load(child / (arm + ".pt"), map_location="cpu", weights_only=True)
            model = runner._new_model(config, arm)
            try:
                runner._checkpoint(
                    config,
                    arm,
                    item["seed"],
                    child / (arm + ".pt"),
                    model,
                    x,
                    y,
                    train,
                    validation,
                    torch.as_tensor(soft[arm]) if arm in soft else None,
                )
                checkpoint_validation = "passed"
            except Exception as error:
                checkpoint_validation = str(error)
                result["backend_reproduction_issues"].append(
                    {
                        "population": item,
                        "arm": arm,
                        "check": "reviewed checkpoint validator",
                        "error": str(error),
                    }
                )
            # Structural and counter checks are repeated independently even if backend reconstruction differs.
            runner._finite(saved, str(child / (arm + ".pt")))
            limit = config.teacher_steps if arm == "teacher" else config.student_steps
            completed = saved["completed_steps"]
            check(
                completed == limit or arm == "teacher" and saved["early_stopped"],
                str(item) + " completed " + arm,
            )
            check(
                not saved["early_stopped"] if arm != "teacher" else True,
                str(item) + " fixed student stopping " + arm,
            )
            check(
                all(state["step"].item() == completed for state in saved["optimizer"]["state"].values()),
                str(item) + " Adam steps " + arm,
            )
            for field in ("history", "completed_steps", "elapsed_seconds"):
                check(
                    training[arm][field] == saved[field],
                    str(item) + " report checkpoint " + arm + " " + field,
                )
            check(
                training[arm]["best_step"]
                == (saved["best_step"] if arm == "teacher" else saved["best_step"].tolist()),
                str(item) + " selected steps report " + arm,
            )
            expected_steps = [0] + list(range(config.eval_every, completed + 1, config.eval_every))
            check(
                [entry["step"] for entry in saved["history"]] == expected_steps,
                str(item) + " validation schedule " + arm,
            )
            best = min(saved["history"], key=lambda h: h["validation_nll"])
            check(
                (saved["best_step"] if arm == "teacher" else saved["population_best_step"]) == best["step"],
                str(item) + " earliest cumulative population min " + arm,
            )
            ar = {
                "completed_steps": completed,
                "early_stopped": saved["early_stopped"],
                "population_selected_step": saved["best_step"]
                if arm == "teacher"
                else saved["population_best_step"],
                "individual_selected_at_limit": None
                if arm == "teacher"
                else int((saved["best_step"] == limit).sum()),
                "individual_best_step_min": None if arm == "teacher" else int(saved["best_step"].min()),
                "individual_best_step_median": None
                if arm == "teacher"
                else float(np.median(saved["best_step"].numpy())),
                "individual_best_step_max": None if arm == "teacher" else int(saved["best_step"].max()),
                "training_seconds": saved["elapsed_seconds"],
                "checkpoint_cpu_validation": checkpoint_validation,
                "initial_train_nll": saved["history"][0]["train_nll"],
                "final_train_nll": saved["history"][-1]["train_nll"],
                "initial_validation_nll": saved["history"][0]["validation_nll"],
                "final_validation_nll": saved["history"][-1]["validation_nll"],
                "best_population_validation_nll": best["validation_nll"],
            }
            row["arms"][arm] = ar
            seconds += saved["elapsed_seconds"]
            for suffix, state in (
                [("", "best_state")]
                if arm == "teacher"
                else [("_individual", "best_state"), ("_population", "population_best_state")]
            ):
                model.load_state_dict(saved[state])
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                    actual = torch.sigmoid(logits).numpy()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
                    val = (loss * validation).sum(1) / validation.sum(1)
                name = arm + suffix
                error = float(np.abs(actual - prediction[name]).max())
                max_probability = max(max_probability, error)
                probe_actual = runner.probe_forward(model, config.n_subjects)
                probe_error = float(np.abs(probe_actual - probes[name]).max())
                max_probe = max(max_probe, probe_error)
                chosen = val.mean().item() if arm == "teacher" or suffix == "_population" else val.numpy()
                expected = (
                    saved["best_val"]
                    if arm == "teacher" or suffix == "_individual"
                    else saved["population_best_val"]
                ).numpy()
                verror = float(np.abs(chosen - expected).max())
                max_validation = max(max_validation, verror)
                per_person_check = arm != "teacher" and suffix == "_individual"
                if per_person_check:
                    max_person_validation = max(max_person_validation, verror)
                else:
                    max_mean_validation = max(max_mean_validation, verror)
                check(
                    verror <= (5e-6 if per_person_check else 5e-7),
                    str(item) + " explicit selected-validation tolerance " + name,
                )
                row["neural_reproduction"].append(
                    {
                        "model": name,
                        "maximum_probability_error": error,
                        "probability_count_over_5e_6": int((abs(actual - prediction[name]) > 5e-6).sum()),
                        "maximum_probe_error": probe_error,
                        "probe_count_over_5e_6": int((abs(probe_actual - probes[name]) > 5e-6).sum()),
                        "maximum_validation_nll_error": verror,
                        "validation_is_per_person": per_person_check,
                    }
                )
                if error > 5e-6 or probe_error > 5e-6:
                    result["backend_reproduction_issues"].append(
                        {
                            "population": item,
                            "model": name,
                            "maximum_probability_error": error,
                            "maximum_probe_error": probe_error,
                        }
                    )
                if arm == "teacher":
                    teacher_error = float(
                        abs(logits.numpy()[data["mask_train"]] - soft["distilled"][data["mask_train"]]).max()
                    )
                    ar["teacher_logits_maximum_cpu_difference"] = teacher_error
                    # Save the discrepancy even when logits differ more than probabilities.
            if arm in ("distilled", "oracle"):
                objectives = {}
                for state_key in ("current_state", "best_state", "population_best_state"):
                    model.load_state_dict(saved[state_key])
                    with torch.no_grad():
                        logits = model(x)
                        values = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, torch.sigmoid(torch.as_tensor(soft[arm])), reduction="none"
                        )
                        objectives[state_key] = float(((values * train).sum(1) / train.sum(1)).mean())
                ar["final_soft_target_mean_training_bce"] = objectives
            del model, saved
        close(
            report["timing"]["cumulative_training_seconds"], seconds, str(item) + " checkpoint training time"
        )
        expected_cost = t4_training_cost_proxy(seconds, config.t4_units_per_hour, cm["sessions"])
        close(
            report["timing"]["t4_equivalent_units_at_user_rate"],
            expected_cost["t4_equivalent_units_at_user_rate"],
            str(item) + " T4 cost",
        )
        checkpoint_seconds += seconds
        unit_values.append(expected_cost["t4_equivalent_units_at_user_rate"])
        print(folder.name, item["kind"], item["seed"], "complete", flush=True)
    close(
        summary["timing"]["cumulative_training_seconds"],
        checkpoint_seconds,
        "summary training time counted once",
    )
    close(
        summary["timing"]["t4_equivalent_units_at_user_rate"],
        sum(unit_values) if all(x is not None for x in unit_values) else None,
        "summary T4 training-only proxy",
    )
    after = fingerprints(folder)
    check(after == before, "all source-result bytes unchanged")
    result.update(
        status="passed"
        if not result["issues"] and not result["backend_reproduction_issues"]
        else "issues_found",
        unchanged=after == before,
        root_listed_hash_checks=root_hash_checks,
        package_inventory_checks=package_count,
        file_count=len(before),
        source_sha256=EXPECTED_SOURCE,
        maximum_cpu_probability_reconstruction_error=max_probability,
        maximum_cpu_probe_reconstruction_error=max_probe,
        maximum_cpu_validation_nll_error=max_validation,
        maximum_cpu_population_mean_validation_nll_error=max_mean_validation,
        maximum_cpu_individual_validation_nll_error=max_person_validation,
        timing=summary["timing"],
        root_invocations=manifest["invocations"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--production", type=Path, default=Path.home() / "Downloads/main-study-first-tranche-v1"
    )
    parser.add_argument("--smoke", type=Path, default=Path.home() / "Downloads/main-study-smoke-b5c5aa888e9b")
    args = parser.parse_args()
    for folder in (args.production, args.smoke):
        if args.output.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Audit output must be outside both input folders")
    torch.set_num_threads(1)
    output = {"production": audit(args.production, "production"), "smoke": audit(args.smoke, "smoke")}
    for value in output.values():
        value["verification_script_sha256"] = sha(Path(__file__))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    key: row[key]
                    for key in (
                        "status",
                        "checks",
                        "file_count",
                        "issues",
                        "backend_reproduction_issues",
                        "maximum_cpu_probability_reconstruction_error",
                        "maximum_cpu_probe_reconstruction_error",
                        "maximum_cpu_validation_nll_error",
                    )
                }
                for name, row in output.items()
            },
            indent=2,
        )
    )

    return 1 if any(value["status"] != "passed" for value in output.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
