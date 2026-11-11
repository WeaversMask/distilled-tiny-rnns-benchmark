"""Execute notebook 06's five actual cells in a fresh CPU kernel.

Working-checkout verification is useful before release; --pinned-archive checks
the exact released local Git commit without a token or network download.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl")


def prepare_archived_project(destination):
    """Safely extract the immutable local Git archive and check its source hash."""
    settings = json.loads((ROOT / "configs/colab_budget_source.json").read_text())
    commit = settings["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The budget-check runtime requires a full Git commit")
    archive = subprocess.run(["git", "archive", "--format=zip", "--prefix=project/", commit],
                             cwd=ROOT, check=True, capture_output=True).stdout
    spec = importlib.util.spec_from_file_location("budget_bootstrap", ROOT / "scripts/colab_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    project = bootstrap._extract_project_archive(archive, destination)
    digest = hashlib.sha256()
    for path in sorted((project / "src/distilled_rnns").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != settings["source_sha256"]:
        raise ValueError("Archived budget-check source does not match its frozen fingerprint")
    return project, settings


def _check_difference(actual, left, right, label):
    if left is None or right is None:
        if actual is not None:
            raise AssertionError("Unavailable support was converted to a number: " + label)
    elif actual is None or not math.isfinite(actual) or not math.isclose(actual, left - right, rel_tol=0, abs_tol=1e-10):
        raise AssertionError("Wrong direction or value: " + label)


def _verify_aggregate(aggregate, values, label, directions=False):
    available = [value for value in values if value is not None]
    if aggregate["supported_seed_count"] != len(available) or aggregate["planned_seed_count"] != len(values):
        raise AssertionError("Aggregate support disagrees with per-seed support: " + label)
    for field, expected in (("mean", sum(available) / len(available) if available else None),
                            ("min", min(available) if available else None),
                            ("max", max(available) if available else None)):
        actual = aggregate[field]
        if expected is None:
            if actual is not None:
                raise AssertionError("Unavailable aggregate was converted to a number: " + label)
        elif actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise AssertionError("Aggregate value disagrees with per-seed values: " + label)
    if directions:
        expected_counts = {"positive_seed_count": sum(value > 0 for value in available),
                           "negative_seed_count": sum(value < 0 for value in available),
                           "tied_seed_count": sum(value == 0 for value in available)}
        if any(aggregate[field] != expected for field, expected in expected_counts.items()):
            raise AssertionError("Aggregate directions disagree with per-seed values: " + label)


def verify_summary(summary):
    """Check all paired changes, original scores, support and incremental compute."""
    if summary["status"] != "complete":
        raise AssertionError("Verification study did not finish")
    if summary["pending_seeds"] or summary["completed_seeds"] != summary["planned_seeds"]:
        raise AssertionError("The summary omitted a planned seed")
    if set(summary["groups"]) != {"rare", "typical"}:
        raise AssertionError("Both rare and typical groups must be reported")
    for group in summary["groups"].values():
        if [row["seed"] for row in group["per_seed"]] != summary["planned_seeds"]:
            raise AssertionError("The per-seed report omitted a planned seed")
        for row in group["per_seed"]:
            for mode in ("population", "individual"):
                for metric in METRICS:
                    gain = metric + "_gain"
                    for stage in ("baseline", "extended"):
                        _check_difference(row[f"{stage}_target_comparisons"][mode][gain],
                                          row[f"{stage}_models"][f"distilled_{mode}"][metric],
                                          row[f"{stage}_models"][f"oracle_{mode}"][metric], stage + " target-source gain")
                    _check_difference(row["paired_gain_change"][mode][gain],
                                      row["extended_target_comparisons"][mode][gain],
                                      row["baseline_target_comparisons"][mode][gain], "paired gain change")
                    for arm in ("distilled", "oracle"):
                        _check_difference(row["arm_changes"][mode][arm][metric],
                                          row["extended_models"][f"{arm}_{mode}"][metric],
                                          row["baseline_models"][f"{arm}_{mode}"][metric], "arm error change")
            for model in ("teacher", "solo", "solo_population", "solo_individual", "coinflip", "frequency"):
                if row["baseline_models"][model] != row["extended_models"][model]:
                    raise AssertionError("A fixed reference changed: " + model)
        for mode in ("population", "individual"):
            for section in ("baseline_target_comparisons", "extended_target_comparisons", "paired_gain_change"):
                for metric in METRICS:
                    key = metric + "_gain"
                    _verify_aggregate(group[section][mode][key],
                                      [row[section][mode][key] for row in group["per_seed"]],
                                      f"{section}.{mode}.{key}", directions=True)
            for arm in ("distilled", "oracle"):
                for metric in METRICS:
                    _verify_aggregate(group["arm_changes"][mode][arm][metric],
                                      [row["arm_changes"][mode][arm][metric] for row in group["per_seed"]],
                                      f"arm_changes.{mode}.{arm}.{metric}", directions=True)
        for section in ("baseline_models", "extended_models"):
            for model, metrics in group[section].items():
                for metric in METRICS:
                    _verify_aggregate(metrics[metric], [row[section][model][metric] for row in group["per_seed"]],
                                      f"{section}.{model}.{metric}")
        for key in ("subject_count", "test_count", "diagnostic_count", "diagnostic_subject_count"):
            _verify_aggregate(group["support"][key], [row[key] for row in group["per_seed"]], key)
    timing = summary["timing"]
    for name in ("incremental_training_seconds", "parent_training_seconds"):
        if not math.isfinite(timing[name]) or timing[name] < 0:
            raise AssertionError("Invalid training time: " + name)
    units = timing["t4_equivalent_units_at_user_rate"]
    if units is not None and (not math.isfinite(units) or units < 0):
        raise AssertionError("Invalid incremental compute-unit proxy")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-archive", action="store_true",
                        help="Use the exact pinned runtime from local Git")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/06_colab_budget_check.executed.ipynb")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="distilled-budget-notebook-") as temporary:
        # macOS commonly exposes the temporary directory through /var ->
        # /private/var. Artifact validation intentionally rejects symlink paths.
        root = Path(temporary).resolve()
        settings = json.loads((ROOT / "configs/colab_budget_source.json").read_text())
        project, settings = prepare_archived_project(root / "project") if args.pinned_archive else (ROOT, settings)
        if not (project / "configs/budget_check_smoke.json").is_file():
            raise RuntimeError("The selected runtime lacks budget_check_smoke.json; pin the reviewed budget-check runtime before archive verification.")
        kernel_folder = root / "kernels/python3"
        kernel_folder.mkdir(parents=True)
        (kernel_folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Budget-check verification", "language": "python",
        }))
        environment = dict(os.environ)
        environment.update({"DISTILLED_PROJECT_ROOT": str(project), "DISTILLED_OUTPUT_ROOT": str(root / "runs"),
                            "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(root / "matplotlib"),
                            "IPYTHONDIR": str(root / "ipython"), "JUPYTER_RUNTIME_DIR": str(root / "jupyter")})
        manager = KernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root / "kernels")]))
        notebook = nbformat.read(ROOT / "notebooks/06_colab_budget_check.ipynb", as_version=4)
        # No code substitutions: all setup, smoke-parent creation, child training and reporting are real.
        NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(project)}}).execute(env=environment)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        if counts != [1, 2, 3, 4, 5]:
            raise AssertionError(f"Expected all five actual code cells, got {counts}")
        for run_name in ("budget-check-smoke-" + settings["source_sha256"][:12], "budget-check-v1"):
            summary = json.loads((root / "runs" / run_name / "summary.json").read_text())
            verify_summary(summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        print(json.dumps({"executed_notebook": str(args.output.resolve()), "code_cells": len(counts),
                          "execution_counts": counts, "configuration": "budget_check_smoke", "device": "cpu",
                          "runtime": "pinned archive" if args.pinned_archive else "working checkout",
                          "runtime_commit": settings["commit"] if args.pinned_archive else None,
                          "runtime_source_sha256": settings["source_sha256"],
                          "real_parent_smoke_resume_and_study_complete": True}, indent=2))


if __name__ == "__main__":
    main()
