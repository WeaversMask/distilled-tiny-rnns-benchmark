"""Execute notebook 05's five shipped cells in a fresh CPU kernel.

The default verifies the frozen runtime from the local Git archive;
no token or network is needed. New modules must not change notebook 05's source.
"""
import argparse
import hashlib
import importlib.util
import json
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


def prepare_archived_project(destination):
    """Safely extract the pinned local Git archive and check its package hash."""
    settings = json.loads((ROOT / "configs/colab_oracle_source.json").read_text())
    commit = settings["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The oracle-control runtime requires a full Git commit")
    archive = subprocess.run(
        ["git", "archive", "--format=zip", "--prefix=project/", commit],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    spec = importlib.util.spec_from_file_location("oracle_bootstrap", ROOT / "scripts/colab_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    project = bootstrap._extract_project_archive(archive, destination)
    digest = hashlib.sha256()
    for path in sorted((project / "src/distilled_rnns").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != settings["source_sha256"]:
        raise ValueError("Archived oracle-control source does not match its frozen fingerprint")
    return project, settings


def verify_summary(summary):
    """Reject completed smoke reports that omit the actual paired oracle metrics."""
    if summary["status"] != "complete":
        raise AssertionError("Verification study did not finish")
    if "calibration" in summary:
        raise AssertionError("The oracle-control study must not calibrate or retune")
    if summary["pending_seeds"] or summary["completed_seeds"] != summary["planned_seeds"]:
        raise AssertionError("The summary omitted a planned seed")
    for group in summary["groups"].values():
        if len(group["per_seed"]) != len(summary["planned_seeds"]):
            raise AssertionError("The per-seed report omitted a planned seed")
        for mode in ("population", "individual"):
            if "diagnostic_kl_gain" not in group["target_comparisons"][mode]:
                raise AssertionError("The runtime omitted matched diagnostic ΔKL")
            for row in group["per_seed"]:
                left = row["models"][f"distilled_{mode}"]["diagnostic_kl"]
                right = row["models"][f"oracle_{mode}"]["diagnostic_kl"]
                actual = row["target_comparisons"][mode]["diagnostic_kl_gain"]
                if left is None or right is None:
                    if actual is not None:
                        raise AssertionError("Unavailable diagnostic support was converted to a number")
                elif actual is None or abs(actual - (left - right)) > 1e-10:
                    raise AssertionError("Diagnostic ΔKL has the wrong direction or value")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--pinned-archive", dest="pinned_archive", action="store_true", default=True,
                         help="Use the frozen runtime from local Git (default)")
    runtime.add_argument("--working-checkout", dest="pinned_archive", action="store_false",
                         help="Use the checkout only if it still matches the frozen source fingerprint")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/05_colab_oracle_control.executed.ipynb")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="distilled-oracle-notebook-") as temporary:
        root = Path(temporary).resolve()
        settings = json.loads((ROOT / "configs/colab_oracle_source.json").read_text())
        project, settings = prepare_archived_project(root / "project") if args.pinned_archive else (ROOT, settings)
        if not (project / "configs/oracle_control_smoke.json").is_file():
            raise RuntimeError("The selected runtime lacks oracle_control_smoke.json; pin the reviewed oracle runtime before archive verification.")
        kernel_folder = root / "kernels/python3"
        kernel_folder.mkdir(parents=True)
        (kernel_folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Oracle-control verification", "language": "python",
        }))
        environment = dict(os.environ)
        environment.update({
            "DISTILLED_PROJECT_ROOT": str(project),
            "DISTILLED_OUTPUT_ROOT": str(root / "runs"),
            "DISTILLED_ORACLE_CONFIG": "oracle_control_smoke",
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(root / "matplotlib"),
            "IPYTHONDIR": str(root / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(root / "jupyter"),
        })
        manager = KernelManager(kernel_name="python3",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root / "kernels")]))
        notebook = nbformat.read(ROOT / "notebooks/05_colab_oracle_control.ipynb", as_version=4)
        # No substitutions: setup, training, saved smoke resume and reporting are real.
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={"metadata": {"path": str(project)}})
        client.execute(env=environment)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        if counts != [1, 2, 3, 4, 5]:
            raise AssertionError(f"Expected all five actual code cells, got {counts}")
        for run_name in ("oracle-control-smoke-" + settings["source_sha256"][:12], "oracle-control-v1"):
            summary = json.loads((root / "runs" / run_name / "summary.json").read_text())
            verify_summary(summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        print(json.dumps({
            "executed_notebook": str(args.output.resolve()), "code_cells": len(counts),
            "execution_counts": counts, "configuration": "oracle_control_smoke", "device": "cpu",
            "runtime": "pinned archive" if args.pinned_archive else "working checkout",
            "runtime_commit": settings["commit"] if args.pinned_archive else None,
            "runtime_source_sha256": settings["source_sha256"],
            "smoke_resume_and_study_complete": True,
        }, indent=2))


if __name__ == "__main__":
    main()
