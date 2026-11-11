"""Execute notebook 03's actual five cells in a fresh CPU kernel.

The default verifies the released runtime from local Git objects;
this requires neither a GitHub token nor a network download.
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
    """Extract exactly the local pinned Git commit and verify its source hash."""
    settings = json.loads((ROOT / "configs/colab_baseline_source.json").read_text())
    commit = settings["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The baseline_audit runtime requires a full Git commit")
    archive = subprocess.run(
        ["git", "archive", "--format=zip", "--prefix=project/", commit],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    spec = importlib.util.spec_from_file_location("baseline_audit_bootstrap", ROOT / "scripts/colab_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    project = bootstrap._extract_project_archive(archive, destination)
    digest = hashlib.sha256()
    for path in sorted((project / "src/distilled_rnns").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != settings["source_sha256"]:
        raise ValueError("Archived baseline_audit source does not match its frozen fingerprint")
    return project, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--pinned-archive", dest="pinned_archive", action="store_true", default=True,
                         help="Use the frozen runtime from local Git (default)")
    runtime.add_argument("--working-checkout", dest="pinned_archive", action="store_false",
                         help="Use the checkout only if it still matches the frozen source fingerprint")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/03_colab_baseline_audit.executed.ipynb",
                        help="Where to save the executed notebook")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="distilled-baseline_audit-notebook-") as temporary:
        root = Path(temporary)
        settings = json.loads((ROOT / "configs/colab_baseline_source.json").read_text())
        if args.pinned_archive:
            project, settings = prepare_archived_project(root / "project")
        else:
            project = ROOT
        if not (project / "configs/baseline_audit_smoke.json").is_file():
            raise RuntimeError("The selected runtime lacks baseline_audit_smoke.json; pin the reviewed baseline_audit runtime before archive verification.")
        kernel_folder = root / "kernels/python3"
        kernel_folder.mkdir(parents=True)
        (kernel_folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Baseline audit verification", "language": "python",
        }))
        environment = dict(os.environ)
        environment.update({
            "DISTILLED_PROJECT_ROOT": str(project),
            "DISTILLED_OUTPUT_ROOT": str(root / "runs"),
            "DISTILLED_BASELINE_CONFIG": "baseline_audit_smoke",
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(root / "matplotlib"),
            "IPYTHONDIR": str(root / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(root / "jupyter"),
        })
        manager = KernelManager(kernel_name="python3",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root / "kernels")]))
        # Run the shipped cells without injecting replacements for setup or training.
        notebook = nbformat.read(ROOT / "notebooks/03_colab_baseline_audit.ipynb", as_version=4)
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={"metadata": {"path": str(project)}})
        client.execute(env=environment)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        if counts != [1, 2, 3, 4, 5]:
            raise AssertionError(f"Expected all five actual code cells, got {counts}")
        for run_name in ("baseline-audit-smoke-" + settings["source_sha256"][:12], "baseline-audit-v1"):
            summary = json.loads((root / "runs" / run_name / "summary.json").read_text())
            if summary["status"] != "complete":
                raise AssertionError(f"Verification run {run_name} did not finish")
            for group in summary["groups"].values():
                for section in ("comparisons", "matched_comparisons"):
                    if not all("diagnostic_fidelity_loss" in metrics for metrics in group[section].values()):
                        raise AssertionError("The actual runtime omitted diagnostic KL loss")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)
        print(json.dumps({
            "executed_notebook": str(args.output.resolve()), "code_cells": len(counts),
            "execution_counts": counts, "configuration": "baseline_audit_smoke", "device": "cpu",
            "runtime": "pinned archive" if args.pinned_archive else "working checkout",
            "runtime_commit": settings["commit"] if args.pinned_archive else None,
            "runtime_source_sha256": settings["source_sha256"],
            "smoke_and_study_complete": True,
        }, indent=2))


if __name__ == "__main__":
    main()
