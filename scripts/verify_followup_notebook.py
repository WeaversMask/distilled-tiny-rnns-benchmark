"""Execute notebook 02 on CPU against its archived, immutable runtime revision."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
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
    """Safely unpack the local pinned Git revision; no network or secret is used."""
    settings = json.loads((ROOT / "configs/colab_followup_source.json").read_text())
    commit = settings["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The follow-up runtime requires a full Git commit")
    archive = subprocess.run(
        ["git", "archive", "--format=zip", "--prefix=project/", commit],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    spec = importlib.util.spec_from_file_location("colab_bootstrap", ROOT / "scripts/colab_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    project = bootstrap._extract_project_archive(archive, destination)
    digest = hashlib.sha256()
    for path in sorted((project / "src/distilled_rnns").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != settings["source_sha256"]:
        raise ValueError("Archived follow-up source does not match its frozen fingerprint")
    return project, settings


def main():
    with tempfile.TemporaryDirectory(prefix="distilled-followup-notebook-") as temporary:
        root = Path(temporary)
        project, settings = prepare_archived_project(root / "project")
        kernel_folder = root / "kernels/python3"
        kernel_folder.mkdir(parents=True)
        (kernel_folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Project verification", "language": "python",
        }))
        environment = dict(os.environ)
        environment.update({
            "DISTILLED_PROJECT_ROOT": str(project),
            "DISTILLED_OUTPUT_ROOT": str(root / "runs"),
            "DISTILLED_FOLLOWUP_CONFIG": "followup_smoke",
            "MPLCONFIGDIR": str(root / "matplotlib"),
            "IPYTHONDIR": str(root / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(root / "jupyter"),
        })
        manager = KernelManager(kernel_name="python3",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root / "kernels")]))
        # Execute the shipped launcher (including its running-kernel import fix)
        # against archived numerical code, rather than its older archived launcher.
        notebook = nbformat.read(ROOT / "notebooks/02_colab_followup.ipynb", as_version=4)
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={"metadata": {"path": str(project)}})
        client.execute(env=environment)
        target = ROOT / "artifacts/02_colab_followup.executed.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        print(json.dumps({"executed_notebook": str(target), "code_cells": len(counts),
                          "execution_counts": counts, "configuration": "followup_smoke", "device": "cpu",
                          "runtime_commit": settings["commit"],
                          "runtime_source_sha256": settings["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
