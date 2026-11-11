"""Rebuild the frozen first-pilot launcher and a current-working-tree source ZIP.

The optional ZIP records current files; it is not the pinned first-pilot runtime.
The launcher always downloads the immutable revision in colab_source.json.
"""
from pathlib import Path
import hashlib
import json
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
github_source = json.loads((ROOT / "configs/colab_source.json").read_text())
bootstrap_source = (ROOT / "scripts/colab_bootstrap.py").read_text()
source_hash = github_source["source_sha256"]
if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
    raise ValueError("colab_source.json must contain the frozen first-pilot source SHA256")

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
code = lambda text: cells.append(nbf.v4.new_code_cell(text))

md("""# Distilled Tiny RNNs — first Colab pilot

**Purpose:** verify the pipeline on your GPU and measure its runtime. This is a short synthetic first-stage, parametric rare-strategy pilot. It is not the proposal's mechanistic gate, a convergence study, or a reproduction of the published method.

Open this notebook in a fresh Colab session and select **Runtime → Change runtime type → T4 GPU**. The first cell downloads a fixed public revision without a GitHub token. Mount your own Google Drive to save results and checkpoints, then run the cells in order.

The notebook runs verification tests before training.

The pilot trains 1,000 synthetic people with 100 trials each. It uses 100 teacher steps and 150 steps per student arm, with a 10-minute limit per phase checked between steps. A phase that hits its time limit is saved as incomplete and resumes when rerun. Setup, checkpoints, and evaluation add time beyond training caps.

Keep the same notebook revision and run name to resume after a disconnect. Use a new run name when settings or code change. Use one runtime per output folder. Disconnect the GPU runtime when finished.
""")
code(bootstrap_source + '\n\n' + '''from pathlib import Path
import os, sys, hashlib, json, subprocess, tempfile
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

EXPECTED_SOURCE_SHA256 = "''' + source_hash + '''"
GITHUB_REPOSITORY = "''' + github_source["repository"] + '''"
PINNED_COMMIT = "''' + github_source["commit"] + '''"
if IN_COLAB:
    destination = Path(tempfile.mkdtemp(prefix="distilled-source-")) / "project"
    PROJECT = fetch_public_project(GITHUB_REPOSITORY, PINNED_COMMIT, destination)
    drive.mount("/content/drive")
    DRIVE_PROJECT = Path("/content/drive/MyDrive/distilled-tiny-rnns")
    RUNS = DRIVE_PROJECT / "runs"
else:
    PROJECT = Path(os.environ.get("DISTILLED_PROJECT_ROOT", Path.cwd())).resolve()
    if not (PROJECT / "pyproject.toml").exists() and (PROJECT.parent / "pyproject.toml").exists():
        PROJECT = PROJECT.parent
    RUNS = Path(os.environ.get("DISTILLED_OUTPUT_ROOT", str(PROJECT / "outputs/notebook")))

digest = hashlib.sha256()
for path in sorted((PROJECT / "src/distilled_rnns").glob("*.py")):
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
assert digest.hexdigest() == EXPECTED_SOURCE_SHA256, "Downloaded source differs from the verified notebook revision."
os.chdir(PROJECT)
RUNS.mkdir(parents=True, exist_ok=True)
print("Project:", PROJECT)
print("Saved runs:", RUNS)
''')
md("""## Install and verify

Installation keeps a compatible preinstalled PyTorch version. Exact installed versions are saved with every run. Tests cover the simulator, causal inputs, test masks, student independence, standard-GRU agreement, and checkpoint resume. GPU-specific checks run when CUDA is present; local CPU verification cannot replace these.
""")
code('''if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"], check=True)
# Editable installs create a startup path file; this kernel is already running.
# Activate the verified source immediately in both Colab and local kernels.
import importlib
source_dir = str((PROJECT / "src").resolve())
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)
importlib.invalidate_caches()
from distilled_rnns.config import ExperimentConfig
print("Project import: ready")
import torch
print("PyTorch:", torch.__version__)
if IN_COLAB and not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before continuing; this notebook will not silently run the Colab pilot on CPU.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU (local verification only)")
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
''')
md("""## Small smoke test

Twenty people and four training steps per model check that your runtime can complete the whole pipeline. The test runs on the selected GPU in Colab, saves its outputs, and verifies that a second invocation resumes successfully.
""")
code('''from distilled_rnns.config import ExperimentConfig
from distilled_rnns.experiment import run_experiment

SMOKE_RUN = RUNS / ("smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = ExperimentConfig.from_file(PROJECT / "configs/smoke.json")
smoke = run_experiment(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
smoke_resumed = run_experiment(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["per_subject"] == smoke["per_subject"], "Resumed predictions differ"
print("Smoke test and resume passed.")
''')
md("""## Bounded pilot

The default configuration is in `configs/pilot.json`. It has a 2% rare group whose model-based weight is zero; the majority uses hybrid strategies. The teacher pools everyone; each student has independent weights. Solo and distilled students start from identical weights. Validation chooses each student's weights separately.

The notebook's short runs may be underfit. Read the losses and timings before deciding how long to train the scientific experiment. **Do not infer an effect or a null from this pilot.**
""")
code('''RUN_NAME = "pilot-v1"  # Keep for resume; change for a new configuration or source revision.
CONFIG_NAME = "pilot" if IN_COLAB else os.environ.get("DISTILLED_NOTEBOOK_CONFIG", "smoke")
config = ExperimentConfig.from_file(PROJECT / "configs" / f"{CONFIG_NAME}.json")
print(json.dumps(config.to_dict(), indent=2))
report = run_experiment(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
''')
md("""## Results and next decision

NLL measures prediction of held-out first-stage choices; lower is better. KL measures disagreement with the known generating policy on the same histories; lower is better. Diagnostic scores include only trials where true and reference policies disagree by more than 0.2, and are `null` if no such test trials exist.

On the plot, positive x means distillation predicts better than solo training; positive y means its policy fidelity is worse. These are descriptive points from one short run. They do not establish statistical significance or strategy recovery on new histories.

The timing report converts training wall time at your rate only if every session in this run used a T4. Mixed-device histories receive no cost estimate. It excludes setup, evaluation, and idle billing. Before expanding the experiment, inspect convergence, GPU memory, diagnostic support, and actual Colab unit consumption. The full grid and scientific gate require additional implementations and validation described in `docs/roadmap.md`.
""")
code('''if report["status"] == "complete":
    from IPython.display import display, Image
    print(json.dumps({"groups": report["groups"], "timing": report["timing"]}, indent=2))
    display(Image(filename=str(Path(report["output_dir"]) / "quadrant.png")))
    print("Saved:", report["output_dir"])
    print("Keep the entire run folder, including report.json, manifest.json, training.json, data.npz, predictions.npz, environments/, and the three .pt checkpoints.")
else:
    print("Pilot incomplete; resume before interpreting metrics.")
''')

for index, cell in enumerate(cells):
    cell["id"] = f"pilot-cell-{index:02d}"

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "01_colab_pilot.ipynb", "provenance": []},
    "accelerator": "GPU",
})
nbf.validate(notebook)
notebook_path = ROOT / "notebooks/01_colab_pilot.ipynb"
nbf.write(notebook, notebook_path)

bundle = ROOT / "artifacts/colab_project.zip"
bundle.parent.mkdir(exist_ok=True)
paths = [ROOT / name for name in ("README.md", "AGENTS.md", "pyproject.toml")]
for directory in ("src", "tests", "configs", "docs", "notebooks", "review", "scripts"):
    paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix != ".pyc"
                 and not any(part.endswith(".egg-info") for part in path.parts))
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        archive.write(path, path.relative_to(ROOT).as_posix())
bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
(ROOT / "artifacts/colab_project.sha256").write_text(f"{bundle_hash}  colab_project.zip\n")
print(json.dumps({"notebook": str(notebook_path), "bundle": str(bundle), "source_sha256": source_hash, "bundle_sha256": bundle_hash, "bundle_scope": "current working tree; not the pinned first-pilot runtime"}, indent=2))
