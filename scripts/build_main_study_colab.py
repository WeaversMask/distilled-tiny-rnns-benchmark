"""Build the five-cell, pinned private-repository main-study launcher."""
import hashlib
import json
from pathlib import Path
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / 'configs/colab_main_study_source.json').read_text())
    if not re.fullmatch('[0-9a-f]{40}', str(settings.get('commit'))):
        raise ValueError('A full immutable runtime commit is required')
    for key in ('source_sha256', 'protocol_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(settings.get(key))):
            raise ValueError('A full ' + key + ' is required')
    if not isinstance(settings.get('ready'), bool):
        raise ValueError('Release readiness must be a boolean')
    bootstrap = (root / 'scripts/colab_bootstrap.py').read_text()
    bootstrap += '\nfrom pathlib import Path\nimport os, sys, hashlib, json, subprocess, tempfile\n'
    for variable, key in (('EXPECTED_SOURCE_SHA256','source_sha256'), ('EXPECTED_PROTOCOL_SHA256','protocol_sha256'),
                          ('GITHUB_REPOSITORY','repository'), ('PINNED_COMMIT','commit')):
        bootstrap += f'{variable} = {json.dumps(settings[key])}\n'
    bootstrap += f'MAIN_STUDY_REVISION_READY = {settings["ready"]!r}\n'
    bootstrap += '''try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and not MAIN_STUDY_REVISION_READY:
    raise RuntimeError("This main-study launcher is awaiting its verified runtime release.")
if IN_COLAB:
    destination = Path(tempfile.mkdtemp(prefix="distilled-main-source-")) / "project"
    PROJECT = fetch_public_project(GITHUB_REPOSITORY, PINNED_COMMIT, destination)
else:
    PROJECT = Path(os.environ.get("DISTILLED_PROJECT_ROOT", Path.cwd())).resolve()
    if not (PROJECT / "pyproject.toml").exists() and (PROJECT.parent / "pyproject.toml").exists():
        PROJECT = PROJECT.parent
source_paths = []
for folder in ("main_study", "foundation", "src/distilled_rnns"):
    paths = list((PROJECT / folder).glob("*.py"))
    if not paths:
        raise RuntimeError("Downloaded source differs from the verified notebook revision.")
    source_paths.extend(paths)
digest = hashlib.sha256()
for path in sorted(source_paths, key=lambda value: value.relative_to(PROJECT).as_posix()):
    digest.update(path.relative_to(PROJECT).as_posix().encode())
    digest.update(path.read_bytes())
if digest.hexdigest() != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("Downloaded source differs from the verified notebook revision.")
if hashlib.sha256((PROJECT / "configs/main_study_v1.json").read_bytes()).hexdigest() != EXPECTED_PROTOCOL_SHA256:
    raise RuntimeError("Downloaded prospective protocol differs from the verified revision.")
if IN_COLAB:
    drive.mount("/content/drive")
    RUNS = Path("/content/drive/MyDrive/distilled-tiny-rnns/runs")
else:
    RUNS = Path(os.environ.get("DISTILLED_OUTPUT_ROOT", str(PROJECT / "outputs/notebook-main-study")))
os.chdir(PROJECT)
RUNS.mkdir(parents=True, exist_ok=True)
print("Verified project:", PROJECT)
print("Results:", RUNS)
'''
    setup = '''# Reject a previously imported revision before activating this source.
for namespace, relative in (("distilled_rnns", "src/distilled_rnns"), ("main_study", "main_study"), ("foundation", "foundation")):
    expected = (PROJECT / relative).resolve()
    for name, module in list(sys.modules.items()):
        if name == namespace or name.startswith(namespace + "."):
            module_file = getattr(module, "__file__", None)
            locations = [Path(module_file)] if module_file else [Path(p) for p in getattr(module, "__path__", [])]
            if not locations or any(not p.resolve().is_relative_to(expected) for p in locations):
                raise RuntimeError("Another project revision is loaded. Restart the runtime and run from cell one.")
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"], check=True)
for path in (PROJECT, PROJECT / "src"):
    if str(path.resolve()) not in sys.path:
        sys.path.insert(0, str(path.resolve()))
import importlib
importlib.invalidate_caches()
import torch
if IN_COLAB and not torch.cuda.is_available():
    raise RuntimeError("Select a T4 GPU runtime before continuing.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU (local verification)")
verification = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(verification.stdout, end="", flush=True)
verification.check_returncode()
from main_study.config import MainStudyConfig
from main_study.runner import run_main_study, main_source_fingerprint
if main_source_fingerprint() != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("Runtime source identity changed during setup.")
'''
    smoke = '''SMOKE_RUN = RUNS / ("main-study-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = MainStudyConfig.from_file(PROJECT / "configs/main_study_smoke.json")
if smoke_config.mode != "smoke":
    raise RuntimeError("The preflight must use smoke mode.")
smoke = run_main_study(smoke_config, SMOKE_RUN, device=DEVICE)
if smoke["status"] != "complete":
    raise RuntimeError("Smoke check incomplete. Rerun this third cell using the same folder.")
def _hash_files(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}
before = _hash_files(SMOKE_RUN)
resumed = run_main_study(smoke_config, SMOKE_RUN, device=DEVICE)
if resumed != smoke or _hash_files(SMOKE_RUN) != before:
    raise RuntimeError("Completed smoke resume changed results or files.")
print("Both rare strategies, every model, behavioral assay and immutable completed resume passed.")
'''
    production = '''RUN_NAME = "main-study-first-tranche-v1"
config = MainStudyConfig.from_file(PROJECT / "configs" / ("main_study_first_tranche.json" if IN_COLAB else "main_study_smoke.json"))
if config.mode != ("production" if IN_COLAB else "smoke"):
    raise RuntimeError("The main run configuration has the wrong role.")
print(json.dumps(config.to_dict(), indent=2))
report = run_main_study(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report.get("message", "Run paused with saved checkpoints."))
    print("Live runtime: rerun ONLY this fourth code cell with the same RUN_NAME.")
    print("After a deleted runtime: restore cells 1–3, then rerun cell 4.")
'''
    reporting = '''if report["status"] == "complete":
    from IPython.display import display, Markdown
    # Full per-person metrics, paired directions, null support, all model scores
    # and population/individual checkpoint results remain in the saved JSON.
    display(Markdown("**Complete.** Positive prediction gain means the teacher-target student predicts held-out choices better than solo. Positive behavioral distortion means it has higher error in the four-cell strategy assay. Read both coordinates together for each rare kind."))
    def _score(a, signed=False):
        if a["mean"] is None:
            return f"unavailable ({a['supported_seed_count']}/{a['planned_seed_count']})"
        return (f"{a['mean']:+.5f}" if signed else f"{a['mean']:.5f}") + f" [{a['min']:.5f}, {a['max']:.5f}] ({a['supported_seed_count']}/{a['planned_seed_count']})"
    for kind_name, kind in report["kinds"].items():
        rows = [f"**{kind_name}**", "", "| Group / selection | Prediction gain | Behavioral distortion | Fraction positive in both |", "| --- | ---: | ---: | ---: |"]
        for group_name, group in kind["groups"].items():
            for mode in ("population", "individual"):
                primary = group["primary"][mode]
                rows.append("| " + " | ".join([group_name + " / " + mode, _score(primary["prediction_gain"], True), _score(primary["behavioral_distortion"], True), _score(primary["quadrant_fraction"])]) + " |")
            rows += ["", f"Diagnostic support, trials / people: {_score(group['diagnostic_count'])} / {_score(group['diagnostic_subject_count'])}", "", f"**{group_name} model scores**", "", "| Model | Test NLL | True-policy KL | Diagnostic KL | Signature RMS | Probe KL |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for model, scores in group["models"].items():
                rows.append("| " + " | ".join([model] + [_score(scores[key]) for key in ("nll", "kl", "diagnostic_kl", "signature_rms_error", "probe_kl")]) + " |")
        display(Markdown("\\n".join(rows)))
        print("Per-population primary comparisons:")
        for population in kind["per_seed"]:
            print(population["seed"], json.dumps({group:values["primary"] for group,values in population["groups"].items()}))
    print(json.dumps(report["timing"], indent=2))
    print("Keep the complete main-study-first-tranche-v1 folder and", SMOKE_RUN.name)
    print("Training-time cost excludes setup, evaluation, saving and idle runtime. Disconnect the GPU when finished.")
else:
    print("All planned populations and models must finish before the tranche is aggregated.")
    print("Rerun only cell 4 in a live runtime. After a deleted runtime, restore cells 1–3 first.")
'''
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md('''# Distilled Tiny RNNs — first main-study comparison

This compares models on **six fresh synthetic populations**. Three contain pure model-free rare agents, and three contain reward-cue rare agents. Each population has **1,000 people, 100 trials and 20 rare people**. Alongside held-out prediction, a fixed behavioral assay tests whether each model preserves the generating policy's reward/transition response.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the five code cells in order. Results save to `MyDrive/distilled-tiny-rnns/runs/main-study-first-tranche-v1`.

The H=32 teacher has at most 500 steps with validation stopping. The H=2 solo, teacher-target and oracle-target students each have 4,000 steps. Classical and hierarchical baselines fit the same training choices. The first-stage project adaptation and the finite-grid populations follow the frozen protocol. The small controlled behavioral assay and the drifting task have separate reported metrics.

Each training phase has a soft ten-minute budget, with a soft thirty-minute invocation budget. Evaluation and saving add time. **At a time-budget pause, rerun only code cell 4 with the same run name.** After a deleted runtime, restore cells 1–3 first. Use one runtime per output folder. Retain the full results and smoke folders when complete and disconnect the GPU.
''')
    code(bootstrap)
    md('## Install and verify\n\nThe pinned source and prospective design are checked before training. The complete preflight suite runs on the selected device.')
    code(setup)
    md('## Small end-to-end check\n\nExercise both rare strategies, every comparison and the behavioral assay, then verify that rerunning a completed smoke leaves its files and time accounting identical.')
    code(smoke)
    md('## Run the first tranche\n\nKeep the fixed run name for every resume. The six production seeds are 202611401, 202611411, 202611421, 202621401, 202621411 and 202621421. The main study uses fresh output folders and its own frozen identities.')
    code(production)
    md('## Results\n\nThe primary comparison has two coordinates for each rare kind. Prediction gain is solo minus teacher-target student test NLL. Behavioral distortion is teacher-target student minus solo signature RMS error. Population checkpoint selection is primary; individual selection is secondary. Positive values in both coordinates describe better sampled prediction together with worse behavior in the declared assay. Inspect every seed, all controls and diagnostic support. Means and ranges are descriptive across three populations per kind.')
    code(reporting)
    for index, cell in enumerate(cells):
        cell.id = f'main-study-{index:02d}'
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
    target = root / 'notebooks/07_colab_main_study.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)
    return target


if __name__ == '__main__':
    print(build())
