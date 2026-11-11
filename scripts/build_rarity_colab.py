"""Build notebook08 for the next prospective100-trial rarity cells."""
import json
from pathlib import Path
import re

import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root=Path(root)
    settings=json.loads((root/'configs/colab_rarity_source.json').read_text())
    if not re.fullmatch('[0-9a-f]{40}',str(settings.get('commit'))):
        raise ValueError('A full immutable runtime commit is required')
    for key in ('source_sha256','protocol_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',str(settings.get(key))):
            raise ValueError('A full '+key+' is required')
    if not isinstance(settings.get('ready'),bool):
        raise ValueError('Release readiness must be a boolean')
    previous=nbf.read(root/'notebooks/07_colab_main_study.ipynb',as_version=4)
    old=[cell.source for cell in previous.cells if cell.cell_type=='code']
    bootstrap=old[0]
    for variable,key in (('EXPECTED_SOURCE_SHA256','source_sha256'),('EXPECTED_PROTOCOL_SHA256','protocol_sha256'),('GITHUB_REPOSITORY','repository'),('PINNED_COMMIT','commit')):
        bootstrap,count=re.subn(rf'^{variable} = .+$',lambda _:f'{variable} = {json.dumps(settings[key])}',bootstrap,flags=re.MULTILINE)
        if count!=1:raise ValueError('Expected one '+variable+' assignment')
    bootstrap=bootstrap.replace('MAIN_STUDY_REVISION_READY','RARITY_REVISION_READY').replace('This main-study launcher','This rarity-study launcher')
    bootstrap,count=re.subn('^RARITY_REVISION_READY = .+$',f'RARITY_REVISION_READY = {settings["ready"]!r}',bootstrap,flags=re.MULTILINE)
    if count!=1:raise ValueError('Expected one release guard')
    bootstrap=bootstrap.replace('("main_study", "foundation", "src/distilled_rnns")','("main_study", "foundation", "src/distilled_rnns", "grid_study")')
    marker='digest = hashlib.sha256()'
    if bootstrap.count(marker)!=1:raise ValueError('Expected one source digest')
    bootstrap=bootstrap.replace(marker,'''source_paths.extend([PROJECT / "configs/main_study_v1.json", PROJECT / "configs/grid_baseline_reference_v1.json"])
digest = hashlib.sha256()''')
    setup=old[1].replace('("foundation", "foundation")):', '("foundation", "foundation"), ("grid_study", "grid_study")):')
    setup=setup.replace('from main_study.runner import run_main_study, main_source_fingerprint\nif main_source_fingerprint() != EXPECTED_SOURCE_SHA256:', '''from main_study.runner import run_main_study
from grid_study.config import GridTrancheConfig
from grid_study.runner import run_rarity_tranche, run_baseline_smoke, grid_source_fingerprint
if grid_source_fingerprint() != EXPECTED_SOURCE_SHA256:''')
    smoke='''BASELINE_SMOKE = RUNS / ("rarity-baseline-smoke-" + EXPECTED_SOURCE_SHA256[:12])
legacy_smoke_config = MainStudyConfig.from_file(PROJECT / "configs/main_study_smoke.json")
if legacy_smoke_config.mode != "smoke":
    raise RuntimeError("The baseline preflight must use smoke mode.")
legacy_smoke = run_baseline_smoke(legacy_smoke_config, BASELINE_SMOKE, device=DEVICE)
if legacy_smoke["status"] != "complete":
    raise RuntimeError("Baseline smoke incomplete. Rerun this third cell with the same folders.")
def _hash_files(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}
parent_before = _hash_files(BASELINE_SMOKE)
SMOKE_RUN = RUNS / ("rarity-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = GridTrancheConfig.from_file(PROJECT / "configs/rarity_smoke.json")
if smoke_config.mode != "smoke":
    raise RuntimeError("The rarity preflight must use smoke mode.")
for _ in range(3):
    smoke = run_rarity_tranche(smoke_config, BASELINE_SMOKE, SMOKE_RUN, device=DEVICE)
    if smoke["status"] == "complete":
        break
if smoke["status"] != "complete":
    raise RuntimeError("Rarity smoke paused. Rerun this third cell with the same folders.")
before = _hash_files(SMOKE_RUN)
resumed = run_rarity_tranche(smoke_config, BASELINE_SMOKE, SMOKE_RUN, device=DEVICE)
if resumed != smoke or _hash_files(SMOKE_RUN) != before or _hash_files(BASELINE_SMOKE) != parent_before:
    raise RuntimeError("Completed resume changed the smoke results or read-only baseline.")
print("All three rarity cells, both strategies, saved baseline reference and immutable completed resume passed.")
'''
    production='''RUN_NAME = "main-study-rarity-v1"
config = GridTrancheConfig.from_file(PROJECT / "configs" / ("rarity_tranche.json" if IN_COLAB else "rarity_smoke.json"))
if config.mode != ("production" if IN_COLAB else "smoke"):
    raise RuntimeError("The rarity run configuration has the wrong role.")
PARENT_RUN = RUNS / "main-study-first-tranche-v1" if IN_COLAB else BASELINE_SMOKE
print(json.dumps(config.to_dict(), indent=2))
report = run_rarity_tranche(config, PARENT_RUN, RUNS / RUN_NAME, device=DEVICE)
if not IN_COLAB:
    # The actual local cell exercises all three tiny cells across real invocations.
    for _ in range(2):
        if report["status"] == "complete":
            break
        report = run_rarity_tranche(config, PARENT_RUN, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report.get("message", "Rarity tranche paused with saved checkpoints."))
    print("Completed cells:", report["completed_cells"])
    print("Pending cells:", report["pending_cells"])
    print("Live runtime: rerun ONLY this fourth code cell with the same RUN_NAME.")
    print("After a deleted runtime: restore cells 1–3, then rerun cell 4.")
'''
    reporting='''if report["status"] == "complete":
    from IPython.display import display, Markdown
    def _score(a, signed=False):
        if a["mean"] is None:
            return f"unavailable ({a['supported_seed_count']}/{a['planned_seed_count']})"
        value = f"{a['mean']:+.5f}" if signed else f"{a['mean']:.5f}"
        return value + f" [{a['min']:.5f}, {a['max']:.5f}] ({a['supported_seed_count']}/{a['planned_seed_count']})"
    cells = [("Verified baseline", report["baseline_reference"]["summary"])] + [(name, cell) for name, cell in report["cells"].items()]
    rows = ["**Rarity comparison at a fixed history length**", "",
            "Within each population, distilled is paired with solo. Different rarity levels use distinct populations.", "",
            "| Rare fraction | Strategy | Prediction gain | Assay error change | Solo diagnostic KL | Distilled diagnostic KL |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for label, cell in cells:
        fraction = cell["config"]["rare_fraction"]
        for kind_name, kind in cell["kinds"].items():
            group = kind["groups"]["rare"]
            primary = group["primary"]["population"]
            rows.append("| " + " | ".join([f"{100*fraction:g}%", kind_name,
                _score(primary["prediction_gain"], True), _score(primary["behavioral_distortion"], True),
                _score(group["models"]["solo_population"]["diagnostic_kl"]),
                _score(group["models"]["distilled_population"]["diagnostic_kl"])]) + " |")
    display(Markdown("\\n".join(rows)))
    for label, cell in cells:
        rows = [f"**{label}: {100*cell['config']['rare_fraction']:g}% rare, {cell['config']['n_trials']} trials**", ""]
        for kind_name, kind in cell["kinds"].items():
            for group_name, group in kind["groups"].items():
                rows += [f"**{kind_name} / {group_name}**", "",
                         f"Diagnostic trials / people: {_score(group['diagnostic_count'])} / {_score(group['diagnostic_subject_count'])}", "",
                         "| Model | Test NLL | True-policy KL | Diagnostic KL | Assay RMS | Probe KL |",
                         "| --- | ---: | ---: | ---: | ---: | ---: |"]
                for model, scores in group["models"].items():
                    rows.append("| " + " | ".join([model]+[_score(scores[key]) for key in ("nll", "kl", "diagnostic_kl", "signature_rms_error", "probe_kl")]) + " |")
                for mode in ("population", "individual"):
                    primary = group["primary"][mode]
                    rows += ["", f"{mode} checkpoint rule: prediction gain {_score(primary['prediction_gain'], True)}; assay error change {_score(primary['behavioral_distortion'], True)}; positive-both fraction {_score(primary['quadrant_fraction'])}."]
            for population in kind["per_seed"]:
                print(label, kind_name, population["seed"], json.dumps({group:v["primary"] for group,v in population["groups"].items()}))
        display(Markdown("\\n".join(rows)))
    print(json.dumps(report["timing"], indent=2))
    print("Keep the full main-study-rarity-v1 and", SMOKE_RUN.name, "folders for review.")
    print("Baseline training is inherited context. New training, setup, evaluation and idle billing have separate accounting.")
    print("Disconnect the GPU when complete.")
else:
    print("The rarity tranche is incomplete. Rerun only cell 4 in the live runtime.")
    print("All18 planned populations must finish before the final rarity tables are displayed.")
'''
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md(f'''# Distilled Tiny RNNs — rarity at 100 trials

This is the next prospectively scheduled part of the frozen main study. It runs **5%, 10% and 25% rare people**, both rare strategies, and three populations per cell: **18 fresh populations**. Every population has **1,000 people and 100 trials**. The architecture, 4,000-step student budget, selection rules, classical controls, diagnostic metrics and behavioral assay follow the existing design.

Keep the complete **`runs/main-study-first-tranche-v1`** folder in Google Drive. Its verified 2% results provide a read-only baseline. See [the rarity run guide](https://github.com/{settings['repository']}/blob/main/docs/rarity-tranche.md) for the exact verified 2% baseline bundle. The new run validates that baseline and saves a portable reference containing its summary, protocol and hash binding. Original baseline checkpoints and predictions stay in their original folder.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the five code cells in order. Results save under **`runs/main-study-rarity-v1`**. Use one runtime per output folder.

Each invocation works on at most one pending rarity cell, with the existing soft 30-minute invocation and 10-minute phase limits. **Rerun only code cell 4 after a pause or cell completion while more cells remain.** After a deleted runtime, restore cells 1–3 before resuming cell 4. Keep the same run name. Final tables appear after all 18 populations finish. Completed cells retain their checkpoints and results.

For scheduling, the prior six-population 100-trial run took 118.86 minutes of measured T4 runner time. Three times that is roughly 6 hours for 18 populations, before additional setup and idle time. This is a rough scheduling reference; actual timing and eligible cost proxies are recorded by this run. The completed first-tranche findings remain visible, including their improved primary assay and worsened diagnostic errors.
''')
    code(bootstrap)
    md('## Install and verify\n\nVerify the pinned runtime and unchanged frozen protocol, activate source, and run the full preflight suite on the selected device.')
    code(setup)
    md('## Small baseline and three-cell smoke\n\nCreate a genuine tiny first-tranche baseline, exercise each scheduled cell with development seeds, and verify completed resume and parent preservation.')
    code(smoke)
    md('## Run the next 18 populations\n\nThe production cells use the original frozen seed formula. All 100-trial populations retain the same training and evaluation recipe. The 2% baseline is inherited context and has separate compute accounting.')
    code(production)
    md('## Rarity comparison\n\nPositive prediction gain means better distilled prediction. Negative assay error change means lower distilled behavioral error. Report the primary pair alongside diagnostic scores, probe fidelity, both checkpoint rules and all controls. Each rarity/strategy cell has three distinct populations. Ranges and directions are descriptive; rarity differences use independent populations.')
    code(reporting)
    for index,cell in enumerate(cells):cell.id=f'rarity-{index:02d}'
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    target=root/'notebooks/08_colab_rarity.ipynb';target.parent.mkdir(parents=True,exist_ok=True);nbf.write(notebook,target)
    return target


if __name__=='__main__':print(build())
