"""Build the second launcher while preserving the first pilot notebook.

The release settings contain both the immutable runtime commit and its source
hash. Updating the working tree alone must never update the launcher's hash.
"""
from pathlib import Path
import json
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/colab_followup_source.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", settings["commit"]):
        raise ValueError("A full pinned runtime commit is required")
    if not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("The pinned runtime's source SHA256 is required")
    original = nbf.read(root / "notebooks/01_colab_pilot.ipynb", as_version=4)
    original_code = [cell.source for cell in original.cells if cell.cell_type == "code"]
    bootstrap = original_code[0]
    # Keep the tested downloader byte-for-byte and replace only release values.
    for variable, key in (
        ("EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("GITHUB_REPOSITORY", "repository"),
        ("PINNED_COMMIT", "commit"),
    ):
        bootstrap, count = re.subn(
            rf"^{variable} = .+$", f"{variable} = {json.dumps(settings[key])}",
            bootstrap, flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Expected exactly one {variable} assignment")
    ready = settings.get("ready", False)
    bootstrap = bootstrap.replace(
        "if IN_COLAB:\n    destination = Path(tempfile.mkdtemp(",
        f"FOLLOWUP_REVISION_READY = {ready!r}\n"
        'if IN_COLAB and not FOLLOWUP_REVISION_READY:\n'
        '    raise RuntimeError("This follow-up launcher is awaiting its verified runtime release. Use the completed reviewed notebook.")\n'
        "if IN_COLAB:\n    destination = Path(tempfile.mkdtemp(",
        1,
    )
    setup = '''# A different notebook revision may already be imported in this kernel.
# Never mix its modules with this verified source download.
expected_package = (PROJECT / "src/distilled_rnns").resolve()
for module_name, module in list(sys.modules.items()):
    if module_name == "distilled_rnns" or module_name.startswith("distilled_rnns."):
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(expected_package):
            raise RuntimeError("Another project revision is loaded. Restart the Colab session, then run this notebook from its first cell.")
''' + original_code[1]
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Distilled Tiny RNNs — longer training and repeated seeds

**Purpose:** check training convergence and the stability of the first pilot's descriptive results. This remains a synthetic, conditional first-stage experiment with a parametric rare group. It does not establish recovery of the generating mechanism or reproduce the published method.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for checkpoints. Run the cells in order.

The follow-up allows 500 teacher steps and 1,000 steps per student arm. A phase can pause after 10 minutes; each study invocation has a soft 30-minute training budget, checked between steps/phases. Setup, saving, and evaluation can add time. Rerun the same notebook and run name to resume. These limits do not guarantee convergence or a particular GPU runtime. Disconnect the runtime when finished.
""")
    code(bootstrap)
    md("""## Install and verify

The notebook checks the downloaded source, activates it immediately in this running kernel, and runs the verification suite before training. The GPU resume test runs on Colab; CPU verification alone cannot substitute for it. Exact package versions are recorded with each run.
""")
    code(setup)
    md("""## Small follow-up smoke test

A tiny study exercises regularization selection, repeated seeds, saved results, and resume. This is a software check, not a scientific result.
""")
    code('''from distilled_rnns.followup import FollowupConfig, run_followup

SMOKE_RUN = RUNS / ("followup-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = FollowupConfig.from_file(PROJECT / "configs/followup_smoke.json")
smoke = run_followup(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
smoke_resumed = run_followup(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["groups"] == smoke["groups"], "Resumed results differ"
assert smoke_resumed["calibration"]["selected_l2"] == smoke["calibration"]["selected_l2"]
print("Follow-up smoke test and resume passed.")
''')
    md("""## Run the bounded follow-up

**Calibration:** the original seed, 20261029, has already been inspected and is exploratory. Train four solo candidates with L2 strengths **0, 0.001, 0.01, and 0.1**. Each person's checkpoint is chosen using observed-choice validation NLL. Choose one strength using the mean validation NLL across all people; ties favor the smaller strength. This population-selected choice is dominated by the 98% typical majority; it is not selected specifically for rare people. The choice is saved before computing test metrics. Rare-group labels and true-policy scores do not choose the strength.

**Repeated runs:** freeze the selected strength for five new simulation seeds, **20261101, 20261111, 20261121, 20261131, 20261141** (integer random seeds, not dates). Compare the teacher, longer-trained unregularized solo students, calibrated solo students, and distilled students. If calibration selects zero, the calibrated and unregularized solo results are identical. The student arms start from the same weights within each seed. Population seeds are spaced to avoid reusing random initialization streams across populations. Training remains masked, and validation selects each person's checkpoint.

Calibration results stay separate from the repeated-seed summary. Keep the default protocol for this run: changing settings or code requires a new output folder. No test score should be used to choose settings for these same seeds.
""")
    code('''RUN_NAME = "followup-v1"  # Keep for resume; use a new name for changed settings or source.
CONFIG_NAME = "followup" if IN_COLAB else os.environ.get("DISTILLED_FOLLOWUP_CONFIG", "followup_smoke")
config = FollowupConfig.from_file(PROJECT / "configs" / f"{CONFIG_NAME}.json")
print(json.dumps(config.to_dict(), indent=2))
report = run_followup(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
''')
    md("""## Results and convergence checks

Inspect the validation curves and the fraction of people whose selected checkpoint is at the training limit. A boundary warning means further convergence checking is needed; it is not evidence for or against the research hypothesis. More training steps alone do not guarantee convergence.

NLL measures prediction of held-out choices and KL measures disagreement with the known generating policy; lower is better for both. Diagnostic scores have their support counts reported and can be unavailable. The summary reports descriptive differences across the five new seeds. It does not pool the calibration run into those results or establish statistical significance.

Review the comparison with both solo baselines. The same observed histories remain available as causal inputs, so these are conditional predictions rather than free-running strategy recovery. Cost conversion uses your supplied **1.07 units per T4-hour** only for recorded T4 training; Colab also bills setup and idle time.
""")
    code('''if report["status"] == "complete":
    from IPython.display import display, Image, Markdown
    print("Frozen solo L2 strength:", report["calibration"]["selected_l2"])
    print("Population-selected L2 is dominated by the typical majority (98% in the default study).")
    if report["calibration"]["selected_l2"] == 0:
        print("Zero was selected: calibrated solo and unregularized solo are the same model.")
    if report["calibration"].get("lambda_at_grid_boundary"):
        print("CHECK REQUIRED: the selected L2 strength is at a search-grid boundary.")
    for warning in report.get("convergence_warnings", []):
        print("CHECK REQUIRED:", warning)
    def _score_text(summary, paired=False):
        support = f"({summary['supported_seed_count']}/{summary['planned_seed_count']} seeds)"
        if summary["mean"] is None:
            return "unavailable " + support
        if paired:
            return f"{summary['mean']:+.4f} [{summary['min']:+.4f}, {summary['max']:+.4f}] {support}"
        return f"{summary['mean']:.4f} {support}"

    display(Markdown("**Scores:** seed means in nats per trial; lower is better. Parentheses show supported/planned seeds. "
                     "**Paired differences:** mean [minimum, maximum] across seeds. Positive NLL gain favors distillation; "
                     "positive KL loss means distillation has worse policy fidelity. These are descriptive summaries."))
    labels = {"teacher": "Teacher", "solo_unregularized": "Solo, no L2",
              "solo": "Calibrated solo", "distilled": "Distilled"}
    for group_name, group in report["groups"].items():
        rows = [f"**{group_name.title()} group**", "", "| Model | NLL mean | KL mean | Diagnostic NLL mean |",
                "| --- | ---: | ---: | ---: |"]
        for name in ("teacher", "solo_unregularized", "solo", "distilled"):
            metrics = group["models"][name]
            values = [_score_text(metrics[metric]) for metric in ("nll", "kl", "diagnostic_nll")]
            rows.append("| " + " | ".join([labels[name], *values]) + " |")
        rows.extend(["", "| Distilled versus | NLL gain | KL loss | Diagnostic NLL gain |",
                     "| --- | ---: | ---: | ---: |"])
        for baseline in ("solo", "solo_unregularized"):
            metrics = group["comparisons"][baseline]
            values = [_score_text(metrics[metric], paired=True)
                      for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain")]
            rows.append("| " + " | ".join([labels[baseline], *values]) + " |")
        support = [f"{row['seed']}: {row['diagnostic_count']} trials / {row['diagnostic_subject_count']} people"
                   for row in group["per_seed"]]
        rows.extend(["", "Diagnostic test support — " + "; ".join(support) + "."])
        display(Markdown("\\n".join(rows)))
    timing = report["timing"]
    print(f"Recorded training time: {timing['cumulative_training_seconds'] / 60:.2f} minutes.")
    units = timing["t4_equivalent_units_at_user_rate"]
    if units is None:
        print("T4 compute-unit proxy unavailable for this device history.")
    else:
        print(f"Training-only T4 compute-unit proxy: {units:.4f} units at your supplied rate.")
    print("Includes calibration candidates and repeated fits; setup, evaluation and idle billing are additional.")
    output_dir = Path(report["output_dir"])
    plots = sorted(output_dir.rglob("convergence.png"))
    for plot in plots[:2]:
        print(plot.parent.name)
        display(Image(filename=str(plot)))
    for plot in plots[2:]:
        print("Additional convergence figure:", plot)
    print("Saved:", output_dir)
    print("Keep the entire followup-v1 folder, including summary.json, the frozen protocol/selection, every seed folder, and all checkpoints and environment records.")
else:
    print("Follow-up incomplete. Resume the same run before interpreting a repeated-seed summary.")
''')
    for index, cell in enumerate(cells):
        cell["id"] = f"followup-cell-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "02_colab_followup.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    nbf.validate(notebook)
    destination = root / "notebooks/02_colab_followup.ipynb"
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build())
