"""Build the baseline-audit launcher without changing the first two notebooks."""
from pathlib import Path
import json
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/colab_baseline_source.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", settings["commit"]):
        raise ValueError("A full pinned runtime commit is required")
    if not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("The pinned runtime's source SHA256 is required")
    if not isinstance(settings.get("ready", False), bool):
        raise ValueError("Release readiness must be a boolean")
    original = nbf.read(root / "notebooks/02_colab_followup.ipynb", as_version=4)
    original_code = [cell.source for cell in original.cells if cell.cell_type == "code"]
    bootstrap = original_code[0]
    # Preserve the reviewed downloader and change only the release constants.
    for variable, key in (
        ("EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("GITHUB_REPOSITORY", "repository"),
        ("PINNED_COMMIT", "commit"),
    ):
        replacement = f"{variable} = {json.dumps(settings[key])}"
        bootstrap, count = re.subn(rf"^{variable} = .+$", lambda _: replacement,
                                  bootstrap, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one {variable} assignment")
    bootstrap = bootstrap.replace("FOLLOWUP_REVISION_READY", "BASELINE_REVISION_READY")
    bootstrap, count = re.subn(
        r"^BASELINE_REVISION_READY = .+$",
        f"BASELINE_REVISION_READY = {settings.get('ready', False)!r}",
        bootstrap, flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Expected the prior notebook's release guard")
    bootstrap = bootstrap.replace(
        "This follow-up launcher is awaiting its verified runtime release.",
        "This baseline-audit launcher is awaiting its verified runtime release.",
    )
    setup = original_code[1]  # Includes the running-kernel import/revision guard.
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Distilled Tiny RNNs — baseline and validation audit

**Purpose:** test a stronger comparison for distillation. The previous small models trained alone predicted worse than a constant 50/50 reference. This audit separates the data used to choose saved model weights from the data used to choose regularization, and checks whether using one training checkpoint for the population is more reliable than choosing one for each person.

This remains a synthetic, conditional first-stage experiment with a parametric rare group. It does not establish recovery of people's reasoning or reproduce the published method.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the cells in order.

Training allows 500 teacher steps and 1,000 student steps. Each phase has a soft 10-minute limit; each study invocation has a soft 30-minute budget. Setup, completed steps, saving, and evaluation can add time. If the study pauses, rerun its cell with the same run name to resume. Disconnect the runtime when finished.
""")
    code(bootstrap)
    md("""## Install and verify

The source fingerprint is checked before training. Setup activates the downloaded package in this running kernel and rejects modules left over from another revision. The test suite runs before the smoke test, including GPU-specific checks when CUDA is present. Exact package versions are recorded with every run.
""")
    code(setup)
    md("""## Small audit smoke test

This tiny study checks the separate tuning split, both checkpoint-selection methods, fixed references, repeated seeds, and resume. Its scores are software checks, not scientific findings.
""")
    code('''from distilled_rnns.baseline_audit import BaselineAuditConfig, run_baseline_audit

SMOKE_RUN = RUNS / ("baseline-audit-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = BaselineAuditConfig.from_file(PROJECT / "configs/baseline_audit_smoke.json")
smoke = run_baseline_audit(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
smoke_resumed = run_baseline_audit(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["groups"] == smoke["groups"], "Resumed results differ"
for field in ("solo_l2", "solo_selection", "distilled_selection"):
    assert smoke_resumed["calibration"][field] == smoke["calibration"][field]
print("Baseline audit smoke test and resume passed.")
''')
    md("""## Run the bounded baseline audit

**Calibration uses a fresh population, seed 20261201.** For each person, 60% of trials train the models, 20% select saved checkpoints, and a separate final 20% choose settings. That last calibration split is **tuning data**, not an independent test set. No calibration test result is claimed.

Solo candidates use L2 strengths **0, 0.001, 0.01, and 0.1**. Each fitted trajectory supplies two alternatives: **individual selection** chooses each person's best checkpoint; **population selection** chooses one step using mean validation NLL across the population, while every person's model weights remain independent. Tracking both alternatives does not train the same trajectory twice. The outer tuning split chooses the solo strength and selection method, and separately chooses the distilled selection method. Exact ties favor smaller L2, then population selection. Settings are selected over the population, dominated by the 98% typical majority; neither rare labels nor true-policy scores choose them.

**Evaluation freezes those choices** before testing five fresh populations: **20261211, 20261221, 20261231, 20261241, 20261251**. These are integer random seeds, not dates. Each evaluation population keeps its own 60% training, 20% checkpoint-selection, and 20% test split. Test scores do not choose settings.

The report includes the teacher, calibrated solo and distilled pipelines, both matched checkpoint-selection comparisons, the previous solo recipe (individual selection with L2 0.01), a fixed **50/50 reference**, and a **choice-frequency reference** estimated only from training choices with Laplace smoothing. The matched comparisons hold the checkpoint rule fixed; the training objective and solo L2 still differ.

Keep the default protocol for this run. Changing source or settings requires a new run name. Do not use these test scores to choose settings and then describe the same seeds as an independent assessment.
""")
    code('''RUN_NAME = "baseline-audit-v1"  # Keep for resume; change for new settings or source.
CONFIG_NAME = "baseline_audit" if IN_COLAB else os.environ.get("DISTILLED_BASELINE_CONFIG", "baseline_audit_smoke")
config = BaselineAuditConfig.from_file(PROJECT / "configs" / f"{CONFIG_NAME}.json")
print(json.dumps(config.to_dict(), indent=2))
report = run_baseline_audit(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
''')
    md("""## Results and what to inspect

Prediction error (NLL) and policy mismatch (KL) are lower when better. A constant 50/50 prediction has NLL **0.6931**. The population-level means and ranges below describe the five fresh seeds; they are not significance tests. Calibration is excluded.

The main comparison uses the independently calibrated solo and distilled pipelines. The matched-method rows hold the checkpoint rule fixed; the training objective and solo L2 still differ, so these rows do not isolate a pure causal effect of distillation. Selected-pipeline rows can duplicate their matching individual/population rows.

The selection-audit figures compare scores on checkpoint-selection targets with separate calibration tuning targets or fresh-population test targets. They help show how well the selected checkpoints generalize beyond the data used to choose them. Calibration tuning scores remain part of model selection, not an independent test result.

Diagnostic scores use the same predeclared subset and report its support. Missing support remains unavailable. Inspect the learning curves and training-boundary warnings before claiming convergence. These models predict choices conditional on observed histories; they do not establish recovery of the generating mechanism on new free-running histories.
""")
    code('''if report["status"] == "complete":
    from IPython.display import display, Image, Markdown
    modes = {"individual": "one checkpoint per person", "population": "one checkpoint for the population"}
    calibration = report["calibration"]
    print("Frozen solo L2 strength:", calibration["solo_l2"])
    print("Solo selection:", modes.get(calibration["solo_selection"], calibration["solo_selection"]))
    print("Distilled selection:", modes.get(calibration["distilled_selection"], calibration["distilled_selection"]))
    print("These settings were selected on separate calibration tuning targets, before fresh-seed test scores.")
    for warning in report.get("convergence_warnings", []):
        print("CHECK REQUIRED:", warning)

    def _score_text(summary, paired=False):
        support = f"({summary['supported_seed_count']}/{summary['planned_seed_count']} seeds)"
        if summary["mean"] is None:
            return "unavailable " + support
        if paired:
            return f"{summary['mean']:+.4f} [{summary['min']:+.4f}, {summary['max']:+.4f}] {support}"
        return f"{summary['mean']:.4f} {support}"

    labels = {"teacher": "Teacher", "solo": "Calibrated solo", "distilled": "Calibrated distilled",
              "coinflip": "Fixed 50/50", "frequency": "Training choice frequency",
              "solo_legacy": "Previous solo recipe",
              "solo_individual": "Solo: individual checkpoint", "solo_population": "Solo: population checkpoint",
              "distilled_individual": "Distilled: individual checkpoint", "distilled_population": "Distilled: population checkpoint"}
    display(Markdown("**Scores:** seed means in nats per trial; lower is better. Parentheses show supported/planned seeds. "
                     "**Paired differences:** mean [minimum, maximum]. Positive NLL gain favors distillation; positive KL loss "
                     "means worse policy fidelity. The fixed 50/50 NLL reference is 0.6931. These are descriptive summaries."))
    for group_name, group in report["groups"].items():
        rows = [f"**{group_name.title()} group**", "", "| Model | NLL | KL | Diagnostic NLL | Diagnostic KL |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for name in labels:
            metrics = group["models"][name]
            values = [_score_text(metrics[metric]) for metric in ("nll", "kl", "diagnostic_nll", "diagnostic_kl")]
            rows.append("| " + " | ".join([labels[name], *values]) + " |")
        rows.extend(["", "| Calibrated distilled versus | NLL gain | KL loss | Diagnostic NLL gain |",
                     "| --- | ---: | ---: | ---: |"])
        for baseline in ("solo", "solo_legacy", "coinflip", "frequency"):
            metrics = group["comparisons"][baseline]
            values = [_score_text(metrics[metric], paired=True)
                      for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain")]
            rows.append("| " + " | ".join([labels[baseline], *values]) + " |")
        rows.extend(["", "| Matched method: distilled versus solo | NLL gain | KL loss | Diagnostic NLL gain |",
                     "| --- | ---: | ---: | ---: |"])
        for mode in ("individual", "population"):
            metrics = group["matched_comparisons"][mode]
            values = [_score_text(metrics[metric], paired=True)
                      for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain")]
            rows.append("| " + " | ".join([modes[mode], *values]) + " |")
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
    print("Includes all calibration and evaluation fits; setup, evaluation and idle billing are additional.")
    output_dir = Path(report["output_dir"])
    selection_plots = sorted(output_dir.rglob("selection_audit.png"))
    calibration_plot = output_dir / "calibration" / "selection_audit.png"
    replication_plots = [plot for plot in selection_plots if plot.parent.name.startswith("seed-")]
    shown_selection = ([calibration_plot] if calibration_plot.is_file() else []) + replication_plots[:1]
    for plot in shown_selection:
        print("Checkpoint-selection audit:", plot.parent.name)
        display(Image(filename=str(plot)))
    for plot in selection_plots:
        if plot not in shown_selection:
            print("Additional selection-audit figure:", plot)
    convergence_plots = sorted(output_dir.rglob("convergence.png"))
    replication_curves = [plot for plot in convergence_plots if plot.parent.name.startswith("seed-")]
    shown_convergence = (replication_curves or convergence_plots)[:1]
    for plot in shown_convergence:
        print("Training curves:", plot.parent.name)
        display(Image(filename=str(plot)))
    for plot in convergence_plots:
        if plot not in shown_convergence:
            print("Additional convergence figure:", plot)
    print("Saved:", output_dir)
    print("Keep the entire baseline-audit-v1 folder, including protocol.json, summary.json, calibration, every seed folder, checkpoints, predictions and environment records.")
else:
    print("Baseline audit incomplete. Resume the same study cell before interpreting a repeated-seed summary.")
''')
    for index, cell in enumerate(cells):
        cell["id"] = f"baseline-cell-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "03_colab_baseline_audit.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    nbf.validate(notebook)
    destination = root / "notebooks/03_colab_baseline_audit.ipynb"
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build())
