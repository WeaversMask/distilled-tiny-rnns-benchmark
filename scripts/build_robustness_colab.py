"""Build the robustness launcher without changing notebooks 01, 02, or 03."""
from pathlib import Path
import json
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/colab_robustness_source.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", settings["commit"]):
        raise ValueError("A full pinned runtime commit is required")
    if not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("The pinned runtime's source SHA256 is required")
    if not isinstance(settings.get("ready", False), bool):
        raise ValueError("Release readiness must be a boolean")
    original = nbf.read(root / "notebooks/03_colab_baseline_audit.ipynb", as_version=4)
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
    bootstrap = bootstrap.replace("BASELINE_REVISION_READY", "ROBUSTNESS_REVISION_READY")
    bootstrap, count = re.subn(
        r"^ROBUSTNESS_REVISION_READY = .+$",
        f"ROBUSTNESS_REVISION_READY = {settings.get('ready', False)!r}",
        bootstrap, flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Expected the prior notebook's release guard")
    bootstrap = bootstrap.replace(
        "This baseline-audit launcher is awaiting its verified runtime release.",
        "This robustness launcher is awaiting its verified runtime release.",
    )
    setup = original_code[1]  # Includes the running-kernel import/revision guard.
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Distilled Tiny RNNs — rare-policy robustness study

**Purpose:** test whether the rare group's diagnostic true-policy KL loss survives stronger solo regularization and longer student training, alongside overall NLL/KL and both checkpoint-selection controls. The teacher was also weak on the diagnostic subset; compare teacher and student scores before interpreting a gap as loss caused by distillation. This run changes only the training budget, solo regularization grid, and population seeds from notebook 03, using the same audited runtime, selection rules, and metrics.

This is a **combined-protocol robustness replication**, not an isolated training-duration experiment or proof of convergence. Report notebook 03 and notebook 04 separately; do not pool their different protocols into a ten-seed result.

This remains a synthetic, conditional first-stage experiment with a parametric rare group. It does not establish recovery of people's reasoning or reproduce the published method.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the cells in order.

The full study uses **1,000 people, 100 trials per person, and a 2% rare group**, with **500 teacher steps, 4,000 student steps, and evaluation every 10 steps**. Each phase has a soft **10-minute** limit; each study invocation has a soft **30-minute** budget. Setup, completed steps, saving, and evaluation can add time. These are stopping limits, not a runtime or cost forecast. If the study pauses while this session is still live, **rerun ONLY the fourth code cell with the same RUN_NAME; do NOT use Run all or restart the runtime to resume**. No repeated-seed summary is available until every planned seed finishes. Disconnect the runtime when finished.

**Recovery after a disconnect:** if Colab has cleared the variables or replaced the runtime, run the first three code cells again to restore the verified source, setup, and smoke check, then resume the fourth code cell with the same RUN_NAME and saved Google Drive folder. The fourth-cell-only rule applies while the same session remains live.
""")
    code(bootstrap)
    md("""## Install and verify

The source fingerprint is checked before training. Setup activates the downloaded package in this running kernel and rejects modules left over from another revision. The test suite runs before the smoke test, including GPU-specific checks when CUDA is present. Exact package versions are recorded with every run.
""")
    code(setup)
    md("""## Small robustness smoke test

This tiny study checks the separate tuning split, both checkpoint-selection methods, fixed references, repeated seeds, and resume. Its scores are software checks, not scientific findings.
""")
    code('''from distilled_rnns.baseline_audit import BaselineAuditConfig, run_baseline_audit

SMOKE_RUN = RUNS / ("robustness-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = BaselineAuditConfig.from_file(PROJECT / "configs/robustness_smoke.json")
smoke = run_baseline_audit(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
smoke_resumed = run_baseline_audit(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["groups"] == smoke["groups"], "Resumed results differ"
for field in ("solo_l2", "solo_selection", "distilled_selection"):
    assert smoke_resumed["calibration"][field] == smoke["calibration"][field]
print("Robustness smoke test and resume passed.")
''')
    md("""## Run the bounded robustness study

**Calibration uses a fresh population, seed 20261301.** For each person, 60% of trials train the models, 20% select saved checkpoints, and a separate final 20% choose settings. That last calibration split is **tuning data**, not an independent test set. No calibration test result is claimed.

Solo candidates retain L2 strengths **0, 0.001, 0.01, and 0.1** and add **0.3, 1.0, and 3.0**. Each fitted trajectory supplies two alternatives: **individual selection** chooses each person's best checkpoint; **population selection** chooses one step using mean validation NLL across the population, while every person's model weights remain independent. Tracking both alternatives does not train the same trajectory twice. The outer tuning split chooses the solo strength and selection method, and separately chooses the distilled selection method. Exact ties favor smaller L2, then population selection. Settings are selected over the population, dominated by the 98% typical majority; neither rare labels nor true-policy scores choose them.

**Evaluation freezes those choices** before testing five fresh populations: **20261311, 20261321, 20261331, 20261341, 20261351**. These are integer random seeds, not dates. Each evaluation population keeps its own 60% training, 20% checkpoint-selection, and 20% test split. Test scores do not choose settings.

The report includes the teacher, calibrated solo and distilled pipelines, both matched checkpoint-selection comparisons, a **fixed solo L2=0.01 control with individual selection and 4,000 training steps**, a fixed **50/50 reference**, and a **choice-frequency reference** estimated only from training choices with Laplace smoothing. The fixed solo control is trained longer here than in notebook 03; it is not the exact old recipe. The matched comparisons hold the checkpoint rule fixed; the training objective and solo L2 still differ.

Keep the default protocol for this run. Changing source or settings requires a new run name. Do not use these test scores to choose settings and then describe the same seeds as an independent assessment.
""")
    code('''RUN_NAME = "robustness-v1"  # Keep for resume; change for new settings or source.
CONFIG_NAME = "robustness" if IN_COLAB else os.environ.get("DISTILLED_ROBUSTNESS_CONFIG", "robustness_smoke")
config = BaselineAuditConfig.from_file(PROJECT / "configs" / f"{CONFIG_NAME}.json")
print(json.dumps(config.to_dict(), indent=2))
report = run_baseline_audit(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
    print("Resume: rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    md("""## Results and what to inspect

Prediction error (NLL) and policy mismatch (KL) are lower when better. A constant 50/50 prediction has NLL **0.6931**. The population-level means and ranges below describe the five fresh seeds; they are not significance tests. Calibration is excluded.

The main comparison uses the independently calibrated solo and distilled pipelines. The matched-method rows hold the checkpoint rule fixed; the training objective and solo L2 still differ, so these rows do not isolate a pure causal effect of distillation. Selected-pipeline rows can duplicate their matching individual/population rows.

The selection-audit figures compare scores on checkpoint-selection targets with separate calibration tuning targets or fresh-population test targets. They help show how well the selected checkpoints generalize beyond the data used to choose them. Calibration tuning scores remain part of model selection, not an independent test result.

Diagnostic scores use the same predeclared subset and report its support. Missing support remains unavailable. **Diagnostic KL loss is distilled minus the stated comparison model**; a positive value indicates worse true-policy fidelity for the distilled model. Both paired tables show this quantity. The per-seed table also compares the teacher and calibrated students on that same subset, because an already weak teacher can limit the interpretation. These descriptive comparisons do not establish causality or statistical significance.

The inherited plots label the fixed solo control **"Previous solo recipe"**. Here that label means L2=0.01 with individual checkpoint selection and this run's student training budget: **4,000 steps in the full study**, not the original 1,000-step recipe.

Inspect the learning curves and training-boundary warnings before claiming convergence. These models predict choices conditional on observed histories; they do not establish recovery of the generating mechanism on new free-running histories. If the run is incomplete and the same session remains live, rerun ONLY the fourth code cell with the same RUN_NAME; do NOT use Run all or restart the runtime to resume. If variables were cleared after a disconnect, follow the recovery instructions above first. Run this results cell again after the study finishes.
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
              "solo_legacy": f"Fixed solo L2=0.01, individual ({config.base.student_steps:,} steps)",
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
        rows.extend(["", "| Calibrated distilled versus | NLL gain | KL loss | Diagnostic NLL gain | Diagnostic KL loss |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        for baseline in ("solo", "solo_legacy", "coinflip", "frequency"):
            metrics = group["comparisons"][baseline]
            values = [_score_text(metrics[metric], paired=True)
                      for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain", "diagnostic_fidelity_loss")]
            rows.append("| " + " | ".join([labels[baseline], *values]) + " |")
        rows.extend(["", "| Matched method: distilled versus solo | NLL gain | KL loss | Diagnostic NLL gain | Diagnostic KL loss |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        for mode in ("individual", "population"):
            metrics = group["matched_comparisons"][mode]
            values = [_score_text(metrics[metric], paired=True)
                      for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain", "diagnostic_fidelity_loss")]
            rows.append("| " + " | ".join([modes[mode], *values]) + " |")
        support = [f"{row['seed']}: {row['diagnostic_count']} trials / {row['diagnostic_subject_count']} people"
                   for row in group["per_seed"]]
        rows.extend(["", "Diagnostic test support — " + "; ".join(support) + "."])
        rows.extend(["", "**Diagnostic true-policy KL by seed** (lower is better; same diagnostic subset)", "",
                     "| Seed | Teacher | Calibrated solo | Calibrated distilled | Distilled minus solo |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        diagnostic_differences = []
        for row in group["per_seed"]:
            values = [row["models"][name]["diagnostic_kl"] for name in ("teacher", "solo", "distilled")]
            scores = ["unavailable" if value is None else f"{value:.4f}" for value in values]
            difference = row["comparisons"]["solo"]["diagnostic_fidelity_loss"]
            difference_text = "unavailable" if difference is None else f"{difference:+.4f}"
            if difference is not None:
                diagnostic_differences.append(difference)
            rows.append("| " + " | ".join([str(row["seed"]), *scores, difference_text]) + " |")
        higher = sum(value > 0 for value in diagnostic_differences)
        lower = sum(value < 0 for value in diagnostic_differences)
        tied = sum(value == 0 for value in diagnostic_differences)
        primary_diagnostic = group["comparisons"]["solo"]["diagnostic_fidelity_loss"]
        rows.extend(["", "Calibrated pipeline diagnostic KL difference, mean [minimum, maximum]: "
                     + _score_text(primary_diagnostic, paired=True) + ".",
                     f"Direction counts: {higher} higher, {lower} lower, {tied} tied; "
                     f"{len(diagnostic_differences)}/{primary_diagnostic['planned_seed_count']} seeds supported. "
                     "Higher means worse true-policy fidelity for distilled students; unavailable seeds are excluded."])
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
    print(f'Plot label note: "Previous solo recipe" means fixed solo L2=0.01 with individual checkpoint selection and {config.base.student_steps:,} training steps in this run, not the original 1,000-step recipe.')
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
    print("Keep the entire robustness-v1 folder, including protocol.json, summary.json, calibration, every seed folder, checkpoints, predictions and environment records.")
else:
    print("Robustness study incomplete. Rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    for index, cell in enumerate(cells):
        cell["id"] = f"robustness-cell-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "04_colab_robustness.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    nbf.validate(notebook)
    destination = root / "notebooks/04_colab_robustness.ipynb"
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build())
