"""Build the oracle-target control launcher without modifying notebooks 01–04."""
from pathlib import Path
import json
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/colab_oracle_source.json").read_text())
    if not isinstance(settings.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", settings["commit"]):
        raise ValueError("A full pinned runtime commit is required")
    if not isinstance(settings.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("The pinned runtime's source SHA256 is required")
    if not isinstance(settings.get("ready", False), bool):
        raise ValueError("Release readiness must be a boolean")
    previous = nbf.read(root / "notebooks/04_colab_robustness.ipynb", as_version=4)
    previous_code = [cell.source for cell in previous.cells if cell.cell_type == "code"]
    bootstrap = previous_code[0]
    # Keep the reviewed archive extraction and source checks intact.
    for variable, key in (("EXPECTED_SOURCE_SHA256", "source_sha256"),
                          ("GITHUB_REPOSITORY", "repository"), ("PINNED_COMMIT", "commit")):
        replacement = f"{variable} = {json.dumps(settings[key])}"
        bootstrap, count = re.subn(rf"^{variable} = .+$", lambda _: replacement,
                                  bootstrap, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one {variable} assignment")
    bootstrap = bootstrap.replace("ROBUSTNESS_REVISION_READY", "ORACLE_REVISION_READY")
    bootstrap, count = re.subn(r"^ORACLE_REVISION_READY = .+$",
                              f"ORACLE_REVISION_READY = {settings.get('ready', False)!r}",
                              bootstrap, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected the previous notebook's release guard")
    bootstrap = bootstrap.replace("This robustness launcher", "This oracle-control launcher")
    setup = previous_code[1]
    # Keep source activation and the GPU guard; explicitly display subprocess
    # output because Colab may otherwise show only CalledProcessError.
    old_preflight = 'subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)'
    if setup.count(old_preflight) != 1:
        raise ValueError("Expected exactly one previous pytest invocation")
    setup = setup.replace(old_preflight, '''verification = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print(verification.stdout, end="", flush=True)
verification.check_returncode()''')
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Distilled Tiny RNNs — oracle-target control

**Question:** does replacing the teacher's soft training labels with the simulator's true choice probabilities improve the same tiny learner's rare-group diagnostic policy fidelity? This follow-up uses five fresh populations. Its primary comparison is the **population-selected teacher-target student versus population-selected oracle-target student**. Individual checkpoint selection is a predeclared secondary comparison.

The oracle is a **trained H=2 student**, evaluated through its learned predictions on held-out test targets. It does not output simulator truth directly at test time. Both soft-target students use the same observed histories, inputs, initialization, optimizer, budget, and observed-choice validation selection. Only their soft labels on **training targets** differ. Simulator probabilities never select checkpoints or settings.

This remains a synthetic, conditional first-stage experiment. Better oracle-target performance would show that better targets help this learner under this recipe. Remaining oracle error does **not** isolate capacity: optimization, finite data, checkpoint selection, and capacity remain possible contributors. Neither result establishes recovery of people's reasoning, the generating mechanism, or a pure causal compression effect. Keep notebooks 03, 04, and 05 separate; do not pool their different protocols.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the five code cells in order.

The full study uses **1,000 people, 100 trials each, and a 2% rare group**. The H=32 teacher has at most **500 steps**, with early stopping; H=2 students have **4,000 steps**, evaluated every **10 steps**. Each phase has a soft **10-minute** limit and each invocation a soft **30-minute** budget. Setup, evaluation, completed steps and saving can add time. These are stopping limits, not runtime or cost forecasts.

**Resume in a live session:** rerun **ONLY the fourth code cell with the same RUN_NAME**. Do **NOT** use Run all or restart the runtime. There is no repeated-seed aggregation until every planned seed finishes. **After a disconnect that clears variables or replaces the runtime:** run the first three cells again to restore verified source, setup and smoke validation, then resume the fourth cell with the same RUN_NAME and saved Drive folder. Disconnect the runtime when finished.
""")
    code(bootstrap)
    md("""## Install and verify

The source fingerprint is checked before training. Setup activates that source in this running kernel and rejects stale imported modules. The test suite, including GPU checks when CUDA is available, runs before the smoke test. Each run records its package versions and device history.
""")
    code(setup)
    md("""## Small oracle-control smoke test

This tiny run exercises teacher-target and oracle-target learning, both checkpoint-selection methods, fixed references, repeated-seed reporting and actual saved-result resume. Its scores validate software; they are not scientific results.
""")
    code('''from distilled_rnns.oracle_control import OracleControlConfig, run_oracle_control

SMOKE_RUN = RUNS / ("oracle-control-smoke-" + EXPECTED_SOURCE_SHA256[:12])
smoke_config = OracleControlConfig.from_file(PROJECT / "configs/oracle_control_smoke.json")
smoke = run_oracle_control(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
smoke_resumed = run_oracle_control(smoke_config, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["status"] == "complete", smoke_resumed
assert smoke_resumed["groups"] == smoke["groups"], "Resumed results differ"
assert smoke_resumed["selection"] == smoke["selection"], "Fixed settings changed on resume"
print("Oracle-control smoke test and resume passed.")
''')
    md("""## Run the bounded oracle-target control

**No calibration or retuning is performed.** Solo L2 **0.1** and the primary **population** checkpoint rule are inherited from notebook 04 and fixed before viewing these test results. The fresh integer seeds are **20261411, 20261421, 20261431, 20261441, 20261451**. Each population uses 60% training targets, 20% observed-choice validation targets for checkpoints, and 20% held-out test targets.

Both soft-target students use teacher proportion **1**, temperature **1**, and student L2 **0**. The ordinary arm learns from the fitted teacher's probabilities; the oracle arm learns from simulator `true_prob` on training targets only. Shared architecture, inputs, initialization and fitting recipe make this a controlled target-source comparison. Both arms select checkpoints by **observed-choice validation NLL**: population selection uses a common step minimizing population-average validation NLL, while individual selection chooses each person's step. Independent weights remain independent in either mode. Both selections come from the same fitted trajectory, without training it twice.

The contextual scores include the teacher, fixed solo L2=0.1, fixed 50/50, and training-only choice-frequency reference. Contextual solo-versus-soft-student differences also change the training objective and L2; they do not isolate target source. Keep the default protocol. A changed source or setting needs a new run name. Do not use these test seeds to tune settings and then call them an independent assessment.
""")
    code('''RUN_NAME = "oracle-control-v1"  # Keep for resume; use a new name for new source or settings.
CONFIG_NAME = "oracle_control" if IN_COLAB else os.environ.get("DISTILLED_ORACLE_CONFIG", "oracle_control_smoke")
config = OracleControlConfig.from_file(PROJECT / "configs" / f"{CONFIG_NAME}.json")
print(json.dumps(config.to_dict(), indent=2))
report = run_oracle_control(config, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
    print("Resume: rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    md("""## Results and interpretation

NLL (observed-choice prediction error) and KL (true-policy mismatch) are lower when better. The focal **rare diagnostic ΔKL = teacher-target student KL minus oracle-target student KL**: positive values favor the learned oracle-target student. The same sign applies to all target-source NLL/KL gains. Teacher-target student means the ordinary distilled student, not the teacher model itself.

The primary comparison uses **population** checkpoints in both arms; the **individual** comparison is secondary. Inspect per-seed diagnostic ΔKL, mean and range, direction counts and support before interpretation. Missing diagnostic support stays unavailable and is excluded from supported-seed means. These are descriptive five-seed summaries, not significance tests. The teacher's own diagnostic KL and solo scores provide context; direct simulator probabilities are never reported as oracle-student predictions.

Improvement supports a contribution from imperfect teacher targets under this fixed recipe. It does not partition inherited teacher error from compression error, prove convergence, or identify a mechanism. Residual oracle error does not isolate capacity. Inspect convergence warnings and saved learning curves before interpreting either outcome. Training-only time and any T4 unit proxy omit setup, evaluation and idle billing; a proxy is unavailable for an incompatible or mixed device history.

If incomplete, follow the live-session resume or disconnected-session recovery instructions above; run this fifth cell again when all seeds finish.
""")
    code('''if report["status"] == "complete":
    from IPython.display import display, Image, Markdown
    selection = report["selection"]
    print("Fixed solo L2:", selection["solo_l2"])
    print("Primary checkpoint rule:", selection["primary_mode"], "; secondary:", selection["secondary_mode"])
    print("Fixed before test; no calibration or retuning in this study.")
    for warning in report.get("convergence_warnings", []):
        print("CHECK REQUIRED:", warning)

    def _score_text(summary, paired=False):
        support = f"({summary['supported_seed_count']}/{summary['planned_seed_count']} seeds)"
        if summary["mean"] is None:
            return "unavailable " + support
        sign = "+" if paired else ""
        return (f"{summary['mean']:{sign}.4f} [{summary['min']:{sign}.4f}, "
                f"{summary['max']:{sign}.4f}] " + support)

    def _value_text(value, paired=False):
        if value is None:
            return "unavailable"
        return f"{value:+.4f}" if paired else f"{value:.4f}"

    labels = {"teacher": "Teacher model", "solo": "Fixed solo L2=0.1 (primary)",
              "distilled": "Teacher-target student (primary)", "oracle": "Oracle-target student (primary)",
              "solo_population": "Solo: population", "solo_individual": "Solo: individual",
              "distilled_population": "Teacher-target student: population",
              "distilled_individual": "Teacher-target student: individual",
              "oracle_population": "Oracle-target student: population", "oracle_individual": "Oracle-target student: individual",
              "coinflip": "Fixed 50/50", "frequency": "Training choice frequency"}
    gain_metrics = ("nll_gain", "kl_gain", "diagnostic_nll_gain", "diagnostic_kl_gain")
    display(Markdown("**Primary outcome: rare-group diagnostic ΔKL with population checkpoints.** "
                     "ΔKL = teacher-target student minus oracle-target student; positive favors oracle-target training. "
                     "All scores are learned-model test predictions, in nats per trial. Means [minimum, maximum] "
                     "describe supported seeds; parentheses show supported/planned seeds. Lower scores are better. "
                     "The fixed 50/50 NLL reference is 0.6931. These summaries are descriptive."))
    for group_name in ("rare", "typical"):
        group = report["groups"][group_name]
        rows = [f"**{group_name.title()} group**", "", "| Model | NLL | KL | Diagnostic NLL | Diagnostic KL |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for name, label in labels.items():
            metrics = group["models"][name]
            values = [_score_text(metrics[metric]) for metric in ("nll", "kl", "diagnostic_nll", "diagnostic_kl")]
            rows.append("| " + " | ".join([label, *values]) + " |")
        rows.extend(["", "**Matched target-source comparison:** teacher-target minus oracle-target student; positive favors oracle.", "",
                     "| Checkpoint rule | NLL gain | KL gain | Diagnostic NLL gain | Diagnostic ΔKL |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        for mode in ("population", "individual"):
            values = [_score_text(group["target_comparisons"][mode][metric], paired=True) for metric in gain_metrics]
            rows.append("| " + " | ".join([mode + (" (primary)" if mode == "population" else " (secondary)"), *values]) + " |")
        rows.extend(["", "**Contextual population-checkpoint comparisons:** positive favors the named student; these do not isolate target source.", "",
                     "| Student versus reference | NLL gain | KL gain | Diagnostic NLL gain | Diagnostic KL gain |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        for name, label in (("distilled_vs_solo", "Teacher-target student versus solo"),
                            ("oracle_vs_solo", "Oracle-target student versus solo"),
                            ("oracle_vs_teacher", "Oracle-target student versus teacher model")):
            values = [_score_text(group["contextual_comparisons"][name][metric], paired=True) for metric in gain_metrics]
            rows.append("| " + " | ".join([label, *values]) + " |")
        rows.extend(["", "**Diagnostic true-policy KL by seed, primary population checkpoints** (lower scores are better)", "",
                     "| Seed | Trials / people | Teacher model | Solo | Teacher-target student | Oracle-target student | Population ΔKL (primary) | Individual ΔKL (secondary) |",
                     "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in group["per_seed"]:
            scores = [_value_text(row["models"][name]["diagnostic_kl"]) for name in ("teacher", "solo", "distilled", "oracle")]
            difference = row["target_comparisons"]["population"]["diagnostic_kl_gain"]
            secondary = row["target_comparisons"]["individual"]["diagnostic_kl_gain"]
            support = f"{row['diagnostic_count']} / {row['diagnostic_subject_count']}"
            rows.append("| " + " | ".join([str(row["seed"]), support, *scores, _value_text(difference, paired=True), _value_text(secondary, paired=True)]) + " |")
        for mode in ("population", "individual"):
            differences = [row["target_comparisons"][mode]["diagnostic_kl_gain"] for row in group["per_seed"]]
            supported = [value for value in differences if value is not None]
            positive = sum(value > 0 for value in supported)
            negative = sum(value < 0 for value in supported)
            tied = sum(value == 0 for value in supported)
            aggregate = group["target_comparisons"][mode]["diagnostic_kl_gain"]
            rows.extend(["", f"{mode.title()} diagnostic ΔKL, mean [minimum, maximum]: " + _score_text(aggregate, paired=True) + ".",
                         f"Direction counts ({mode}): {positive} favor oracle, {negative} favor teacher-target, {tied} tied; "
                         f"{len(supported)}/{aggregate['planned_seed_count']} seeds supported. Unavailable seeds are excluded."])
        display(Markdown("\\n".join(rows)))
    timing = report["timing"]
    print(f"Recorded training time: {timing['cumulative_training_seconds'] / 60:.2f} minutes.")
    units = timing["t4_equivalent_units_at_user_rate"]
    if units is None:
        print("T4 compute-unit proxy unavailable for this device history; no proxy is inferred from mixed or unrecognized devices.")
    else:
        print(f"Training-only T4 compute-unit proxy: {units:.4f} units at your supplied rate.")
    print("Includes all study fits; setup, evaluation and idle billing are additional.")
    output_dir = Path(report["output_dir"])
    for filename, label in (("selection_audit.png", "Checkpoint-selection audit"), ("convergence.png", "Training curves")):
        plots = sorted(output_dir.rglob(filename))
        for plot in plots[:1]:
            print(label + ":", plot.parent.name)
            display(Image(filename=str(plot)))
        for plot in plots[1:]:
            print("Additional " + label.lower() + " figure:", plot)
    print("Saved:", output_dir)
    print("Keep the entire oracle-control-v1 folder, including protocol.json, summary.json, every seed folder, checkpoints, predictions and environment records.")
else:
    print("Oracle-control study incomplete. Rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    for index, cell in enumerate(cells):
        cell["id"] = f"oracle-control-cell-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "05_colab_oracle_control.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    nbf.validate(notebook)
    destination = root / "notebooks/05_colab_oracle_control.ipynb"
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build())
