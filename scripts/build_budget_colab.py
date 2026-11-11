"""Build notebook 06 without changing the five earlier launchers."""
from pathlib import Path
import json
import re

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/colab_budget_source.json").read_text())
    if not isinstance(settings.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", settings["commit"]):
        raise ValueError("A full pinned runtime commit is required")
    if not isinstance(settings.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("The pinned runtime's source SHA256 is required")
    if not isinstance(settings.get("ready", False), bool):
        raise ValueError("Release readiness must be a boolean")
    previous = nbf.read(root / "notebooks/05_colab_oracle_control.ipynb", as_version=4)
    previous_code = [cell.source for cell in previous.cells if cell.cell_type == "code"]
    bootstrap = previous_code[0]
    for variable, key in (("EXPECTED_SOURCE_SHA256", "source_sha256"),
                          ("GITHUB_REPOSITORY", "repository"), ("PINNED_COMMIT", "commit")):
        bootstrap, count = re.subn(rf"^{variable} = .+$", lambda _: f"{variable} = {json.dumps(settings[key])}",
                                  bootstrap, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one {variable} assignment")
    bootstrap = bootstrap.replace("ORACLE_REVISION_READY", "BUDGET_REVISION_READY")
    bootstrap, count = re.subn(r"^BUDGET_REVISION_READY = .+$",
                              f"BUDGET_REVISION_READY = {settings.get('ready', False)!r}",
                              bootstrap, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected the previous notebook's release guard")
    bootstrap = bootstrap.replace("This oracle-control launcher", "This budget-check launcher")
    setup = previous_code[1]  # Reviewed source activation, stale-module/GPU checks, visible pytest output.
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md("""# Distilled Tiny RNNs — one bounded budget check

**Question:** how do the saved notebook 05 results change when the same teacher-target and oracle-target H=2 students continue from **4,000 to 8,000 total steps**? This is one predeclared extension of both arms on all five existing populations. The teacher, fixed solo L2=0.1, data, splits, initialization, optimizer and target sources remain fixed. It is not a fresh five-seed replication.

Finish Notebook 05 in your own Drive before running the full study. Its folder **`runs/oracle-control-v1` is a read-only parent**. Keep it complete, including every checkpoint and saved prediction. Notebook 06 validates the finished parent, copies continuation state into **`runs/budget-check-v1`**, and writes only the child. Do not rename or delete the parent. Do not run two notebooks against the same child folder at once.

Open this notebook in a fresh Colab session and select a **T4 GPU**. The first cell downloads a fixed public revision without a GitHub token and mounts your Google Drive for results. Run the five code cells in order.

The full study keeps **1,000 people, 100 trials, a 2% rare group**, and seeds **20261411, 20261421, 20261431, 20261441, 20261451**. Population checkpoint selection is primary; individual selection is secondary. Both select cumulatively from the original **0–4,000** and added **4,001–8,000** trajectory using observed-choice validation NLL, evaluated every 10 steps. This tests extra training **and more checkpoint candidates**; it does not prove convergence or isolate model capacity. Simulator truth and held-out test scores select neither checkpoints nor the training horizon.

Each phase has a soft **10-minute** limit and each invocation a soft **30-minute** limit; setup, evaluation, saving and a completed step can add time. These are stopping limits, not runtime or cost forecasts. **Live-session resume: rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime.** If a disconnect clears variables or replaces the runtime, run the first three cells again, then the fourth using the same parent and child folders. A completed child is final: reruns only verify/reload it, without more training. Do not repeatedly extend the horizon in response to these test scores. Disconnect the runtime when finished.

This remains a synthetic, conditional first-stage experiment, not a reproduction claim, mechanism recovery or significance test. Keep notebook 05's original results and notebook 06's paired extension results separate.
""")
    code(bootstrap)
    md("""## Install and verify

Activate the verified source in the running kernel, reject stale imports, require a GPU in Colab, and display the complete test output before continuing. A failed check stops this notebook before training.
""")
    code(setup)
    md("""## Small parent-to-child smoke test

Create a separate tiny, real oracle-control parent using this revision, then extend its students from 4 to 8 steps. Check that completed resume changes neither predictions nor recorded compute and that the parent stays byte-for-byte unchanged. These smoke folders are separate from the production parent and child; smoke scores are software checks, not scientific evidence.
""")
    code('''from dataclasses import replace
from distilled_rnns.oracle_control import OracleControlConfig, run_oracle_control
from distilled_rnns.budget_check import BudgetCheckConfig, run_budget_check, validate_parent

def _file_fingerprints(folder, pattern="*"):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob(pattern)) if path.is_file()}

SMOKE_PARENT = RUNS / ("budget-check-parent-smoke-" + EXPECTED_SOURCE_SHA256[:12])
SMOKE_RUN = RUNS / ("budget-check-smoke-" + EXPECTED_SOURCE_SHA256[:12])
parent_smoke_config = OracleControlConfig.from_file(PROJECT / "configs/oracle_control_smoke.json")
smoke_config = BudgetCheckConfig.from_file(PROJECT / "configs/budget_check_smoke.json")
# Only this newly generated smoke parent uses the current runtime fingerprint.
# Production always trusts the frozen notebook 05 fingerprint from its config file.
smoke_config = replace(smoke_config, expected_parent_source_sha256=EXPECTED_SOURCE_SHA256)
parent_summary_path = SMOKE_PARENT / "summary.json"
parent_smoke = json.loads(parent_summary_path.read_text()) if parent_summary_path.is_file() else None
if parent_smoke is None or parent_smoke.get("status") != "complete":
    if SMOKE_RUN.exists():
        raise RuntimeError("The smoke child already exists but its parent is incomplete. Preserve both folders and investigate; do not rebuild the parent.")
    parent_smoke = run_oracle_control(parent_smoke_config, SMOKE_PARENT, device=DEVICE)
assert parent_smoke["status"] == "complete", parent_smoke
# Reuse a completed parent without calling its runner: that would rewrite its environment log.
validate_parent(smoke_config, SMOKE_PARENT, device=DEVICE)
parent_before = _file_fingerprints(SMOKE_PARENT)
smoke = run_budget_check(smoke_config, SMOKE_PARENT, SMOKE_RUN, device=DEVICE)
assert smoke["status"] == "complete", smoke
predictions_before = _file_fingerprints(SMOKE_RUN, "*predictions.npz")
assert predictions_before, "Smoke run did not save predictions"
smoke_resumed = run_budget_check(smoke_config, SMOKE_PARENT, SMOKE_RUN, device=DEVICE)
assert smoke_resumed["status"] == "complete", smoke_resumed
assert smoke_resumed["groups"] == smoke["groups"], "Resumed results differ"
assert smoke_resumed["timing"] == smoke["timing"], "Completed resume added compute"
assert _file_fingerprints(SMOKE_RUN, "*predictions.npz") == predictions_before, "Completed resume changed predictions"
assert _file_fingerprints(SMOKE_PARENT) == parent_before, "The parent was modified"
print("Budget-check smoke test and completed resume passed; parent unchanged.")
''')
    md("""## Continue the fixed 4,000 → 8,000 protocol

The production configuration comes directly from `configs/budget_check.json`. The parent must contain the finished, original notebook 05 protocol and all five seeds. A missing, incomplete, changed or incompatible parent stops the run. **Do not fix a failed parent check by changing the trusted source hash or weakening validation.**

Only teacher-target and oracle-target students continue. The teacher and solo references are inherited, with no refitting, calibration or retuning. The new child preserves the original 4,000-step results for side-by-side comparisons. No aggregate is reported until both arms finish for every planned seed.
""")
    code('''RUN_NAME = "budget-check-v1"  # Keep this fixed for the one authorized extension and all resumes.
if IN_COLAB:
    config = BudgetCheckConfig.from_file(PROJECT / "configs/budget_check.json")
    PARENT_RUN = RUNS / "oracle-control-v1"
else:
    # Local five-cell verification uses the actual tiny parent created above.
    config = smoke_config
    PARENT_RUN = SMOKE_PARENT
print(json.dumps(config.to_dict(), indent=2))
report = run_budget_check(config, PARENT_RUN, RUNS / RUN_NAME, device=DEVICE)
if report["status"] != "complete":
    print(report["message"])
    print("Resume: rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    md("""## Original versus extended results

NLL and KL are lower when better. **Target-source ΔKL = teacher-target student KL minus oracle-target student KL**: positive favors the learned oracle-target student. The main outcome is rare-group diagnostic ΔKL with **population** checkpoints; **individual** selection is secondary. Typical-group results and overall scores provide context.

The paired gain change is **extended ΔKL minus baseline ΔKL**; a positive value means the oracle advantage grew. Each arm's error change is **extended error minus baseline error**; negative means improvement. A smaller target-source gap alone does not imply both models improved. Means and ranges include only supported seeds; unavailable diagnostic scores remain unavailable. Inspect support, every seed and training-boundary warnings. These are paired descriptive comparisons, not significance tests or proof of convergence. A remaining oracle error does not isolate capacity.

Only **incremental extension training time** and its eligible T4 unit proxy are charged to this check. Original parent training is shown separately. Setup, evaluation, saving and idle billing are additional; no proxy is inferred for an incompatible or mixed device history.
""")
    code('''if report["status"] == "complete":
    from IPython.display import display, Markdown

    def _score_text(summary, signed=False):
        support = f"({summary['supported_seed_count']}/{summary['planned_seed_count']} seeds)"
        if summary["mean"] is None:
            return "unavailable " + support
        sign = "+" if signed else ""
        return (f"{summary['mean']:{sign}.4f} [{summary['min']:{sign}.4f}, "
                f"{summary['max']:{sign}.4f}] " + support)

    def _value_text(value, signed=False):
        return "unavailable" if value is None else (f"{value:+.4f}" if signed else f"{value:.4f}")

    baseline_steps = config.parent_config.base.student_steps
    extended_steps = config.student_steps
    display(Markdown("**Primary outcome: rare-group diagnostic ΔKL with population checkpoints.** "
                     "ΔKL = teacher-target minus oracle-target student; positive favors oracle. "
                     "Individual checkpoints are secondary. Values are seed means [minimum, maximum], "
                     "with supported/planned seed counts. These paired summaries are descriptive."))
    for warning in report.get("convergence_warnings", []):
        print("CHECK REQUIRED:", warning)
    for group_name in ("rare", "typical"):
        group = report["groups"][group_name]
        rows = [f"**{group_name.title()} group: original versus extended**", "",
                f"| Checkpoint rule | Metric | Baseline Δ ({baseline_steps:,}) | Extended Δ ({extended_steps:,}) | Paired gain change |",
                "| --- | --- | ---: | ---: | ---: |"]
        for mode in ("population", "individual"):
            label = mode + (" (primary)" if mode == "population" else " (secondary)")
            for metric in ("diagnostic_kl_gain", "kl_gain", "diagnostic_nll_gain", "nll_gain"):
                values = [_score_text(group[section][mode][metric], signed=True)
                          for section in ("baseline_target_comparisons", "extended_target_comparisons", "paired_gain_change")]
                rows.append("| " + " | ".join([label, metric, *values]) + " |")
        rows.extend(["", "**Each arm's error: negative change means improvement. Teacher and solo stay fixed.**", "",
                     f"| Checkpoint rule / model | Metric | Baseline ({baseline_steps:,}) | Extended ({extended_steps:,}) | Error change |",
                     "| --- | --- | ---: | ---: | ---: |"])
        for mode in ("population", "individual"):
            for arm, label in (("distilled", "Teacher-target student"), ("oracle", "Oracle-target student")):
                for metric in ("diagnostic_kl", "kl", "diagnostic_nll", "nll"):
                    values = [_score_text(group[section][f"{arm}_{mode}"][metric])
                              for section in ("baseline_models", "extended_models")]
                    values.append(_score_text(group["arm_changes"][mode][arm][metric], signed=True))
                    rows.append("| " + " | ".join([f"{mode}: {label}", metric, *values]) + " |")
        for model, label in (("teacher", "Fixed teacher"),
                             ("solo_population", "Fixed solo L2=0.1 (population)"),
                             ("solo_individual", "Fixed solo L2=0.1 (individual)")):
            for metric in ("diagnostic_kl", "kl", "diagnostic_nll", "nll"):
                values = [_score_text(group[section][model][metric])
                          for section in ("baseline_models", "extended_models")]
                rows.append("| " + " | ".join([label, metric, *values, "fixed reference"]) + " |")
        rows.extend(["", "**Diagnostic support and paired ΔKL by seed**", "",
                     "| Seed | Trials / people | Checkpoint rule | Baseline ΔKL | Extended ΔKL | Gain change | Teacher-target error change | Oracle-target error change |",
                     "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in group["per_seed"]:
            for mode in ("population", "individual"):
                values = [_value_text(row[section][mode]["diagnostic_kl_gain"], signed=True)
                          for section in ("baseline_target_comparisons", "extended_target_comparisons", "paired_gain_change")]
                values.extend(_value_text(row["arm_changes"][mode][arm]["diagnostic_kl"], signed=True)
                              for arm in ("distilled", "oracle"))
                rows.append("| " + " | ".join([str(row["seed"]),
                            f"{row['diagnostic_count']} / {row['diagnostic_subject_count']}", mode, *values]) + " |")
        for mode in ("population", "individual"):
            values = [row["paired_gain_change"][mode]["diagnostic_kl_gain"] for row in group["per_seed"]]
            supported = [value for value in values if value is not None]
            rows.extend(["", f"Paired diagnostic gain change ({mode}): "
                         f"{sum(value > 0 for value in supported)} positive, {sum(value < 0 for value in supported)} negative, "
                         f"{sum(value == 0 for value in supported)} tied; {len(supported)}/{len(values)} seeds supported."])
            for arm, label in (("distilled", "Teacher-target"), ("oracle", "Oracle-target")):
                errors = [row["arm_changes"][mode][arm]["diagnostic_kl"] for row in group["per_seed"]]
                supported_errors = [value for value in errors if value is not None]
                rows.extend(["", f"{label} diagnostic KL error change ({mode}): "
                             f"{sum(value < 0 for value in supported_errors)} improved, "
                             f"{sum(value > 0 for value in supported_errors)} worsened, "
                             f"{sum(value == 0 for value in supported_errors)} tied; "
                             f"{len(supported_errors)}/{len(errors)} seeds supported."])
        display(Markdown("\\n".join(rows)))
    timing = report["timing"]
    print(f"Incremental extension training: {timing['incremental_training_seconds'] / 60:.2f} minutes.")
    print(f"Original parent training (not new work): {timing['parent_training_seconds'] / 60:.2f} minutes.")
    units = timing["t4_equivalent_units_at_user_rate"]
    if units is None:
        print("Incremental T4 compute-unit proxy unavailable; no proxy is inferred from mixed or unrecognized devices.")
    else:
        print(f"Incremental training-only T4 compute-unit proxy: {units:.4f} units at your supplied rate.")
    print("Setup, evaluation, saving and idle billing are additional.")
    print("Saved:", report["output_dir"])
    print("Keep the full oracle-control-v1 parent and budget-check-v1 child folders. The completed extension is final; reruns add no training.")
else:
    print("Budget check incomplete. Rerun ONLY the fourth code cell with the same RUN_NAME. Do NOT use Run all or restart the runtime. No aggregation until all planned seeds complete.")
''')
    for index, cell in enumerate(cells):
        cell["id"] = f"budget-check-cell-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "06_colab_budget_check.ipynb", "provenance": []}, "accelerator": "GPU",
    })
    nbf.validate(notebook)
    destination = root / "notebooks/06_colab_budget_check.ipynb"
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build())
