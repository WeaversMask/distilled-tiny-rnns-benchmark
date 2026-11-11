# Focused robustness replication — notebook 04

## Question and prior evidence

Does the rare diagnostic policy-mismatch penalty persist with a wider solo
regularization grid and a longer student training horizon, while average
prediction and average policy fidelity remain better for distilled models?

The completed `baseline-audit-v1` used 1,000 student steps, not 4,000. Its five
fresh populations showed better rare-group observed-choice prediction in four
of five seeds and better average rare-group true-policy KL in all five. Rare
diagnostic KL was worse for distillation in all five under the frozen primary
pipeline. With individual checkpoint selection, that deterioration occurred in
three of five. The teacher also had worse diagnostic KL than the selected solo
baseline in all five. These are descriptive results on simulated observed
histories, not proof that compression causes loss of a reasoning mechanism.

The previous solo selection chose the largest tested L2 (0.1), and the selected
population checkpoint for distillation was at the final step in three of five
replications. Notebook 04 addresses these boundaries with a predeclared combined
protocol change. It is not a controlled ablation of training duration or L2.

## Frozen design

Use the existing, reviewed `BaselineAuditConfig` / `run_baseline_audit` engine.
No numerical model, optimizer, scoring, simulator, or resume code changes for
this study. Configuration is in `configs/robustness.json`.

| Setting | Notebook 03 | Notebook 04 |
| --- | --- | --- |
| Calibration RNG seed | 20261201 | 20261301 |
| Evaluation RNG seeds | 20261211, 20261221, 20261231, 20261241, 20261251 | 20261311, 20261321, 20261331, 20261341, 20261351 |
| Student steps | 1,000 | 4,000 |
| Solo L2 candidates | 0, 0.001, 0.01, 0.1 | 0, 0.001, 0.01, 0.1, 0.3, 1, 3 |

These are integer random seeds, not dates. Every population still has 1,000
simulated people, 100 trials per person, and exactly 20 parametric rare agents.
Teacher size stays 32, student size 2. The teacher still has a 500-step maximum,
with early stopping after 20 evaluations without improvement. Both student arms
use the fixed 4,000-step horizon, evaluated every 10 steps. Other training settings
and the diagnostic definition remain unchanged. This longer horizon offers more
checkpoint-selection opportunities; it does not guarantee convergence.

Calibration uses separate 60% fitting, 20% checkpoint-selection, and 20% outer
tuning targets. Outer observed-choice NLL chooses one global solo (L2, checkpoint
rule), and independently chooses the distilled checkpoint rule. Choices freeze
before scoring fresh-seed tests. Exact tuning ties prefer lower L2, then population
selection. No oracle probabilities, diagnostic metrics, rare labels, or test
outcomes select settings. Calibration has no independent test result. The typical
98% majority dominates the population-level tuning objective.

Fresh evaluation populations each use their own 60% fitting, 20% checkpoint, and
20% test targets. Original seed offsets for splits and model initialization are
retained. Their ranges do not overlap the preceding studies. Tests are interspersed
within observed histories, not held-out sessions: past observations can be inputs
to later predictions without being direct fitting targets.

## Outcomes specified before running

The focal outcome is rare-group **diagnostic KL difference = distilled minus
solo**, using the independently calibrated pipelines. Positive values mean worse
fidelity for distillation. Report its five paired seed values, equal-weight mean,
range, number of positive/negative/tied seeds, and supported trials and people.
Keep the existing diagnostic criterion: a held-out trial where the generating
policy differs by more than 0.2 from the fixed majority-like policy (weight 0.75)
on the same history and other parameters. This is not an estimated population
mean. Missing diagnostic support remains null, never zero.

Read this focal result alongside observed-choice NLL and overall true-policy KL
in both groups. NLL gain is solo minus distilled (positive favors distillation);
KL loss is distilled minus solo (positive means worse fidelity). Also report
diagnostic NLL separately: actual sampled-choice error and generating-probability
mismatch are different outcomes and may disagree on a small subset.

The planned controls are retained:

1. Individual versus individual and population versus population checkpoint rules,
   using the globally selected solo L2. Do not choose the more favorable rule
   after seeing results. These controls match the checkpoint rule, not the training
   objectives or regularization.
2. Teacher scores on the same trials. Teacher-versus-solo and student-versus-teacher
   gaps are descriptive comparisons, not a causal decomposition of inherited bias
   and compression error.
3. Fixed solo L2 0.01 with individual checkpoints, a fair-coin predictor, and a
   train-only Beta(1,1) smoothed choice-frequency predictor. The fixed solo control
   now also receives 4,000 steps, so it is not an exact rerun of notebook 03's
   1,000-step legacy recipe. The saved machine-readable key stays `solo_legacy`
   for compatibility with the unchanged engine.

No significance test or acceptance threshold is introduced. The five new seeds
assess repeatability conditional on one new calibration population, not stability
across many calibrations. Keep this study separate from notebook 03; do not pool
the two different protocols and describe them as a ten-seed replication.
Five seeds and small
rare diagnostic subsets still limit inference. A smaller or reversed difference
is a valid result. Even a consistent difference would support only a localized
conditional-prediction finding, not overall fidelity loss or mechanism recovery.

## Boundaries and running

Keep teacher and student convergence warnings visible. If the selected L2 is still
the largest candidate, or selected checkpoints remain at the horizon, report that
limitation. Do not adapt this run's grid/budget after seeing its test scores and
reuse those scores as independent evidence. A new calibration and new evaluation
seeds would be needed for another adaptation.

There is a soft 30-minute study invocation limit and a soft 10-minute phase limit.
The study is expected to need multiple invocations; no GPU-time guarantee is made.
When it pauses, **rerun only the fourth code cell with the same `RUN_NAME`**
(`robustness-v1`). Keep the runtime connected while continuing. If Colab has
disconnected and cleared the Python session, rerun setup and smoke first to restore
the session, then resume the unchanged study and run name. Do not rename a run to
resume it. Use only one runtime per output folder.

The entire study saves to `MyDrive/distilled-tiny-rnns/runs/robustness-v1`;
software checks use a separate `robustness-smoke-<source hash>` folder. Retain both
folders, including checkpoints, predictions, source/config identities, environment
inventories, and plots. No complete aggregate is produced before all five seeds
finish. Training cost is counted once per optimizer trajectory, including every
calibration candidate. Only entirely T4 sessions receive a training-time conversion
at the supplied 1.07 units/hour; setup, evaluation, and idle billing are additional.
The reported notebook 03 cost is not a guaranteed forecast for this longer study.

Open `notebooks/04_colab_robustness.ipynb` in a fresh T4 Colab session and run its
five code cells in order. The launcher downloads the pinned source from public
GitHub. Authorize the Google Drive mount when prompted. This study keeps earlier
notebook pins and saved result folders intact. It does not launch the full research grid, change the
teacher objective, or add human-data or free-running evaluations.
