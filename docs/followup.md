# Follow-up protocol: longer training and frozen solo regularization

This is an exploratory synthetic, conditional first-stage study. It checks the
first pilot's sensitivity to training length and solo overfitting. It does not
establish the proposal's mechanistic hypothesis or reproduce the authors' method.

## Why this follow-up

The initial T4 run used 1,000 people, 100 trials, and 20 rare people. Distillation
improved mean rare-group NLL and true-policy KL relative to solo training. Solo
training showed overfitting, and 28.8% of distilled students selected the final
available step. The observed diagnostic difference had only 53 test trials in
14 rare people. These observations motivated this protocol; they are not
independent evidence for its results.

## Fixed plan

`configs/followup.json` specifies the entire plan. Before training, the runner
writes `protocol.json` with its configuration, metric conventions, and numerical
source hash. It rejects a different protocol or source in the same run folder.

- Calibration uses the already inspected seed **20261029**. It is exploratory.
- Teacher: up to **500** steps, validation every **10**, early stopping after
  **20** evaluations without improvement. Best pooled validation weights are used.
- Independent students: **1,000** steps each, validation every **10**, with each
  person's best validation checkpoint selected separately, including step zero.
- Solo candidates: L2 strengths **0, 0.001, 0.01, 0.1**. The penalty is
  `(lambda / 2) * sum(parameter²)`, including biases, added to the sum of
  per-person mean training losses. It is not AdamW decay. Validation NLL is
  unpenalized. Summing preserves equivalence to independent separate fits.
- Select one global strength using the mean of the candidates' selected
  per-person observed-choice validation NLL. Exact ties favor smaller strengths.
  This is a **population-selected** baseline, dominated by the 98% typical group;
  it is not selected specifically for rare people. Group labels, test scores,
  oracle probabilities, and diagnostic subsets do not select it.
- Save `calibration/selection.json` before any test or oracle scoring. Freeze that
  strength across five new simulation seeds: **20261101, 20261111, 20261121, 20261131, 20261141** (integer RNG seeds, not dates).
- Within every seed, all solo candidates and distilled students start from
  identical weights. Distillation retains the original full soft-target weight
  and temperature of one; it has no extra L2 penalty.
- Report teacher, selected solo, longer unregularized solo, and distilled models.
  If zero wins, reuse its model rather than fitting an identical arm again.

Population seeds are spaced to avoid reusing the simulator/split/initialization seed offsets across populations.

The simulator, population size, splits, features, learning rate, and network sizes
match the first pilot. All phases retrain in new folders; the original notebooks,
checkpoints, and scores are retained. This run does not add mechanisms, larger
populations, a full parameter grid, or human data.

## What the results mean

Calibration is excluded from the repeated-seed aggregate. Within each seed,
subject scores are averaged within group. Across seeds, report equal-weight
descriptive means and ranges of paired differences; do not pool all people as
independent replication units. There are no significance tests or hypothesis
acceptance thresholds.

`solo` in dataset reports means the frozen population-selected baseline;
`solo_unregularized` means the longer unregularized control. For each baseline:

- Prediction gain = baseline NLL minus distilled NLL; positive favors distillation.
- Fidelity loss = distilled KL minus baseline KL; positive means worse fidelity.
- Diagnostic prediction gain uses only supported diagnostic test trials.

Diagnostic means omit unsupported subjects within a seed, and unsupported seeds
across seeds. Reports preserve per-seed trial/subject support and the supported
seed count; absent support remains `null`, never zero. The diagnostic reference
is the same fixed majority-like policy as the original pilot.

These scores condition on observed histories. Earlier held-out observations can
enter later causal inputs, but held-out target losses do not train weights.
Evaluation therefore does not demonstrate recovery on free-running histories.

## Convergence and compute controls

Each dataset saves `convergence.png`, per-phase learning histories, selected-step
summaries, and the fraction selected at the step limit. The runner flags more
than 10% selecting the limit or a raw validation improvement exceeding 0.0001
over the last up-to-five evaluation intervals at the limit. These are inspection
heuristics, not a test proving convergence. Curves show the current weights;
reported scores use the best selected weights. Choosing the largest tested L2
also triggers a request to inspect the grid in a separate future protocol.

The study has a **soft 30-minute invocation budget**, and a **10-minute per-phase
budget**, checked between steps/phases. Setup, checkpoint writes, prediction,
plotting, and tests can add time; these are not billing caps. A time-limited phase
saves its state and returns incomplete. Rerunning the same cell resumes the same
study. No complete aggregate is published until every planned seed completes.
Use one runtime per output folder.

The cost report includes all calibration candidates plus all replication fits.
Conversion at the user's supplied 1.07 units/T4-hour is available only when all
recorded sessions used T4 GPUs. Setup, evaluation, and idle billing are additional.
The first pilot's timing is not a guarantee of this longer study's runtime.

## Running and retaining results

Open `notebooks/02_colab_followup.ipynb` in a fresh T4 Colab session and run cells
in order. The launcher downloads the pinned source from public GitHub. Authorize
the Google Drive mount when prompted. Tests and a tiny follow-up/resume check
run before the study.
The pinned runtime commit and source hash are in `configs/colab_followup_source.json`.

Results go to `MyDrive/distilled-tiny-rnns/runs/followup-v1`. Keep the whole folder:
`protocol.json`, `summary.json`, `calibration/`, and all `seed-*` folders, including
their selection, data, predictions, checkpoints, reports, and environment records.
Read convergence warnings before interpreting mean differences. Any subsequent
protocol change needs a new run name and, for an independent assessment, fresh
evaluation seeds.
