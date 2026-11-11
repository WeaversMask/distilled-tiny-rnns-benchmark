# Baseline and validation audit

The preceding five-seed study found that distillation improved both average
prediction and true-policy fidelity over the solo students. Both solo recipes
still predicted worse than a constant 50/50 reference (NLL = log(2), about 0.6931).
Selecting each person's checkpoint from only 20 validation trials may contribute
to a large selection advantage on validation that does not generalize to test.
That is a hypothesis to audit, not a diagnosed implementation error.

Notebook `03_colab_baseline_audit.ipynb` tests this question using a fixed new
protocol. The earlier notebooks, pins, results, and downloaded folders are retained.

## Separate target roles

Calibration uses a fresh population with integer RNG seed **20261201**. Every
person has 100 trials, partitioned into 60 fitting targets, 20 checkpoint-selection
targets, and 20 outer tuning targets. Models see fitting targets in their loss;
checkpoint targets select saved weights. Only the separate outer tuning targets
choose regularization and checkpoint-selection rules. Calibration therefore has
**no independent test score**. Small smoke configurations use rounded counts.

After those choices are frozen, five fresh populations use seeds **20261211,
20261221, 20261231, 20261241, 20261251**. Each has its own 60/20/20 fitting,
checkpoint-selection, and test targets. The last split reports results; it never
chooses settings. Seeds are spaced to avoid overlapping data/split/initialization
seed offsets. They are RNG integers, not calendar dates.

Splits remain interspersed. These are conditional predictions on observed histories:
past held-out observations may enter subsequent causal inputs. The split roles
separate direct target use, not entire histories. This is not a new-session or
free-running test of reasoning recovery.

## Two checkpoint rules from one fit

Each independent student trajectory records two possible final models:

- **Individual:** each person gets their own checkpoint with the lowest observed
  choice NLL on that person's checkpoint-selection targets.
- **Population:** all people use weights from one shared training step, chosen by
  mean checkpoint-selection NLL across people. Their parameter sets remain
  independent. Only the decision about training duration is shared.

Both snapshots are tracked at the same evaluation times. Tracking does not change
the optimization trajectory or train a duplicate model. Strict improvements replace
saved snapshots; exact checkpoint-score ties retain the earlier step. Both snapshot
types, optimizer state, RNG state, and completed step count survive resume.

The teacher retains pooled checkpoint selection and early stopping. Budgets remain
500 teacher steps, 1,000 student steps, validation every 10 steps, and teacher
patience of 20 evaluations. Student trajectories run their fixed step budget.

## Frozen settings and comparisons

Calibration fits solo L2 strengths **0, 0.001, 0.01, 0.1**. The established penalty
is `(lambda/2) * sum(parameter²)`, including biases, added to summed per-person
mean fitting losses. Validation/tuning scores are unpenalized observed-choice NLL.
Distillation retains its original soft-target weight and temperature of one, with
no added L2 penalty. Arms use identical student initialization within a seed.

Outer tuning NLL chooses one global solo `(L2, checkpoint rule)` and independently
chooses the distilled checkpoint rule. Exact tuning ties favor lower L2 and then
population selection. Neither rare labels, oracle probabilities, diagnostic
subsets nor fresh-seed test scores select these choices. Population-level choices
are dominated by the 98% typical majority; they are not rare-group-optimal tuning.

The predeclared comparisons are:

1. **Primary:** independently calibrated solo and distilled pipelines. This measures
   pipeline performance; it does not isolate the effect of distillation alone.
2. **Matched checkpoint rules:** individual versus individual, and population versus
   population, each using the frozen selected solo L2. These hold the checkpoint
   rule fixed; the training objectives and solo regularization still differ.
3. **Previous solo recipe:** individual checkpoints with fixed L2 = 0.01. Its weights
   are fitted on each fresh population; no old evaluation outputs are reused.
4. **Fixed references:** probability 0.5, and each person's observed fitting-choice
   frequency with fixed Beta(1,1) smoothing, `(number of ones + 1)/(fit count + 2)`.
   These references have no validation/tuning choices. Frequency uses fitting
   targets only, estimated once; it does not update from held-out outcomes.

Do not choose which comparison to headline after inspecting test results. The
matched-rule controls are part of the planned report, not optional discoveries.

## Artifacts and interpretation

`protocol.json` records settings, source fingerprint, split roles and conventions
before fitting. Calibration `selection.json` records every candidate's outer tuning
score and the choices; fresh-seed folders copy the frozen selection before scoring
test targets. Changed source/configuration or missing provenance prevents resume.

Every dataset stores observations and explicitly named masks, both checkpoint
types in each student `.pt` file, individual/population candidate predictions,
learning histories, environment inventories, and plots. Calibration data has
`mask_tuning`, not `mask_test`; its report contains tuning scores without test groups.

The summary reports per-seed and equal-weight seed means/ranges for NLL and KL.
Diagnostic NLL and diagnostic true-policy KL are distinct planned metrics, using
the existing threshold/reference. Missing diagnostic support stays null, with
subject/trial/seed counts. Calibration never enters the five-seed aggregate. No
complete aggregate is emitted until every planned seed completes.

`selection_audit.png` compares scores on the checkpoint-selection targets with
outer tuning (calibration) or test (fresh populations), using selected weights.
`convergence.png` shows training trajectories. Selection gaps are descriptive;
they do not establish a causal explanation by themselves. Training-boundary flags
request inspection and do not prove or disprove convergence.

## Compute and running

The audit keeps a soft 30-minute invocation limit and a soft 10-minute phase limit.
Complete steps, setup, checkpoint writes, predictions and figures can add time.
Rerun the same study cell and run name to resume. Use one runtime per output folder.
Timing counts each optimizer trajectory once, including every calibration candidate;
two checkpoint rules do not count as two fits. If selected and legacy L2 coincide,
their single trajectory is reused. T4 conversion uses the supplied 1.07 units/hour
only for histories entirely attributed to T4. Setup/evaluation/idle billing is extra.

Open notebook 03 in a fresh T4 Colab session and run cells in order. The launcher
downloads the pinned source from public GitHub. Authorize the Google Drive mount
when prompted. Tests and the small audit/resume check precede the full study.
Keep the entire output folder at
`MyDrive/distilled-tiny-rnns/runs/baseline-audit-v1` for verification.

This remains an exploratory first-stage parametric simulation. The mechanistic
cohort, published-code equivalence, free-running probes, human-data arm, and full
research grid remain outside this audit.
