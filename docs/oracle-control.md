# Training-target control — notebook 05

## Question and motivation

Does replacing the ordinary teacher's training probabilities with the simulator's
true probabilities reduce the tiny student's error on revealing rare-group trials,
under a fixed architecture, optimizer, training budget and checkpoint-selection
recipe?

Notebook 04's five fresh populations showed better average choice prediction and
true-policy fidelity with distillation, but worse rare diagnostic KL in every seed
under both checkpoint rules. The teacher's diagnostic KL was close to the
distilled student's: 0.1485 versus 0.1506, compared with 0.1267 for the selected solo
model. The teacher was worse than solo in four of five seeds; the student was worse
than the teacher in three of five. These descriptive gaps motivate checking the
targets supplied by the teacher; they are not a causal decomposition of its error
and compression error. Diagnostic support was 267 trials across 69 simulated people.

Notebook 05 adds a learned **oracle-target student**. Here "oracle" means the
simulator can supply the correct conditional choice probabilities at fitting
positions. The student is still a two-unit GRU and must predict the held-out
positions itself. It is not the simulator's perfect predictions substituted into
the evaluation array. This privileged training signal is a synthetic control,
not a claim that the same information exists for human data.

## Fixed design before this study

Five new populations use RNG seeds **20261411, 20261421, 20261431, 20261441,
20261451**. These are integers, not dates. Each has 1,000 people, 100 trials per
person, and 20 parametric rare agents. Data, split and model-initialization seed
offsets do not overlap earlier studies. The base seed is the first replication's
seed, not a calibration population.

There is **no new calibration or grid search**. Fix solo L2 at 0.1 and the primary
checkpoint rule at population selection, based on the preceding studies before
examining these new populations. Keep individual selection as a predeclared
secondary control. This isolates a target replacement within a fixed recipe; it
does not retune each arm to its own optimum or claim calibration stability.

Each population fits four trajectories:

| Model | Fitting targets | Extra L2 |
| --- | --- | --- |
| Shared teacher (32 units) | Observed choices | 0 |
| Solo student (2 units) | Observed choices | 0.1 |
| Teacher-target student (2 units; saved as `distilled`) | Teacher probabilities | 0 |
| Oracle-target student (2 units; saved as `oracle`) | True simulator probabilities | 0 |

All students start from the same parameter values within a population and use
the same histories, train/checkpoint/test split, Adam learning rate 0.01, and
4,000-step horizon. Both soft-target students require weight 1 and temperature 1,
so their fitting loss differs only through the supplied soft targets. The teacher
retains its existing 500-step maximum and early stopping after 20 evaluations
without improvement. Evaluate every 10 steps. A finite budget is not a convergence
guarantee; keep boundary warnings and raw curves visible.

There is one optimizer trajectory per student arm. Tracking individual and
population checkpoints does not train a duplicate arm. Each student's parameter
set remains independent even when the selected training step is shared across
the population. All fitting time, including teacher fitting, is counted once.

## Target access and checkpoint selection

Each person's interspersed trial targets are split into 60% fitting, 20%
checkpoint selection, and 20% testing. No outer tuning split is needed because
the settings above are already fixed. Small software checks use rounded counts.

Both teacher-target and oracle-target logits are neutralized to zero outside
the fitting mask. Oracle probabilities are clipped to [1e-12, 1-1e-12] and converted
to logits in float64 before casting to the model's float32 dtype. This provides
finite logits for exact endpoints; float32 reconstruction can introduce ordinary
rounding. Fitting uses only masked soft targets. Neither held-out oracle
probabilities nor diagnostic labels enter gradients or the checkpoint criterion.

Checkpoint selection for every student uses **observed-choice NLL**, including
the oracle-target student. Population selection chooses the shared step with the
lowest mean checkpoint NLL; individual selection chooses each person's lowest
checkpoint NLL. Strict improvements replace snapshots; ties retain earlier steps.
The arms use the same selection rule but need not select the same training step.
This comparison includes the consequences of target replacement under that rule.

Inputs remain causal, constructed from earlier observed trials. As in earlier
notebooks, a past held-out observation can enter a later prediction's history,
while its target is excluded from the fitting objective. These are conditional
predictions on observed histories, not independent-session or free-running tests.
At scoring time the generating probabilities are used to calculate KL and
diagnostic support; they are not fed into the student's test-time forward pass.

## Outcomes specified before running

The focal outcome is the rare group's diagnostic true-policy KL reduction:

**teacher-target student diagnostic KL minus oracle-target student diagnostic KL**.

A positive value means the oracle-target student is closer to the true policy.
The primary comparison uses population checkpoints. Show each seed, equal-weight
seed mean and range, positive/negative/tied counts, and supported people/trials.
Use the same definition of diagnostic trials: held-out positions where the true
probability differs by more than 0.2 from a fixed majority-like policy (weight
0.75) on the same observed history and other parameters. This is not an empirical
population mean. Missing support stays null and is excluded from diagnostic
averages, with supported/planned seed counts shown.

Report observed-choice NLL, overall KL, diagnostic NLL and diagnostic KL for both
groups and every model. Average over people equally within a group, then over
seeds equally. Do not pool this study with earlier protocols. The individual
checkpoint contrast is a mandatory secondary control, not an alternative chosen
after inspecting the results.

All new paired gain fields use left error minus right error, so positive always
favors the right-hand model named in the comparison. In target comparisons the
right-hand model is the oracle-target student. The saved fields are `nll_gain`,
`kl_gain`, `diagnostic_nll_gain`, and `diagnostic_kl_gain`. This sign convention
differs from the earlier notebooks' `fidelity_loss` fields; their names and labels
must not be interchanged.

Context includes the teacher, fixed solo, teacher-target and oracle-target
students, both checkpoint rules, a probability-0.5 reference, and the train-only
Beta(1,1) smoothed choice-frequency reference. The contextual oracle-versus-teacher
and student-versus-solo comparisons are descriptive. Different architecture and
regularization make them different questions from the matched target comparison.

## What the result can establish

An oracle advantage would show that the training target affects this student's
performance under the fixed recipe, supporting further investigation of ordinary
teacher-target quality. It does not establish a cognitive mechanism or identify
exactly how much of the previous penalty was "inherited" versus "compressed".

Residual oracle-student error can reflect limited capacity, optimization, limited
training histories, or the observed-choice checkpoint rule. It is not by itself
proof of a capacity limit. A null or reversed oracle advantage is also valid and
would not by itself prove the ordinary teacher unbiased. No significance test or
hypothesis-acceptance threshold is introduced by this five-population exploratory
study. Population selection remains dominated by the 98% typical majority, and
rare diagnostic support can be sparse.

## Running and preserving results

Open `notebooks/05_colab_oracle_control.ipynb` in a fresh T4 Colab session and run
the five code cells in order. The launcher downloads the pinned source from
public GitHub. Authorize the Google Drive mount when prompted. Setup validates
the pinned numerical source and runs tests, including CUDA-only checks on the
GPU, before the smoke and resume checks.

Use `RUN_NAME = "oracle-control-v1"`. Results and checkpoints go to
`MyDrive/distilled-tiny-rnns/runs/oracle-control-v1`; smoke checks use a separate
`oracle-control-smoke-<source hash>` folder. Keep both complete folders for review.
The previous notebooks, source pins and results remain separate.

Each study invocation has a soft 30-minute limit, and each phase has a soft
10-minute limit. Setup, completing a step, writing checkpoints and evaluation can
add time. While the session remains alive, rerun **only the fourth code cell**
with the same run name after a time-budget pause. If the runtime is deleted or
its variables are cleared, rerun setup and smoke first, then resume the same study.
Drive checkpoints survive runtime deletion. Use only one runtime per output folder.

No complete aggregate is emitted until every planned population finishes. Source,
configuration and data identities prevent silently resuming a changed experiment.
Session environments, optimizer/RNG state, current and both selected student
states, observations, predictions, and training histories are retained. T4 unit
conversion covers recorded training time only at the supplied 1.07 units/hour;
setup, evaluation and idle billing are extra. No runtime or cost forecast is
guaranteed. Disconnect the GPU after completion and downloading the results.
