# Prospective main-study protocol, version 1

Frozen 2026-11-04 after the bounded budget check and foundation fixtures.
All evaluation populations in this protocol are fresh. The machine-readable
record specifies the full design, seed formula, fitting and selection rules.
Notebook 07 implements these settings with a separately fingerprinted runtime
and read-only checks of the frozen design. [Running instructions](main-study-running.md). [Frozen settings](../configs/main_study_v1.json).

## Question and first tranche

Can a teacher-target student predict a rare person's observed choices better
than a tiny student fitted directly to that person's choices, while producing
less faithful strategy-dependent behavior? The primary outcome has two
coordinates. Prediction gain is solo minus distilled held-out NLL. Behavioral
distortion is distilled minus solo error in the four-cell behavioral assay
below. Positive values for both describe the proposed tradeoff.

Start with six populations of 1,000 people and 100 trials, with exactly 20 rare
people in each. Three populations use pure model-free rare people among hybrid
agents. Three use the reward-cue rule verified in the foundation checks.
Independent discrete uniform draws choose the generating parameters from
supports contained in the baseline grids. This makes the first tranche a
benchmark where the fitted candidate set contains the generating policy.
Continuous-parameter generalization requires a subsequent off-grid design.
[Mechanism and recovery check](rare-mechanism.md).

The two rare kinds use data seeds 202611401, 202611411, 202611421 and
202621401, 202621411, 202621421 respectively. Split seeds add one, teacher
initialization adds two and all student arms share the same seed plus three.
The versioned seed formula also fixes all 96 populations in the proposed
rarity/history grid, including the initial six and 90 additional populations. After the six-population tranche, measured complete
runtime determines the scheduling of broader compute. Every outcome from a
completed tranche is reported before moving on.

The broader grid retains the proposal's rarities 2%, 5%, 10%, 25% and history
lengths 100, 200, 350, 1,000 across both rare kinds, with three populations per
cell. Initial foundation and notebook populations remain development evidence.

## Fitting and fair comparisons

Retain the procedure fixed before notebook 06. The teacher is H=32 with a
500-step maximum and observed-choice validation patience of 20 evaluations.
The H=2 students train for 4,000 steps with Adam at 0.01, evaluating every ten
steps. Solo uses L2 0.1, and the two soft-target arms use zero L2, pure soft
targets and temperature one. This choice preserves the earlier procedure and
bounded compute. Notebook 06's examined test scores do not choose a preferred
training horizon.

Each person has interspersed 60/20/20 training, validation and test masks.
All models receive the causal observed history. Training loss uses training
targets only. The pooled teacher is selected by validation NLL. Students
retain both the population-wide and per-person observed-choice validation
minimum, with strict improvement and earlier ties. Population selection is
primary, and individual selection is reported from the same trajectories.
Teacher, solo, distilled and oracle arms use the same populations and splits.
Oracle probabilities enter student fitting only at training positions.

The baseline comparisons are individual maximum likelihood on the hybrid grid,
one pooled hybrid triple, hierarchical hybrid predictions and the hierarchical
hybrid/cue mixture. The mixture averages likelihood within each family before
estimating the two population family weights. This gives the families equal
prior mass despite having different candidate counts. All fits use first-stage
training choices. True group labels and parameters enter recovery and
evaluation after fitting. Include coinflip and training-frequency references.
[Baseline specification](classical-baselines.md).

The project uses the verified first-stage GRU adaptation. Its target encoding,
size and student selection differ from the authors' Gillan recipe, and those
differences accompany every report. [Published-method comparison](published-method-alignment.md).

## Behavioral preservation

The held-out prediction metrics are accompanied by an exact three-choice assay.
Enumerate all 256 two-trial histories of first action, state, second action and
reward. Starting from the initial state, compute each fitted policy's first-stage
choice probabilities at all three decisions. Integrate the first two choices
under that policy, 70/30 transitions, fair second-stage actions and fair rewards.
Hierarchical and mixture predictions hold each person's fitted training-posterior
weights fixed throughout the assay. At every history, average the candidates'
conditional action probabilities with those weights. Each candidate updates
its value states from the reached history. Assay choices neither refit the
posterior nor select a persistent latent candidate for the trajectory. This
evaluates the marginal action policy exposed by the fitted prediction API.

The four signature cells are the probability of repeating the previous
first-stage action after unrewarded rare, unrewarded common, rewarded rare and
rewarded common transitions.

This assay integrates over histories generated by the fitted first-stage policy.
Its controlled environment makes the behavioral comparison reproducible and
independent of which histories happened to occur in a person's sample. The
drifting-reward training task remains a separate evaluation environment.
[Executable assay](../foundation/strategy_probes.py).

For every person and model, report the four cells and their RMS difference from
the true generating-policy signature. Report the signed projection toward the
mean true signature of that population's typical people. A squared direction
norm of at most 1e-12 gives a null pull score with support recorded. Values below
zero or above one remain visible. Also report KL on the third decision, weighting
the 256 fixed histories equally, as a conditional probe complement.

A strategy-preservation claim requires the declared behavioral measurements,
their seed directions, and prediction scores together. Matching test NLL alone
does not establish behavioral preservation. Agreement in this short controlled
assay establishes fidelity for its particular perturbations and horizon.

## Analysis and stopping rules

People receive equal weight within each group. Populations then receive equal
weight. Primary summaries compare teacher-target students with solo on rare
people using population-selected states. Report both coordinates, seed
directions, and the descriptive fraction of rare people in each quadrant.
Each rare kind and grid cell has its own three-population summary.

Secondary reports include the same comparisons in typical people, the teacher
and oracle arm, individual checkpoint selection, overall NLL and true-policy KL,
diagnostic NLL and KL, all four assay cells, pull, probe KL and family uncertainty.
Diagnostic trials satisfy an absolute true/reference probability difference
above 0.2 on held-out positions. The reference is hybrid weight 0.75 on the same
history with the person's stage-two learning rate and inverse temperature.
People without diagnostic trials retain null diagnostic values.

Complete all arms for the planned tranche. Time-budget pauses retain current
weights, optimizer, RNG, learning history and cumulative selected states.
An incomplete or numerically failed trajectory is logged and repaired before
aggregation. Missing seeds are identified explicitly. There is no significance
declaration from three populations per cell, and an early selected checkpoint
does not establish convergence.

Version and hash the implemented generator, fitting code, assay and full
configuration before evaluation. Record the device and complete package
inventories for every session. Preserve the frozen protocol and prior outcomes
when making later amendments. Broader history/rarity experiments use the same
settings. Extra capacities, teacher interventions, off-grid populations and
human data follow separate prospective designs.
