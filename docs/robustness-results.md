# Notebook 04 results and verification — 2026-11-01

This describes the downloaded `robustness-v1` study separately from notebook 03.
The result motivated notebook 05; it is not a result from that new experiment.

## Completion and validation

The five planned seeds 20261311, 20261321, 20261331, 20261341 and 20261351 completed,
including the population interrupted by runtime deletion. Every student trajectory
reached 4,000 steps. The earlier unfinished session in seed 20261311 lacks a final
session status, as expected after an abrupt interruption; its subsequent sessions
completed and final checkpoint state is internally consistent.

Across the full and smoke folders, 42 checkpoints and 75 selected model states
were verified. Observations and masks were regenerated, manifest/config/source
identities and environment-inventory hashes checked, all selected weights reloaded,
probabilities and selected validation scores recomputed on CPU, and Adam counters
checked against completed steps. GPU/CPU numerical differences in full-study group
NLL were below 6e-8. Direct independent NumPy formulas reproduced every saved
per-person and group metric, paired contrast, calibration tuning score, reference
prediction and aggregate to floating-point rounding (maximum difference below
7e-16). Source SHA256 was
`63c789f6b9ef1cb7c8d32bd726ecfbfcf2c44aa3ac48ed5da18c5096cb008210`, matching notebook
04's pinned runtime `bba8ce3e712173802f705650b27356fedbdea8fd`. No downloaded results
were modified.

## Fresh-population results

Equal-weight means across the five seeds; errors are in nats and lower is better.
People are weighted equally within groups. The two protocols are not pooled.

| Rare-group outcome | Solo | Teacher-target student | Student versus solo |
| --- | ---: | ---: | --- |
| Overall observed-choice NLL | 0.675423 | 0.657426 | Better 5/5 |
| Overall true-policy KL | 0.071453 | 0.049425 | Better 5/5 |
| Diagnostic observed-choice NLL | 0.651336 | 0.685669 | Worse 4/5 |
| Diagnostic true-policy KL | 0.126660 | 0.150620 | Worse 5/5 |

Primary rare diagnostic KL loss (student minus solo) was +0.023960 on average,
with a +0.002142 to +0.067114 range. There were 267 diagnostic test trials across
69 supported rare people. The individual-checkpoint control also worsened
diagnostic KL in all five seeds: +0.014943 mean, +0.002859 to +0.032507 range.
This contrasts with notebook 03's less consistent individual-mode pattern, without
establishing that extra training alone caused the difference.

The teacher's rare diagnostic KL was 0.148515. It was worse than solo in four of
five seeds. Student minus teacher diagnostic KL was +0.002105 on average, with
three positive and two negative differences. Similar teacher/student errors
motivate checking training targets, but do not identify a causal mechanism or a
percentage of error attributable to the teacher.

Typical-group overall NLL and KL also favored distillation in all five seeds.
Typical diagnostic KL worsened in all five primary comparisons, but only 23
trials from 23 people supported this subset. The diagnostic weakness has not been
shown to be exclusive to the rare group.

## Tuning, training and compute

Fresh calibration selected solo L2 0.1 and population checkpoints for both arms.
L2 0.1 is now inside the expanded grid; higher values did not improve the declared
outer tuning criterion. Selected distilled population checkpoints in the five
test populations were 2470, 1540, 1490, 1320 and 3510, none exactly at 4000.
Calibration selected 3960, close to its limit. Between 1.7% and 3.0% of individual
student selections reached 4000. These observations reduce the earlier endpoint
concern without proving convergence. Remaining summary warnings mainly involve
weaker solo candidates and controls.

The 29 full-study optimizer trajectories total 8301.205 recorded seconds
(138.35 minutes), approximately 2.4673 T4 units at the supplied rate. Every
recorded session used a T4 and the same package inventory. This is training-time
accounting, not the full Colab bill: setup, evaluation, idle time and any unsaved
work before interruption can add cost. Smoke outcomes are software checks, not
additional scientific replication.
