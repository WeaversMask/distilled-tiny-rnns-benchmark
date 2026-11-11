# Notebook 05 results and verification — 2026-11-02

## Finding

The same H=2 student learned the simulated rare behaviour substantially better
when trained on the simulator's true probabilities instead of the learned
teacher's probabilities. The predeclared primary rare diagnostic KL fell from
0.141839 to 0.028140 nats: a 0.113700 reduction, or about 80.2% of the
teacher-target mean. Every one of the five fresh populations improved.

This is evidence that training-target quality matters under the fixed recipe.
The student's small size does not impose the error level reached with teacher
targets. The oracle uses privileged synthetic supervision: these results do not
establish that an ordinary teacher could infer equally good targets from the
available observations, isolate a percentage of error caused by compression,
or establish recovery of a human cognitive mechanism.

## Completion and verification

Both the full run and the software smoke run are complete. Full-run seeds are
20261411, 20261421, 20261431, 20261441 and 20261451, each with 1,000 people,
100 trials, and 20 rare people. All student fits reached 4,000 steps.

- Protocols and configuration, numerical source fingerprint, session package
  hashes, observations, split masks and checkpoint signatures were verified.
- All 28 checkpoints and 77 model states across both folders were finite.
  Saved weights reproduce prediction streams on CPU; the largest CPU/T4
  probability difference is 3.16e-6, and selected validation NLL differs by at
  most 1.79e-7. Regenerated simulation values differ by at most 4.44e-16.
- A separate agent independently recomputed 176,397 numeric fields and 167,790
  null fields from the raw arrays using NumPy formulas, including per-person
  and group scores, diagnostic support, comparisons and aggregate summaries.
  Maximum discrepancy: 2.22e-16; no failures.
- Both soft-target arrays are zero outside fitting positions. Saved oracle
  targets match generating probabilities only on fitting positions. Learned
  test predictions were reconstructed from saved weights, not substituted with
  simulator truth.
- The recorded source SHA256 is
  `c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`, matching
  the reviewed code. All recorded sessions used a Tesla T4 and the same package
  inventory: PyTorch 2.11.0+cu130, NumPy 2.1.3 and matplotlib 3.10.0.

The audit preserved the raw inputs byte-for-byte. The input folders were
`oracle-control-v1` and `oracle-control-smoke-c68c9b2918c7`.

## Rare-group comparison

Scores are nats; lower is better. Group scores weight people equally, and
reported means then weight the five populations equally. Population checkpoint
selection is primary; individual selection is a prespecified secondary check.

| Outcome | Teacher-target student | Oracle-target student | Gain | Direction |
| --- | ---: | ---: | ---: | --- |
| Diagnostic KL, primary | 0.141839 | 0.028140 | 0.113700 | Better 5/5 |
| Diagnostic KL, individual selection | 0.143665 | 0.053766 | 0.089899 | Better 5/5 |
| Overall observed-choice NLL | 0.641752 | 0.615892 | 0.025860 | Better 5/5 |
| Overall true-policy KL | 0.047023 | 0.018653 | 0.028371 | Better 5/5 |
| Diagnostic observed-choice NLL | 0.654386 | 0.532098 | 0.122288 | Better 5/5 |

| Seed | Diagnostic trials / people | Teacher-target diagnostic KL | Oracle-target diagnostic KL | Primary gain |
| --- | ---: | ---: | ---: | ---: |
| 20261411 | 61 / 18 | 0.151970 | 0.025212 | 0.126757 |
| 20261421 | 61 / 12 | 0.143790 | 0.060353 | 0.083437 |
| 20261431 | 43 / 14 | 0.127870 | 0.020312 | 0.107558 |
| 20261441 | 53 / 13 | 0.162851 | 0.022899 | 0.139952 |
| 20261451 | 39 / 13 | 0.122716 | 0.011922 | 0.110794 |

The diagnostic subset comprises 257 test trials across 70 of the 100 rare
people in the five populations. The five-seed descriptive gain range is
0.083437–0.139952; it is not a confidence interval. Smoke results are software
checks and are excluded from the scientific comparison.

## Controls and qualifications

The larger teacher's rare diagnostic KL is 0.140574; the ordinary teacher-target
student is close at 0.141839. The solo student's score is 0.142691. Thus, the
previous notebook's distillation-versus-solo diagnostic penalty does not recur
uniformly here: distillation is worse in three seeds and better in two, with a
near tie on average. Notebook 04 and 05 protocols are not pooled.

Oracle training also improves typical-group overall NLL and KL in all five
seeds. Typical diagnostics have only 25 trials across 23 people. The benefit
has not been shown to be exclusive to rare people.

No population-selected student checkpoint was exactly at the 4,000-step limit.
The teacher-target selection steps were 2750, 2580, 3250, 2980 and 3790; oracle
steps were 2390, 3740, 2070, 900 and 1310. One teacher-target fit (20261411) has
a training-boundary inspection warning. Individual selections at the limit
range from 1.8–3.0% for teacher targets and 1.6–3.1% for oracle targets.
Convergence is not established. Saved learning curves show observed-choice
NLL, not the actual soft-target objective optimized by these two arms.

Recorded full-study training totals 5,510.205 seconds (91.84 minutes), equivalent
to 1.6378 T4 units at the user's 1.07 units/hour rate. This excludes setup,
evaluation, idle billing and any unsaved work. It is not the complete Colab bill.

## Recommended next step

First close the remaining optimization-budget question with the single
[4,000→8,000 sensitivity check](budget-check.md) applied to both soft-target
arms, preserving the current result and cumulative selection rule. Then follow
the [agreed foundation plan](foundation-plan.md): published-method comparison,
stronger scientific baselines, one mechanistically different rare agent, and a
frozen main-study protocol. The current result motivates examining supervision
quality; it does not identify the best teacher architecture or justify selecting
a teacher intervention from these already examined test populations. Practical
teacher remedies belong to the later main study with fresh evaluation data.
