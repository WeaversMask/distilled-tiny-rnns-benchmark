# Independent interpretation review of notebook 07 results

Reviewed 2026-11-05 against the frozen main-study-v1 design and the downloaded
production summary, all six population reports and their four neural
checkpoints. This review reads downloaded files only. Separate agents are
checking hashes, structural integrity and independent numerical reconstruction.
The judgments below concern interpretation and scope of the saved results.

## Main interpretation

The predeclared population-level pattern of better held-out prediction together
with greater four-cell behavioral-assay error did not occur in this tranche.
Teacher-target distillation improved the assay RMS error over solo in every one
of the six populations. It improved observed-choice prediction in all three
model-free populations and two of three reward-cue populations. Both kinds had
positive average prediction gain and negative average behavioral distortion.

| Rare kind | Mean prediction gain (solo minus distilled NLL) | Mean behavioral distortion (distilled minus solo RMS) | Population directions |
| --- | ---: | ---: | --- |
| Parametric model-free | +0.023313 | -0.014819 | Prediction improves 3/3, assay improves 3/3 |
| Reward cue | +0.021778 | -0.025741 | Prediction improves 2/3, assay improves 3/3 |

Each kind contains three populations and 60 rare people. The separate-person
checkpoint sensitivity gives the same directions of the kind-level means.
Its prediction gain is +0.024194 and +0.031145, with assay distortion -0.014428
and -0.020861 respectively. Under this secondary selection, prediction improves
in all six populations.

This is a result about comparisons with the particular solo recipe, including
its fixed L2=0.1. The distilled arm changes both objective and regularization.
It does not isolate the causal effect of soft targets alone.

## Absolute fidelity and classical comparisons

Improvement over solo leaves substantial absolute rare-strategy error,
especially for the reward-cue mechanism.

| Model, population-selected neural states | Model-free NLL / assay RMS | Reward-cue NLL / assay RMS |
| --- | ---: | ---: |
| Solo | 0.666510 / 0.099317 | 0.696419 / 0.245217 |
| Teacher-target student | 0.643197 / 0.084499 | 0.674641 / 0.219476 |
| Teacher | 0.640186 / 0.082757 | 0.673235 / 0.210090 |
| Oracle-target student | 0.614645 / 0.054420 | 0.589732 / 0.087675 |
| Individual hybrid-grid ML | 0.599478 / 0.022643 | 0.665596 / 0.214836 |
| Hierarchical hybrid | 0.606601 / 0.040672 | 0.662722 / 0.216184 |
| Hybrid/cue mixture | 0.600320 / 0.028910 | 0.544619 / 0.066211 |

For the cue group, allowing the correct family materially changes prediction
and assay fidelity. The mixture beats the distilled student on both scores in
all three populations. The true generating policies lie on these candidate
grids by design. This is a useful controlled benchmark for sparse-data and
pooling effects, with a deliberately specified model family. The result does
not establish general superiority on off-grid policies or people.

The teacher itself has substantial error and its student's scores are nearby.
Oracle targets improve rare conditional true-policy KL and assay RMS in every
population compared with ordinary teacher targets. Observed test NLL is mixed
in one model-free population, so avoid saying oracle improves every score in
every seed. Oracle is a trained student with privileged simulator training
probabilities. Neither residual oracle error nor the oracle advantage uniquely
separates target, optimization, selection and capacity contributions.

## Keep the secondary disagreement visible

A blanket conclusion that distillation preserves rare strategies would be too
broad. On the observed task histories, diagnostic KL worsens relative to solo,
even while the declared behavioral assay improves.

| Rare kind | Solo diagnostic KL | Distilled diagnostic KL | Distilled has higher error |
| --- | ---: | ---: | --- |
| Model-free | 0.134172 | 0.144456 | 3/3 populations |
| Reward cue | 0.202218 | 0.207003 | 2/3 populations |

Diagnostic observed-choice NLL also worsens on average for both rare kinds.
The model-free means are 0.630126 solo versus 0.651752 distilled. The cue means
are 0.707602 versus 0.710884. Overall prediction and a brief controlled assay can
therefore improve while errors on some policy-disagreement histories increase.
This is a secondary signal relevant to masking. The frozen primary outcome
remains the joint prediction/assay contrast, and it must not be relabeled after
viewing these results.

A further disagreement appears for oracle students in model-free populations.
Their assay RMS improves over distilled in all three populations, but their
uniform probe-history KL is higher in all three (mean 0.049841 versus 0.037652).
The four-cell signature and uniform conditional probe KL weight different
aspects of behavior. Neither substitutes for the full policy.

The directional pull score also need not track the scalar RMS error. For
model-free rare people, mean pull increases from 0.948400 solo to 1.065054
distilled despite improved RMS. For cue people it falls from 1.043429 to
0.926868. Pull is a projection with person-specific denominators, and values
outside zero to one remain legitimate. Avoid equating it with a global measure
of strategy loss or comparing percentages as calibrated prevalence.

## Support and individual exceptions

Seven of 60 model-free rare people and two of 60 cue rare people occupy the
primary better-prediction/worse-assay quadrant. These are descriptive counts
from noisy short histories, without a significance or practically-important
threshold. They coexist with average assay improvements. The individual
selection sensitivity gives six and three people respectively.

The same quadrant is also populated among typical people, at roughly 18.9%
for the model-free populations and 21.6% for cue populations under the primary
selection. Its mere presence does not establish a rare-specific effect.
Typical-group prediction and assay RMS means improve with distillation in all
six populations.

Rare diagnostic support is 184 test trials from 35 of the 60 model-free people,
and 612 trials from 52 of the 60 cue people. Unsupported individual diagnostic
scores must remain null. The supported conditional diagnostic comparison and
the all-person assay comparison therefore have different support as well as
different history distributions.

## Histories and convergence

All 18 student trajectories reached the fixed 4,000 steps. The teachers stopped
under declared validation patience after 250–330 steps, selecting steps 50–130.
No population-selected neural checkpoint lands at the 4,000-step boundary.
Some individual selections do. Among the 120 rare people, the boundary counts
are one solo, one distilled and seven oracle. Across all 6,000 people, the
corresponding counts are eight, 129 and 145.

Selected steps and nearly flat observed-choice validation curves do not prove
convergence. The two soft arms optimize a soft-target objective, while the saved
learning curves summarize observed-choice NLL. The fixed recipe should remain
visible and any boundary concern should be disclosed, without another
results-driven training extension.

## Assay scope

The behavioral assay resets state and integrates three first-stage choices over
all 256 two-trial histories under fair rewards and second-stage actions. It is
a defined short controlled intervention. The main task has drifting rewards and
longer histories. A shared teacher has the same cold-start probe policy for all
people in a population because it has no person input or embedding. The tiny
students can have different policies through their independently learned
weights. Good performance on this assay establishes agreement on its declared
perturbations, rather than full cognitive mechanism recovery or long-run task
behavior. Matching or failing it can coexist with the conditional diagnostic
findings above.

## Recommended next step

Report and retain this tranche as mixed evidence. Its primary group-level
hypothesis was not observed, while meaningful secondary diagnostic errors and
absolute failures on the cue mechanism remain. Keep both rare kinds and all
classical controls visible. Use the existing frozen seed and recipe plan for
subsequent rarity/history cells, staged using the recorded complete T4 runner
time. Choose the scheduling tranche prospectively according to budget and
coverage, without tuning the horizon or teacher to these six test populations.

Three populations per kind support descriptive means, ranges and directions.
This tranche does not establish a rarity trend or a causal distillation-specific
mechanism. Stronger baselines and teacher/oracle comparisons sharpen the
question without changing those limits.

## Final report review

Reviewed the completed results draft and figure at
`docs/main-study-first-tranche-results.md` in the results staging checkout.
The primary means, six paired seed values, individual quadrant counts,
diagnostic KL and NLL directions, full rare-group model tables, selection
sensitivity, support counts and checkpoint-boundary summaries agree with my
independent extraction from the downloads. I separately confirmed the mixture
family classification counts from saved posterior metadata: 33/60 cue people
and 6/2,940 typical people exceed posterior cue probability 0.5. The primary
figure has the correct signs and population points, with diamonds marking the
reported means.

The integrity and raw-metric auditors separately reported passing complete
checks with unchanged inputs and numerical reconstruction agreement. The draft
keeps the lack of the primary group-level tradeoff and the secondary diagnostic
worsening visible together. It also retains the assay environment, on-grid
scope, descriptive sample size, oracle/probe metric disagreement and stronger
classical comparisons. I found no material numerical or interpretive error in
the final draft. Use “first tranche” in its introductory sentence where the
current draft says “first cell,” since it spans two rare-strategy kinds at the
same rarity/history setting. The linked review already explains the teacher's
shared cold-start assay policy, which is helpful context for interpreting its
similar scores across people.
