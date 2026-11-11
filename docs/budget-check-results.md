# Notebook 06: verified bounded-budget results

Reviewed 2026-11-04. Inputs are the unchanged downloads
`budget-check-smoke-a7394cdff009` and `budget-check-v1`, including their
embedded parent snapshots. This closes the single planned 4,000→8,000-step
check.

## Plain-English finding

Students taught the simulator's true probabilities still predicted the rare
agents' choices much more faithfully after twice as much training. The size of
that advantage barely changed.

The comparison uses the version of each student chosen on separate validation
trials. All five teacher-target populations and four of the five oracle-target
populations kept their earlier selection. The fifth oracle population selected
step 4,120 instead of 3,740. Its validation score improved, while its diagnostic
test error rose slightly.

The simulator can supply the exact probability behind a choice. In these five
synthetic populations, access to that information mattered more to this
comparison than another 4,000 training steps. Whether an ordinary teacher can
learn those targets and whether a human uses the same strategy remain open.

## Main comparison

Lower KL means the predicted probabilities are closer to the simulator's true
probabilities. These are equal-person, then equal-population averages in nats.
“Diagnostic” means held-out trials where the true policy differs by more than 0.2
from the fixed reference policy. Population checkpoint selection is primary.

| Rare-group metric | Teacher targets, 4,000 | Teacher targets, 8,000 | Oracle targets, 4,000 | Oracle targets, 8,000 |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic KL, primary | 0.141839 | 0.141839 | 0.028140 | 0.028223 |
| Diagnostic observed-choice NLL | 0.654386 | 0.654386 | 0.532098 | 0.532281 |
| Overall true-policy KL | 0.047023 | 0.047023 | 0.018653 | 0.019694 |
| Overall observed-choice NLL | 0.641752 | 0.641752 | 0.615892 | 0.617404 |

The primary oracle advantage (teacher-target KL minus oracle-target KL) is
**0.113616**, compared with **0.113700** before extension. The change is
**−0.00008349 nats**. The advantage remains positive in all five populations.
Oracle diagnostic KL is about **80.1% lower** than teacher-target KL. The small
narrowing comes entirely from the increase in oracle error in seed 20261421.

| Population seed | Teacher-target diagnostic KL, both budgets | Oracle diagnostic KL, 4,000 | Oracle diagnostic KL, 8,000 |
| --- | ---: | ---: | ---: |
| 20261411 | 0.151970 | 0.025212 | 0.025212 |
| 20261421 | 0.143790 | 0.060353 | 0.060770 |
| 20261431 | 0.127870 | 0.020312 | 0.020312 |
| 20261441 | 0.162851 | 0.022899 | 0.022899 |
| 20261451 | 0.122716 | 0.011922 | 0.011922 |

The diagnostic comparison includes 257 test trials from 70 of the 100 rare
people. The other 30 had no qualifying test trials and their scores remain
null. Both budgets were tested on the same five populations, so the comparison
is paired and descriptive.

## Secondary checkpoint rule and majority group

Choosing a checkpoint separately for each person gives small improvements for
both rare-group arms. It does not reverse the main conclusion.

| Rare-group metric, individual selection | Teacher targets, 4,000 | Teacher targets, 8,000 | Oracle targets, 4,000 | Oracle targets, 8,000 |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic KL | 0.143665 | 0.142305 | 0.053766 | 0.051582 |
| Diagnostic NLL | 0.658097 | 0.654897 | 0.534200 | 0.527402 |
| Overall KL | 0.055961 | 0.055215 | 0.032729 | 0.031032 |
| Overall NLL | 0.652001 | 0.650805 | 0.631794 | 0.628259 |

The individual-selection diagnostic advantage changes from 0.089899 to
0.090724 nats, remaining positive in all five populations. This advantage grew
in two populations and shrank in three.

| Typical-group metric | Teacher, population 4k→8k | Oracle, population 4k→8k | Teacher, individual 4k→8k | Oracle, individual 4k→8k |
| --- | ---: | ---: | ---: | ---: |
| Overall NLL | 0.669379→0.669379 | 0.665535→0.665571 | 0.677448→0.677240 | 0.674463→0.674178 |
| Overall KL | 0.011417→0.011417 | 0.006172→0.006196 | 0.019913→0.019696 | 0.015758→0.015576 |
| Diagnostic NLL | 0.826823→0.826823 | 0.707784→0.708206 | 0.815010→0.811764 | 0.714642→0.713939 |
| Diagnostic KL | 0.068963→0.068963 | 0.024610→0.024656 | 0.069255→0.069921 | 0.031086→0.031002 |

Typical-group overall NLL and KL also favour oracle targets in all five
populations. The benefit therefore extends to the majority group in this pilot.
Only 25 trials from 23 of the 4,900 typical people qualify for their diagnostic
comparison. Complete per-seed metrics,
directions, supports, and nulls remain in the original summary and verification
record.

The comparison with tiny students trained directly on observed choices (solo)
remains mixed. Teacher-target distillation has higher rare diagnostic error in
three populations and lower error in two. Its mean diagnostic KL (0.141839) is
close to solo (0.142691). The fixed teacher scores 0.140574.

## Training and cost

Both continued arms reached 8,000 steps for every planned seed. The child retains
the original histories and selects cumulatively over steps 0–8,000. The extra
budget increases both optimization and the number of checkpoint candidates.
Every population selected a checkpoint before 8,000. Of the 5,000 individual
fits per arm, 993 teacher-target and 1,184 oracle-target fits selected a step
after 4,000. Only 43 and 51 respectively selected step 8,000. The original
boundary warnings flag oracle trajectories in seeds 20261421 and 20261451.

The current weights fit the training probabilities better at 8,000 in all five
oracle arms and three of the five teacher-target arms. In the other two
teacher-target arms the soft-target loss rose slightly. Validation uses
observed choices. These different objectives explain how additional learning
can coexist with selection of an earlier checkpoint. Convergence remains open.

Incremental checkpoint training time is **4,302.3389 seconds (71.71 minutes)**,
counting each child-minus-parent arm once across resumes. Inherited parent
training is **5,509.5106 seconds (91.83 minutes)** and is kept separate. With
the recorded T4 sessions and the user's supplied 1.07 units/hour rate, the
incremental training proxy is **1.279 compute units**. The complete Colab
bill includes setup, evaluation, copying, saving, and idle runtime.

## Read-only verification

The smoke run completed both seeds (201 and 211) from four to eight steps.
Only one rare person in one seed contributed one diagnostic trial. Its role is
to check the software. All 53 smoke files and all 125 production files matched
their pre-audit hashes. All 77 original production-parent files also matched
the embedded snapshot byte for byte.

The audit checked the frozen configurations and source fingerprints, generated
data and masks, package inventories, optimizer counters, cumulative selections,
prefixes of learning histories, and fixed prediction arrays. Independent
calculations from saved choices, masks and probabilities reproduced the
individual scores, group averages, paired contrasts and support. The largest
production probability difference when reconstructing the saved T4 predictions
on CPU was 0.00000316, within the original 0.000005 tolerance. Smoke predictions
agreed to within 0.000000477 across both parent and child selections.

Run the read-only checker from the project with its test dependencies available.
Supply the downloaded paths and choose new report paths outside the inputs.

```sh
python scripts/verify_budget_results.py /path/to/budget-check-smoke-a7394cdff009 --kind smoke --output /tmp/smoke-audit.json
python scripts/verify_budget_results.py /path/to/budget-check-v1 --kind production --original-parent /path/to/oracle-control-v1 --output /tmp/production-audit.json
```

Detailed records include all seed directions, support, soft-target losses,
and checkpoint selections. [Production audit](../review/budget-check-results-verification.json)
and [smoke audit](../review/budget-smoke-results-verification.json).

## Decision

Close the budget check at its declared 8,000-step limit. The main-study recipe
will be frozen before fresh evaluation populations, using the scientific design
and available compute. The next foundation milestones are the published-method
comparison and classical baselines. [Foundation plan](foundation-plan.md).
