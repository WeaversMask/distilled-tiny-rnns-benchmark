# Independent review of notebook 07 and the main-study runner

Reviewer model: GPT-6.1 Sol. Runtime review verdict: PASS, no open findings. The reference design is the
prospective `main_study_v1.json` protocol frozen before fresh evaluation.
Production populations have not been generated or evaluated during this review.

The review covers implementation of the six-population first tranche, its
resumable fitting and outputs, the controlled three-choice assay, and the actual
five-cell Colab launcher. Development experiments use separate smoke seeds.
Earlier numerical modules and the six released launchers are checked for
preservation. Findings, resolutions and executed evidence are recorded below.

## Reference design

The primary comparison is a pair of signed quantities for population-selected
teacher-target students versus solo. Prediction gain is solo minus distilled
held-out NLL. Behavioral distortion is distilled minus solo RMS error over four
stay-signature cells. Comparisons remain separate for parametric-model-free and
reward-cue populations. People and then populations receive equal weight.

The assay averages histories generated under the fitted first-stage policy in a
controlled 70/30-transition environment with fair second-stage actions and fair
rewards. Hierarchical policies retain each person's training-posterior weights
while candidate states advance on the simulated history. This convention differs
from drawing a persistent latent candidate for each trajectory.

The selected recipe remains H32/H2, at most 500 teacher updates, 4,000 student
updates, Adam at 0.01 and the fixed solo L2 of 0.1. Baseline candidates contain the
first tranche's generating policies. Reported scope is an on-grid synthetic
benchmark and the documented first-stage adaptation of the published method.

## Simulator and scientific evaluation

Using development seeds 777 and 888, I independently replayed every first-stage
probability with scalar action/state-value updates and the explicit cue rule.
Both kinds agreed with saved truth to at most `2.22e-16`. The causal signed input
arrays agreed exactly with the previous trial's four observations. Parameter
sampling, rounded rare counts, the different cue mechanism and stage-two rules
match the frozen supports and task assumptions.

For the baseline API I removed every privileged field, retaining only choices,
states, second choices and rewards. Its fitted predictions, probe predictions
and metadata were identical. I independently expanded the two-family probe
mixture from the saved family and within-family posterior weights and candidate
policies. The largest discrepancy was `2.22e-16`. The weights remain fixed over
probe histories as declared.

I independently recomputed raw per-person observed-choice NLL and every group
mean from returned per-person scores, checking NLL, KL, diagnostic scores,
signature RMS error, pull and probe KL. The primary prediction and distortion
coordinates have the specified signs. A perfect true-policy assay has zero RMS
error. A deliberately diagnostic-free fixture preserves per-person and group
null diagnostic scores with zero support.

## Runner findings and corrections under verification

The first implementation had recoverability gaps after abrupt process loss. A
session appended before training could lack `wall_seconds`, causing final
accounting to fail after a valid resume. It now retains that interrupted session
with unknown wall time and explicit coverage information. Checkpoint training
time remains cumulative and is counted once per arm.

The original output path used lexical absolute paths. Equivalent paths
containing `..` could evade source-overlap checks and use distinct lock keys.
The runner now rejects symlinks on the original path and canonicalizes it before
checking overlap and acquiring its lock.

Checkpoint signatures alone did not verify saved Adam hyperparameters or the
student early-stop flag. Resumption could otherwise load changed optimizer
settings or retain an invalid shortened student trajectory. The validator now
checks effective Adam settings, parameter mapping, moment dtypes and counters,
CPU RNG structure, student stopping flags, selection/history consistency and
chosen-device validation losses. Dedicated regressions passed in the final independent invocation.

I independently compared batched student assay forwarding against five separate
native PyTorch GRUs over all 256 histories, with a chunk boundary through the
subject order. Maximum probability discrepancy was `5.96e-8`; teacher assay
copies were exactly equal.

Initialization now validates and adopts only matching protocol/data records and
prefixes of deterministic atomic JSON staging writes. Unknown files and
nonmatching staging bytes are rejected. The initializer writes population
identity before inputs. The regression suite covers interrupted protocol,
run-manifest and population-manifest writes, along with invalid orphan content.
For reconnect portability, only regenerated float64 true/reference policy
probabilities use the established absolute tolerance of 1e-12. All other input
arrays remain exact, and fitting continues from the immutable saved arrays.
Thus accepted backend roundoff never silently replaces checkpoint inputs.

I also independently enumerated the three-choice assay by a scalar branching
calculation. The four conditioning-cell path masses were 0.15, 0.35, 0.15 and
0.35. Reconstructed true and fitted signatures, RMS errors and the signed pull
projection matched the evaluator to `2.22e-16`. Eight scientific/configuration
tests passed in a separate invocation.


## Final executed verification

The complete new independent suite passed **38 tests**, with **one CUDA-only
skip**, in 24.51 seconds. It covers scientific/configuration checks, actual
resumption, file/config/source corruption rejection, optimizer and stopping
integrity, partial initialization, canonical paths and locks, notebook role and
release guards, source tampering, stale namespaces, and the reproducible builder.

I independently injected losses during first population-report finalization and
the final root manifest write. Both retries completed. The four and sixteen
already-written checkpoints respectively retained their exact bytes and were
never refitted. Each completed retry then returned the same summary without
changing any result file on a further invocation. Interrupted wall-time coverage
remains explicitly incomplete while checkpoint training time stays cumulative.

I inspected `07-main-working.executed.ipynb`. All five actual code
cells executed in order, and their source equals the shipped notebook's source.
The setup preflight passed **414 tests**, with **five GPU-only skips** and one
expected duplicate-ZIP fixture warning. The two-kind smoke and its immutable
completed rerun passed, followed by the local smoke-sized study invocation and
compact reporting. Both kinds, primary coordinates, control score tables and
diagnostic support were present. No production population was simulated.

The reviewed main-source fingerprint is
`b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372`.
The prospective protocol SHA256 is
`1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.
All original numerical modules and notebooks 01–06 match the starting Git
revision byte for byte. The prior notebook source identities remain intact.

The runtime implementation and working five-cell launcher pass review in this
bounded synthetic scope. Final release verification must additionally execute
the exact pinned local archive and retain its result. CUDA integration is
included in Colab's preflight and remains to be exercised on a T4.


## Exact pinned-runtime release follow-up

Independent final review verdict: **PASS, no open findings**.

The shipped notebook pins runtime commit

`29eb07627f0d8ea20e5b5964e7c44287d464f857`.

Its release readiness flag is true. The full runtime SHA, scientific source SHA
and prospective protocol SHA match the reviewed identities. The secret-access
guard precedes runtime setup. The launcher has five clean code
cells with no stored execution outputs.

I independently reconstructed the main-source digest from all 20 relevant Python
files in the immutable Git commit and verified the prospective JSON hash.
The digests are
`b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372`
and
`1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.

I inspected `07-main-pinned.executed.ipynb`. Every cell source equals
the current shipped notebook, and the five code-cell execution counts are
1, 2, 3, 4, 5. No error output is present. The exact archived runtime preflight
reported **411 passed, 8 skipped, 1 warning in 35.84s**. Five skipped checks
require CUDA; the other three verify local Git archives and require Git metadata
absent from the downloaded archive. The warning comes from the intentional
duplicate-member archive fixture. The two-kind smoke, every fitted comparison, controlled
behavioral assay and immutable completed resume passed. The local smoke-sized
production-cell invocation and all reporting cells completed. Both rare-kind
tables, primary signs, diagnostic support and the hierarchical family-mixture
control are present. This execution used CPU-only development fixtures.

The working-source and exact-pinned-archive checks both pass. The released
notebook's CUDA integration test and complete preflight run on the selected
Colab GPU before production training. No production population
was generated or evaluated in this review.
