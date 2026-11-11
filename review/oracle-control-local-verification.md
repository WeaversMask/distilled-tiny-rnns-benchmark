# Oracle-target control verification — 2026-11-01

Notebook 05 adds a new study module and reporting workflow. All pre-existing
numerical modules, notebooks 01–04, and their runtime pins remain byte-identical
to release `efc394a`. This release does not run the full scientific study locally.

## Local checks before pinning

- The new runtime, production configuration and launcher tests passed: **72 tests**.
- Executed all five actual notebook cells in a fresh CPU kernel, without replacing
  setup, training, resume or reporting code. Its full setup suite reported **270
  passed, 3 skipped**, with one intentional duplicate-ZIP fixture warning. The
  three skipped checks require CUDA and run during Colab GPU setup.
- The configured two-population smoke, completed saved-result resume, separate
  small study, and final results cell all completed. Execution counts were 1–5
  with no errors. Artifact: `05_colab_oracle_control.executed.ipynb`.
  The tables preserve real newlines, signed contrasts, support and unavailable
  values. The displayed four-panel learning-curve figure was visually inspected.
- A separate six-person, 15-trial CPU software check ran all three student arms
  through the full 4,000-step horizon. Every current, individually selected and
  population-selected state was finite; each student history had 401 evaluations,
  and Adam counters reached 4,000. This is a numerical stress check, not scientific
  evidence or a GPU runtime forecast. Its record is
  `oracle-long-horizon-validation.json`.
- Leakage tests perturb held-out generating probabilities, diagnostic references
  and group labels while checking unchanged fitting signatures, checkpoints and
  predictions. Perturbing only training probabilities changes only the oracle arm.
  Equal training targets and initialization give exactly equal soft-student
  trajectories. Observed-choice checkpoint selection is independently reconstructed
  with a fixture in which true-policy KL would select different checkpoints.
- Interrupted resume restores optimizer, latest state and both selected states;
  a resumed run matches an uninterrupted reference. Incomplete fifth populations
  prevent aggregation. Fixed identities reject changed settings, missing provenance
  and modified selection. The four fits per population are counted once each.
- Raw metric tests cover equal-person/equal-seed aggregation, positive/negative/tied
  contrast counts, both modes, missing diagnostic support and contextual signs.
- A validation-score assertion was narrowly adjusted to 3e-7 absolute tolerance:
  stored BCE uses float32 logits while the independent score uses float64 logs of
  rounded float32 probabilities. No model or metric implementation was changed to
  address this few-ulp numerical difference.

The new package source SHA256 is
`c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`.
Interpreter: `python`.

## Preservation of prior notebooks

Adding a package module changes the aggregate source fingerprint. The local
verifiers for notebooks 03 and 04 now default to their exact pinned Git archives,
as the earlier 01/02 verifiers already do. Neither shipped notebook was altered.
Both revised verifiers executed all five actual cells successfully:

- Notebook 03 archived runtime: **183 passed, 3 CUDA-only skipped**.
- Notebook 04 archived runtime: **197 passed, 4 skipped** (three CUDA-only and one
  checkout-only archive test). Its lack of `.git` matches downloaded Colab archives.

Artifacts: `oracle-legacy03.executed.ipynb` and
`oracle-legacy04.executed.ipynb`.

## Immutable release verification

The reviewed runtime is commit
`9370417a5b131d712f1c01e88b4785e213fb7c5a`. The historical launcher pins that
commit and the source hash above. Its readiness check precedes runtime setup.

The final shipped five code cells executed in a fresh CPU kernel against an
extracted archive of that exact runtime. All five completed in order, with no
errors; smoke, saved-result resume and the separate small study passed. Its
preflight reported **268 passed, 5 skipped**, one expected duplicate-ZIP warning.
Three skips require CUDA; two require a Git checkout for legacy-archive checks.
Those two checks pass in the working checkout and are intentionally unavailable
in a downloaded archive without `.git`.

Final artifact: `05_colab_oracle_control.final-pinned.executed.ipynb`.
Independent GPT-6.1 Sol review confirmed this artifact's code matches every shipped
cell, the immutable pin/hash/readiness agree, and both-mode signed/null-support
reporting is correct. Final verdict: PASS with no open findings.

Local verification used CPU. Colab must pass its
GPU preflight and smoke check before the full scientific run.
