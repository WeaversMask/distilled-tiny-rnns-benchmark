# Bounded training-budget check verification — 2026-11-03

Notebook 06 continues copies of notebook 05's two soft-target student
trajectories from 4,000 to 8,000 total steps on the same five populations. The
teacher and solo models remain fixed. This is one descriptive sensitivity
check, not a new replication, a convergence claim or a horizon search.

## Working checkout

- All 16 pre-existing tracked numerical-package and notebook files are
  byte-identical to release `393a8ec03791cb6a2f32cbd63b2f0638d667935a`.
  The original package fingerprint remains
  `c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9` when
  the new, separate `budget_check.py` module is excluded.
- Focused continuation tests: **44 passed**. These include genuine interrupted
  continuation versus uninterrupted training, retained current/Adam/RNG and
  both selected states, cumulative history, completed-arm reuse, immutable
  parents, atomic-copy interruption, changed-artifact rejection, complete-seed
  aggregation, paired signs/null support, and incremental compute accounting.
- Executed all five actual notebook cells in a fresh CPU kernel, without code
  replacements. The preflight ran the full suite: **357 passed, 3 skipped**,
  with one intentional duplicate-ZIP-fixture warning. The three skips require
  CUDA and are exercised by the Colab GPU preflight.
- The notebook created a real four-step oracle parent, extended its copied
  students to eight steps, verified a completed resume, completed a separate
  tiny study and rendered its final results. Execution counts were 1–5 with
  no errors. The verifier independently checked every paired difference,
  aggregate mean/range, support, direction counts and fixed reference score.
- The reconnect regression executes the actual third cell repeatedly in both
  live and fresh namespaces. A completed parent is reused without rewriting
  its environment log; predictions and compute remain unchanged. A missing
  parent with an existing child is rejected instead of silently regenerated.

Working execution artifact:
`06_colab_budget_check.working.executed.ipynb`.
Interpreter: `python`;
Python 3.13, PyTorch 2.15.0, CPU. The new package fingerprint is
`243a4e1c70f78dbe8d31ec0ebe48a65f6eeba60c54719c597b40081e44a48822`.

## Previous notebook preservation

The notebook 05 verifier now defaults to its original pinned archive, because
adding a module changes the full-package fingerprint. All five unchanged
notebook 05 cells executed successfully against runtime
`27fddc6c10ec711b3124e465199c03965c9d36b6`: **269 passed, 5 skipped** in its
preflight, one expected duplicate-ZIP warning. Three skips require CUDA; two
require Git metadata unavailable in a downloaded archive. Smoke, completed
resume and the separate tiny study completed. Artifact:
`05_colab_oracle_control.frozen.executed.ipynb`.

## Immutable runtime verification

The release launcher pins runtime
`baaf93518fae4eec86b3580b74aa2bbcdac0698b` and the new full-package fingerprint
above. The release-ready guard is true and precedes runtime setup.

All five final shipped cells executed against an extracted archive of that
exact commit. Preflight: **354 passed, 6 skipped**, one expected duplicate-ZIP
fixture warning. Three skips require CUDA; three require Git metadata for
archive checks, which the downloaded source intentionally does not contain.
These checkout checks passed in the working-tree preflight. Smoke creation,
real continuation, completed resume, separate tiny study and full reporting
all completed with execution counts 1–5 and no errors.

Final artifact:
`06_colab_budget_check.final-pinned.executed.ipynb`.

The full 4,000→8,000 scientific extension has not been run locally. Local CPU
smoke results are software checks only. GPU behavior, actual T4 timing and the
scientific outcome must be measured in Colab; no GPU runtime forecast follows
from these local checks.
