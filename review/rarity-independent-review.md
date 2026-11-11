# Independent review of notebook 08 rarity tranche

Reviewed 2026-11-05 using the frozen main-study-v1 protocol, verified
first-tranche results, source inspection and separate CPU executions. The
review covers the new grid adapter, read-only baseline binding and notebook 08.

## Scientific design and source preservation

The next tranche contains the remaining 100-trial rarity levels, 5%, 10% and
25%. Both rare kinds contribute three populations at each level, giving 18
fresh populations. I independently enumerated all 18 seeds from the original
formula and verified their 72 data/split/teacher/student role IDs are distinct.
The production plans match those values exactly and exclude the first six
populations. The 1,000 people, model sizes, 500/4,000-step limits, masks, fitting
and selection rules, all controls and behavioral assay match the original
configuration. Source checks preserve all 28 original numerical, foundation,
main-study, protocol and notebook 01–07 files byte for byte.

The adapter subclasses the original cell configuration and calls the unchanged
training engine. Each cell identity adds its indices and the grid-source digest.
The digest covers the adapter and all underlying numerical modules, together
with the prospective JSON and trusted baseline specification. It leaves release
pin metadata outside the digest, avoiding a self-reference. The reviewed grid
source is `a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df`.

Within a population, distilled-versus-solo contrasts remain paired. Different
rarity levels use different prospectively fixed populations. Every rarity and
strategy keeps its separate three-population scores and support. The wrapper
has no pooled cross-rarity group summary. The displayed rarity comparison
states this distinction and includes the primary pair alongside diagnostic
scores and all controls.

## Baseline validation and portability

I independently validated the production baseline against all 87 hashes from
the existing independent first-tranche audit. I created a separate three-file
context snapshot, checked the protocol and summary bytes against the original,
loaded it through the portable reference API and repeated the completed
snapshot. All original and copied bytes remained unchanged. Metadata paths
retain their original provenance. The source data and model checkpoints remain
in the original baseline folder.

The validator checks the frozen configuration, source and protocol, exact
planned populations, root and child artifact manifests, package inventories
and finished optimization counters. Production hashes match a separately
fingerprinted specification. Dynamically generated smoke baselines use the
original tiny protocol and complete manifests, and cannot substitute for a
production baseline. The reference payloads are bound to the validated input
hashes. Partial atomic copies retry only when existing payloads or staging
prefixes agree. Corrupt, incomplete, overlapping and symlink paths are rejected.
The original baseline is required at its current supplied location for each
production invocation. Saved reference paths themselves are portable.

The seven baseline property tests passed in a separate invocation. The combined
independent release suite later repeated them with the runner and notebook
guards.

## Resumption and failure recovery

The wrapper advances at most one pending rarity cell per invocation. Existing
cell checkpoints retain current and selected weights, optimizer/RNG and history.
Timing sums each cell's cumulative checkpoint training once, and shows old
baseline compute separately. Interrupted wall-time coverage remains explicitly
unknown rather than being inferred. Final tables wait for the complete tranche.

Inspection identified an inherited startup edge case. The shared atomic JSON
writer stages `protocol.json.tmp`, `run_manifest.json.tmp` and
`manifest.json.tmp`, while the original runner recognizes shorter legacy
`.tmp` names during initialization. The new adapter validates exact expected
prefixes and narrowly normalizes these actual staging suffixes. It recovers
those new-cell failures without changing the original engine. A helper
restricted to the exact original smoke configuration provides the same safe
startup retry for the notebook's genuine tiny baseline. It rejects production
and grid-cell configurations before creating output.

Two additional review findings were resolved. Unknown existing output files are
rejected before baseline context is copied. Adopted staging files are cleaned
in a `finally` path if a later training failure occurs, and owned leftovers from
a prior hard interruption are checked before cleanup. Actual atomic-replace
failure tests cover these cases and both root/cell initialization layers.

The runtime suite checks interrupted versus uninterrupted optimizer/state,
history, RNG and predictions, complete-cell preservation, immutable completed
reruns, corruption and source/configuration rejection, canonical source/parent
separation and local concurrent writers. Its CUDA integration test runs in the
Colab preflight when CUDA is available.

I additionally trained a separate genuine tiny baseline and all three tiny
rarity cells. A fresh independent failure injector interrupted the actual
atomic replacement of the outer summary and the final outer manifest, after
all cell checkpoints existed. Both retries recovered while every one of the
48 existing neural checkpoint files remained byte-identical. I replaced the
cell writer with an exception during those retries, confirming no completed
cell entered a writer or refit path. New training time stayed equal to the sum
of cell checkpoint totals, missing wall time remained null with coverage, and
subsequent completed reruns changed no files or parent bytes.

## Launcher and actual cells

The historical launcher checks the full grid source and prospective protocol
before Drive and results setup. Setup rejects stale grid and legacy namespaces
and requires CUDA in Colab before preflight. The baseline smoke, grid smoke,
and production cells have separate roles. Changes to the adapter, source, or
trusted baseline specification stop execution before it creates results.

I inspected the actual working execution artifact and verified exact cell-type
and source equality with the shipped notebook, execution counts 1 through 5,
clean shipped outputs and absence of execution errors. The actual preflight
passed 446 tests with six CUDA-only skips. Its duplicate-ZIP-member warning is
from the existing malformed-archive fixture. All three development rarity cells,
both rare kinds, baseline preservation, completed smoke resume and the final
primary/control/support tables executed. The local verifier uses only the
smoke fractions 0.3, 0.4 and 0.5 and development seeds. It does not generate the
18 production populations.

A separate combined independent invocation of the baseline, runtime and
notebook suites passed 32 tests with one CUDA-only skip in 19.11 seconds.

## Verdict and release follow-up

The reviewed runtime and working launcher pass with no open findings in this
bounded scope. The archived-runtime execution verifies the CPU path. CUDA and
Drive behavior are checked in Colab. The measured previous T4 duration
supplies a rough scheduling reference for this same-history tranche, with
additional setup, preflight and idle billing recorded separately.

## Final pinned release review

The final launcher passes independent review with no open findings. I checked
the actual shipped notebook's full runtime pin
`87f5068a637377089fe0ea7087a13211a9ea8d2b`, ready flag, source and prospective
protocol assignments, and readiness before runtime setup. All five
shipped code cells have empty execution counts and outputs.

I independently reconstructed the grid-source fingerprint directly from the
immutable runtime commit's numerical and adapter files plus both bound JSON
specifications. It matches
`a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df`.
The frozen protocol digest remains
`1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.
Every reviewed runtime file matched those committed bytes.

The exact pinned archive executed all five final shipped cells successfully.
I compared every cell type and source with the executed artifact and confirmed
execution counts 1–5 and no error outputs. Pinned preflight passed **443 tests**
with **nine expected skips**, comprising six CUDA-only checks and three checks
requiring Git metadata absent from an extracted archive. The duplicate archive
member warning is the existing deliberately malformed test fixture.

The artifact includes the actual baseline and three-cell smoke, immutable
completed resume, all control-model and support tables and the explicit
statement that rarity levels use distinct populations. Local fixtures use only
development smoke populations. The production baseline remains read-only.

The source, scientific specification, checkpoint resumption, and pinned launcher passed the internal verification.
