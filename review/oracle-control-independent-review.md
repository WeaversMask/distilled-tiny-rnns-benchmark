# Independent oracle-target control review

Internal reviewer: GPT-6.1 Sol. Review date: 2026-11-01.

## Conclusion

**PASS with no open findings** in the design, implementation, tests and final
immutable release. Archived five-cell execution and final launcher identity
passed independent review. This is review of a bounded exploratory target-control
experiment, not acceptance of a scientific hypothesis, convergence or a capacity
or cognitive-mechanism explanation.

## Reviewed behavior

Reviewed the new `oracle_control.py`, production/smoke configs, substantive runtime
tests, frozen configuration tests, generated notebook, builder, verifier, launcher
tests and protocol documentation, together with reused training/scoring helpers.

- True probabilities at fitting positions form the oracle targets through stable
  float64 clipping/logit conversion and float32 casting. Both soft tensors are
  exactly neutral outside the fitting mask. Held-out probabilities and rare or
  diagnostic labels do not affect fitting or selection.
- All students use the same seed+3 initialization, inputs and fitting/validation
  masks. The paired soft students have the same optimizer, horizon and zero extra
  L2, with teacher proportion and temperature exactly one. The teacher is fitted
  first using the existing observed-choice recipe. There are four fits per seed.
- Selection is frozen before simulation/fitting: inherited solo L2, population
  checkpoints primary, individual checkpoints secondary, no new calibration.
  Both modes use observed-choice checkpoint NLL. Learned student forward passes
  produce test probabilities; true test probabilities are not substituted for them.
- Latest and both selected snapshots, optimizer/RNG state, protocol identities,
  session environments and artifacts support resumability. Complete planned seed
  coverage is required before aggregation. Accounting counts each trajectory once.
- All paired gains have explicit left-minus-right signs, including diagnostic KL.
  Equal-person/equal-seed summaries retain missing support as null, with supported
  seed counts and per-seed people/trials. The primary contrast is specified before
  running, and both checkpoint modes appear in the report.
- The launcher preserves the secure downloader, stale-kernel import guard and
  release gate before runtime setup. Previous notebooks and pins remain unchanged;
  old verifier defaults now use their frozen archives after the source hash changes.

## Independent execution and resolutions

The reviewer ran a separate 16-person, 30-trial, two-seed CPU study (601/611), using
threshold 0.05 solely to exercise diagnostic support in a software check. It
provided 15 and 16 rare diagnostic trials. Independently recomputed 96 group,
model and mode NLL/KL/diagnostic values from saved observations and predictions to
1e-12; regenerated exact predictions from individual/population checkpoint states;
verified float64 logit conversion and zero off-fitting targets; and confirmed
oracle predictions came from the learned model rather than copied truth.

Independent configuration/launcher tests passed **24 tests**. A broader focused
run initially had 71 passes and one numerical tolerance failure: float32-logit
BCE and float64 probability-based NLL differed by about 1.12e-7 against a 1e-7
tolerance. The reviewer accepted a documented 3e-7 tolerance, with runtime
unchanged; the corrected focused suite passed **72 tests**.

Two reporting requests were resolved: signed seed-direction counts were added
to the saved comparison aggregates, and the notebook now displays per-seed
individual-mode diagnostic differences alongside the primary population contrast.

The review explicitly retains these interpretation limits: improvement with oracle
targets measures a target-source contribution under the chosen recipe; remaining
error can also involve optimization, finite data and selection, and cannot alone
identify limited capacity. No significance, free-running mechanism recovery or
percentage decomposition of teacher/compression errors is asserted.

## Final release check

**PASS.** Independently verified:

- Local Git archive `9370417a5b131d712f1c01e88b4785e213fb7c5a` exactly contains the
  reviewed numerical package. Its SHA256 is
  `c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`, matching the
  final source settings and executed bootstrap. The readiness check
  precedes runtime setup.
- Only the new oracle study module changes the numerical package relative to
  `efc394a`. Notebooks 01–04 and their source pins remain unchanged. Their local
  frozen-archive verification behavior was reviewed and executed successfully.
- The working-checkout notebook executed all five actual cells with **270 passed,
  3 CUDA-only skipped** in setup. The reviewer inspected its signed contrasts,
  both checkpoint modes, support handling, actual smoke/resume confirmation and
  absence of an invented T4 proxy on CPU.
- Final artifact `05_colab_oracle_control.final-pinned.executed.ipynb`
  matches all five shipped code cells exactly, with execution counts 1–5 and no
  errors. It ran against the full pinned archive and completed smoke, saved-result
  resume and a separate small study.
- Final archived preflight: **268 passed, 5 skipped**, one expected duplicate-ZIP
  fixture warning. Three skips need CUDA; two legacy-archive checks need a Git
  checkout, which a downloaded ZIP does not contain. Those checkout checks passed
  separately before release.
- Final tables correctly render both-mode per-seed diagnostic differences,
  positive/negative/tied counts, supported and unavailable values, and CPU timing
  without a T4 conversion. The displayed learning curves were also inspected
  during local validation.

The local verification ran on CPU. The Colab launcher requires the GPU
preflight and smoke check before the full study.
