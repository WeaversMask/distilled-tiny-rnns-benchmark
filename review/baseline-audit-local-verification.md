# Baseline-audit local verification — 2026-10-31

The baseline audit was verified before release for a new Colab run. No production
simulation or CUDA measurement was performed locally.

- The full suite executed from the actual notebook: **183 passed, 3 skipped**.
  All three skips require CUDA (two existing checkpoint-resume cases and the new
  population-snapshot resume case). The only warning is the intentional duplicate
  ZIP safety fixture.
- All five code cells of notebook 03 completed in a fresh CPU kernel using the
  10-person, 15-trial smoke configuration. This exercised full preflight tests,
  calibration, frozen settings, both repeated seeds, exact resumed outputs, a
  separate small study, and the actual results cell.
- Displayed tables and all three figures were inspected. Audit figure legends were
  subsequently moved outside the axes and labels clarified so data remain visible;
  the affected audit/notebook tests passed (27 tests), and the revised plot was rendered
  and visually checked. This display adjustment does not alter fitting or scoring.
- Independent GPT-6.1 Sol review ran its own small audit and resume, and independently
  recomputed split masks, reference probabilities, outer tuning scores, saved
  population checkpoint minima, and fresh-seed reported metrics. See its signed
  record in `baseline-audit-independent-review.md`.
- Training tests establish unchanged optimizer trajectories when tracking the
  extra population snapshot, independent per-person gradients, manual snapshot
  equivalence, exact ties, signature protection, and resumed state equality.
- Audit tests establish that tuning targets cannot change weights, fresh-seed test
  targets/oracle/group labels cannot change models or frozen settings, partial
  candidate grids cannot freeze selection, incomplete studies cannot aggregate,
  each trajectory is counted once, and unsupported diagnostic metrics remain null.
- Notebooks 01 and 02 remain byte-identical. Notebook 02's updated local verifier
  uses its frozen Git archive: all five cells passed with 144 tests and two CUDA
  skips from that historical runtime.

Final package source SHA256:
`63c789f6b9ef1cb7c8d32bd726ecfbfcf2c44aa3ac48ed5da18c5096cb008210`.

Local interpreter: `python`.
Executed notebook artifacts remain in the ignored `artifacts/` directory. The
final launcher's immutable commit and readiness are checked separately at release.
Colab runs the GPU checks and small audit/resume check before the default study.
