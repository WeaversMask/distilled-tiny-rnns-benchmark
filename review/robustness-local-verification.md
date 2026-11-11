# Robustness notebook local verification — 2026-10-31

Notebook 04 is a configuration and reporting release using the unchanged numerical
engine reviewed for notebook 03. No full scientific study or GPU measurement was
performed locally for this release.

## Checks before pinning

- The actual five code cells executed in a fresh CPU kernel with the new 10-person,
  15-trial smoke configuration. Setup ran the complete suite: **198 passed, 3
  skipped**. The skips require CUDA; the single warning is the intentional
  duplicate-ZIP safety fixture. Smoke, completed resume, a separate small study,
  and results rendering all passed.
- The independent GPT-6.1 Sol reviewer separately ran the full suite with the same
  198/3 result. Its independent smoke/resume and raw-array metric audit passed.
  See `robustness-independent-review.md` for scope and release status.
- Production configuration checks enforce the declared changes from notebook 03
  and disjoint data/split/initialization seed ranges. The software smoke includes
  the new maximum L2 strength 3. The existing audit tests cover separate tuning,
  leakage perturbations, incomplete candidate/study behavior, dual checkpoint
  resume, references, null diagnostic support, and trajectory accounting.
- Two six-person, 15-trial CPU stress checks exercised the full **4,000-step**
  horizon: solo L2=3 and distillation from fixed finite teacher logits. Each was
  deliberately paused (at steps 24 and 66 respectively), then resumed to step 4,000.
  Both had 401 scheduled history rows, finite current and selected states, Adam
  counters at 4,000, and identical predictions/selection/history on completed
  reload. These are numerical software checks, not scientific results or runtime
  forecasts. Elapsed bookkeeping can include checkpoint-write overhead on reload.
- The actual new results cell was additionally rendered against the previously
  verified notebook 03 downloaded scores, without modifying those files. It
  correctly displayed the positive rare diagnostic difference +0.0206, its
  [+0.0041, +0.0379] range, five positive seeds, matched individual-mode difference
  +0.0092 with a range spanning zero, and per-seed trial/person support. This is
  display validation only; those older scores are not new robustness results.
- All three PNGs displayed in the new smoke execution were visually inspected.
  Labels and legends are readable and do not obscure the data. The unchanged
  engine's inherited fixed-control plot label is explained in notebook 04; the
  fixed L2=0.01 control receives this run's longer horizon.
- Notebooks 01, 02 and 03 and all numerical package files were compared directly
  against release `31bd42b` and are byte-identical. Package source SHA256 remains
  `63c789f6b9ef1cb7c8d32bd726ecfbfcf2c44aa3ac48ed5da18c5096cb008210`.

Local Python: `python`. Local
execution uses CPU and temporary outputs. Colab performs CUDA-specific preflight
and the small GPU smoke/resume before allowing the full study.

## Immutable release verification

The reviewed runtime is commit
`bba8ce3e712173802f705650b27356fedbdea8fd`. The historical launcher pins that
commit and the source hash above. Its readiness check precedes runtime setup.

Executed the final shipped notebook against a fresh local Git archive of that
exact runtime, using `scripts/verify_robustness_notebook.py --pinned-archive`.
All five code cells completed in order, execution counts 1–5. Smoke and its
completed resume passed, and the separate tiny study completed. The archived
runtime's setup suite reported **197 passed, 4 skipped**: three CUDA-only checks
and one local-Git archive check that is intentionally skipped for a downloaded
archive without `.git`. The same archive-only check passed in the working checkout.
The sole test warning was the intentional duplicate-ZIP fixture.

Executed artifact: `04_colab_robustness.final-pinned.executed.ipynb`.
The final caption clarification and release constants were included in this run.
This verifies the released notebook’s local CPU path. The full study and CUDA preflight run in Colab.
