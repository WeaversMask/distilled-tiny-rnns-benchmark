# Independent baseline-audit review

Reviewer: independent GPT-6.1 Sol agent. Reviewed 2026-10-31.

## Current conclusion

No actionable correctness or protocol findings were identified in the implementation reviewed. This is approval of a bounded exploratory audit, not evidence that the scientific comparison will succeed or that training converges. The final immutable runtime pin, launcher readiness and archived-runtime notebook execution passed the release check below. The notebook is ready for the user’s first bounded Colab audit run, with CUDA validation deferred to its required GPU preflight.

## Scope and verified behavior

Reviewed project `AGENTS.md`, the new `src/distilled_rnns/baseline_audit.py`, the optional population-checkpoint changes in `training.py`, simulator/input construction, existing evaluation and provenance helpers, baseline configs, notebook builder/generated notebook, notebook verification script, relevant tests, and `docs/baseline-audit.md`. No implementation changes, commits, or publication were made by this reviewer.

- Calibration has distinct fitting/checkpoint/outer-tuning targets. Only outer-tuning observed-choice NLL selects solo `(L2, mode)` and distilled mode. Exact setting ties use lower L2, then population mode. The selected settings are written before fresh-seed test scoring. Calibration has `mask_tuning` and no test group report.
- The five planned replication seeds are fresh, exclude calibration, and avoid overlapping data/split/initialization seed offsets. Each replication fits its own models and freezes the calibration-selected settings. Group labels, generating probabilities and diagnostic support enter reporting only.
- Inputs remain causal: current predictions use previous-trial history only. The protocol explicitly permits past held-out observations in later causal inputs and describes conditional prediction, rather than independent histories or free rollouts.
- Optional population tracking captures a whole-population state at the minimum equal-person checkpoint NLL. It leaves individual best slices, gradients, evaluation times and student stopping unchanged. Both modes share one optimizer trajectory and initialization; resumed checkpoints retain both snapshots, optimizer state and RNG state. Strict comparison preserves earlier steps on exact ties.
- Primary results compare independently calibrated pipelines. Matched-mode controls hold checkpoint mode fixed while retaining the chosen solo L2 and zero extra distilled L2. The docs explicitly state that these controls still differ in objective and regularization. Fixed legacy individual L2=0.01 is retrained on fresh populations; selected/legacy fits are shared when their L2 values coincide.
- Coin-flip probability is exactly 0.5. Frequency uses fixed Beta(1,1) smoothing of fitting targets only. Neither reference has a held-out tuning choice.
- Group metrics average people equally; seed summaries average paired per-seed results equally and exclude calibration. Diagnostic NLL and diagnostic generating-policy KL share the predeclared subset, preserve unavailable support as null, and expose trial/person/seed support. No complete aggregate is emitted for an incomplete planned study.
- Source/configuration identities protect resume. Time limits are soft and checked between complete steps; completed phases reload their existing trajectory instead of retraining it. Training accounting sums each trajectory's cumulative recorded time once, and T4 conversion is unavailable for mixed or unknown device histories.
- The historical launcher checks readiness, validates and verifies the pinned archive, and guards imports in the running kernel.
- The first two shipped notebooks were verified byte-for-byte equal to their HEAD versions. Their SHA256 values at review: notebook 01 `e57185ca0405abb911b9ab3aaeb56aa8fe55af207f2c0f8109fdc67c33e28f0c`; notebook 02 `e62d4bdbb9d686698f80403a91a8e525de028a02a6bac00fe6c7e005b1c8bf58`.

## Independent execution

Used `python`, `PYTHONPATH=src`, CPU, and a temporary output folder. CUDA was unavailable. Ran the configured 10-person/15-trial smoke audit across calibration seed 71 and fresh seeds 81/91, then resumed the completed audit.

Independently checked saved arrays and reports rather than relying only on the implementation's assertions:

- All three populations had disjoint, exhaustive 9/3/3 fitting/checkpoint/held-out masks.
- Recomputed every solo calibration candidate's tuning NLL and the declared tie ordering; the selected pipeline agreed.
- Checked each saved student's population best step belonged to the evaluation schedule and its value equaled the minimum recorded raw population checkpoint NLL.
- Recomputed frequency probabilities from fitting targets and verified exact coin-flip arrays.
- Recomputed every reported fresh-seed model's group NLL, KL and supported diagnostic KL directly from saved data/predictions; all agreed exactly.
- Confirmed calibration omitted test groups/per-subject test results.
- Completed resume returned identical calibration settings and group aggregates, with bitwise-identical saved prediction arrays.

The focused population-tracking tests were also inspected for manual Adam snapshot selection, trajectory equivalence, exact ties, signature changes and resume. The final full-suite and fresh-kernel notebook execution independently performed by this reviewer are recorded below; no GPU timing or CUDA result is claimed by this review.

## Findings and resolutions

No unresolved implementation findings at this stage. The following interpretation limits are deliberate and adequately disclosed:

1. Population-wide tuning is dominated by the typical majority and is not rare-group-optimal selection.
2. The primary pipeline comparison and matched-mode controls do not by themselves isolate a causal distillation effect.
3. Calibration tuning is not an independent test; later changes based on the five fresh-seed test scores require a new assessment population.
4. Soft time caps are resumability safeguards, not a guarantee of billed duration. Boundary warnings request curve inspection and do not establish convergence.

## Final release check

Passed on 2026-10-31:

- Final runtime commit `5c933f45b140f7752bcb517c7f6abadd94b6612f` exists locally and was extracted directly with `git archive` into a fresh temporary project. Its archived package fingerprint is exactly `63c789f6b9ef1cb7c8d32bd726ecfbfcf2c44aa3ac48ed5da18c5096cb008210`, matching `configs/colab_baseline_source.json`, the final shipped launcher and the current package source. The only numerical-package change since the earlier smoke inspection was selection-figure presentation; reviewed labels and external legend placement do not affect fitting or selection.
- The readiness check precedes runtime setup. The final notebook references the full verified commit and fingerprint. Notebook 01/02 SHA256 values remain the unchanged values listed above.
- Executed the actual final shipped notebook in a fresh CPU kernel against the extracted pinned runtime, with temporary smoke outputs. All five code cells completed in order (execution counts 1–5), with no error outputs. Its runtime preflight reported **183 passed, 3 skipped**; the skipped tests require CUDA. The single warning came from the intentional duplicate-entry unsafe-archive test.
- Its smoke audit and completed resume assertions passed. The report produced both selection-audit figures and one learning-curve figure. Independently viewed all three PNG outputs: labels are readable, the external selection legends do not obscure bars, and no clipping or table-newline failure was found.
- Saved the verified executed notebook to `artifacts/03_colab_baseline_audit.executed.ipynb`. No full scientific audit, GPU timing estimate or scientific result was produced by this local validation.

Final verdict: no unresolved review findings. The pinned launcher and runtime are ready for the user’s first bounded GPU audit run; the notebook itself requires a GPU in Colab and runs the CUDA-specific preflight there.
