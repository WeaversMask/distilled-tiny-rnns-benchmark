# Local verification before independent review

This records the initial verification. See `resolutions.md` for the final post-review verification of the delivered notebook and source.

Date: 2026-10-29 (Europe/Berlin).

- Test suite: **73 passed, 1 skipped**. The skipped check requires CUDA.
- Actual notebook execution through nbclient and an IPython kernel: **all 5 code cells completed**, using the small CPU configuration. This execution reran the full test suite, performed a smoke run, reloaded its checkpoints, and completed the notebook's pilot cell using the smoke configuration.
- Full pilot dimensions checked on CPU: 1,000 subjects × 100 trials, teacher hidden size 32, student hidden size 2, two training steps in each of three arms. All outputs were produced. This is a shape/integration check, not adequate training and not a GPU timing measurement.
- Simulator tests include independent hand calculations and policy replay, transition frequencies, exact group counts, probability bounds, and causal inputs.
- Model tests compare batched student logits, gradients and optimizer updates with independent standard PyTorch GRUs; they also test subject isolation, loss masks, temperature/mixing endpoints, individual validation selection, final-step evaluation, resume equivalence and changed-checkpoint rejection.
- Evaluation tests include known metric values, quadrant signs, empty diagnostic support, and disjoint/exhaustive splits.
- Pipeline tests cover saved artifacts, identical resumed predictions, config mismatch refusal, incomplete status on time limits, and explicit CUDA requirements.

Environment: macOS ARM64, Python 3.13.0, PyTorch 2.15.0, NumPy 2.5.4. No CUDA GPU is available locally. Colab Drive mounting, package installation in the current Colab image, CUDA execution, T4 timing, and billed unit consumption remain checks for the first Colab run. The notebook fails before pilot training if its tests or GPU smoke test fail.

Evidence: `artifacts/01_colab_pilot.executed.ipynb`; `outputs/local-shape-check/` (local, excluded from upload bundle). The source fingerprint is recorded in each run's manifest and report.

Independent review and its resolutions will be recorded separately in this folder before handoff.
