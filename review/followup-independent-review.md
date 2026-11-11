# Independent follow-up implementation review

Internal reviewer: GPT-6.1 Sol.
Date: 2026-10-30.
Repository: `.`.
Status: final local review passed; pinned launcher and archived runtime verified. GPU verification remains for the Colab preflight.

## Scope and verdict

Read the repository AGENTS.md, the uncommitted follow-up implementation and tests, optional L2 changes in `training.py`, launchers, downloader, and first-notebook preservation changes. I did not modify implementation files. No unresolved correctness blocker was found in the final implementation. The second launcher is enabled (`ready=true`) and pins runtime commit `4f8e2d196574fe7f59c768d0906d890ee7478dee`, source SHA256 `1b4d8e6c58188e9e16b6526a0f24bd9ec2ef20ddb0021d54a433db3e778b8255`. I independently verified the source fingerprint from a safely extracted local Git archive, matching production/smoke configurations, final seed list, deterministic notebook rebuilding, and all five final launcher cells in a fresh CPU kernel against that archived runtime.

The follow-up is an exploratory, conditional first-stage simulation study. Calibration uses the already-inspected original seed, chooses one population-wide L2 strength, and is excluded from descriptive summaries over the five fresh seeds. This review does not validate mechanistic recovery, significance, scientific convergence, or GPU performance.

## Evidence inspected

- The L2 objective adds `(lambda/2) * sum(parameter**2)` to the sum of per-person mean losses. The actual fitter is tested against separate ordinary PyTorch GRUs and Adam optimizers, including regularization. Validation selection uses observed-choice NLL without the penalty. Each person's parameters and validation checkpoint remain independent; arms restart with the same initialization within a seed.
- Calibration selection uses mean selected validation NLL over all people with a deterministic smaller-L2 tie rule. `selection.json` is written before test/oracle evaluation. Tests perturb held-out target labels, oracle probabilities, and rare-group labels while holding causal inputs fixed and confirm unchanged selection. Shared observed histories are intentional conditional inputs.
- A study protocol binds the serialized configuration and source fingerprint. Dataset manifests bind their protocol identity. Checkpoint signatures include regularization, data, masks, teacher targets, and optimizer settings. Unknown source/configuration or unprovenanced checkpoints are rejected. Latest optimizer weights and selected validation weights are stored separately.
- The invocation deadline and per-phase limit are soft limits checked between complete operations. A timed-out candidate returns incomplete without freezing a partial grid. Replications retain the calibration choice, and no aggregate is available until every planned seed is complete. Calibration is excluded and equal-weight seed summaries preserve paired comparisons.
- Diagnostic values without support remain null, with trial/subject/seed support counts. Summaries distinguish predictive NLL gain from policy-fidelity KL loss and include both selected-L2 and zero-L2 baselines. Convergence flags request inspection rather than claiming convergence.
- GPU peak memory is recorded only from CUDA execution. T4 conversion requires every recorded dataset session to be attributed to a T4; CPU or mixed-device histories do not produce a cost estimate. Arms are trained sequentially and no population-size-dependent gradient normalization is introduced.
- The second setup cell rejects modules imported from another source location, activates the verified source in the running kernel, and runs tests before smoke and study execution.
- Notebook 01 remains unchanged. Its frozen source fingerprint was independently recomputed from archived commit `bc4080578412d729b5ff86f10141152e0a2de653` and matched `e161fb5f9f7215c05d7fa1bfe22edf3225809c5c8258013e75fa2be051b95436`.

## Independent executions

The checks ran in a separate CPU environment.

1. Focused follow-up, notebook, model/training, simulation, evaluation, downloader, and legacy notebook tests: **119 passed, 1 skipped**. The skip was CUDA-only. A harmless duplicate-ZIP fixture warning occurred; the workspace cache write warning was avoided in the subsequent full run.
2. Full suite with cache provider disabled: **143 passed, 1 skipped** in 16.99 seconds. The skip was CUDA-only, and the only warning came from the intentional duplicate-ZIP fixture.
3. An actual follow-up notebook execution was attempted in a fresh local CPU kernel with all run outputs under a temporary directory. It halted in the first cell because the working-tree source changed while the launcher still contained its earlier provisional fingerprint. This confirms the integrity guard; it is not a successful full notebook execution. The final verification below supersedes that provisional integrity failure.

4. Final notebook and preserved first-notebook static checks: **8 passed**.
5. Executed the unmodified final `02_colab_followup.ipynb` in a fresh CPU kernel against a safe extraction of pinned runtime commit `4f8e2d196574fe7f59c768d0906d890ee7478dee`. **All five code cells passed with execution counts 1–5**. Its embedded archived-runtime suite passed **144 tests, 2 skipped** (both CUDA-only) in 14.93 seconds; the intentional duplicate-ZIP fixture produced the only pytest warning. Follow-up smoke and its second invocation passed; the separate tiny study completed both smoke seeds 51/61 and produced its descriptive summary. Both CPU summaries have null T4 cost proxies. I visually inspected the calibration convergence PNG and found readable axes, labels, curves, and legends.

Final executed artifact: `independent-followup-final-4d499h_u/executed.ipynb`. The artifact and generated studies are temporary, separate from the source project and user runs. This was the tiny verification configuration, not the production 1,000-person follow-up or a measured GPU experiment.

## Review refinements resolved before freeze

A subsequent design check identified that consecutive population seeds reuse an exact Torch initialization seed across different roles: teacher(seed + 1) and student(seed) both use seed + 3. I recommended spacing population seeds before freezing the protocol. The final intended fresh seeds are **20261101, 20261111, 20261121, 20261131, 20261141**, explicitly integer RNG seeds, not calendar dates. I inspected the updated default/configuration, smoke seeds 51/61, notebook builder, documentation, and configuration guard requiring population seeds to differ by at least four. This removes the exact cross-population offset reuse without selection based on observed results.

The CUDA checkpoint resume test was also parameterized for zero and nonzero L2 (0.01), resolving the suggested coverage improvement. A focused independent rerun of configuration guards and those CUDA cases passed **12 tests with 2 CUDA skips**. CPU tests already verify nonzero-L2 resumed fitting and independent optimization; neither CUDA case was executed here.

No GPU was tested. No claim is made about T4 wall time, memory headroom, billing, or full-size convergence. The Colab GPU preflight and smoke run must pass before the bounded follow-up runs. No source was published and no commit was created by this reviewer.
