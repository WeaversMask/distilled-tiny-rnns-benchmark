# Notebook 06 parent-validation device correction — 2026-11-03

## Failure and evidence

The reported production continuation attempt stopped during read-only parent
validation, before this invocation could create a child protocol or train. The Colab
runtime was PyTorch 2.11.0+cu130 on a Tesla T4. The validator reconstructed all
parent predictions on CPU despite the selected CUDA run device.

User-run read-only diagnostics found an `oracle_individual` CPU discrepancy:
maximum absolute probability difference `2.8371810913085938e-05`, mean
`1.567767933011055e-08`, 12 of 100,000 values above the fixed `5e-6` tolerance,
with the largest at person index 805, trial index 71. CPU threads: 1.

The user then reconstructed the saved oracle individual predictions on the
T4 for **all five production seeds**. Every comparison was exact: maximum
difference **0.0**, zero values outside tolerance. All five printed checkpoint
SHA256 values also matched the previously audited local originals. This
established a CPU/backend reproduction issue for the reported failure, not
changed oracle checkpoint files. The original local ARM CPU verification had
not exposed it; all five oracle comparisons there stayed within tolerance.

## Correction

`validate_parent` accepts an explicit device, defaulting to CPU for existing
read-only callers. The continuation runner passes its resolved run device;
the notebook's smoke-parent check passes `DEVICE` as well. Saved checkpoint
loading, signature bytes, state structure/history and Adam validation remain
on CPU. Only the model and forward-evaluation tensors/masks move to the
requested device; returned probabilities and saved-value comparisons return
to CPU as needed.

No numerical tolerances, training rules, original models, original source
identity or parent artifacts change. The device is not added to the immutable
parent audit identity, so device selection alone does not change that identity.
There is no retry on another device to accept a failed comparison.

New full runtime-package source SHA256:
`a7394cdff009e97f1130e135e8439e5ae70295f0fb7b29e01eba771831a659a0`.
Original numerical source remains
`c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`.

The corrected smoke folder is `budget-check-smoke-a7394cdff009`. The production
parent and output names remain `oracle-control-v1` and `budget-check-v1`.
The observed validation failure occurred before this invocation could create production output.
If a user has a pre-existing child protocol from another runtime revision,
preserve and inspect it; never delete or rewrite its source identity to force
resume. This correction is not a migration of already-started child runs.

## Verification before runtime pinning

Focused runtime and notebook tests: **91 passed, 1 CUDA-only skipped**, also
independently executed by a fresh GPT-6.1 Sol reviewer. New regressions check
CPU/CUDA/auto device routing before child writes, CPU signature/Adam auditing,
forward reconstruction on the chosen device, immutable parents and rejection
of changed candidate and selected prediction files at the original tolerances.
The CUDA case creates and audits a real GPU smoke parent in Colab preflight.
Another regression executes the actual smoke cell to check device routing.

All five actual corrected notebook cells executed in a fresh CPU kernel:
**362 passed, 4 CUDA-only skipped** in the full preflight, plus the expected
duplicate-ZIP fixture warning. Real smoke continuation, completed resume,
separate tiny study and reporting passed. Artifact:
`06_colab_budget_check.device-fix-working.executed.ipynb`.

The corrected CPU validator also revalidated the actual original production
parent: 77 files, all five planned populations, 5,509.51061769 inherited
checkpoint seconds. Original numerical modules and notebooks 01–05 are unchanged.

Independent GPT-6.1 Sol implementation and regression review: **PASS, no open
findings**. The user's
exact five-seed T4 reconstruction is direct device evidence; the new complete
CUDA integration/preflight has not been run on this local CPU-only machine.

## Final pinned execution

Corrected immutable runtime: `a97cb3f2e92fbce06c240484b3a52d012f56a7b9`.
The final launcher pins this commit and the source hash above with readiness
true.

All five final shipped cells executed against that exact archived runtime:
**359 passed, 7 skipped** in preflight (4 CUDA-only and 3 Git-checkout-only
checks), one expected duplicate-ZIP fixture warning. Smoke continuation,
completed resume, the separate tiny study and reporting all passed. Every
shipped cell, including Markdown, matches the executed artifact, with code
execution counts 1–5 and no errors.

Artifact:
`06_colab_budget_check.device-fix-pinned.executed.ipynb`.

Independent GPT-6.1 Sol final release verdict: **PASS, no open findings**.
The reviewer verified the exact archive, legacy and full source hashes, readiness
guard, deterministic builder, and absence
of shipped outputs, exact shipped/executed cells and successful final results.
Shipped notebook SHA256:
`71875c4f4a853f68a4b619625fa294947ba4b8eedc5db2181c9f4016c832bd1c`.
Executed artifact SHA256:
`8ee98decc8b0b9cfa9911acd0f813aacd39d04c523dacd6f994861d191d084da`.
The corrected validator's actual T4 execution remains for Colab preflight and
production validation. The diagnostic exercised reconstruction with the original saved weights.
