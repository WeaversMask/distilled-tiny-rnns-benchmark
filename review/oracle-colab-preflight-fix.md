# Colab preflight correction — 2026-11-02

## Reported failure and cause

The second cell of notebook 05 imported the project correctly on Colab with
Python 3.13, PyTorch 2.11.0+cu130 and a Tesla T4. Pytest then failed in
`test_actual_bootstrap_rejects_changed_package_before_training`: the subprocess
hit the release guard instead of the intended changed-source assertion.

That fixture executes the archived notebook's bootstrap with a deliberately
changed local package. Its `-I` subprocess still loaded system site packages,
including `google.colab` on Colab. The archived runtime notebook has its original
pre-release guard, so the child selected the Colab branch and stopped there.
The same local-fixture test passed on the Mac, where Colab is not installed.

## Correction and scope

Add `-S` to this standard-library-only test subprocess so it cannot import
installed site packages. The actual notebook retains its Colab detection,
release, secret-access, source-fingerprint and GPU guards. No test is skipped
and no numerical tolerance is relaxed. Notebook 05 now captures and prints
merged pytest output before raising on failure, so Colab users see the cause.
The added setup case verifies visible failure output and continued rejection of
failed verification. Notebooks 01–04, all numerical source files and experiment
settings remain unchanged.

A live-session repair changes only the test invocation in the downloaded
runtime and reruns the complete suite with visible output. It neither reloads
project modules nor touches Drive results. Continue with the original third
code cell only after verification succeeds; this does not require a restart.

## Verification and independent review

- Reproduced the exact reported failure using the original pinned bootstrap
  (`9370417a5b131d712f1c01e88b4785e213fb7c5a`) and a fake installed
  `google.colab` package in a disposable virtual environment. The original
  child hit the release guard; the corrected child reached the changed-source
  rejection and did not create the run directory.
- Full working-checkout suite: **271 passed, 3 CUDA-only tests skipped**.
  The one warning comes from the deliberate duplicate-ZIP safety fixture.
- Independent **GPT-6.1 Sol** review: **PASS, no blocking findings** in the
  correction and the exact live-session repair. Independent focused run:
  **23 tests passed**. Failure output is printed before the exception; failed
  checks still stop setup.
- Local checks do not validate CUDA execution. The user's complete Colab
  preflight must pass before continuing to the smoke/training cells.

## Corrected release validation

The runtime is pinned to `27fddc6c10ec711b3124e465199c03965c9d36b6`.
The numerical source SHA256 remains
`c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`.

- Executed all five final launcher cells against that exact archived runtime
  in a fresh CPU kernel. Setup, smoke, saved-checkpoint resume, separate small
  study and results display completed successfully. Setup: **269 passed,
  5 skipped** (three CUDA cases and two local-Git archive cases).
- Independently executed the exact live repair cell on the user's original
  pinned archive: **268 passed, 5 skipped**. Before/after file hashes show that
  only `tests/test_oracle_notebook.py` changed. The source fingerprint remained
  identical, and the cell printed continuation only after successful tests.

The local executed notebook is
`05_colab_oracle_control.preflight-fixed.executed.ipynb`.
All GPU checks remain required in the user's Colab session.
