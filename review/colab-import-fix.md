# Colab running-kernel import fix

Date: 2026-10-30 (Europe/Berlin).

The first live Colab run reached the smoke cell but could not import `distilled_rnns`. The setup cell installed the project in editable mode, then only added the source path explicitly for local execution. New editable-install path files are not automatically processed by a notebook kernel that is already running. The earlier local verification had already added the path and did not expose this Colab-specific omission.

The setup cell now activates the verified `PROJECT/src` path in both Colab and local kernels, invalidates import caches, and imports `ExperimentConfig` before preflight tests. The numerical source and pinned runtime commit are unchanged.

A new regression test runs the actual setup cell in a fresh isolated Python process with no preexisting project import path. It simulates a newly written editable path file and the Colab branch, then verifies that the configuration imports from the downloaded source in that same running process. Network installation and GPU detection are mocked in this focused test; it does not claim a live Colab run.

The full local suite passes: 107 passed, one CUDA-only test skipped. The updated notebook is also executed using the small CPU smoke configuration. Users with an already open notebook can add `import sys` and `sys.path.insert(0, str((PROJECT / "src").resolve()))` above the third cell's imports and rerun that cell.
