# Independent-review resolutions and final verification

Date: 2026-10-29 (Europe/Berlin).

The independent GPT-6.1 Sol review found no blocker to a guarded first Colab feasibility pilot and identified two reporting issues. The original review is preserved unchanged in `independent-review.md`.

## Resolved findings

- **P2, mixed-device timing:** the T4 cost proxy is now emitted only when every recorded session used a T4. CPU, another GPU, missing attribution, or mixed histories produce null plus an explicit reason. Regression tests exercise T4-only and mixed/unknown histories without claiming access to a local GPU.
- **P3, overwritten environment history:** every session now references a content-addressed package inventory in `environments/`. Earlier inventories remain available after resumes. Tests verify preservation across changed inventories, hashes, duplicate normalization, and rejection of altered inventory contents. Missing historical inventories are explicitly marked unavailable. `environment-freeze.txt` remains a latest-copy convenience only.

The notebook and README describe these behaviors. No scientific model, generating process, loss, or metric definition was changed to address the review.

## Final verification

- Installed the package and test dependencies in a fresh Python environment outside iCloud. This avoids the evicted local library files that limited the reviewer's full-suite rerun.
- Fresh complete test suite: **83 passed, 1 skipped**. The skipped test requires CUDA; it is included in the Colab preflight.
- Executed the final notebook through a real IPython kernel: **all five code cells completed**. Its own preflight again reported **83 passed, 1 skipped**, followed by successful smoke training, checkpoint reload, and the pilot cell using the small CPU verification configuration.
- The package's dependency check reported **no broken requirements**.
- The earlier 1,000-person × 100-trial shape check and independent scalar/gradient checks remain evidence for unchanged numerical components. These were feasibility checks, not adequately trained research results.
- Rebuilt the upload archive with the final source, matching notebook, tests, and both review records. The archive checksum is in `artifacts/colab_project.sha256`.

Final runtime-source SHA256:

`e161fb5f9f7215c05d7fa1bfe22edf3225809c5c8258013e75fa2be051b95436`

Final executed notebook evidence is `artifacts/01_colab_pilot.executed.ipynb`. The clean upload notebook is `notebooks/01_colab_pilot.ipynb`.

The review examined the pre-fix source fingerprint recorded in its report. The fixes above were regression-tested locally after that review; this document does not imply a second independent sign-off on the changed source.

## Remaining first-Colab checks

Actual Drive mounting and checkpoint persistence, current Colab-image installation, CUDA execution, T4 speed/memory, and billed account-unit consumption have not been measured locally. The notebook requires its full runtime tests and GPU smoke/resume checks to pass before the pilot. Its output remains a preliminary parametric first-stage feasibility experiment, as specified in the README and methodology.
