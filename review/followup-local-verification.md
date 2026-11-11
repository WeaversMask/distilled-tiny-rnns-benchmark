# Follow-up verification — 2026-10-30

The follow-up was built and checked locally before a Colab recommendation. This
record describes software checks, not new scientific results or GPU measurements.

- Full suite from the actual second notebook: **144 passed, 2 skipped**. Both skips
  require CUDA (checkpoint resume with zero and nonzero L2). The one warning is
  the intentional duplicate-ZIP safety fixture.
- All five code cells of `02_colab_followup.ipynb` executed in a fresh CPU kernel
  using `followup_smoke.json`. This ran the full test suite, a complete small study,
  a resume with identical groups and chosen L2, and a separate small study through
  the same final report cell used in Colab. Rendered score tables and both displayed
  convergence figures were inspected. Unsupported diagnostic values stayed unavailable.
- All five code cells of the unchanged first notebook executed against its archived
  runtime `bc4080578412d729b5ff86f10141152e0a2de653`. That runtime's suite passed
  **83 tests**, with its one CUDA-only test skipped.
- The original notebook is byte-identical to the previous shipped version. Its
  generator now reads its frozen numerical source hash rather than the current tree.
- New tests verify L2 independence, regularized resume, validation-only selection,
  exact tie rules, frozen protocol rejection, partial-study reporting, zero-L2 model
  reuse, diagnostic support, seed separation, immediate kernel imports, and stale
  revision rejection. The source hash is checked before imports/training.

Local interpreter: `python`.
Numerical follow-up source SHA256:
`1b4d8e6c58188e9e16b6526a0f24bd9ec2ef20ddb0021d54a433db3e778b8255`.
Executed notebooks are retained in the ignored `artifacts/` folder.

Independent review used GPT-6.1 Sol. It identified avoidable cross-role random
stream reuse from consecutive seed values. The final plan spaces population seeds
by ten and rejects overlapping seed-offset ranges before any scientific run.
The suggested extra nonzero-L2 CUDA resume test was also added. See
`followup-independent-review.md` for the reviewer's own evidence and release checks.

No full-size follow-up or CUDA run was performed locally. Colab runs the GPU checks
and its small follow-up/resume study before starting the default longer study.
