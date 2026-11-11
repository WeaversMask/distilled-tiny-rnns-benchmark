# Verification records

This directory contains the project’s internal implementation reviews, execution logs, and numerical checks. Historical records titled “independent review” document separate internal verification passes, including AI-assisted checks.

- [Notebook 08 report](notebook08-audit/report.md) summarizes the completed 100-trial sweep and its evidence.
- [File integrity and timing](notebook08-audit/verification/integrity-notes.md) describes the artifact inventories, source revisions, and compute record.
- [Checkpoint and simulator reproduction](notebook08-audit/reproduction/README.md) documents the CPU replay, including differences from saved CUDA outputs.
- [First-tranche results](../docs/main-study-first-tranche-results.md) describes the original 2% comparison.

These records accompany the clean release copy of the 100-trial benchmark. Historical commit identifiers document the source of the original runs. The public notebooks pin their own runtime snapshots, and the [historical source manifest](notebook08-audit/verification/historical-source.json) records the digests that bind the released numerical code to the original Notebook 08 runtime.
