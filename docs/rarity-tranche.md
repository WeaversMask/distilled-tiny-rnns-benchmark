# Prospective 100-trial rarity tranche

Scheduled 2026-11-05 before evaluating any of its populations. This advances the
existing frozen main-study design by varying rarity while holding history length
at 100 trials. The 18 new populations use 5%, 10% and 25% rare people, both rare
strategies and three populations per rarity/strategy cell. Every population has
1,000 people. The verified 2% first tranche supplies the baseline reference.

The schedule follows the measured runtime of the 100-trial first tranche and
covers the remaining levels of one frozen factor. The H=32 teacher, H=2
students, original 500/4,000-step limits, optimizer, regularization, targets,
splits, checkpoint criteria, classical controls and behavioral assay retain the
prospective version 1 definitions. [Frozen design](main-study-protocol.md).

## Planned cells and seeds

| Rare fraction | Strategy | Three data seeds |
| --- | --- | --- |
| 5% | Model-free | 202612401, 202612411, 202612421 |
| 5% | Reward cue | 202622401, 202622411, 202622421 |
| 10% | Model-free | 202613401, 202613411, 202613421 |
| 10% | Reward cue | 202623401, 202623411, 202623421 |
| 25% | Model-free | 202614401, 202614411, 202614421 |
| 25% | Reward cue | 202624401, 202624411, 202624421 |

These are rarity indices 1, 2, 3 and history index 0 of the frozen formula. Split,
teacher and student initialization seeds add 1, 2 and 3 respectively. The 18 data
seeds and all their role offsets are disjoint from the completed first tranche.
Each rarity/strategy cell retains its own three-population summaries. Model
contrasts are paired within a population. Comparisons across rarity levels use
distinct simulated populations and are descriptive.

[Replay the completed 100-trial results from saved predictions](result-replay.md).

## Running notebook 08

For a smoke check, open the notebook and run the first three code cells. The
smoke check generates its own small baseline.

For the production comparison, download and extract the audited 2% baseline
ZIP (about 117 MB). Copy the extracted `main-study-first-tranche-v1` folder to
`MyDrive/distilled-tiny-rnns/runs/`, with `summary.json` and `protocol.json`
directly inside that folder. Preserve its contents as extracted. [Download the audited baseline](https://github.com/WeaversMask/distilled-tiny-rnns-benchmark/releases/download/rarity-baseline-v1/main-study-first-tranche-v1.zip)

The ZIP SHA-256 is
`543c1ed858eb467852a14bd09d1b6f3f620320cb19037c7ded58fb8ba18f3547`.
Notebook 08 checks all 87 baseline file hashes from the independent audit.
Use the archived bundle to reproduce this comparison against the exact baseline.
The new output includes its summary, protocol and verification binding.

Open `notebooks/08_colab_rarity.ipynb` in a fresh T4 Colab runtime and run all
five code cells in order. The launcher downloads the pinned source from public
GitHub. Authorize the Google Drive mount when prompted. New results save to
`runs/main-study-rarity-v1` and smoke results to `runs/rarity-smoke-<source-hash-prefix>`.

Each invocation advances at most one pending rarity cell using the existing
soft 30-minute invocation and 10-minute phase limits. After a pause, or a finished
cell when more remain, rerun only code cell 4 with the same run name. A deleted
runtime requires restoring cells 1–3 first. Use one runtime per output folder.
The completed-tranche summary verifies and reloads without adding training.
After all 18 populations finish, retain the complete new result and smoke folders
for review, then disconnect the GPU.

## Timing and reporting

The previous six-population 100-trial T4 run recorded 117.15 minutes of training
and 118.86 minutes of runner wall time. Three times those observations gives a
rough scheduling reference of about 6 hours for 18 populations and approximately
6.27 training-only units at the supplied 1.07 units/T4-hour rate. This is an
estimate for planning. The completed 18-population run recorded 456.70 minutes
of training and 465.13 minutes of runner wall time on Tesla T4. Setup, preflight,
evaluation and idle billing contribute additional cost. [Verified timing](../review/notebook08-audit/report.md#5-compute-and-study-status)

The final display includes the verified 2% baseline and separate 5%, 10% and 25%
results. Preserve the primary pair of prediction gain and assay error change,
with both selection rules, every seed and all controls. Keep secondary diagnostic
NLL/KL, probe fidelity, support/nulls and directional pull visible. The first
tranche improved the primary mean assay comparison while worsening rare
diagnostic scores. Both findings remain part of the research record.
[Verified first-tranche findings](main-study-first-tranche-results.md).

The completed rarity sweep covers 24 of the 96 frozen populations at 100 trials.
The remaining 72 use 200, 350 and 1,000 trials. They require their own measured
runtime scheduling before release.
