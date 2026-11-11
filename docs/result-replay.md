# Replay the 100-trial results

Use the two release archives to recalculate the primary comparison for all
24 simulated populations on CPU from the original saved predictions. The replay
calculates test negative log likelihood, behavioral assay error, and the number
of players in each prediction and assay outcome. It verifies the resulting
contrasts against the saved per-population reports.

## Download and verify the inputs

Save these archives in the same directory.

| Populations | Archive | SHA-256 |
| --- | --- | --- |
| Six at 2% rarity | [`main-study-first-tranche-v1.zip`](https://github.com/WeaversMask/distilled-tiny-rnns-benchmark/releases/download/rarity-baseline-v1/main-study-first-tranche-v1.zip) | `543c1ed858eb467852a14bd09d1b6f3f620320cb19037c7ded58fb8ba18f3547` |
| Eighteen at 5%, 10%, and 25% rarity | [`main-study-rarity-replay-v1.zip`](https://github.com/WeaversMask/distilled-tiny-rnns-benchmark/releases/download/v0.1.0/main-study-rarity-replay-v1.zip) | `f8d28189d769f7336003937e419d5b24f7283e7da67482ad103b5924d774330d` |

Compare the SHA-256 values using `shasum -a 256 *.zip` on macOS or
`sha256sum *.zip` on Linux. Extract both archives into the same directory. The
extracted folders should be named `main-study-first-tranche-v1` and
`main-study-rarity-v1`.

The 2% archive also serves as the exact production baseline for Notebook 08.
The rarity replay archive contains the four inputs needed per population
(`data.npz`, `predictions.npz`, `probe_predictions.npz`, and `report.json`).
The [bundle builder](../scripts/build_rarity_replay_bundle.py) verifies each input against the audited SHA-256 inventory before packaging. Checkpoint replay and full integrity checks use the original run folders.
[Notebook 08 production setup](rarity-tranche.md#running-notebook-08) · [Full audit](../review/notebook08-audit/report.md#4-verification)

Both archives contain synthetic data and use CC BY 4.0. [Data and results license](../DATA-LICENSE.md)

## Run the replay

From the repository root, use Python 3.10+ and NumPy. Pass the path to the
directory that contains both extracted folders.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.26,<3'
.venv/bin/python review/notebook08-audit/primary_reconstruction.py \
  --downloads /path/to/extracted \
  --output /path/to/primary-replay.json
```

The expected primary results are 23 of 24 rare-group averages with better
held-out prediction and 24 of 24 with lower assay error. The JSON records
every population, both checkpoint selection rules, and both player groups.
The saved comparisons use an absolute tolerance of `1e-12`. [Expected table](../review/notebook08-audit/primary-tables.md) · [Committed reconstruction](../review/notebook08-audit/primary-reconstruction.json)
