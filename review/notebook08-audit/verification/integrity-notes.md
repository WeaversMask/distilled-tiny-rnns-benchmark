# Notebook 08 file integrity and compute record

The integrity pass recorded 6,855 successful checks across 537 downloaded files and 36 saved populations. It compared the checkout's 30 runtime files with the pinned Git revision and compared the released notebook and its source binding with the release revision. Both embedded baseline references loaded successfully. The full 2% baseline matched the trusted 87-file hash inventory, including exact copies of the protocol and summary embedded in production. The separate raw smoke parent was absent. Its portable reference verified its binding and the copied protocol and summary and listed 59 raw parent files for four planned populations.

## Input inventories

| Canonical input | Files | Directories below root | SHA-256 of canonical relative-file hash map |
| --- | ---: | ---: | --- |
| main-study-rarity-v1 | 267 | 47 | `c1be31a49857c16f823a663dab090e8812b8b739d732557664f95f95061eb6b2` |
| main-study-first-tranche-v1 | 87 | 14 | `504f92bd0de798299010d17209f20cb21d262370f481131a2723e7b7051f44f8` |
| rarity-smoke-a4582314a2b3 | 183 | 35 | `178c4d5dcd07448fe19bcb48bc1ba3ba8523b3d4acd53843b1cfa90bc49d032d` |

The maps used sorted relative paths and compact JSON. The post-integrity inventory matched the starting inventory byte for byte, including file and directory names. The audit checked for symlinks, special entries, partial or staging files, unsafe manifest paths, and extraneous or missing artifacts. All checks passed.

## Provenance and completion

- Historical audit checkout HEAD `3a943d9d1c8c2ceebfa2173b96fa8e54b98806f5`.
- Pinned runtime Git revision `87f5068a637377089fe0ea7087a13211a9ea8d2b`.
- Notebook release Git revision `92521ec26f1030443cde325530f50f3f479b18b4`. Released notebook file SHA-256 `764feeac07e7f9aa22e6f07bd60c539257af05c6d0b26ae20d5eee9673fbf332`.
- Grid source SHA-256 `a4582314a2b34289883f526366298f6604bb4c1e327856d068ee91e2c494c9df`.
- Core source SHA-256 `b5c5aa888e9b9ba00e9e8c69486c740a40efaf134bcda41397d8980fc4944372`.
- Frozen design file SHA-256 `1e8762df6fc01919dd3cfc432e96b78ed1d3fe1a25c26859edd9674cd501821e`.

| Protocol artifact | SHA-256 of saved `protocol.json` | Completed populations |
| --- | --- | ---: |
| Notebook 07 raw baseline, 2% | `43112ad9a9ec3c052cb4cc229ada2027403152910f5f6df6cc02886777ff948a` | 6 |
| Notebook 08 production outer | `8c02ac39eaf98fee12f5d8f46389598c4b16854f633c2e3c4758d038a62e22e3` | 18 |
| Production `rarity-05_history-100` | `557dc057f22ee219a7b5b6c2f038deda5074e5dd96be3528acd2171f0c391559` | 6 |
| Production `rarity-10_history-100` | `bdc89c13aad2ffeb657d7ae4b546368d0b6cbcd7d241ebeb5a14ead8470c10a8` | 6 |
| Production `rarity-25_history-100` | `be80b24de11be3dc277095cc7568c22f7b918c1d4fc8df4940ae45ab9b3031ae` | 6 |
| Notebook 08 smoke outer | `e8596f10cf561a555cdde6d4dfe6799c233f3b07e4d755d89566b05b0b7393fb` | 12 |
| Smoke `rarity-05_history-100`, actual 30% | `5db6a8b3f2d44dd2487648342e71eeafe3b014c90c7f2a497075a067a39bef17` | 4 |
| Smoke `rarity-10_history-100`, actual 40% | `005c8e2fe642f4ce03c38633df064006087b734e1a434bc72057c3e95c286f09` | 4 |
| Smoke `rarity-25_history-100`, actual 50% | `459c5ee7868191383d44243339558d44c36b627c449e047bd71269b9d0ecb4cf` | 4 |
| Notebook 08 smoke parent's portable copy | `c88e5fd9be6e70ca0ccd3d8137e644c3a80bccb3124610385533aa822d09ebde` | 4 documented |

The outer manifests matched the pinned rarity tranche configuration and the embedded cell summaries matched their on-disk copies. The inner cells retained the core first-tranche study identifier and source hash. Every scheduled cell and seed was marked complete, pending lists were empty, and each population matched its cell protocol hash. The production and smoke outer completion lists covered 266 and 182 files, respectively, excluding their own manifests. The core completion lists covered exactly 80 entries per production or baseline cell and 54 per smoke cell, including the per-session content-addressed package snapshots. The audit checked each latest environment freeze copy against its population's last session. Every final population session was complete, and phase records matched their population reports.

## Measured compute

| Run | Checkpoint training, min | Inner runner wall, min | Outer runner wall, min | Outer invocations recorded | Training-only T4 units |
| --- | ---: | ---: | ---: | ---: | ---: |
| New 5% cell | 152.675 | 154.447 | 155.746 | 6/6 | 2.722699 |
| New 10% cell | 151.301 | 153.093 | 153.863 | 6/6 | 2.698192 |
| New 25% cell | 152.720 | 154.459 | 155.519 | 6/6 | 2.723511 |
| All 18 new production populations | 456.695 | 461.999 | 465.128 | 18/18 | 8.144403 |
| Prior 2% baseline, separate | 117.146 | 118.858 | — | 4/4 core | 2.089106 |
| Notebook 08 software smoke | 0.200 | 0.596 | 0.672 | 3/3 | 0.003558 |

The production total counts 27,401.728675114 cumulative checkpoint seconds once across 72 fits. It records 27,907.668607373 outer runner seconds. All 33 population sessions recorded wall time, with 18 completed sessions and 15 resumable pauses (five solo, six distilled and four oracle). Every core invocation recorded a duration. Fifteen returned at the time budget before the relevant cell completed. Each outer invocation returned with a saved result and the outer manifest completed only after all three cells finished. No recorded invocation had an unknown duration.

All 33 new production sessions and 12 grid-smoke sessions recorded Tesla T4, CUDA 13.0, cuDNN 92800, PyTorch 2.14.1+cu130, NumPy 2.4.6 and Matplotlib 3.10.0. The nine earlier baseline sessions recorded Tesla T4, CUDA 13.0, cuDNN 92700 and PyTorch 2.11.0+cu130, with the same NumPy and Matplotlib versions. The 1.07 units per T4-hour conversion applies to measured checkpoint training time. Setup, idle time and actual billing are separate.

Timing was recomputed using CPU checkpoint loads with `weights_only=True`. The recorded and recalculated per-arm elapsed times agreed exactly. Timing aggregates were checked to an absolute tolerance of `1e-10`. See [integrity.json](integrity.json) for every seed, arm, invocation coverage count and recorded session environment. The reproducible file checks are in [verify_integrity.py](verify_integrity.py), with full [before](inventory-before.json) and [after](inventory-after.json) hash maps.

## Rerun the artifact checks

Use a checkout or extracted source archive of the public benchmark with the [project dependencies](../../../README.md#verify-and-run) installed. The audit CLI supports Python 3.10+. Set the three input variables to the absolute paths of the extracted original output folders listed above. The baseline release is linked in the [Notebook 08 setup](../../../docs/rarity-tranche.md#running-notebook-08). The complete check also requires the original Notebook 08 production and smoke output folders, including their checkpoints and manifests. Choose a new output directory outside the source checkout and inputs.

```sh
audit_dir=review/notebook08-audit
python -B "$audit_dir/verification/inventory.py" \
  --new "$new_run" --baseline "$baseline_run" --smoke "$smoke_run" \
  --output "$out/inventory-before.json" \
  --compare-to "$audit_dir/verification/inventory-before.json"
python -B "$audit_dir/verification/verify_integrity.py" --source "$PWD" \
  --new "$new_run" --baseline "$baseline_run" --smoke "$smoke_run" \
  --inventory "$out/inventory-before.json" --output "$out/integrity.json"
python -B "$audit_dir/metrics/verify_raw_metrics.py" --source "$PWD" \
  --new "$new_run" --baseline "$baseline_run" --smoke "$smoke_run" \
  --output "$out/raw-metrics.json"
python -B "$audit_dir/metrics/cross_check_primary.py" --metrics "$out/raw-metrics.json"
python -B "$audit_dir/verification/inventory.py" \
  --new "$new_run" --baseline "$baseline_run" --smoke "$smoke_run" \
  --output "$out/inventory-after.json" --compare-to "$out/inventory-before.json"
```

The original audit inspected the runtime and launcher directly at their historical commits. The [historical source manifest](historical-source.json) records the 30 runtime file hashes and original launcher digests from that checkout. The public audit compares the checkout’s runtime files to those recorded hashes and checks the public Notebook 08 bindings against its source configuration. The original launcher digest is archival evidence. Rechecking its bytes would require the original archived notebook. The audit can verify the published numerical source and saved outputs from an extracted source archive.

Run the following command with the included metrics and reconstruction JSON to render the checked-in tables and primary figure to a separate folder. Pass `--metrics` and `--primary` to use freshly replayed JSON.

```sh
python -B "$audit_dir/present_results.py" --output-dir "$out/figures"
```
