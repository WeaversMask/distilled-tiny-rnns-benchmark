# Checkpoint and simulator reproduction

All 36 populations scheduled in the three supplied folders were audited from the pinned numerical source. The audit covered 18 new, six prior 2% and 12 software-smoke populations, with four checkpoints and seven selected neural outputs per population. It ran 7,039 checks and found zero structural issues. Training, resume and notebook execution were excluded. Every loaded checkpoint was verified against its population completion hash and opened on CPU with `weights_only=True`. All NPZ loads used `allow_pickle=False`.

The CPU environment was arm64 macOS, Python 3.11.5, NumPy 1.26.3 and PyTorch 2.4.1 with one thread. Saved Notebook 08 execution used Tesla T4 with PyTorch 2.14.1+cu130. The prior 2% execution used PyTorch 2.11.0+cu130. A matching CUDA environment was unavailable for this audit. The local PyTorch version is below the project’s nominal 2.5 runtime floor. The audit exercised read-only CPU inference and verified its baseline results against the prior 2.15 CPU audit.

## Simulator, targets and trajectory checks

Regenerated discrete fields, dtypes, shapes, assignments, feature arrays and seed-plus-one masks matched exactly. True and reference probabilities reproduced within 2.22e-16 at a 1e-12 absolute tolerance. Each person's four causal input features were zero at the first trial and matched the encoded observations of the preceding trial thereafter. The train/validation/test masks were disjoint and exhaustive with 60/20/20 trials in production and 9/3/3 in smoke. Rare counts matched the plan. The 13 expected model outputs in both prediction archives had valid shapes and finite probabilities for every population.

All 144 checkpoint signatures, weights, Adam settings and moments, counters, validation schedules, selected step fields and CPU score checks passed the reviewed validator. The audit also checked the complete phase reports against the checkpoint contents and reproduced initialization scores from data seed +2 for the teacher and +3 shared by the three student arms. For a person whose selected step coincided with the initial, final or population-selected step, the corresponding saved weights matched exactly. Their selected scores were at least as good as the available common-step scores within the per-person validation tolerance. The saved histories provide the earliest recorded population minima. Establishing each person's minimum across the full trajectory would require their unsaved intermediate scores or a fresh fit.

Both soft-target tensors are finite float32 with exact zero outside the training mask. The oracle's simulator-derived logits reproduce exactly. The sigmoid of the saved teacher training logits matches the saved teacher probabilities at those positions within 1.19209e-7 in production and 5.96046e-8 in smoke, against a 5e-7 threshold. The direct CPU teacher-logit replay exceeds the runner's 5e-6 resume threshold in all 18 new and all six prior production populations, with 70,414 and 5,336 entries respectively. The different magnitudes of probability and logit differences matter for applying this tolerance.

## Maximum absolute numerical discrepancies

| Check | Absolute tolerance | New 18 | Prior 6 | Smoke 12 |
| --- | ---: | ---: | ---: | ---: |
| Observed-history probabilities | 5e-6 | 3.88622e-5 | 4.11272e-6 | 8.94070e-7 |
| 256-history probe probabilities | 5e-6 | 4.29153e-6 | 1.28150e-6 | 8.34465e-7 |
| Selected mean validation NLL | 5e-7 | 1.19209e-7 | 5.96046e-8 | 2.38419e-7 |
| Selected individual validation NLL | 5e-6 | 3.21865e-6 | 5.36442e-7 | 1.78814e-7 |
| Direct CPU teacher training logits | 5e-6 runner resume check | 1.21474e-4 | 2.01464e-5 | 3.78489e-6 |

All 126 per-model probability, probe and validation maxima for the 2% baseline exactly matched those recorded in the prior CPU audit using PyTorch 2.15.0 and NumPy 2.5.4. The direct teacher logits are tabulated separately from the three original thresholds.

Eleven new observed-history arrays exceeded the 5e-6 probability tolerance. There were 677 affected array entries in total, including 396 train, 139 validation and 142 test entries. The saved predictions and fitted weights agreed within tolerance for all primary solo and distilled model arrays. Float64 CPU replay also differed from the saved CUDA outputs at some of these positions. The observed pattern is consistent with numerical differences across backends or versions. The exact source of the discrepancies remains unverified.

| Seed | Array in predictions.npz | Entries above threshold | Maximum absolute difference |
| --- | --- | ---: | ---: |
| 202613421 | teacher | 2 | 6.49691e-6 |
| 202623411 | oracle_individual | 2 | 5.00679e-6 |
| 202623421 | teacher | 4 | 7.03335e-6 |
| 202623421 | distilled_individual | 22 | 2.64645e-5 |
| 202623421 | oracle_individual | 3 | 8.97050e-6 |
| 202614421 | teacher | 6 | 7.45058e-6 |
| 202624401 | teacher | 224 | 2.01762e-5 |
| 202624411 | teacher | 163 | 3.88622e-5 |
| 202624411 | distilled_individual | 16 | 2.00272e-5 |
| 202624421 | teacher | 234 | 2.31266e-5 |
| 202624421 | distilled_individual | 1 | 3.23355e-5 |

The separate CPU sensitivity calculation recomputed test NLL, KL, diagnostic NLL/KL and assay RMS from CPU forwards and from the saved probabilities. All 24 scientific populations retained both paired signs for each group and selection rule. The largest change to a new rare-group primary pair was 4.68434e-9 in prediction gain and 7.52884e-10 in assay error change. Across all new neural model group means the largest absolute changes were 7.95552e-8 for test NLL, 2.51175e-8 for test KL, 4.12848e-7 for diagnostic NLL, 1.55233e-7 for diagnostic KL and 5.64889e-8 for assay RMS.

## Training and checkpoint selection

All 18 new teachers stopped at the declared patience, after 250–370 updates with selected steps 50–170. All 54 new students finished 4,000 updates. Population-selected steps ranged from 90 to 3,870 for solo, 220 to 3,840 for distilled and 720 to 3,480 for oracle. The two selections at or after step 3,600 were solo at 3,870 and distilled at 3,840 in seed 202622421. No population-selected step reached the cap.

| Student | Step 4,000 out of 18,000 | Step 4,000 out of 2,400 rare | Step 3,600 or later out of 18,000 | Step 3,600 or later out of 2,400 rare |
| --- | ---: | ---: | ---: | ---: |
| Solo | 23 | 2 | 521 | 103 |
| Distilled | 382 | 45 | 1,837 | 259 |
| Oracle | 463 | 133 | 2,145 | 310 |

The earlier 2% student boundary counts reproduced at 8, 129 and 145 across 6,000 people per arm. The logs and sensitivity calculation describe the frozen training budget and selection rule. Convergence remains unresolved. Cumulative checkpoint seconds, counted once per arm, were 27,401.728675 new, 7,028.766202 prior, and 11.970970 smoke.

## Supporting files

- [Checkpoint audit script](audit_checkpoints.py), [full per-population results](checkpoints-cpu.json), [run log](checkpoints-cpu.log).
- [CPU sensitivity script](check_cpu_sensitivity.py), [per-seed and per-model sensitivity](cpu-sensitivity.json), [run log](cpu-sensitivity.log).

## Rerun on CPU

Use the project environment and original output folders described in the [file integrity instructions](../verification/integrity-notes.md#rerun-the-artifact-checks). Run from the repository root. Keep the output directory outside the input folders and checkout.

```sh
audit_dir=review/notebook08-audit/reproduction
python -B "$audit_dir/audit_checkpoints.py" --source "$PWD" \
  --new "$new_run" --baseline "$baseline_run" --smoke "$smoke_run" \
  --output "$out/checkpoints-cpu.json"
python -B "$audit_dir/check_cpu_sensitivity.py" --source "$PWD" \
  --new "$new_run" --baseline "$baseline_run" \
  --checkpoints-audit "$out/checkpoints-cpu.json" --output "$out/cpu-sensitivity.json"
```

The checkpoint script returns a nonzero status if any numerical tolerance is exceeded, even if it successfully wrote the report. Check its structural issue and tolerance fields separately before running the sensitivity calculation. The sensitivity script resolves each population from the explicit input folders and the audited population identity. It can also use the included [historical checkpoint report](checkpoints-cpu.json) with the original output folders.
