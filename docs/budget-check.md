# Notebook 06: bounded training-budget sensitivity

## Fixed question

Does notebook 05's teacher-target versus oracle-target comparison materially
change when the same two training trajectories continue from 4,000 to 8,000
steps? This is one declared extension, across all five existing populations.
There is no calibration, architecture change, new seed selection, or search for
a favourable result. Stop at 8,000 regardless of what the results show.

The parent is the completed `oracle-control-v1` run. Production parent settings
are exactly those in `configs/oracle_control.json`, with source fingerprint
`c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9`.
The five seeds are 20261411, 20261421, 20261431, 20261441 and 20261451. The
1,000 people, 100 trials, 2% rare group, H=32 teacher, H=2 students, fitting and
selection masks, optimizer, learning rate, temperature and targets remain fixed.
Teacher and solo predictions are inherited context; those models are not refit.

## Continuation and selection

Continue **current** checkpoint weights and Adam/RNG state. Inherit the full
history plus both population and per-person best states. Each selected snapshot
therefore competes over steps 0–8,000. Do not restart from selected weights or
reset the optimizer. Population selection remains primary; individual selection
is a mandatory secondary comparison. Both use observed-choice checkpoint NLL.

This varies optimization budget and the number of checkpoint candidates
together. Selected checkpoint NLL cannot increase when previous candidates are
retained, but held-out error can. It is not a pure capacity or convergence test.
The stored observed-choice learning curves do not, by themselves, measure the
soft-target objective optimized by the two arms.

## Results to report

Keep 4,000-step baseline and 8,000-step extended scores visible side by side.
For each group, model and selection rule report observed-choice NLL, true-policy
KL and their diagnostic counterparts, with the original equal-person then
equal-seed weighting and unavailable diagnostics left null.

The primary paired gain at each horizon is rare diagnostic KL of the
teacher-target student minus that of the oracle-target student. Positive gains
favour oracle targets. The sensitivity contrast is extended gain minus baseline
gain. Each arm's error change is extended minus baseline, so negative values
mean improvement. Report seed directions and support alongside means and ranges.
These are paired repeated measurements of the same five populations, not ten
independent replications. No arbitrary numerical pass threshold or significance
claim is introduced.

The outcome informs the interpretation of the current exploratory result.
Do not pick a preferred horizon, teacher recipe or new hyperparameter because
it scores best on these already examined test data. Freeze any subsequent
training procedure before evaluating fresh main-study populations.

## Input protection and resumption

The original parent folder is read-only. Validate its complete planned seed set,
protocol/config/source, generated observations and masks, target masking,
checkpoint signatures, steps, finite states and selected predictions before
continuing. Bind the child to hashes of the critical parent files and retain a
verified parent snapshot inside its separate output folder. Reject changed or
incomplete parents and overlapping/symlink-escaping paths.

Checkpoint identity and optimizer structure are audited on their saved CPU
representation. Forward predictions and their validation losses are reproduced
on the selected run device (T4 in Colab), with the original fixed tolerances.
This avoids comparing an original T4 computation against a different CPU
backend. There is no relaxed-tolerance or alternative-device acceptance fallback.
Read-only local `validate_parent` calls default to CPU; callers can explicitly
select the original device when available.

Results go to `runs/budget-check-v1`. Only the two copied soft-student checkpoints
are extended. A resumed invocation retains their completed steps, optimizer and
both selected states; completed trajectories are not retrained. Incremental
training time is the child checkpoint's cumulative time minus its immutable
parent checkpoint's time, counted once per trajectory across all invocations.
Keep inherited compute separate. A T4-unit conversion applies only to recorded
incremental training attributed to T4 sessions; setup, checking, evaluation,
snapshot copying and idle billing are additional.

## Running

Complete Notebook 05 and keep its `oracle-control-v1` output folder in
`MyDrive/distilled-tiny-rnns/runs/`. Open notebook 06 in a fresh T4 Colab session
and run its five code cells in order. The launcher downloads the pinned source
from public GitHub. Authorize the Google Drive mount when prompted. The smoke
check creates a genuine four-step parent with the verified runtime and extends
copied checkpoints to eight steps using the matching smoke configuration.

Each phase has a soft ten-minute limit and each invocation a soft thirty-minute
limit. During a live session rerun only the fourth code cell at a time-budget
pause, keeping the same parent and run names. After a deleted runtime, restore
the first three cells before resuming the fourth. Use one runtime per output
folder. Once complete, retain the full `budget-check-v1` and
`budget-check-smoke-<source hash>` folders for review and disconnect the GPU.

After this bounded check, the agreed next foundation work is the
[published-method comparison and stronger baselines](foundation-plan.md).
