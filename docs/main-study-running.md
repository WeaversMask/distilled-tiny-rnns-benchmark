# Running the first main-study comparison

Notebook 07 evaluates the frozen six-population first tranche. Each population
has 1,000 people, 100 trials and 20 rare people. Three populations contain pure
model-free rare agents and three contain reward-cue rare agents. The primary
comparison pairs held-out prediction gain with behavioral distortion for the
teacher-target student against solo. The teacher, oracle student, classical
and hierarchical models provide the declared comparisons. [Prospective design](main-study-protocol.md).

## Colab

Open `notebooks/07_colab_main_study.ipynb` in a fresh T4 Colab runtime and run all
five code cells in order. The launcher downloads the pinned source from public
GitHub and verifies it. Authorize the Google Drive mount when prompted. The
complete test suite and a small end-to-end smoke run precede the production tranche.

Results save to `MyDrive/distilled-tiny-rnns/runs/main-study-first-tranche-v1`.
The smoke folder is `main-study-smoke-<source-hash-prefix>`. Preserve both complete
folders, including protocol, manifests, checkpoint state, data, predictions,
probe predictions and reports. Once complete, disconnect the GPU and provide
both folders for read-only verification.

At a soft time-budget pause in a live runtime, rerun only code cell 4 using the
same run name. After a disconnect that replaces the runtime, restore cells 1–3
before resuming cell 4. One runtime may write each folder at a time. Current
weights, optimizer, RNG, selected checkpoints and validation history resume
from their saved states. A completed run verifies and reloads its saved output.

Each phase has a soft ten-minute training limit, and each invocation a soft
thirty-minute limit. Baseline fitting, probes, evaluation and saving add time.
The recorded training cost proxy uses the user's supplied 1.07 units per
T4-hour only when all recorded contributing sessions used a T4. Recorded
checkpoint time and observed wall-time coverage accompany the result. Setup,
evaluation, saving and idle runtime contribute additional Colab billing.

## Reading the output

Read each rare kind separately. Positive prediction gain means the distilled
student assigned higher probability to the held-out choices than solo did.
Positive behavioral distortion means higher error in the four-cell strategy
assay. Report both coordinates, their directions in every population, all
comparison models and the support counts. Population selection is primary,
and the individual checkpoint rule is a sensitivity comparison on the same
trajectories. The summaries have three populations per kind and are descriptive.

The baseline's frequency reference retains the pilot's fixed Beta(1,1)
smoothing. Hierarchical policies use their training posterior unchanged at each
history in the behavioral assay. That assay integrates three first-stage
choices under fair second actions and fair rewards with 70/30 transitions.
Conditional predictions and diagnostic scores use the separate drifting task.
The finite parameter grids contain the generating policies in this benchmark.
The report describes this implementation's first-stage adaptation of the
published method. [Method comparison](published-method-alignment.md).

## Local verification

The shipped notebook's five actual cells can be executed with the working
checkout or its pinned Git archive. Local execution uses the tiny smoke
configuration for the production cell and a separate temporary results folder.
It preserves all main-study evaluation populations for the Colab run.

```sh
python scripts/verify_main_study_notebook.py --output /tmp/main-working.executed.ipynb
python scripts/verify_main_study_notebook.py --pinned-archive --output /tmp/main-pinned.executed.ipynb
```

The runtime uses the existing numerical training module and a separate
`main_study` package. Its fingerprint covers that package, the foundation
policies and the original numerical package. The frozen protocol has its own
SHA256 identity. Configurations permit choosing the production or smoke mode
and the two runtime budgets. Scientific settings come from the frozen design.
