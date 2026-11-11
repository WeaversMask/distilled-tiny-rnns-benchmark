# Comparison with the authors’ tinyRNN implementation

On 2026-11-04 we fitted the authors’ teacher and students, and the project’s teacher and students, to the same four simulated people. Their predictions can be scored on the same first-stage choices. The fitted models answer different training questions. The authors’ Gillan recipe learns both stages of each trial and selects each distilled student by how well it copies the teacher on validation trials. Our pilot learns first-stage choices and selects students by how well they predict observed validation choices. Those differences matter even when both models are called a “two-unit GRU.” [Authors’ Gillan split-trial experiment](https://github.com/jil095/tinyRNN/blob/a8fc159a2b2d451be6756ae244b29b7d991aac1f/training_experiments/exp_Gillan_splittrial.py) · [Authors’ teacher-target loader](https://github.com/jil095/tinyRNN/blob/a8fc159a2b2d451be6756ae244b29b7d991aac1f/training_experiments/training.py#L379-L501)

## What was run

We retrieved the authors’ public code at commit `a8fc159a2b2d451be6756ae244b29b7d991aac1f` and checked the working-file hashes against the project pin. The exercise uses the released Gillan split-trial configuration as its reference. The original study used human data. Here all histories come from our existing simulator. [Authors’ source](https://github.com/jil095/tinyRNN/tree/a8fc159a2b2d451be6756ae244b29b7d991aac1f) · [Original paper](https://doi.org/10.1038/s41586-025-09142-4) · [Pin and fixture settings](../configs/published_method_pin.json)

Each of the four people had 200 trials. Both implementations trained on the same 150 trials per person, selected models on the same 25 validation trials and reported the same 25 held-out first-stage choices. Trials remained in chronological order as model inputs. Both models could use an observed choice as history for a later prediction. The shared split and 100-update candidate horizon make this a small comparison fixture, separate from notebooks 05 and 06.

| Design choice | Authors’ Gillan split-trial recipe | Project pilot |
| --- | --- | --- |
| Inputs and predictions | Two recurrent steps per trial. Ten inputs encode state, action and reward. Six action scores cover both stages, with two valid actions at each decision. The first prediction comes from the initial hidden state. | One update per trial. Four signed inputs encode the previous trial. A single logit predicts the current first choice, including the first prediction from an all-zero input. |
| Training examples | The teacher sees four versions of each training history under consistent action/state relabeling. Each student sees its own four versions in training and validation. Test scores use the original history. | Each person contributes one history. Students have independent parameters, evaluated in a batch. |
| Teacher and student sizes | H=50 teacher. The released sweep includes H=1, 2, 3, 4, 5 and 20. We ran H=2 and H=5. Native H=2 has 102 parameters and H=5 has 291. | H=32 teacher and H=2 students in notebook 06. The H=2 student has 51 parameters. We also fitted H=5 with 171 parameters in this comparison. |
| Fitting | Float64, AdamW at 0.005, zero weight decay and a recurrent-weight L1 penalty of 0.00001. The reference sets up to 10,000 epochs with patience. | Float32, Adam at 0.01, no penalty on the teacher-target arm. The production arms use their own frozen horizon. |
| Selection | Observed two-stage choices select the teacher. Teacher probabilities select individual students. The trainer logs test loss each epoch and restores the first validation minimum. | Observed first-stage choices select both teacher and students. The primary student checkpoint uses a common population-wide step, with an individual checkpoint as a secondary result. The comparison evaluated every ten steps as in the pilot. |
| Splits | Shared seeded trial indices, 150/25/25 over 200 trials. | Production uses independently seeded masks for each person, approximately 60/20/20. The companion fit here deliberately uses the authors’ indices. |

We executed the authors’ dataset transform, augmentation, GRU agent, trainer, trial-splitting function and teacher-score loader. We supplied synthetic arrays in memory and used freshly generated teacher scores. Temporary checkpoint paths replaced machine-specific paths. The two orchestration functions were loaded from their pinned definitions without launching the upstream cluster or human-data scripts. One CPU initialization and a bounded run replaced the full seed and capacity sweeps. [Dataset encoder](https://github.com/jil095/tinyRNN/blob/a8fc159a2b2d451be6756ae244b29b7d991aac1f/datasets/GillanHumanDataset.py) · [Trainer and selection](https://github.com/jil095/tinyRNN/blob/a8fc159a2b2d451be6756ae244b29b7d991aac1f/agents/RNNAgentTrainer.py)

## What agreed and what differed

First we gave both implementations the same binary problem, previous-trial inputs and mapped H=2 weights. The pilot’s one logit equals the difference between the two upstream action scores. The largest difference was `1.11e-16` in float64. The masked losses agreed for observed choices, teacher probabilities and a mixture of the two. The maximum difference across recurrent and mapped readout gradients was `4.17e-17`. This checks a shared calculation at a fixed model state. An independently trainable second output score, different encodings and different validation targets can all change a fitted model. [Numerical record](../review/published-method-verification.json)

The native Gillan transform yielded 400 stage events per person with ten inputs each. Decoding the channels recovered each person’s original simulated actions, states and rewards, the six-class targets and the valid-action masks. Both stages of each trial occupied the intended split. Changing every observation from trial index 71 onward left every prediction through that trial’s first decision unchanged. The authors’ restored weights reproduced the first validation minimum for all five fits.

The table reports mean held-out first-stage negative log likelihood in nats. Smaller values mean that the model assigned more probability to the choices that occurred. The teacher row averages all four people. Student rows show the first person in each simulated group, chosen by index. Each student row covers only 25 choices.

| Scored model | Authors’ selected model | Project, population selection | Project, individual selection |
| --- | ---: | ---: | ---: |
| Pooled teacher (authors H=50, project H=32) | 0.646 | 0.661 | same shared teacher |
| Rare person, H=2 | 0.658 | 0.580 | 0.589 |
| Typical person, H=2 | 0.677 | 0.682 | 0.698 |
| Rare person, H=5 | 0.582 | 0.582 | 0.582 |
| Typical person, H=5 | 0.649 | 0.692 | 0.692 |

The direction changed across people and model sizes. The native teacher selected epoch 90, and all four native students selected epoch 100, the last allowed candidate. The pilot H=2 students selected step 70 under the population rule, while individual selection chose steps 60 and 0 for the two displayed people. These numbers verify that both recipes produce predictions and implement distinct selection rules on the same cases. The fixture is too small and the fits too brief to rank the methods or reproduce the published human-data findings. [Full scores and selected checkpoints](../review/published-method-verification.json)

## Rerun

Use a clean project environment with the project dependencies and the two optional upstream packages shown below. On the recorded machine the environment used Python 3.13.0, PyTorch 2.15.0 and NumPy 2.5.4. The script verifies the upstream commit and ten source files before importing the authors’ code. PyTorch needs its standard local shared-memory helper on macOS. Use normal Python with assertions enabled.

```sh
python -m pip install joblib==1.6.0 tqdm==4.70.1
git clone https://github.com/jil095/tinyRNN.git /tmp/tinyrnn-upstream
git -C /tmp/tinyrnn-upstream checkout --detach a8fc159a2b2d451be6756ae244b29b7d991aac1f
python scripts/check_published_alignment.py --authors-checkout /tmp/tinyrnn-upstream --output /tmp/published-method-verification.json
```

The script writes its temporary fitted models in a fresh directory and emits a JSON record at `--output`. The record includes the source pin, script hash, environment, trial splits, candidate selections and results for both pipelines. [Comparison harness](../scripts/check_published_alignment.py)
