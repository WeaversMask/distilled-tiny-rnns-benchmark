# Reward as a cue in a two-step task

In a fixed simulated population, a model allowed to choose between the two learning rules identified all four people using the reward cue when it had 1,200 training choices per person. With 60 choices it identified two of the four. It correctly classified all 20 hybrid learners at both lengths. This is a recovery check in a population whose possible rules are known.

## The cue and the behavioural probe

The cue learner aims to return to the state where it last received a reward. After an unrewarded trial it aims for the other state. It picks the first-stage action that commonly leads to its desired state with probability `rho`. Its first choice in an empty history is a fair coin. Its next first-stage choice depends on the preceding state and reward alone. The hybrid learner instead accumulates learned action and state values across trials. The fixture uses cue reliabilities of 0.60, 0.75 and 0.90, all fixed before fitting. This is the specific rule chosen for the proposal's reward-as-cue example.

Both groups encounter the same task. Each first-stage action has a 70% chance of reaching its usual state. Within a state, both groups learn about second-stage action values and use the same softmax rule. Reward probabilities start between 0.25 and 0.75, drift independently and reflect at those bounds. The simulator uses the existing pilot task functions for those dynamics. Across the long fixture, 69.95% of transitions reached the usual state.

Consider one observed trial. First action 0 reaches state 1, an uncommon transition, and second action 0 either earns a reward or does not. With `rho=0.75`, the cue learner's next probability of choosing first action 1 is 0.75 following reward and 0.25 otherwise. The predicted contrast is **0.50**.

For a hybrid with model-based weight 0.75, learning rate 0.4, inverse temperature 4 and initial values 0.5, the rewarded trial raises the chosen first-stage value to 0.6 and the chosen second-stage value to 0.7. Its next action-1 log odds are 0.14. With no reward the corresponding values are 0.4 and 0.3, and the log odds are 0.10. The two probabilities are 0.53494 and 0.52498, a contrast of **0.00996**. Executable checks reproduce both predictions. The calculation distinguishes these two specified policies on a controlled history. Other parameter settings can change a hybrid's response.

## Fitting the two families

The fit uses the twelve fixed hybrid parameter triples and three cue reliabilities. Each family starts with equal weight, and every candidate within its family starts with equal weight. For each person the model scores only observed first-stage training choices, then averages the candidate likelihoods within each family. Averaging gives the two families equal standing at the start despite their different grid sizes. Across people, empirical Bayes estimates the two population weights with the same 0.1 pseudocount per family. Each person's predictive probability averages candidate policies using their posterior weights. The hybrid-only and cue-only controls use the corresponding within-family posterior under identical masks.

Evaluation uses simulator truth and the pilot's fixed majority-like reference. That reference replays a hybrid of weight 0.75 on each person's exact observed history, with that person's second-stage learning rate and inverse temperature. Those privileged parameters are available only in simulation. A diagnostic test trial has an absolute difference greater than 0.2 between the true and reference first-choice probabilities. The fit receives the four observable history arrays and a training mask. The verified causal encoding starts with zeros and includes only the previous trial's events when predicting a choice.

## Recovery and support

The generator seed is 99309 and the split seed is 99310. The short condition takes the first 100 trials of the same 2,000-trial histories. Each length receives its own 60% training, 20% validation and 20% test masks. The fits use only the training portion. Saved simulator probabilities agree exactly with policy replay for every person and trial. Fixed reference replay also agrees exactly. On each person's long fitting history, every listed candidate from the other family differs from the generating probability. The smallest mean gap in true-policy KL is 0.01522 nats per trial.

| Quantity | 100 trials per person | 2,000 trials per person |
| --- | ---: | ---: |
| Training choices per person | 60 | 1,200 |
| Hybrid people classified correctly | 20 / 20 | 20 / 20 |
| Cue people classified correctly | 2 / 4 | 4 / 4 |
| Average posterior probability of the cue family among cue people | 0.599 | >0.999999 |
| Estimated cue share (actual share 4/24) | 0.123 | 0.169 |
| Cue people with diagnostic test support | 3 / 4 | 4 / 4 |
| Diagnostic test trials among cue people | 34 | 747 |
| Cue-group overall KL, mixture / hybrid only / cue only | 0.01630 / 0.10912 / 0.01154 | <0.00001 / 0.09953 / 0.00000 |
| Cue-group diagnostic KL, mixture / hybrid only / cue only | 0.01973 / 0.15742 / 0.01539 | <0.00001 / 0.13244 / 0.00000 |

KL measures the difference between predicted and true first-choice probabilities. Lower is better. Allowing both learning rules reduced cue-group prediction error relative to fitting every person as a hybrid. At the short horizon, classifying half the cue learners as hybrids and estimating a smaller cue share illustrate the uncertainty that remains with sparse choices. The long check establishes recoverability for these candidate sets, parameters, histories and prevalence. Extrapolation to other strategies, prevalence levels or data from people requires additional evidence. The fixture tests classical recovery and the behavioural probe. The RNN comparison belongs to the subsequent study.

Run `python scripts/check_rare_mechanism.py` to print the complete check. To regenerate the saved record, add `--output review/rare-mechanism-check.json`. `python -m pytest tests/test_rare_mechanism.py -q` checks all four state/reward cues, the hand calculation, truth and reference replay, causal encoding, equal family weight with uninformative data, and exclusion of held-out final choices and privileged fields. The record includes source hashes, support, observed-choice likelihood and subgroup scores for the mixture and both controls. [Verification record](../review/rare-mechanism-check.json)
