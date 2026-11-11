# Classical and hierarchical baseline recovery

With 6,000 observed choices per person, both the individual fit and the hierarchical fit recovered the generating parameter triple for all 24 simulated people. With 60 choices they recovered eight and seven respectively. These counts describe a controlled recovery exercise with twelve possible parameter triples. They show how a short choice history can leave the underlying strategy ambiguous even when the correct model is among the candidates.

## What was fitted

Each candidate replays a person's observed history under the two-step reinforcement-learning model used in the synthetic pilot. The three free parameters are the weight placed on model-based values, the learning rate and the inverse choice temperature. This exercise uses all 12 combinations of weights `(0, 0.4, 0.8)`, learning rates `(0.2, 0.6)` and inverse temperatures `(2, 6)`. Each fit scores the same first-stage training choices. Earlier actions, states and rewards remain visible as causal history on every split. The current trial is predicted before its outcome updates either value table.

| Fit | How people share information | Prediction |
| --- | --- | --- |
| Individual | Each person selects the parameter triple that maximizes their training likelihood. | That person's selected policy. |
| Pooled | Everybody shares the single triple that maximizes the sum of training log likelihoods. | The common policy evaluated on each person's own history. |
| Hierarchical | The population shares an estimated distribution over triples. Each person's training likelihood updates that distribution into their own posterior. | The weighted mean of candidate choice probabilities under that person's posterior. |

The hierarchical fit estimates population weights from the marginal likelihood with a symmetric Dirichlet penalty. Each component receives a fixed pseudocount of 0.1, equivalent to a Dirichlet concentration of 1.1 per component. The EM updates start from equal weights and stop when the change in penalized log likelihood is at most `1e-10 × (1 + |previous objective|)`, with a limit of 2,000 iterations. The saved short and long fits stopped after 96 and 2 iterations. All three fits use observable histories and training masks. Simulator probabilities and parameters enter evaluation only.

## Fixed synthetic check

An independently written generator sampled actions and drifting rewards for two people at each of the twelve grid points. The short condition takes the first 100 trials of each 10,000-trial history. At each length, the pilot's mask generator assigns 60% of trials to training, 20% to validation and 20% to testing. The validation portion is reserved. Both conditions use the same fixed parameter grid and fitting settings. The generator seed is 99304 and the split seed is 99305. The two lengths are paired prefixes with separately generated masks.

On each history, the simulator's true first-choice probabilities matched the generating triple's replay to within `5.56e-16`. On the long histories, every alternative grid point had positive average true-policy KL on the fitting trials. The smallest such difference was 0.001117 nats per trial. This checks distinguishability on these histories within this finite candidate set.

| Quantity | 100 trials per person | 10,000 trials per person |
| --- | ---: | ---: |
| Training choices per person | 60 | 6,000 |
| Independent fits selecting the true triple | 8 / 24 | 24 / 24 |
| Hierarchical fits with the true triple most probable | 7 / 24 | 24 / 24 |
| Average posterior probability on the true triple | 0.252 | 0.997 |
| Test KL, individual / pooled / hierarchical | 0.00956 / 0.02853 / 0.00781 | <0.00001 / 0.03113 / <0.00001 |
| Test observed-choice NLL, individual / pooled / hierarchical | 0.63832 / 0.65509 / 0.63931 | 0.62391 / 0.65621 / 0.62392 |

Lower values are better for both scores, measured in nats per test choice. With short histories, averaging over uncertainty improved fidelity to the known simulator relative to the individual point fit. The individual point fit predicted the sampled test choices slightly better. A single population parameter triple had the largest average errors at both lengths. All counts and scores describe this fixed synthetic sample.

The finite grid approximates the continuous parameters used in the pilot. The controlled recovery case deliberately includes the truth on the grid and uses equal representation, so it tests implementation and information limits under favourable conditions. Quantifying error for off-grid people or a rare subgroup requires separate populations. These hierarchical components all share one learning rule. A population containing genuinely different mechanisms requires candidate policies for those mechanisms before a between-mechanism mixture comparison can be made. Prediction of a new person's choices with no fitting trials is outside this check.

Run `python scripts/check_classical_baselines.py` from a project checkout to print the record. To regenerate the saved JSON, add `--output review/classical-baselines-check.json`. Run `python -m pytest tests/test_classical_baselines.py -q` for the causal timing, mask, likelihood, hierarchy and input-preservation checks. Source hashes, numerical settings and the per-person recovery information accompany the run. [Verification record](../review/classical-baselines-check.json)
