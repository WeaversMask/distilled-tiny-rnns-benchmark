import numpy as np
import pytest

from foundation.strategy_probes import rollout_stay_signature, two_trial_probe_bank


def test_reward_as_cue_has_analytic_rollout_signature():
    histories, inputs = two_trial_probe_bank()
    np.testing.assert_array_equal(inputs[:, 0], 0)
    np.testing.assert_array_equal(inputs[:, 1:], 2 * histories - 1)
    # Independently construct stage-one probabilities under a stochastic cue.
    target_state = np.where(histories[:, :, 3], histories[:, :, 1], 1 - histories[:, :, 1])
    p = np.column_stack([np.full(256, .5), np.where(target_state == 1, .9, .1)])
    np.testing.assert_allclose(rollout_stay_signature(p), [.9, .1, .1, .9], atol=1e-12)


def test_rollout_weights_the_first_actions_by_the_fitted_policy():
    # Constant P(action one)=.8 -> P(same action twice)=.8²+.2².
    np.testing.assert_allclose(rollout_stay_signature(np.full((256, 3), .8)), .68)
    p = np.full((256, 3), .8)
    p[0, 1] = .1
    with pytest.raises(ValueError, match="shared history"):
        rollout_stay_signature(p)
