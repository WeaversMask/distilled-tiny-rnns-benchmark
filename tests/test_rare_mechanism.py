import numpy as np
import pytest

from foundation.classical_baselines import DEFAULT_GRID, candidate_probabilities
from foundation.reward_cue import CUE_GRID, cue_probabilities, fit_family_mixture, simulate_fixture
from distilled_rnns.simulation import replay_policy


def test_all_four_cues_and_one_trial_memory():
    state = np.array([[0, 0, 1, 1, 1]])
    reward = np.array([[0, 1, 0, 1, 0]])
    p = cue_probabilities({'state':state,'reward':reward}, [.75])[0,0]
    np.testing.assert_array_equal(p, [.5,.75,.25,.25,.75])
    # Distinct older histories with the same most recent cue predict alike.
    alt = {'state':1-state.copy(), 'reward':1-reward.copy()}
    alt['state'][:, -2:] = state[:, -2:]
    alt['reward'][:, -2:] = reward[:, -2:]
    np.testing.assert_array_equal(cue_probabilities(alt,[.75])[0,0,-1], p[-1])
    with pytest.raises(ValueError): cue_probabilities({'state':state,'reward':reward}, [.5])


def test_hand_derived_uncommon_transition_reward_probe():
    h = {'choice':np.zeros((2,2),int),'state':np.array([[1,0],[1,0]]),
         'second_choice':np.zeros((2,2),int),'reward':np.array([[1,0],[0,0]])}
    np.testing.assert_array_equal(cue_probabilities(h,[.75])[:,0,1], [.75,.25])
    expected = 1/(1+np.exp(-np.array([.14,.10])))
    np.testing.assert_allclose(replay_policy(h,.75,.4,4)[:,1], expected, rtol=0, atol=1e-14)


def test_simulated_truth_and_causal_histories_match_replay():
    d = simulate_fixture(75, 99306)
    hybrid=candidate_probabilities(d); cue=cue_probabilities(d)
    np.testing.assert_array_equal(d['x'][:,0], 0)
    history=np.stack([d[k] for k in ('choice','state','second_choice','reward')],axis=-1)
    np.testing.assert_array_equal(d['x'][:,1:], 2*history[:,:-1]-1)
    for i in range(len(d['rare'])):
        predicted = (cue[i, CUE_GRID.index(d['cue_reliability'][i])] if d['rare'][i]
                     else hybrid[i,d['hybrid_grid_index'][i]])
        np.testing.assert_allclose(predicted, d['true_prob'][i],rtol=0,atol=1e-12)
    np.testing.assert_allclose(replay_policy(d,.75,d['alpha'],d['beta']), d['reference_prob'],rtol=0,atol=1e-12)


def test_unequal_family_sizes_have_equal_prior_without_evidence():
    # At the initial trial all hypotheses give probability 0.5, regardless of family size.
    d = simulate_fixture(1,99307)
    mask = np.ones_like(d['choice'],dtype=bool)
    f = fit_family_mixture(d,mask,DEFAULT_GRID,CUE_GRID)
    np.testing.assert_allclose(f.family_prior,[.5,.5])
    np.testing.assert_allclose(f.family_posterior,.5)
    np.testing.assert_allclose(f.probability,.5)


def test_unscored_final_actions_and_privileged_fields_do_not_change_fit():
    d = simulate_fixture(50,99308)
    mask=np.ones_like(d['choice'],dtype=bool);mask[:,-1]=False
    fit=fit_family_mixture(d,mask)
    changed={k:d[k].copy() for k in ('choice','state','second_choice','reward')}
    changed['choice'][:,-1]=1-changed['choice'][:,-1]
    other=fit_family_mixture(changed,mask)
    np.testing.assert_array_equal(fit.probability,other.probability)
    np.testing.assert_array_equal(fit.family_prior,other.family_prior)
