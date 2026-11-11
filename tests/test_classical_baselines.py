"""Scientific properties of the separate finite-grid baseline foundation."""
import numpy as np
import pytest

from foundation.classical_baselines import (candidate_probabilities,
    empirical_bayes_weights, fit_baselines, training_log_likelihood)
from distilled_rnns.simulation import simulate_population


def test_replay_matches_independently_generated_truth_and_is_causal():
    data = simulate_population(6, 31, 0.5, 99301)
    grid = np.column_stack([data[k] for k in ('model_based_weight', 'alpha', 'beta')])
    probabilities = candidate_probabilities(data, grid)
    np.testing.assert_allclose(probabilities[np.arange(6), np.arange(6)], data['true_prob'], rtol=0, atol=1e-12)
    altered = {k: np.array(v, copy=True) for k, v in data.items()}
    for key in ('choice', 'state', 'second_choice', 'reward'):
        altered[key][:, 17:] = 1 - altered[key][:, 17:]
    # The choice at t=17 is predicted before t=17's history update.
    np.testing.assert_array_equal(candidate_probabilities(altered, grid)[:, :, :18], probabilities[:, :, :18])


def test_heldout_targets_do_not_enter_likelihood():
    p = np.array([[[.1, .8, .2], [.9, .3, .6]]])
    mask = np.array([[True, False, True]])
    first = training_log_likelihood(p, np.array([[0, 0, 1]]), mask)
    second = training_log_likelihood(p, np.array([[0, 1, 1]]), mask)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [[np.log(.9)+np.log(.2), np.log(.1)+np.log(.6)]])


def test_hierarchy_matches_closed_form_separated_components_and_is_permutation_equivariant():
    # With overwhelming evidence the pseudo-count solution is analytic.
    ll = np.array([[0., -1000.], [0., -1000.], [-1000., 0.]])
    prior, posterior, trace, converged = empirical_bayes_weights(ll, pseudocount=.5)
    np.testing.assert_allclose(prior, [2.5/4, 1.5/4], atol=1e-12)
    np.testing.assert_array_equal(posterior, [[1.,0.],[1.,0.],[0.,1.]])
    assert converged and np.diff(trace).min() >= -1e-10
    p2, r2, _, _ = empirical_bayes_weights(ll[::-1, ::-1], pseudocount=.5)
    np.testing.assert_allclose(p2, prior[::-1])
    np.testing.assert_allclose(r2, posterior[::-1, ::-1])


def test_no_information_preserves_uniform_prior():
    prior, posterior, trace, converged = empirical_bayes_weights(np.zeros((4, 3)))
    np.testing.assert_allclose(prior, np.ones(3)/3)
    np.testing.assert_allclose(posterior, np.ones((4,3))/3)
    assert converged


def test_fit_ignores_privileged_simulator_fields_and_preserves_inputs():
    data = simulate_population(4, 20, .5, 99302)
    mask = np.zeros((4,20), dtype=bool); mask[:, :12] = True
    before = {k: v.copy() for k,v in data.items()}
    fit = fit_baselines(data, mask)
    for k in data: np.testing.assert_array_equal(data[k], before[k])
    altered = dict(data)
    for k in ('true_prob', 'rare', 'model_based_weight', 'alpha', 'beta'):
        altered[k] = None
    refit = fit_baselines(altered, mask)
    np.testing.assert_array_equal(fit.training_log_likelihood, refit.training_log_likelihood)
    np.testing.assert_array_equal(fit.hierarchical_probability, refit.hierarchical_probability)
    assert fit.converged and np.isfinite(fit.hierarchical_probability).all()
    np.testing.assert_allclose(fit.posterior_weights.sum(1), 1)
    assert (fit.hierarchical_probability >= 0).all() and (fit.hierarchical_probability <= 1).all()


def test_impossible_masks_and_nonidentifiable_duplicate_grid_are_rejected():
    with pytest.raises(ValueError):
        training_log_likelihood(np.full((2,3,4), .5), np.zeros((2,4)), np.zeros((2,4), dtype=bool))
    data = simulate_population(2, 3, .5, 99303)
    with pytest.raises(ValueError): candidate_probabilities(data, [[0,.2,2],[0,.2,2]])
    with pytest.raises(ValueError): empirical_bayes_weights([[0, np.nan]])
    with pytest.raises(ValueError): empirical_bayes_weights([[0, 1]], pseudocount=0)
