"""A one-trial reward cue and a two-family conditional prediction baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from distilled_rnns import simulation as task
from foundation.classical_baselines import (DEFAULT_GRID, candidate_probabilities,
    empirical_bayes_weights, training_log_likelihood)


CUE_GRID = (.60, .75, .90)


def cue_probabilities(history, reliabilities=CUE_GRID):
    """P(first action=1) before each trial, [people, hypotheses, trials]."""
    state, reward = (np.asarray(history[k]) for k in ('state', 'reward'))
    if (state.ndim != 2 or min(state.shape) < 1 or state.shape != reward.shape
            or not np.isin(state, (0, 1)).all() or not np.isin(reward, (0, 1)).all()):
        raise ValueError("state and reward must be matching nonempty binary histories")
    rho = np.asarray(reliabilities, dtype=np.float64)
    if (rho.ndim != 1 or len(rho) < 1 or not np.isfinite(rho).all()
            or ((rho <= .5) | (rho >= 1)).any() or len(np.unique(rho)) != len(rho)):
        raise ValueError("cue reliabilities must be distinct values strictly between .5 and 1")
    out = np.full((len(state), len(rho), state.shape[1]), .5)
    # A rewarded state 1 and an unrewarded state 0 both cue action 1.
    cues_one = (state[:, :-1] == reward[:, :-1])
    out[:, :, 1:] = np.where(cues_one[:, None, :], rho[None, :, None], 1-rho[None, :, None])
    return out


def simulate_fixture(n_trials, seed):
    """Generate 20 hybrid and four cue people in a common synthetic task.

    This fixed recovery fixture uses balanced coverage of familiar grid points
    and one small mixed population. All people use the same second-stage rule.
    The rare group differs only in the mechanism selecting first-stage actions.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    rng = np.random.default_rng(seed)
    n = 24
    order = rng.permutation(n)
    rare = np.zeros(n, dtype=bool)
    rare[order[-4:]] = True
    # Hybrid reference for every person uses their actual stage-two alpha,beta.
    # Cue people have stage-two parameters but no first-stage hybrid weight.
    grid = np.asarray(DEFAULT_GRID)
    idx = np.empty(n, dtype=int)
    idx[order] = np.arange(n) % len(grid)
    w, alpha, beta = grid[idx].T
    reliability = np.full(n, np.nan)
    reliability[order[-4:]] = [.60, .75, .90, .60]
    arrays = {k: np.empty((n, n_trials), dtype=int) for k in ('choice', 'state', 'second_choice', 'reward')}
    truth = np.empty((n, n_trials))
    reference = np.empty_like(truth)
    cue_next = np.full(n, .5)
    q1 = np.full((n, 2), .5)
    q2 = np.full((n, 2, 2), .5)
    rewards = rng.uniform(task.REWARD_MIN, task.REWARD_MAX, q2.shape)
    people = np.arange(n)
    reference_weight = np.full(n, task.REFERENCE_WEIGHT)
    for t in range(n_trials):
        hybrid_probability = task._first_stage_probability(q1, q2, w, beta)
        truth[:, t] = np.where(rare, cue_next, hybrid_probability)
        reference[:, t] = task._first_stage_probability(q1, q2, reference_weight, beta)
        a = (rng.random(n) < truth[:, t]).astype(int)
        state = np.where(rng.random(n) < task.COMMON_PROBABILITY, a, 1-a)
        p2 = task._probability_one(q2[people, state], beta)
        a2 = (rng.random(n) < p2).astype(int)
        reward = (rng.random(n) < rewards[people, state, a2]).astype(int)
        for name, values in zip(('choice','state','second_choice','reward'), (a,state,a2,reward)):
            arrays[name][:, t] = values
        task._update_values(q1, q2, a, state, a2, reward, alpha)
        rewards = task._reflect(rewards + rng.normal(0, task.REWARD_DRIFT_SD, rewards.shape),
                                task.REWARD_MIN, task.REWARD_MAX)
        # Generative state update. The vectorized replay uses a separate formula.
        desired = np.where(reward[rare] == 1, state[rare], 1-state[rare])
        cue_next[rare] = np.where(desired == 1, reliability[rare], 1-reliability[rare])
    x = np.zeros((n, n_trials, 4), dtype=np.float32)
    x[:, 1:] = 2 * np.stack(tuple(arrays.values()), axis=-1)[:, :-1] - 1
    return {**arrays, 'x': x, 'true_prob': truth, 'reference_prob': reference, 'rare': rare,
            'hybrid_grid_index': np.where(rare, -1, idx), 'alpha': alpha.copy(),
            'beta': beta.copy(), 'cue_reliability': reliability}


@dataclass(frozen=True)
class FamilyFit:
    family_prior: np.ndarray
    family_posterior: np.ndarray
    hybrid_posterior: np.ndarray
    cue_posterior: np.ndarray
    probability: np.ndarray
    hybrid_only_probability: np.ndarray
    cue_only_probability: np.ndarray
    iterations: int
    converged: bool


def _family_evidence(ll):
    """Log mean likelihood under a uniform within-family prior and posterior."""
    peak = ll.max(1, keepdims=True)
    scaled = np.exp(ll - peak)
    weights = scaled / scaled.sum(1, keepdims=True)
    log_evidence = peak[:, 0] + np.log(scaled.mean(1))
    return log_evidence, weights


def fit_family_mixture(history, training_mask, hybrid_grid=DEFAULT_GRID, cue_grid=CUE_GRID):
    """Fit two family weights from training choices. Column 0=hybrid, 1=cue.

    Within each family grid points have a fixed uniform prior. Averaging their
    likelihoods gives each family equal starting weight regardless of grid
    size. The shared population family weights use symmetric 0.1 pseudocounts.
    """
    hybrid = candidate_probabilities(history, hybrid_grid)
    cue = cue_probabilities(history, cue_grid)
    hybrid_ll = training_log_likelihood(hybrid, history['choice'], training_mask)
    cue_ll = training_log_likelihood(cue, history['choice'], training_mask)
    hybrid_ev, hybrid_posterior = _family_evidence(hybrid_ll)
    cue_ev, cue_posterior = _family_evidence(cue_ll)
    prior, families, trace, converged = empirical_bayes_weights(np.column_stack((hybrid_ev, cue_ev)))
    hybrid_pred = np.einsum('ng,ngt->nt', hybrid_posterior, hybrid)
    cue_pred = np.einsum('ng,ngt->nt', cue_posterior, cue)
    pred = families[:, :1] * hybrid_pred + families[:, 1:] * cue_pred
    return FamilyFit(prior, families, hybrid_posterior, cue_posterior, pred, hybrid_pred, cue_pred,
                     len(trace)-1, converged)
