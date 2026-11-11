"""Bounded generating-family baselines; no changes to frozen pilot modules.

All likelihoods score first-stage training choices only. Observed earlier
held-out trials remain causal history, just as in the pilot RNN comparison.
The hierarchical model is an empirical-Bayes distribution over a finite grid,
not a continuous-parameter fit or a mixture of distinct mechanism classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from distilled_rnns.simulation import replay_policy


# A fixed small engineering/recovery grid, not a frozen main-study baseline.
DEFAULT_GRID = tuple(product((0.0, 0.4, 0.8), (0.2, 0.6), (2.0, 6.0)))
HISTORY_KEYS = ("choice", "state", "second_choice", "reward")


@dataclass(frozen=True)
class GridFit:
    grid: np.ndarray
    training_log_likelihood: np.ndarray
    individual_index: np.ndarray
    pooled_index: int
    prior_weights: np.ndarray
    posterior_weights: np.ndarray
    individual_probability: np.ndarray
    pooled_probability: np.ndarray
    hierarchical_probability: np.ndarray
    marginal_objective: tuple[float, ...]
    converged: bool


def _grid_array(grid):
    value = np.asarray(grid, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 1 or not np.isfinite(value).all():
        raise ValueError("grid must be finite [candidates, 3] in (weight, alpha, beta) order")
    if ((value[:, 0] < 0) | (value[:, 0] > 1) | (value[:, 1] <= 0)
            | (value[:, 1] > 1) | (value[:, 2] <= 0)).any():
        raise ValueError("invalid generating-family parameters")
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError("duplicate grid points make component weights nonidentifiable")
    return value.copy()


def candidate_probabilities(history, grid=DEFAULT_GRID):
    """Return [people, candidates, trials] using the unchanged pilot replay.

    Only the four observable history arrays are read. Simulator probabilities,
    group labels and generating parameters are neither accepted as fit targets
    nor used to initialize the fits.
    """
    points = _grid_array(grid)
    arrays = {name: np.asarray(history[name]) for name in HISTORY_KEYS}
    shape = arrays["choice"].shape
    if len(shape) != 2 or min(shape) < 1 or any(a.shape != shape for a in arrays.values()):
        raise ValueError("history arrays must share nonempty [people, trials] shape")
    if any(not np.isin(a, (0, 1)).all() for a in arrays.values()):
        raise ValueError("history observations must be binary")
    n, t = shape
    expanded = {name: np.repeat(a, len(points), axis=0) for name, a in arrays.items()}
    parameters = np.tile(points, (n, 1))
    return replay_policy(expanded, *parameters.T).reshape(n, len(points), t)


def training_log_likelihood(probabilities, choices, training_mask):
    """Score only masked choices; candidate histories are supplied separately."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(choices)
    mask = np.asarray(training_mask)
    if (p.ndim != 3 or min(p.shape) < 1 or y.shape != (p.shape[0], p.shape[2])
            or mask.shape != y.shape or mask.dtype != np.bool_):
        raise ValueError("expected [people, candidates, trials], matching choices and boolean mask")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y, (0, 1)).all():
        raise ValueError("invalid probabilities or choices")
    if not mask.any(1).all():
        raise ValueError("each person needs at least one training choice")
    q = np.clip(p, 1e-12, 1 - 1e-12)
    logp = np.where(y[:, None, :] == 1, np.log(q), np.log1p(-q))
    return (logp * mask[:, None, :]).sum(axis=2)


def _posterior(log_likelihood, prior):
    scores = log_likelihood + np.log(prior)[None, :]
    maximum = scores.max(axis=1, keepdims=True)
    exp = np.exp(scores - maximum)
    denominator = exp.sum(axis=1, keepdims=True)
    return exp / denominator, (maximum + np.log(denominator)).sum()


def empirical_bayes_weights(log_likelihood, *, pseudocount=0.1, max_iterations=2000, tolerance=1e-10):
    """EM for a shared finite mixing distribution and individual posteriors.

    The objective is sum_i log(sum_g pi_g L_ig) + c sum_g log(pi_g).
    c>0 corresponds to a symmetric Dirichlet(1+c) MAP prior over pi and
    prevents unsupported components receiving exactly zero mass. All people
    enter symmetrically. There is no rare-group input or held-out tuning.
    """
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.ndim != 2 or min(ll.shape) < 1 or not np.isfinite(ll).all():
        raise ValueError("log likelihood must be finite [people, candidates]")
    if not np.isfinite(pseudocount) or pseudocount <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("pseudocount and tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    n, g = ll.shape
    prior = np.full(g, 1 / g)
    responsibilities, marginal = _posterior(ll, prior)
    trace = [float(marginal + pseudocount * np.log(prior).sum())]
    converged = False
    for _ in range(max_iterations):
        prior = (responsibilities.sum(0) + pseudocount) / (n + g * pseudocount)
        responsibilities, marginal = _posterior(ll, prior)
        objective = float(marginal + pseudocount * np.log(prior).sum())
        improvement = objective - trace[-1]
        if improvement < -1e-8 * (1 + abs(trace[-1])):
            raise RuntimeError("empirical-Bayes objective decreased")
        trace.append(objective)
        if abs(improvement) <= tolerance * (1 + abs(trace[-2])):
            converged = True
            break
    return prior, responsibilities, tuple(trace), converged


def fit_baselines(history, training_mask, grid=DEFAULT_GRID, *, pseudocount=0.1,
                  max_iterations=2000, tolerance=1e-10):
    """Fit individual ML, a single pooled point, and a shared finite prior.

    Exact likelihood ties choose the first predeclared grid point. Posterior
    predictive probabilities average candidate probabilities, not parameters.
    Validation/test choices are never scored by this fitting objective.
    """
    points = _grid_array(grid)
    p = candidate_probabilities(history, points)
    ll = training_log_likelihood(p, history["choice"], training_mask)
    individual = ll.argmax(axis=1)
    pooled = int(ll.sum(0).argmax())
    prior, posterior, trace, converged = empirical_bayes_weights(
        ll, pseudocount=pseudocount, max_iterations=max_iterations, tolerance=tolerance)
    return GridFit(points, ll, individual, pooled, prior, posterior,
                   p[np.arange(len(p)), individual], p[:, pooled],
                   np.einsum("ng,ngt->nt", posterior, p), trace, converged)
