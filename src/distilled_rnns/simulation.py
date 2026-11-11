"""Causal, synthetic two-step histories for a first-stage prediction pilot.

This is a specified synthetic task, not a reproduction of the Gillan dataset.
There are two first-stage actions, two second-stage states, and two actions in
each second-stage state. Action ``a`` reaches state ``a`` with probability 0.7.
Each subject has four reward probabilities, initialized uniformly on [0.25,
0.75] and independently perturbed after each trial by N(0, 0.025**2), with
reflection at the bounds.

Both value tables start at 0.5. First-stage values combine model-free values
Q1 with model-based values ``transition_matrix @ max(Q2, axis=action)`` using
weight w. Choices at both stages use a softmax with inverse temperature beta.
After a trial, using the OLD values, delta1 = Q2[s,a2] - Q1[a1] and
delta2 = reward - Q2[s,a2]. The chosen values update as
``Q1[a1] += alpha * (delta1 + 0.5 * delta2)`` and
``Q2[s,a2] += alpha * delta2``. The eligibility trace is fixed at 0.5.

The rare group uses w=0; other subjects use w~Uniform(0.5,0.9). All subjects
use alpha~Uniform(0.15,0.65) and beta~Uniform(2,6). The reference policy has
w=0.75 with the same learning parameters and the same observed history; it
is a fixed majority-like comparator, not the population's average policy.

``true_prob`` and ``reference_prob`` are P(first action=1), evaluated before
the current action is sampled. The input x at trial t contains ONLY the
previous trial's (first action, state, second action, reward), encoded as
-1/+1; the first input is all zeros. No current outcome enters its prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np


COMMON_PROBABILITY = 0.7
REWARD_MIN = 0.25
REWARD_MAX = 0.75
REWARD_DRIFT_SD = 0.025
ELIGIBILITY_TRACE = 0.5
REFERENCE_WEIGHT = 0.75
_TRANSITIONS = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64)


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _reflect(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Reflect even arbitrarily large excursions, without clipping mass at edges."""
    width = high - low
    folded = np.remainder(values - low, 2.0 * width)
    return low + np.minimum(folded, 2.0 * width - folded)


def _probability_one(values: np.ndarray, beta: np.ndarray) -> np.ndarray:
    difference = np.clip(beta * (values[:, 1] - values[:, 0]), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-difference))


def _first_stage_probability(
    q1: np.ndarray, q2: np.ndarray, weight: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    model_based = np.max(q2, axis=2) @ _TRANSITIONS.T
    combined = (1.0 - weight[:, None]) * q1 + weight[:, None] * model_based
    return _probability_one(combined, beta)


def _update_values(
    q1: np.ndarray,
    q2: np.ndarray,
    choice: np.ndarray,
    state: np.ndarray,
    second_choice: np.ndarray,
    reward: np.ndarray,
    alpha: np.ndarray,
) -> None:
    rows = np.arange(len(q1))
    old_second_value = q2[rows, state, second_choice].copy()
    delta1 = old_second_value - q1[rows, choice]
    delta2 = reward - old_second_value
    q1[rows, choice] += alpha * (delta1 + ELIGIBILITY_TRACE * delta2)
    q2[rows, state, second_choice] += alpha * delta2


def simulate_population(
    n_subjects: int, n_trials: int, rare_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    """Generate independent synthetic subjects with exactly rounded group sizes.

    Rare count is rounded to the nearest integer (half upward), then subjects
    are randomly assigned to groups. At least one subject in each group is
    required. ``seed`` controls all draws without touching global RNG state.
    """
    n_subjects = _integer("n_subjects", n_subjects, 2)
    n_trials = _integer("n_trials", n_trials, 1)
    seed = _integer("seed", seed, 0)
    if (
        isinstance(rare_fraction, (bool, np.bool_))
        or not isinstance(rare_fraction, numbers.Real)
        or not np.isfinite(rare_fraction)
        or not 0.0 < rare_fraction < 1.0
    ):
        raise ValueError("rare_fraction must be finite and strictly between 0 and 1")
    rare_count = int(np.floor(n_subjects * rare_fraction + 0.5))
    if not 0 < rare_count < n_subjects:
        raise ValueError("rounded rare count must leave at least one subject in each group")

    rng = np.random.default_rng(seed)
    rare = np.zeros(n_subjects, dtype=bool)
    rare[rng.permutation(n_subjects)[:rare_count]] = True
    weights = rng.uniform(0.5, 0.9, n_subjects)
    weights[rare] = 0.0
    alpha = rng.uniform(0.15, 0.65, n_subjects)
    beta = rng.uniform(2.0, 6.0, n_subjects)
    reference_weights = np.full(n_subjects, REFERENCE_WEIGHT)

    shape = (n_subjects, n_trials)
    choice = np.empty(shape, dtype=np.int64)
    state = np.empty(shape, dtype=np.int64)
    second_choice = np.empty(shape, dtype=np.int64)
    reward = np.empty(shape, dtype=np.int64)
    true_prob = np.empty(shape, dtype=np.float64)
    reference_prob = np.empty(shape, dtype=np.float64)
    q1 = np.full((n_subjects, 2), 0.5, dtype=np.float64)
    q2 = np.full((n_subjects, 2, 2), 0.5, dtype=np.float64)
    reward_probabilities = rng.uniform(REWARD_MIN, REWARD_MAX, (n_subjects, 2, 2))
    rows = np.arange(n_subjects)

    for trial in range(n_trials):
        true_prob[:, trial] = _first_stage_probability(q1, q2, weights, beta)
        reference_prob[:, trial] = _first_stage_probability(q1, q2, reference_weights, beta)
        choice[:, trial] = rng.random(n_subjects) < true_prob[:, trial]
        common = rng.random(n_subjects) < COMMON_PROBABILITY
        state[:, trial] = np.where(common, choice[:, trial], 1 - choice[:, trial])
        second_prob = _probability_one(q2[rows, state[:, trial]], beta)
        second_choice[:, trial] = rng.random(n_subjects) < second_prob
        selected_reward_probability = reward_probabilities[
            rows, state[:, trial], second_choice[:, trial]
        ]
        reward[:, trial] = rng.random(n_subjects) < selected_reward_probability
        _update_values(
            q1, q2, choice[:, trial], state[:, trial], second_choice[:, trial],
            reward[:, trial], alpha,
        )
        reward_probabilities = _reflect(
            reward_probabilities + rng.normal(0.0, REWARD_DRIFT_SD, reward_probabilities.shape),
            REWARD_MIN,
            REWARD_MAX,
        )

    x = np.zeros((n_subjects, n_trials, 4), dtype=np.float32)
    history = np.stack((choice, state, second_choice, reward), axis=-1)
    x[:, 1:] = 2.0 * history[:, :-1].astype(np.float32) - 1.0
    return {
        "x": x,
        "choice": choice,
        "state": state,
        "second_choice": second_choice,
        "reward": reward,
        "true_prob": true_prob,
        "reference_prob": reference_prob,
        "rare": rare,
        "model_based_weight": weights,
        "alpha": alpha,
        "beta": beta,
    }


def replay_policy(
    history: Mapping[str, np.ndarray],
    model_based_weight: np.ndarray | float,
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray:
    """Evaluate a policy on observed histories, without resampling any actions.

    The four history arrays may have shape [subjects, trials] or [trials] for
    a single subject. Parameters may be scalars or one value per subject.
    Returned probabilities have the same shape as the history arrays.
    """
    histories = [np.asarray(history[key]) for key in ("choice", "state", "second_choice", "reward")]
    shape = histories[0].shape
    if len(shape) not in (1, 2) or any(array.shape != shape for array in histories):
        raise ValueError("history arrays must share shape [trials] or [subjects, trials]")
    if any(dimension < 1 for dimension in shape):
        raise ValueError("history must contain at least one subject and trial")
    if any(not np.isin(array, (0, 1)).all() for array in histories):
        raise ValueError("history values must be binary 0 or 1")
    single_subject = len(shape) == 1
    histories = [array.reshape(1, -1) if single_subject else array for array in histories]
    histories = [array.astype(np.int64) for array in histories]
    n_subjects, n_trials = histories[0].shape

    parameters = []
    for name, value in (("model_based_weight", model_based_weight), ("alpha", alpha), ("beta", beta)):
        try:
            parameter = np.broadcast_to(np.asarray(value, dtype=np.float64), (n_subjects,))
        except ValueError as error:
            raise ValueError(f"{name} must be scalar or have one value per subject") from error
        if not np.isfinite(parameter).all():
            raise ValueError(f"{name} must be finite")
        parameters.append(parameter)
    weights, learning_rates, inverse_temperatures = parameters
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("model_based_weight must lie in [0, 1]")
    if np.any((learning_rates <= 0) | (learning_rates > 1)):
        raise ValueError("alpha must lie in (0, 1]")
    if np.any(inverse_temperatures <= 0):
        raise ValueError("beta must be positive")

    q1 = np.full((n_subjects, 2), 0.5)
    q2 = np.full((n_subjects, 2, 2), 0.5)
    probabilities = np.empty((n_subjects, n_trials), dtype=np.float64)
    for trial in range(n_trials):
        probabilities[:, trial] = _first_stage_probability(q1, q2, weights, inverse_temperatures)
        _update_values(q1, q2, *(array[:, trial] for array in histories), learning_rates)
    return probabilities[0] if single_subject else probabilities
