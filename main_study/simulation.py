"""Discrete generating populations for the frozen synthetic main study."""
from __future__ import annotations

import numpy as np

from distilled_rnns import simulation as task


HISTORY_KEYS = ("choice", "state", "second_choice", "reward")
KINDS = ("parametric_model_free", "reward_cue")


def simulate_population(config, kind, seed):
    """Sample independent on-grid people and their causal two-step histories.

    All population-parameter draws use a local generator. Cue people share the
    stage-two learning rule and task with hybrid people. Their first-stage rule
    uses the immediately preceding state and reward, instead of learned values.
    Unused parameters are finite: cue reliability zero denotes a hybrid person,
    and model-based weight zero on cue people is ignored by their choice rule.
    """
    if kind not in KINDS:
        raise ValueError("Unknown rare-agent kind")
    n, trials = config.n_subjects, config.n_trials
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n < 2
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials < 3
    ):
        raise ValueError("Population requires at least two people and three trials")
    fraction = config.rare_fraction
    if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Rare fraction must lie strictly between zero and one")
    rare_count = int(np.floor(n * fraction + 0.5))
    if not 0 < rare_count < n:
        raise ValueError("Rounded rare count must leave both groups represented")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    design = config.design["population"]
    rng = np.random.default_rng(seed)
    rare = np.zeros(n, dtype=bool)
    rare[rng.permutation(n)[:rare_count]] = True
    weights = rng.choice(design["typical_weight"], n).astype(np.float64)
    weights[rare] = rng.choice(design["rare_parametric_weight"], rare_count)
    alpha = rng.choice(design["alpha"], n).astype(np.float64)
    beta = rng.choice(design["beta"], n).astype(np.float64)
    is_cue = rare.copy() if kind == "reward_cue" else np.zeros(n, dtype=bool)
    reliability = np.zeros(n, dtype=np.float64)
    reliability[is_cue] = rng.choice(design["cue_reliability"], int(is_cue.sum()))
    weights[is_cue] = 0

    shape = (n, trials)
    history = {name: np.empty(shape, dtype=np.int64) for name in HISTORY_KEYS}
    truth, reference = np.empty(shape), np.empty(shape)
    first_values = np.full((n, 2), 0.5)
    second_values = np.full((n, 2, 2), 0.5)
    reward_probabilities = rng.uniform(task.REWARD_MIN, task.REWARD_MAX, (n, 2, 2))
    cue_next = np.full(n, 0.5)
    people = np.arange(n)
    reference_weights = np.full(n, task.REFERENCE_WEIGHT)
    for trial in range(trials):
        hybrid = task._first_stage_probability(first_values, second_values, weights, beta)
        truth[:, trial] = np.where(is_cue, cue_next, hybrid)
        reference[:, trial] = task._first_stage_probability(
            first_values, second_values, reference_weights, beta
        )
        first_choice = (rng.random(n) < truth[:, trial]).astype(np.int64)
        state = np.where(rng.random(n) < task.COMMON_PROBABILITY, first_choice, 1 - first_choice)
        second_probability = task._probability_one(second_values[people, state], beta)
        second_choice = (rng.random(n) < second_probability).astype(np.int64)
        reward = (rng.random(n) < reward_probabilities[people, state, second_choice]).astype(np.int64)
        for name, value in zip(HISTORY_KEYS, (first_choice, state, second_choice, reward)):
            history[name][:, trial] = value
        task._update_values(first_values, second_values, first_choice, state, second_choice, reward, alpha)
        reward_probabilities = task._reflect(
            reward_probabilities + rng.normal(0, task.REWARD_DRIFT_SD, reward_probabilities.shape),
            task.REWARD_MIN,
            task.REWARD_MAX,
        )
        target_state = np.where(reward == 1, state, 1 - state)
        cue_next = np.where(target_state == 1, reliability, 1 - reliability)

    inputs = np.zeros((n, trials, 4), dtype=np.float32)
    inputs[:, 1:] = 2 * np.stack([history[name] for name in HISTORY_KEYS], axis=-1)[:, :-1] - 1
    return {
        **history,
        "x": inputs,
        "true_prob": truth,
        "reference_prob": reference,
        "rare": rare,
        "is_cue": is_cue,
        "model_based_weight": weights,
        "alpha": alpha,
        "beta": beta,
        "cue_reliability": reliability,
    }
