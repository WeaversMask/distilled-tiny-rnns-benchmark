"""Exact three-choice behavioral assay used by the prospective synthetic study."""

from itertools import product

import numpy as np


def two_trial_probe_bank():
    """Every two-trial history in (first action, state, second action, reward) order."""
    histories = np.array(list(product((0, 1), repeat=8)), dtype=np.int64).reshape(256, 2, 4)
    inputs = np.zeros((256, 3, 4), dtype=np.float32)
    inputs[:, 1:] = 2 * histories - 1
    return histories, inputs


SIGNATURE_ORDER = ("unrewarded_rare", "unrewarded_common", "rewarded_rare", "rewarded_common")


def rollout_stay_signature(probability_one):
    """Four probabilities of repeating the second first-stage choice on trial three.

    The policy supplies all three first-stage probabilities for each history in
    the fixed bank. Integrate over its own earlier choices, 70/30 transitions,
    fair second-stage actions and fair rewards. No observed subject is replayed.
    """
    probabilities = np.asarray(probability_one, dtype=np.float64)
    if probabilities.shape != (256, 3) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite P(first action = 1) at all 256 histories and 3 steps")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Choice probabilities must lie in [0, 1]")
    if np.ptp(probabilities[:, 0]) > 5e-6 or np.ptp(probabilities[:, 1].reshape(16, 16), axis=1).max() > 5e-6:
        raise ValueError("Predictions depend on events later than their shared history")

    histories, _ = two_trial_probe_bank()
    choices = histories[:, :, 0]
    common = histories[:, :, 1] == choices
    p_chosen = np.where(choices == 1, probabilities[:, :2], 1 - probabilities[:, :2])
    history_mass = np.prod(p_chosen * np.where(common, .7, .3) * .5 * .5, axis=1)
    if abs(history_mass.sum() - 1.0) > 5e-6:
        raise ValueError("Generated trajectory weights do not sum to one")
    p_stay = np.where(choices[:, -1] == 1, probabilities[:, 2], 1 - probabilities[:, 2])
    signature = []
    for reward, was_common in product((0, 1), repeat=2):
        cell = (histories[:, -1, 3] == reward) & (common[:, -1] == was_common)
        signature.append(np.average(p_stay[cell], weights=history_mass[cell]))
    return np.array(signature)
