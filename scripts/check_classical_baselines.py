"""Reproduce the fixed finite-grid recovery experiment without reading pilot outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from distilled_rnns.evaluation import binary_kl, binary_nll, make_trial_masks
from foundation.classical_baselines import DEFAULT_GRID, candidate_probabilities, fit_baselines


GENERATOR_SEED = 99304
SPLIT_SEED = 99305
LENGTHS = (100, 10000)
COPIES_PER_POINT = 2


def generate_history(points, trials, seed):
    """Independently generate the two-step task with explicit on-grid parameters.

    The 0.4 factor below is the action-1 versus action-0 difference in
    model-based state values under a 0.7 common transition probability.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    person = np.arange(n)
    weight, alpha, beta = points.T
    first = np.full((n, 2), .5)
    second = np.full((n, 2, 2), .5)
    reward_probability = rng.uniform(.25, .75, (n, 2, 2))
    history = {k: np.empty((n, trials), dtype=np.int64) for k in ("choice", "state", "second_choice", "reward")}
    truth = np.empty((n, trials))
    for t in range(trials):
        state_values = np.max(second, axis=2)
        advantage = ((1 - weight) * (first[:, 1] - first[:, 0])
                     + .4 * weight * (state_values[:, 1] - state_values[:, 0]))
        probability = 1 / (1 + np.exp(-beta * advantage))
        a = (rng.random(n) < probability).astype(int)
        state = np.where(rng.random(n) < .7, a, 1 - a)
        second_advantage = second[person, state, 1] - second[person, state, 0]
        action = (rng.random(n) < 1 / (1 + np.exp(-beta * second_advantage))).astype(int)
        reward = (rng.random(n) < reward_probability[person, state, action]).astype(int)
        truth[:, t] = probability
        for key, values in zip(history, (a, state, action, reward)):
            history[key][:, t] = values
        old = second[person, state, action].copy()
        first[person, a] += alpha * ((old + .5 * (reward - old)) - first[person, a])
        second[person, state, action] += alpha * (reward - old)
        # Triangular-wave reflection between 0.25 and 0.75.
        drifted = reward_probability + rng.normal(0, .025, reward_probability.shape)
        reward_probability = .75 - np.abs((drifted - .25) % 1.0 - .5)
    return history, truth


def run_length(history, truth, true_index, grid, length):
    obs = {name: value[:, :length] for name, value in history.items()}
    truth = truth[:, :length]
    n = len(truth)
    masks = make_trial_masks(n, length, SPLIT_SEED)
    train, test = masks['train'], masks['test']
    fit = fit_baselines(obs, train, grid)
    if not fit.converged:
        raise RuntimeError("Hierarchical fit did not reach its fixed stopping tolerance")
    # Ground truth is used only after fitting, to score this recovery exercise.
    candidates = candidate_probabilities(obs, grid)
    error = float(np.max(np.abs(candidates[np.arange(n), true_index] - truth)))
    if error > 1e-12:
        raise AssertionError("Generator and independently replayed policies disagree")
    # Expected choice loss on these histories separates grid points when even
    # the closest alternative has positive true-policy KL.
    separations = []
    for i in range(n):
        rows = candidates[i][:, train[i]]
        expected = np.broadcast_to(truth[i, train[i]], rows.shape)
        point_kl = binary_kl(expected, rows).mean(1)
        gap = np.delete(point_kl, true_index[i]).min()
        separations.append(float(gap))
    if min(separations) <= 0:
        raise AssertionError("A generating grid point is observationally indistinguishable")
    scores = {}
    for name, probabilities in [('individual', fit.individual_probability), ('pooled', fit.pooled_probability),
                                ('hierarchical', fit.hierarchical_probability)]:
        scores[name] = {'observed_choice_nll': float(binary_nll(obs['choice'], probabilities)[test].mean()),
                        'true_policy_kl': float(binary_kl(truth, probabilities)[test].mean())}
    return {
        'trials_per_person': length, 'training_trials_per_person': int(train[0].sum()),
        'test_trials_per_person': int(test[0].sum()), 'validation_trials_unused_per_person': int(masks['val'][0].sum()),
        'maximum_truth_replay_difference': error, 'minimum_nearest_wrong_point_mean_training_kl': min(separations),
        'individual_exact_joint_recoveries': int((fit.individual_index == true_index).sum()),
        'individual_recovered_grid_indices': fit.individual_index.tolist(),
        'hierarchical_map_exact_joint_recoveries': int((fit.posterior_weights.argmax(1) == true_index).sum()),
        'hierarchical_mean_posterior_mass_on_truth': float(fit.posterior_weights[np.arange(n), true_index].mean()),
        'hierarchical_prior_weights': fit.prior_weights.tolist(), 'hierarchical_em_iterations': len(fit.marginal_objective)-1,
        'hierarchical_em_converged': fit.converged, 'pooled_grid_index': fit.pooled_index, 'held_out_scores': scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON file. The default prints the result.')
    args = parser.parse_args()
    grid = np.array(DEFAULT_GRID)
    true_index = np.tile(np.arange(len(grid)), COPIES_PER_POINT)
    history, truth = generate_history(grid[true_index], max(LENGTHS), GENERATOR_SEED)
    runs = [run_length(history, truth, true_index, grid, length) for length in LENGTHS]
    record = {
        'scope': 'Fixed synthetic foundation recovery check with first-stage training choices only.',
        'generator_seed': GENERATOR_SEED, 'split_seed': SPLIT_SEED, 'people': len(true_index),
        'grid_columns': ['model_based_weight','alpha','beta'], 'grid': grid.tolist(),
        'true_grid_indices': true_index.tolist(), 'shared_history_prefix': True,
        'hierarchical_pseudocount': .1, 'hierarchical_tolerance': 1e-10, 'hierarchical_max_iterations': 2000,
        'numpy_version': np.__version__, 'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('foundation/classical_baselines.py', 'scripts/check_classical_baselines.py',
                         'src/distilled_rnns/simulation.py', 'src/distilled_rnns/evaluation.py')},
        'runs': runs,
    }
    encoded = json.dumps(record, indent=2, allow_nan=False) + '\n'
    if args.output is None:
        print(encoded, end='')
    else:
        args.output.write_text(encoded)


if __name__ == '__main__':
    main()
