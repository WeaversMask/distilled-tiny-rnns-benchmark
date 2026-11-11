import numpy as np
import pytest

from distilled_rnns.simulation import _reflect, replay_policy, simulate_population


def test_deterministic_seed_and_global_rng_isolation():
    np.random.seed(999)
    expected_next_global_draw = np.random.random()
    np.random.seed(999)
    first = simulate_population(20, 40, 0.1, 12)
    second = simulate_population(20, 40, 0.1, 12)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.random.random() == expected_next_global_draw
    other = simulate_population(20, 40, 0.1, 13)
    assert not np.array_equal(first["choice"], other["choice"])


def test_exact_groups_and_valid_probabilities():
    data = simulate_population(1000, 100, 0.02, 33)
    assert data["rare"].sum() == 20
    assert data["x"].shape == (1000, 100, 4)
    assert data["x"].dtype == np.float32
    assert data["choice"].dtype == np.int64
    assert data["rare"].dtype == bool
    assert np.all(data["model_based_weight"][data["rare"]] == 0)
    typical_weights = data["model_based_weight"][~data["rare"]]
    assert np.all((typical_weights >= 0.5) & (typical_weights <= 0.9))
    for key in ("true_prob", "reference_prob"):
        assert np.isfinite(data[key]).all()
        assert np.all((data[key] > 0) & (data[key] < 1))
        np.testing.assert_array_equal(data[key][:, 0], 0.5)
    common_frequency = (data["choice"] == data["state"]).mean()
    assert abs(common_frequency - 0.7) < 0.006


def test_inputs_are_strictly_previous_trial_history():
    data = simulate_population(20, 40, 0.1, 12)
    np.testing.assert_array_equal(data["x"][:, 0], 0)
    for feature, key in enumerate(("choice", "state", "second_choice", "reward")):
        np.testing.assert_array_equal(data["x"][:, 1:, feature], 2 * data[key][:, :-1] - 1)


def test_replay_recovers_saved_true_and_reference_policies():
    data = simulate_population(20, 40, 0.1, 12)
    actual = replay_policy(data, data["model_based_weight"], data["alpha"], data["beta"])
    reference = replay_policy(data, 0.75, data["alpha"], data["beta"])
    np.testing.assert_allclose(actual, data["true_prob"], atol=0, rtol=0)
    np.testing.assert_allclose(reference, data["reference_prob"], atol=0, rtol=0)


def test_hand_calculated_rewarded_common_transition():
    # After rewarded action 0 -> state 0 -> action 1, alpha=.4 gives
    # Q1=[.6,.5], max Q2=[.7,.5]. Thus Q_MB=[.64,.56].
    # With w=.75, mixed values are [.63,.545], so P(action1)=sigmoid(-.17).
    history = {
        "choice": np.array([0, 1]),
        "state": np.array([0, 0]),
        "second_choice": np.array([1, 0]),
        "reward": np.array([1, 0]),
    }
    probability = replay_policy(history, 0.75, 0.4, 2.0)
    np.testing.assert_allclose(probability, [0.5, 1 / (1 + np.exp(0.17))], atol=1e-14)
    model_free = replay_policy(history, 0.0, 0.4, 2.0)
    np.testing.assert_allclose(model_free[1], 1 / (1 + np.exp(0.2)), atol=1e-14)


def test_independent_scalar_reference_for_multistep_learning():
    data = simulate_population(4, 12, 0.25, 42)
    for person in range(4):
        q1 = [0.5, 0.5]
        q2 = [[0.5, 0.5], [0.5, 0.5]]
        w, alpha, beta = (data[key][person] for key in ("model_based_weight", "alpha", "beta"))
        for trial in range(12):
            future = [max(q2[0]), max(q2[1])]
            mb = [0.7 * future[0] + 0.3 * future[1], 0.3 * future[0] + 0.7 * future[1]]
            values = [(1 - w) * q1[action] + w * mb[action] for action in range(2)]
            probability = 1 / (1 + np.exp(-beta * (values[1] - values[0])))
            assert data["true_prob"][person, trial] == pytest.approx(probability, abs=1e-14)
            a1, state, a2, reward = (int(data[key][person, trial]) for key in ("choice", "state", "second_choice", "reward"))
            old_q2 = q2[state][a2]
            first_error = old_q2 - q1[a1]
            reward_error = reward - old_q2
            q1[a1] += alpha * (first_error + 0.5 * reward_error)
            q2[state][a2] += alpha * reward_error


def test_replay_current_and_future_outcomes_cannot_change_current_prediction():
    data = simulate_population(20, 40, 0.1, 12)
    altered = {key: data[key].copy() for key in ("choice", "state", "second_choice", "reward")}
    for key in altered:
        altered[key][:, 10:] = 1 - altered[key][:, 10:]
    changed = replay_policy(altered, data["model_based_weight"], data["alpha"], data["beta"])
    np.testing.assert_array_equal(changed[:, :11], data["true_prob"][:, :11])


def test_reflection_handles_multiple_boundary_crossings():
    values = np.array([0.25, 0.75, 0.24, 0.76, -0.26, 1.26, -100.1, 100.1])
    reflected = _reflect(values, 0.25, 0.75)
    np.testing.assert_allclose(reflected[:6], [0.25, 0.75, 0.26, 0.74, 0.74, 0.26])
    assert np.all((reflected >= 0.25) & (reflected <= 0.75))


@pytest.mark.parametrize("args", [
    (1, 10, 0.5, 1), (20, 0, 0.1, 1), (20, 10, 0, 1),
    (20, 10, 1, 1), (20, 10, 0.001, 1), (20, 10, 0.999, 1),
    (20, 10, float("nan"), 1), (20.0, 10, 0.1, 1),
    (20, 10, 0.1, -1), (True, 10, 0.1, 1),
])
def test_invalid_population_arguments_fail_clearly(args):
    with pytest.raises(ValueError):
        simulate_population(*args)


def test_one_trial_population_is_valid():
    data = simulate_population(2, 1, 0.5, 0)
    assert data["rare"].sum() == 1
    np.testing.assert_array_equal(data["x"], 0)
    np.testing.assert_array_equal(data["true_prob"], 0.5)


def test_replay_rejects_malformed_histories_and_parameters():
    data = simulate_population(4, 12, 0.25, 42)
    with pytest.raises(ValueError, match="model_based_weight"):
        replay_policy(data, 1.1, 0.5, 2)
    with pytest.raises(ValueError, match="alpha"):
        replay_policy(data, 0.5, 0, 2)
    with pytest.raises(ValueError, match="beta"):
        replay_policy(data, 0.5, 0.5, np.nan)
    malformed = dict(data)
    malformed["reward"] = np.full((4, 12), 2)
    with pytest.raises(ValueError, match="binary"):
        replay_policy(malformed, 0.5, 0.5, 2)
