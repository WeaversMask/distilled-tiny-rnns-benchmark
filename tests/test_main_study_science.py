"""Independent scientific checks for the prospective simulation and readout."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from distilled_rnns.evaluation import make_trial_masks
from main_study.config import MainStudyConfig
from main_study.evaluation import baseline_predictions, evaluate, true_probe_predictions
from main_study.simulation import HISTORY_KEYS, simulate_population


def scalar_truth(data, person):
    """An independent scalar replay, without calling the simulator or replay."""
    q_first = [0.5, 0.5]
    q_second = [[0.5, 0.5], [0.5, 0.5]]
    alpha, beta, weight = (data[k][person] for k in ("alpha", "beta", "model_based_weight"))
    result = []
    reference = []
    for t in range(data["choice"].shape[1]):
        advantage = q_first[1] - q_first[0]
        state_advantage = max(q_second[1]) - max(q_second[0])
        logit = beta * ((1 - weight) * advantage + weight * 0.4 * state_advantage)
        p = 1 / (1 + np.exp(-logit))
        if data["is_cue"][person]:
            if t == 0:
                p = 0.5
            else:
                old_state = data["state"][person, t - 1]
                old_reward = data["reward"][person, t - 1]
                desired = old_state if old_reward else 1 - old_state
                rho = data["cue_reliability"][person]
                p = rho if desired else 1 - rho
        result.append(p)
        ref_logit = beta * (0.25 * advantage + 0.75 * 0.4 * state_advantage)
        reference.append(1 / (1 + np.exp(-ref_logit)))
        action, state, second, reward = (data[k][person, t] for k in HISTORY_KEYS)
        old = q_second[state][second]
        q_first[action] += alpha * ((old - q_first[action]) + 0.5 * (reward - old))
        q_second[state][second] += alpha * (reward - old)
    return np.array(result), np.array(reference)


@pytest.mark.parametrize("kind", ["parametric_model_free", "reward_cue"])
def test_seeded_generating_probabilities_match_independent_replay_and_on_grid_parameters(kind):
    config = MainStudyConfig("smoke")
    before = np.random.get_state()
    data = simulate_population(config, kind, 707)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert data["rare"].sum() == 2 and np.all(data["true_prob"][:, 0] == 0.5)
    assert data["is_cue"].sum() == (2 if kind == "reward_cue" else 0)
    for person in range(config.n_subjects):
        true, reference = scalar_truth(data, person)
        np.testing.assert_allclose(data["true_prob"][person], true, rtol=0, atol=1e-12)
        np.testing.assert_allclose(data["reference_prob"][person], reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(data["x"][:, 0], 0)
    raw = np.stack([data[k] for k in HISTORY_KEYS], axis=-1)
    np.testing.assert_array_equal(data["x"][:, 1:], 2 * raw[:, :-1] - 1)
    again = simulate_population(config, kind, 707)
    for key in data:
        np.testing.assert_array_equal(data[key], again[key])
        assert np.isfinite(data[key]).all()


def test_baselines_use_training_observations_and_keep_fixed_posterior_in_probes():
    config = MainStudyConfig("smoke")
    data = simulate_population(config, "reward_cue", 708)
    masks = make_trial_masks(10, 15, 709)
    p, probes, metadata = baseline_predictions(config, data, masks["train"])
    clean = {k: data[k].copy() for k in HISTORY_KEYS}
    clean["true_prob"] = None
    clean["rare"] = None
    p_clean, q_clean, meta_clean = baseline_predictions(config, clean, masks["train"])
    for key in p:
        np.testing.assert_array_equal(p[key], p_clean[key])
        np.testing.assert_array_equal(probes[key], q_clean[key])
        assert probes[key].shape == (10, 256, 3)
    assert metadata == meta_clean
    # Held-out target perturbation with candidate probabilities frozen is tested
    # in the foundation suite; causal histories may legitimately carry actions.
    posterior = np.array(metadata["family_posterior_hybrid_cue"])
    np.testing.assert_allclose(posterior.sum(1), 1)
    from main_study.evaluation import _probe_history
    from foundation.classical_baselines import candidate_probabilities
    from foundation.reward_cue import cue_probabilities

    hybrid = candidate_probabilities(_probe_history(), metadata["hybrid_grid"]).transpose(1, 0, 2)
    cue = cue_probabilities(_probe_history(), metadata["cue_grid"]).transpose(1, 0, 2)
    person = 2
    expected = np.zeros((256, 3))
    for j, model in enumerate(hybrid):
        expected += posterior[person, 0] * metadata["family_mixture_hybrid_posterior"][person][j] * model
    for j, model in enumerate(cue):
        expected += posterior[person, 1] * metadata["family_mixture_cue_posterior"][person][j] * model
    np.testing.assert_allclose(probes["hierarchical_hybrid_cue_mixture"][person], expected, atol=1e-15)
    counts = masks["train"].sum(1)
    expected_frequency = ((data["choice"] * masks["train"]).sum(1) + 1) / (counts + 2)
    np.testing.assert_allclose(p["training_frequency"][:, 0], expected_frequency)
    assert metadata["converged"]


def test_group_metrics_weight_people_equally_preserve_nulls_and_apply_primary_signs():
    config = MainStudyConfig("smoke")
    data = simulate_population(config, "reward_cue", 710)
    # Deliberately unequal held-out counts expose trial weighting errors.
    mask = np.zeros_like(data["choice"], dtype=bool)
    for i in range(10):
        mask[i, : i + 1] = True
    # No diagnostic support should be null rather than a fabricated zero.
    data["reference_prob"] = data["true_prob"].copy()
    true_probes = true_probe_predictions(data)
    solo = np.full_like(data["true_prob"], 0.5)
    learned = 0.8 * data["true_prob"] + 0.2 * solo
    probabilities = {"solo_population": solo, "distilled_population": learned}
    # Deliberately reversed strategy fidelity: solo is perfect on probes.
    probes = {"solo_population": true_probes, "distilled_population": np.full_like(true_probes, 0.5)}
    report = evaluate(config, data, probabilities, probes, mask)
    rare_rows = [row for row in report["per_subject"] if row["rare"]]
    raw_kl = data["true_prob"] * np.log(data["true_prob"] / 0.5) + (1 - data["true_prob"]) * np.log(
        (1 - data["true_prob"]) / 0.5
    )
    expected_kl = np.mean([raw_kl[row["subject_id"], mask[row["subject_id"]]].mean() for row in rare_rows])
    np.testing.assert_allclose(report["groups"]["rare"]["models"]["solo_population"]["kl"], expected_kl)
    for group in report["groups"].values():
        assert group["diagnostic_count"] == 0
        for metrics in group["models"].values():
            assert metrics["diagnostic_nll"] is None and metrics["diagnostic_kl"] is None
        group_rows = (
            rare_rows
            if group is report["groups"]["rare"]
            else [row for row in report["per_subject"] if not row["rare"]]
        )
        expected_gain = np.mean(
            [
                row["models"]["solo_population"]["nll"] - row["models"]["distilled_population"]["nll"]
                for row in group_rows
            ]
        )
        assert group["primary"]["population"]["prediction_gain"] == expected_gain
        assert group["primary"]["population"]["behavioral_distortion"] >= 0
        assert group["models"]["solo_population"]["signature_rms_error"] == 0
        assert group["models"]["solo_population"]["probe_kl"] == 0
    assert report["groups"]["rare"]["primary"]["population"]["behavioral_distortion"] > 0


def test_uniform_true_signatures_give_null_pull_with_explicit_support():
    config = MainStudyConfig("smoke")
    data = simulate_population(config, "reward_cue", 712)
    # Everyone becomes the same valid hybrid before calculating true probes.
    data["is_cue"][:] = False
    data["model_based_weight"][:] = 0.4
    data["alpha"][:] = 0.2
    data["beta"][:] = 2.0
    probes = true_probe_predictions(data)
    predicted = np.full_like(data["true_prob"], 0.5)
    report = evaluate(
        config,
        data,
        {"solo_population": predicted, "distilled_population": predicted},
        {"solo_population": probes, "distilled_population": probes},
        np.ones_like(predicted, dtype=bool),
    )
    for group in report["groups"].values():
        for model in group["models"].values():
            assert model["pull"] is None and model["pull_subject_count"] == 0


def test_invalid_probabilities_or_misaligned_models_are_rejected():
    config = MainStudyConfig("smoke")
    data = simulate_population(config, "reward_cue", 713)
    mask = np.ones_like(data["choice"], dtype=bool)
    with pytest.raises(ValueError, match="models must match"):
        evaluate(config, data, {"solo_population": data["true_prob"]}, {}, mask)
    with pytest.raises(ValueError, match="Unknown rare-agent"):
        simulate_population(config, "unknown", 713)
