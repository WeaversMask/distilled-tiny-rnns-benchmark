import json

import numpy as np
import pytest

from distilled_rnns.evaluation import binary_kl, binary_nll, evaluate_predictions, make_trial_masks, save_quadrant_plot


def test_bernoulli_metrics_match_known_values():
    choice = np.array([1, 0, 1])
    probability = np.array([0.8, 0.8, 0.5])
    np.testing.assert_allclose(binary_nll(choice, probability), [-np.log(.8), -np.log(.2), np.log(2)])
    p, q = np.array([.8]), np.array([.3])
    expected = .8 * np.log(.8 / .3) + .2 * np.log(.2 / .7)
    np.testing.assert_allclose(binary_kl(p, q), [expected])
    assert binary_kl(p, q)[0] != pytest.approx(binary_kl(q, p)[0])


def test_equal_policies_have_exactly_zero_kl_and_endpoints_stay_finite():
    probabilities = np.array([0., 1e-20, .2, .5, .9, 1.])
    np.testing.assert_array_equal(binary_kl(probabilities, probabilities), np.zeros(6))
    assert np.isfinite(binary_kl(np.array([0., 1.]), np.array([1., 0.]))).all()
    assert np.isfinite(binary_nll(np.array([1, 0]), np.array([0., 1.]))).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1, 1.1])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        binary_nll(np.array([1]), np.array([bad]))
    with pytest.raises(ValueError):
        binary_kl(np.array([bad]), np.array([.5]))
    with pytest.raises(ValueError):
        binary_kl(np.array([.5]), np.array([bad]))


def test_probability_shapes_must_match_and_choices_are_binary():
    with pytest.raises(ValueError, match="shapes"):
        binary_nll(np.ones((2, 1)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="shapes"):
        binary_kl(np.ones((2, 1)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="only 0 or 1"):
        binary_nll(np.array([.5]), np.array([.6]))


@pytest.mark.parametrize("n_trials", [3, 4, 5, 19, 100])
def test_masks_partition_every_trial_and_give_each_subject_support(n_trials):
    masks = make_trial_masks(8, n_trials, 11)
    assert set(masks) == {"train", "val", "test"}
    np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones((8, n_trials)))
    for mask in masks.values():
        assert mask.dtype == bool
        assert (mask.sum(axis=1) > 0).all()
    if n_trials == 100:
        np.testing.assert_array_equal(masks["train"].sum(axis=1), np.full(8, 60))
        np.testing.assert_array_equal(masks["test"].sum(axis=1), np.full(8, 20))


def test_split_reproducibility_and_independent_subject_masks():
    masks = make_trial_masks(5, 100, 123)
    again = make_trial_masks(5, 100, 123)
    changed = make_trial_masks(5, 100, 124)
    for key in masks:
        np.testing.assert_array_equal(masks[key], again[key])
        assert not np.array_equal(masks[key], changed[key])
    assert not np.array_equal(masks["test"][0], masks["test"][1])


@pytest.mark.parametrize("args", [(0, 10, 1), (2, 2, 1), (2.5, 10, 1), (2, 10, -1), (True, 10, 1)])
def test_invalid_mask_config_rejected(args):
    with pytest.raises(ValueError):
        make_trial_masks(*args)


def example_report_inputs():
    data = {
        "choice": np.array([[1, 1, 1, 0], [0, 1, 0, 1]]),
        "true_prob": np.full((2, 4), .6),
        "reference_prob": np.array([[.2, .2, .6, .6], [.6, .6, .6, .6]]),
        "rare": np.array([True, False]),
    }
    # On subject 0's two held-out ones, overconfident .9 predicts the realized
    # sample better than the .6 oracle but recovers the known policy worse.
    predictions = {"teacher": np.full((2, 4), .7), "distilled": np.full((2, 4), .9), "solo": np.full((2, 4), .6)}
    test_mask = np.array([[True, True, False, False], [True, True, False, False]])
    return data, predictions, test_mask


def test_report_quadrant_signs_and_heldout_only_scoring():
    data, predictions, test_mask = example_report_inputs()
    report = evaluate_predictions(data, predictions, test_mask)
    rare = report["per_subject"][0]
    assert rare["likelihood_gain"] == pytest.approx(np.log(.9 / .6))
    assert rare["fidelity_loss"] > 0
    assert rare["models"]["solo"]["kl"] == 0
    assert rare["diagnostic_count"] == 2
    assert rare["models"]["distilled"]["diagnostic_nll"] == pytest.approx(-np.log(.9))
    for probabilities in predictions.values():
        probabilities[:, 2:] = .0001
    assert report == evaluate_predictions(data, predictions, test_mask)


def test_empty_diagnostic_support_is_explicit_and_serializable():
    data, predictions, test_mask = example_report_inputs()
    report = evaluate_predictions(data, predictions, test_mask)
    assert report["per_subject"][1]["diagnostic_count"] == 0
    assert report["groups"]["typical"]["diagnostic_count"] == 0
    for model in predictions:
        assert report["per_subject"][1]["models"][model]["diagnostic_nll"] is None
        assert report["groups"]["typical"]["models"][model]["diagnostic_nll"] is None
        assert report["groups"]["typical"]["models"][model]["diagnostic_subject_count"] == 0
    json.dumps(report, allow_nan=False)


def test_group_means_are_subject_weighted_and_missing_group_is_supported():
    data, predictions, test_mask = example_report_inputs()
    data["rare"][:] = True
    test_mask[1, 2] = True
    report = evaluate_predictions(data, predictions, test_mask)
    group = report["groups"]["rare"]
    assert group["subject_count"] == 2
    assert group["test_count"] == 5
    assert group["models"]["solo"]["nll"] == pytest.approx(np.mean([row["models"]["solo"]["nll"] for row in report["per_subject"]]))
    assert group["models"]["solo"]["diagnostic_subject_count"] == 1
    assert report["groups"]["typical"]["subject_count"] == 0
    assert report["groups"]["typical"]["models"]["solo"]["nll"] is None
    json.dumps(report, allow_nan=False)


def test_report_validation():
    data, predictions, mask = example_report_inputs()
    for invalid_mask in (mask.astype(int), np.ones((1, 4), bool), np.zeros_like(mask)):
        with pytest.raises(ValueError):
            evaluate_predictions(data, predictions, invalid_mask)
    for threshold in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            evaluate_predictions(data, predictions, mask, threshold)
    with pytest.raises(ValueError, match="include"):
        evaluate_predictions(data, {"teacher": predictions["teacher"]}, mask)
    predictions["solo"] = predictions["solo"][:1]
    with pytest.raises(ValueError, match="shape"):
        evaluate_predictions(data, predictions, mask)


def test_quadrant_plot_written(tmp_path):
    report = evaluate_predictions(*example_report_inputs())
    path = save_quadrant_plot(report, tmp_path / "plots" / "quadrant.png")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
