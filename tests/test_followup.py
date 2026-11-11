from dataclasses import replace
import copy
import json

import numpy as np
import pytest
import torch

from distilled_rnns.config import ExperimentConfig
import distilled_rnns.followup as followup
from distilled_rnns.evaluation import make_trial_masks


@pytest.fixture
def config():
    return followup.FollowupConfig(
        base=ExperimentConfig(n_subjects=10, n_trials=15, rare_fraction=.2,
                              teacher_hidden=3, teacher_steps=2, student_steps=2,
                              seed=41, eval_every=1, max_seconds_per_phase=120),
        solo_l2_grid=(0.0, .01), replication_seeds=(51, 61), max_seconds_per_invocation=120)


@pytest.fixture(autouse=True)
def skip_figures(monkeypatch):
    # Numerical/provenance tests; actual PNG rendering is exercised by the notebook.
    monkeypatch.setattr(followup, "_plot_convergence", lambda *args: None)


def test_select_l2_ties_and_finite_scores():
    assert followup.select_l2({.1: .6, 0: .6, .01: .7}) == 0
    assert followup.select_l2({.1: .6, 0: .8, .01: .7}) == .1
    for values in ({}, {0: float("nan")}, {0: float("inf")}):
        with pytest.raises(ValueError):
            followup.select_l2(values)


@pytest.mark.parametrize("settings", [
    {"solo_l2_grid": ()}, {"solo_l2_grid": (.1,)}, {"solo_l2_grid": (0, .1, .01)},
    {"solo_l2_grid": (0, 0)}, {"solo_l2_grid": (0, float("nan"))},
    {"replication_seeds": ()}, {"replication_seeds": (51, 51)}, {"replication_seeds": (51, 52)},
    {"replication_seeds": (20261029,)}, {"replication_seeds": (True,)},
    {"max_seconds_per_invocation": 0}, {"max_seconds_per_invocation": float("inf")},
])
def test_config_rejects_unsafe_protocol(settings):
    with pytest.raises(ValueError):
        followup.FollowupConfig(**settings)


def test_full_study_resume_protocol_and_accounting(tmp_path, config, monkeypatch):
    original_evaluate = followup.evaluate_predictions
    calls = []

    def assert_frozen_before_test(*args, **kwargs):
        # Each dataset must freeze its selection before calling test/oracle evaluation.
        folders = [tmp_path / "calibration", *(tmp_path / f"seed-{s}" for s in config.replication_seeds)]
        folder = folders[len(calls) % len(folders)]
        assert (folder / "selection.json").exists()
        assert (tmp_path / "protocol.json").exists()
        calls.append(folder)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(followup, "evaluate_predictions", assert_frozen_before_test)
    first = followup.run_followup(config, tmp_path, "cpu")
    assert first["status"] == "complete"
    assert first["completed_seeds"] == [51, 61] and not first["pending_seeds"]
    assert first["timing"]["t4_equivalent_units_at_user_rate"] is None
    selection = json.loads((tmp_path / "calibration/selection.json").read_text())
    assert first["calibration"]["selected_l2"] == selection["selected_l2"]
    total_seconds = 0
    saved = {}
    for folder in calls:
        report = json.loads((folder / "report.json").read_text())
        total_seconds += sum(p["elapsed_seconds"] for p in report["phases"].values())
        assert report["selection"]["selected_l2"] == selection["selected_l2"]
        with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
            saved[str(folder)] = {key: arrays[key].copy() for key in arrays.files}
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["identity"]["protocol_sha256"] == first["protocol_sha256"]
    assert first["timing"]["cumulative_training_seconds"] == pytest.approx(total_seconds)
    # Calibration must not contribute a third seed to the descriptive summary.
    for group in first["groups"].values():
        assert [row["seed"] for row in group["per_seed"]] == [51, 61]
        pairs = [row["comparisons"]["solo"]["likelihood_gain"] for row in group["per_seed"]]
        assert group["comparisons"]["solo"]["likelihood_gain"]["mean"] == pytest.approx(np.mean(pairs))
    resumed = followup.run_followup(config, tmp_path, "cpu")
    assert resumed["groups"] == first["groups"]
    assert resumed["calibration"] == first["calibration"]
    for folder, predictions in saved.items():
        with np.load(folder + "/predictions.npz", allow_pickle=False) as arrays:
            for name, value in predictions.items():
                np.testing.assert_array_equal(arrays[name], value)
    with pytest.raises(ValueError, match="configuration or source"):
        followup.run_followup(replace(config, solo_l2_grid=(0, .1)), tmp_path, "cpu")
    monkeypatch.setattr(followup, "source_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="configuration or source"):
        followup.run_followup(config, tmp_path, "cpu")


def test_test_choices_oracle_and_group_labels_cannot_select_lambda(tmp_path, config, monkeypatch):
    baseline = followup.run_followup(config, tmp_path / "original", "cpu")
    original_simulation = followup.simulate_population

    def changed_test_and_oracle(n, t, rare_fraction, seed):
        data = original_simulation(n, t, rare_fraction, seed)
        mask = make_trial_masks(n, t, seed + 1)["test"]
        data["choice"][mask] = 1 - data["choice"][mask]
        data["true_prob"] = 1 - data["true_prob"]
        data["reference_prob"] = np.full_like(data["reference_prob"], .99)
        data["rare"] = ~data["rare"]
        # Keep x fixed: this isolates access to held-out labels/oracle/group data.
        return data

    monkeypatch.setattr(followup, "simulate_population", changed_test_and_oracle)
    changed = followup.run_followup(config, tmp_path / "changed", "cpu")
    assert baseline["calibration"]["selected_l2"] == changed["calibration"]["selected_l2"]
    left = json.loads((tmp_path / "original/calibration/selection.json").read_text())
    right = json.loads((tmp_path / "changed/calibration/selection.json").read_text())
    assert left == right
    assert baseline["groups"] != changed["groups"]


def test_interruption_never_selects_partial_candidates_or_aggregates(tmp_path, config, monkeypatch):
    real_fit = followup.fit_model
    with monkeypatch.context() as patch:
        patch.setattr(followup, "fit_model", lambda *a, **kw: real_fit(*a, **{**kw, "max_seconds": 1e-12}))
        result = followup.run_followup(config, tmp_path, "cpu")
    assert result["status"] == "incomplete"
    assert "groups" not in result and "calibration" not in result
    assert result["pending_seeds"] == [51, 61]
    assert not (tmp_path / "calibration/selection.json").exists()
    assert (tmp_path / "calibration/teacher.pt").exists()
    completed = followup.run_followup(config, tmp_path, "cpu")
    assert completed["status"] == "complete"
    for phase in json.loads((tmp_path / "calibration/training.json").read_text()).values():
        assert phase["completed_steps"] == 2


def test_replication_partial_keeps_frozen_lambda_but_no_aggregate(tmp_path, config, monkeypatch):
    real_dataset = followup._dataset

    def stop_replication(c, folder, *args):
        if c.seed != config.base.seed:
            return {"status": "incomplete", "phase": "teacher"}
        return real_dataset(c, folder, *args)

    monkeypatch.setattr(followup, "_dataset", stop_replication)
    result = followup.run_followup(config, tmp_path, "cpu")
    assert result["status"] == "incomplete" and "groups" not in result
    assert "selected_l2" in result["calibration"]
    assert result["completed_seeds"] == []


def test_zero_diagnostic_support_is_null_and_partial_aggregate_rejected(tmp_path, config, monkeypatch):
    original_simulation = followup.simulate_population

    def no_diagnostics(*args):
        data = original_simulation(*args)
        data["reference_prob"] = data["true_prob"].copy()
        return data

    monkeypatch.setattr(followup, "simulate_population", no_diagnostics)
    result = followup.run_followup(config, tmp_path, "cpu")
    for group in result["groups"].values():
        diagnostic = group["comparisons"]["solo"]["diagnostic_likelihood_gain"]
        assert diagnostic == {"mean": None, "min": None, "max": None,
                              "supported_seed_count": 0, "planned_seed_count": 2}
        assert all(row["diagnostic_count"] == 0 for row in group["per_seed"])
    reports = [json.loads((tmp_path / f"seed-{s}/report.json").read_text()) for s in config.replication_seeds]
    with pytest.raises(ValueError, match="Every planned seed"):
        followup.aggregate_replicates(reports[:1], config.replication_seeds)
    incomplete = copy.deepcopy(reports)
    incomplete[-1]["status"] = "incomplete"
    with pytest.raises(ValueError, match="Every planned seed"):
        followup.aggregate_replicates(incomplete, config.replication_seeds)


def test_zero_winner_reuses_single_unregularized_fit(tmp_path, config, monkeypatch):
    config = replace(config, solo_l2_grid=(0.0,))
    result = followup.run_followup(config, tmp_path, "cpu")
    assert result["calibration"]["selected_l2"] == 0
    for folder in (tmp_path / "calibration", tmp_path / "seed-51", tmp_path / "seed-61"):
        assert len(list(folder.glob("solo*.pt"))) == 1
        with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["solo"], arrays["solo_unregularized"])


def test_unprovenanced_folder_rejected(tmp_path, config):
    (tmp_path / "old.pt").write_bytes(b"old")
    with pytest.raises(ValueError, match="without a protocol"):
        followup.run_followup(config, tmp_path, "cpu")
