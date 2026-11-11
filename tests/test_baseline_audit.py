from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distilled_rnns.config import ExperimentConfig
from distilled_rnns.evaluation import binary_nll, make_trial_masks
import distilled_rnns.baseline_audit as audit


@pytest.fixture
def config():
    return audit.BaselineAuditConfig(
        base=ExperimentConfig(n_subjects=10, n_trials=15, rare_fraction=.2,
                              teacher_hidden=3, teacher_steps=2, student_steps=2,
                              seed=71, eval_every=1, max_seconds_per_phase=120),
        solo_l2_grid=(0.0, .01), replication_seeds=(81, 91),
        legacy_solo_l2=.01, max_seconds_per_invocation=120)


@pytest.fixture(autouse=True)
def skip_figures(monkeypatch):
    # Numerical/provenance checks; the actual notebook verifies rendering.
    monkeypatch.setattr(audit, "_plot_convergence", lambda *args: None)
    monkeypatch.setattr(audit, "_selection_plot", lambda *args: None)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def assert_saved_weights_equal(left, right):
    """Compare actual optimizer, individual, and population states after fitting."""
    left_files = sorted(Path(left).glob("*.pt"))
    assert [p.name for p in left_files] == [p.name for p in sorted(Path(right).glob("*.pt"))]
    for path in left_files:
        a = torch.load(path, map_location="cpu", weights_only=True)
        b = torch.load(Path(right) / path.name, map_location="cpu", weights_only=True)
        for key in ("current_state", "best_state", "population_best_state"):
            assert (key in a) == (key in b)
            if key in a:
                assert a[key].keys() == b[key].keys()
                for parameter in a[key]:
                    torch.testing.assert_close(a[key][parameter], b[key][parameter], rtol=0, atol=0)
        assert a["history"] == b["history"]
        assert a["completed_steps"] == b["completed_steps"]
        for parameter in a["optimizer"]["state"]:
            for key, value in a["optimizer"]["state"][parameter].items():
                torch.testing.assert_close(value, b["optimizer"]["state"][parameter][key], rtol=0, atol=0)


def test_outer_pipeline_ties_are_lower_l2_then_population():
    candidates = [
        {"l2": .01, "mode": "population", "tuning_nll": .6},
        {"l2": 0., "mode": "individual", "tuning_nll": .6},
        {"l2": 0., "mode": "population", "tuning_nll": .6},
    ]
    expected = candidates[-1]
    assert audit.choose_pipeline(candidates) == expected
    assert audit.choose_pipeline(list(reversed(candidates))) == expected
    candidates[0]["tuning_nll"] = .59
    assert audit.choose_pipeline(candidates) == candidates[0]


@pytest.mark.parametrize("candidates", [
    [], [{"l2": 0., "mode": "population", "tuning_nll": float("nan")}],
    [{"l2": 0., "mode": "population", "tuning_nll": float("inf")}],
    [{"l2": -.1, "mode": "population", "tuning_nll": .5}],
    [{"l2": 0., "mode": "unknown", "tuning_nll": .5}],
])
def test_invalid_pipeline_candidates_rejected(candidates):
    with pytest.raises(ValueError):
        audit.choose_pipeline(candidates)


@pytest.mark.parametrize("changes", [
    {"legacy_solo_l2": .002}, {"legacy_solo_l2": float("nan")},
    {"legacy_solo_l2": True}, {"solo_l2_grid": (0, .01, .01)},
    {"replication_seeds": (20261201,)}, {"replication_seeds": (81, 82)},
    {"max_seconds_per_invocation": 0},
])
def test_invalid_audit_protocol_rejected(changes):
    with pytest.raises(ValueError):
        audit.BaselineAuditConfig(**changes)


def test_references_are_fixed_and_use_only_training_choices():
    choice = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
    mask = np.array([[True, True, True, False, False]] * 2)
    result = audit.reference_predictions(choice, mask)
    np.testing.assert_array_equal(result["coinflip"], np.full_like(choice, .5, dtype=float))
    np.testing.assert_allclose(binary_nll(choice, result["coinflip"]), np.log(2), rtol=0, atol=0)
    np.testing.assert_allclose(result["frequency"], [[.2] * 5, [.8] * 5], rtol=0, atol=0)
    changed = choice.copy()
    changed[~mask] = 1 - changed[~mask]
    heldout_changed = audit.reference_predictions(changed, mask)
    for key in result:
        np.testing.assert_array_equal(result[key], heldout_changed[key])
    changed[0, 0] = 1
    assert audit.reference_predictions(changed, mask)["frequency"][0, 0] == .4


def test_full_audit_freezes_before_test_and_resumes_identically(tmp_path, config, monkeypatch):
    actual_test = audit._test_report
    calls = []

    def checked_test(*args, **kwargs):
        # Calibration is never routed through the test/oracle report function.
        seed = config.replication_seeds[len(calls) % len(config.replication_seeds)]
        calibration_selection = read_json(tmp_path / "calibration/selection.json")
        current_selection = read_json(tmp_path / f"seed-{seed}/selection.json")
        assert calibration_selection == current_selection
        assert (tmp_path / "protocol.json").exists()
        calls.append(seed)
        return actual_test(*args, **kwargs)

    monkeypatch.setattr(audit, "_test_report", checked_test)
    result = audit.run_baseline_audit(config, tmp_path, "cpu")
    assert result["status"] == "complete" and result["pending_seeds"] == []
    assert result["completed_seeds"] == [81, 91] and calls == [81, 91]
    assert result["timing"]["t4_equivalent_units_at_user_rate"] is None
    protocol = read_json(tmp_path / "protocol.json")
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert result["protocol_sha256"] == digest
    calibration = read_json(tmp_path / "calibration/report.json")
    assert calibration["role"] == "calibration"
    assert "groups" not in calibration and "per_subject" not in calibration
    assert "tuning_scores" in calibration
    selection = calibration["selection"]
    calibration_data = read_arrays(tmp_path / "calibration/data.npz")
    calibration_candidates = read_arrays(tmp_path / "calibration/candidate_predictions.npz")
    tuning_mask = calibration_data["mask_tuning"]
    for name, probability in calibration_candidates.items():
        # Recompute the selection criterion from stored predictions and the
        # independent third split, rather than trusting candidate metadata.
        elementwise = binary_nll(calibration_data["choice"], probability)
        outer_nll = float(((elementwise * tuning_mask).sum(1) / tuning_mask.sum(1)).mean())
        assert calibration["tuning_scores"][name] == pytest.approx(outer_nll, abs=1e-12)
    for candidate in selection["solo_candidates"]:
        index = config.solo_l2_grid.index(candidate["l2"])
        name = f"solo_{index}_{candidate['mode']}"
        assert candidate["tuning_nll"] == calibration["tuning_scores"][name]
    for candidate in selection["distilled_candidates"]:
        assert candidate["tuning_nll"] == calibration["tuning_scores"][f"distilled_{candidate['mode']}"]
    expected_solo = min(selection["solo_candidates"],
                        key=lambda row: (row["tuning_nll"], row["l2"], ("population", "individual").index(row["mode"])))
    expected_distilled = min(selection["distilled_candidates"],
                             key=lambda row: (row["tuning_nll"], row["l2"], ("population", "individual").index(row["mode"])))
    assert (selection["solo_l2"], selection["solo_selection"]) == (expected_solo["l2"], expected_solo["mode"])
    assert selection["distilled_selection"] == expected_distilled["mode"]
    saved = {}
    for name in ("calibration", "seed-81", "seed-91"):
        folder = tmp_path / name
        data = read_arrays(folder / "data.npz")
        role = "tuning" if name == "calibration" else "test"
        mask_names = {key for key in data if key.startswith("mask_")}
        assert mask_names == {"mask_train", "mask_checkpoint", f"mask_{role}"}
        masks = [data[key] for key in mask_names]
        assert all(mask.dtype == bool and np.all(mask.sum(1) > 0) for mask in masks)
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks), np.ones_like(data["choice"]))
        manifest = read_json(folder / "manifest.json")
        assert manifest["identity"]["third_split_role"] == role
        assert manifest["identity"]["protocol_sha256"] == digest
        assert read_json(folder / "selection.json") == selection
        saved[name] = read_arrays(folder / "predictions.npz")
        np.testing.assert_array_equal(saved[name]["solo"], saved[name][f"solo_{selection['solo_selection']}"])
        np.testing.assert_array_equal(saved[name]["distilled"], saved[name][f"distilled_{selection['distilled_selection']}"])
    for group in result["groups"].values():
        assert [row["seed"] for row in group["per_seed"]] == [81, 91]
    resumed = audit.run_baseline_audit(config, tmp_path, "cpu")
    assert resumed["groups"] == result["groups"]
    assert resumed["calibration"] == result["calibration"]
    for name, predictions in saved.items():
        after = read_arrays(tmp_path / name / "predictions.npz")
        for model, values in predictions.items():
            np.testing.assert_array_equal(values, after[model])
    with pytest.raises(ValueError, match="configuration or source"):
        audit.run_baseline_audit(replace(config, legacy_solo_l2=0), tmp_path, "cpu")
    monkeypatch.setattr(audit, "source_fingerprint", lambda: "changed-source")
    with pytest.raises(ValueError, match="configuration or source"):
        audit.run_baseline_audit(config, tmp_path, "cpu")


def test_tuning_labels_change_scores_but_not_fits_or_checkpoints(tmp_path, config, monkeypatch):
    audit.run_baseline_audit(config, tmp_path / "original", "cpu")
    actual_simulation = audit.simulate_population

    def flip_calibration_tuning(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        if seed == config.base.seed:
            mask = make_trial_masks(n, t, seed + 1)["test"]
            data["choice"][mask] = 1 - data["choice"][mask]
            # Keep the recorded causal histories fixed to isolate label access.
        return data

    monkeypatch.setattr(audit, "simulate_population", flip_calibration_tuning)
    audit.run_baseline_audit(config, tmp_path / "changed", "cpu")
    left, right = (tmp_path / name / "calibration" for name in ("original", "changed"))
    assert_saved_weights_equal(left, right)
    assert read_json(left / "report.json")["tuning_scores"] != read_json(right / "report.json")["tuning_scores"]
    assert read_json(left / "report.json")["checkpoint_scores"] == read_json(right / "report.json")["checkpoint_scores"]


def test_replication_test_oracle_and_group_data_cannot_change_models(tmp_path, config, monkeypatch):
    original = audit.run_baseline_audit(config, tmp_path / "original", "cpu")
    actual_simulation = audit.simulate_population

    def change_replication_evaluation_data(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        if seed != config.base.seed:
            mask = make_trial_masks(n, t, seed + 1)["test"]
            data["choice"][mask] = 1 - data["choice"][mask]
            data["true_prob"] = 1 - data["true_prob"]
            data["reference_prob"] = np.full_like(data["reference_prob"], .99)
            data["rare"] = ~data["rare"]
        return data

    monkeypatch.setattr(audit, "simulate_population", change_replication_evaluation_data)
    changed = audit.run_baseline_audit(config, tmp_path / "changed", "cpu")
    assert original["calibration"] == changed["calibration"]
    assert original["groups"] != changed["groups"]
    for name in ("calibration", "seed-81", "seed-91"):
        left, right = (tmp_path / root / name for root in ("original", "changed"))
        assert_saved_weights_equal(left, right)
        assert read_json(left / "selection.json") == read_json(right / "selection.json")
        before, after = read_arrays(left / "predictions.npz"), read_arrays(right / "predictions.npz")
        for model in before:
            np.testing.assert_array_equal(before[model], after[model])


def test_each_dual_selected_trajectory_is_trained_and_counted_once(tmp_path, config, monkeypatch):
    calls = []
    actual_fit = audit.fit_model

    def counted_fit(*args, **kwargs):
        calls.append(Path(kwargs["checkpoint_path"]))
        result = actual_fit(*args, **kwargs)
        # Distinct deterministic durations make accidental double counting visible.
        result["elapsed_seconds"] = float(len(calls))
        return result

    monkeypatch.setattr(audit, "fit_model", counted_fit)
    summary = audit.run_baseline_audit(config, tmp_path, "cpu")
    selected_l2 = summary["calibration"]["solo_l2"]
    expected_replicate_fits = 2 + len({selected_l2, config.legacy_solo_l2})
    assert len(calls) == 2 + len(config.solo_l2_grid) + len(config.replication_seeds) * expected_replicate_fits
    assert len(calls) == len(set(calls))
    assert summary["timing"]["cumulative_training_seconds"] == sum(range(1, len(calls) + 1))
    for name in ("calibration", "seed-81", "seed-91"):
        folder = tmp_path / name
        phases = read_json(folder / "training.json")
        assert len(phases) == len(list(folder.glob("*.pt")))
        for phase_name, phase in phases.items():
            if phase_name != "teacher":
                saved = torch.load(folder / f"{phase_name}.pt", weights_only=True, map_location="cpu")
                assert "population_best_state" in saved and "best_state" in saved
                assert saved["population_best_step"] == phase["population_best_step"]


def test_partial_candidate_grid_never_freezes_or_aggregates(tmp_path, config, monkeypatch):
    actual_fit = audit.fit_model

    def interrupt_first_solo(*args, **kwargs):
        if Path(kwargs["checkpoint_path"]).stem.startswith("solo_"):
            kwargs["max_seconds"] = 1e-12
        return actual_fit(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(audit, "fit_model", interrupt_first_solo)
        result = audit.run_baseline_audit(config, tmp_path, "cpu")
    assert result["status"] == "incomplete" and "groups" not in result
    assert "calibration" not in result and result["pending_seeds"] == [81, 91]
    assert not (tmp_path / "calibration/selection.json").exists()
    assert (tmp_path / "calibration/solo_0.pt").exists()
    resumed = audit.run_baseline_audit(config, tmp_path, "cpu")
    uninterrupted = audit.run_baseline_audit(config, tmp_path / "reference", "cpu")
    assert resumed["status"] == "complete"
    assert resumed["groups"] == uninterrupted["groups"]
    assert resumed["calibration"] == uninterrupted["calibration"]
    for name in ("calibration", "seed-81", "seed-91"):
        assert_saved_weights_equal(tmp_path / name, tmp_path / "reference" / name)


def test_partial_replication_keeps_selection_but_has_no_aggregate(tmp_path, config, monkeypatch):
    actual_dataset = audit._dataset

    def stop_last_seed(audit_config, experiment, folder, *args):
        if experiment.seed == config.replication_seeds[-1]:
            return {"status": "incomplete", "phase": "teacher"}
        return actual_dataset(audit_config, experiment, folder, *args)

    monkeypatch.setattr(audit, "_dataset", stop_last_seed)
    result = audit.run_baseline_audit(config, tmp_path, "cpu")
    assert result["status"] == "incomplete" and "groups" not in result
    assert result["completed_seeds"] == [81] and result["pending_seeds"] == [91]
    assert "solo_l2" in result["calibration"]
    assert read_json(tmp_path / "summary.json")["status"] == "incomplete"


def test_unsupported_diagnostics_stay_null_and_aggregation_requires_all_seeds(tmp_path, config, monkeypatch):
    actual_simulation = audit.simulate_population

    def no_diagnostic_difference(*args):
        data = actual_simulation(*args)
        data["reference_prob"] = data["true_prob"].copy()
        return data

    monkeypatch.setattr(audit, "simulate_population", no_diagnostic_difference)
    result = audit.run_baseline_audit(config, tmp_path, "cpu")
    empty = {"mean": None, "min": None, "max": None, "supported_seed_count": 0, "planned_seed_count": 2}
    for group in result["groups"].values():
        for model in group["models"].values():
            assert model["diagnostic_nll"] == empty and model["diagnostic_kl"] == empty
        for section in ("comparisons", "matched_comparisons"):
            for comparison in group[section].values():
                assert comparison["diagnostic_likelihood_gain"] == empty
                assert comparison["diagnostic_fidelity_loss"] == empty
        assert all(row["diagnostic_count"] == 0 and row["diagnostic_subject_count"] == 0 for row in group["per_seed"])
    reports = [read_json(tmp_path / f"seed-{seed}/report.json") for seed in config.replication_seeds]
    with pytest.raises(ValueError, match="Every planned replication"):
        audit.aggregate_audit(reports[:1], config.replication_seeds)
    damaged = copy.deepcopy(reports)
    damaged[0]["status"] = "incomplete"
    with pytest.raises(ValueError, match="Every planned replication"):
        audit.aggregate_audit(damaged, config.replication_seeds)
    damaged = copy.deepcopy(reports)
    damaged[0]["role"] = "calibration"
    with pytest.raises(ValueError, match="Every planned replication"):
        audit.aggregate_audit(damaged, config.replication_seeds)


def test_predeclared_matched_comparisons_use_matching_modes(tmp_path, config):
    summary = audit.run_baseline_audit(config, tmp_path, "cpu")
    for group in summary["groups"].values():
        for mode in ("population", "individual"):
            values = {key: [] for key in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain", "diagnostic_fidelity_loss")}
            for row in group["per_seed"]:
                solo = row["models"][f"solo_{mode}"]
                distilled = row["models"][f"distilled_{mode}"]
                expected = {"likelihood_gain": solo["nll"] - distilled["nll"],
                            "fidelity_loss": distilled["kl"] - solo["kl"],
                            "diagnostic_likelihood_gain": None if solo["diagnostic_nll"] is None else solo["diagnostic_nll"] - distilled["diagnostic_nll"],
                            "diagnostic_fidelity_loss": None if solo["diagnostic_kl"] is None else distilled["diagnostic_kl"] - solo["diagnostic_kl"]}
                assert row["matched_comparisons"][mode] == expected
                for metric, value in expected.items():
                    if value is not None:
                        values[metric].append(value)
            for metric, supported in values.items():
                aggregate = group["matched_comparisons"][mode][metric]
                assert aggregate["supported_seed_count"] == len(supported)
                assert aggregate["planned_seed_count"] == 2
                assert aggregate["mean"] == (pytest.approx(np.mean(supported)) if supported else None)
                assert aggregate["min"] == (min(supported) if supported else None)
                assert aggregate["max"] == (max(supported) if supported else None)
