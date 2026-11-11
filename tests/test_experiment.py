from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distilled_rnns.config import ExperimentConfig
from distilled_rnns.experiment import (resolve_device, run_experiment, save_environment_inventory,
                                      source_fingerprint, t4_training_cost_proxy)


@pytest.fixture
def tiny_config():
    return ExperimentConfig(n_subjects=10, n_trials=15, rare_fraction=0.2,
                            teacher_hidden=3, teacher_steps=2, student_steps=2,
                            eval_every=1, max_seconds_per_phase=120)


def test_pipeline_outputs_and_resume(tmp_path, tiny_config):
    first = run_experiment(tiny_config, tmp_path, "cpu")
    expected = {"report.json", "manifest.json", "training.json", "data.npz", "predictions.npz",
                "teacher.pt", "solo.pt", "distilled.pt", "quadrant.png", "environment-freeze.txt"}
    assert expected <= {p.name for p in tmp_path.iterdir()}
    assert first["status"] == "complete"
    assert first["timing"]["t4_equivalent_units_at_user_rate"] is None
    assert first["source_sha256"] == source_fingerprint()
    with np.load(tmp_path / "predictions.npz") as arrays:
        saved = {key: arrays[key].copy() for key in arrays.files}
    resumed = run_experiment(tiny_config, tmp_path, "cpu")
    assert resumed["per_subject"] == first["per_subject"]
    assert resumed["groups"] == first["groups"]
    with np.load(tmp_path / "predictions.npz") as arrays:
        for key, value in saved.items():
            np.testing.assert_array_equal(arrays[key], value)
    for phase in resumed["phases"].values():
        assert phase["completed_steps"] == 2
    sessions = json.loads((tmp_path / "manifest.json").read_text())["sessions"]
    assert len(sessions) == 2
    for session in sessions:
        inventory = session["package_inventory"]
        assert inventory["status"] == "recorded"
        saved_bytes = (tmp_path / inventory["path"]).read_bytes()
        assert hashlib.sha256(saved_bytes).hexdigest() == inventory["sha256"]
    with pytest.raises(ValueError, match="configuration or source"):
        run_experiment(replace(tiny_config, seed=99), tmp_path, "cpu")


def test_incomplete_phase_is_not_reported_as_complete(tmp_path, tiny_config, monkeypatch):
    import distilled_rnns.experiment as experiment
    monkeypatch.setattr(experiment, "fit_model", lambda *args, **kwargs: {
        "stopped_reason": "time_limit", "completed_steps": 0, "elapsed_seconds": 0.01,
    })
    result = run_experiment(tiny_config, tmp_path, "cpu")
    assert result["status"] == "incomplete"
    assert result["phase"] == "teacher"
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "predictions.npz").exists()


@pytest.mark.parametrize("settings", [
    {"n_subjects": 0}, {"n_trials": 2}, {"n_subjects": 2}, {"teacher_prop": 1.1},
    {"temperature": float("nan")}, {"seed": -1}, {"learning_rate": 0},
    {"max_seconds_per_phase": 0}, {"rare_fraction": 1}, {"student_hidden": True},
])
def test_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_unknown_config_key_fails(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text('{"typo_steps": 100}')
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_file(file)


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        resolve_device("cuda")
    assert str(resolve_device("auto")) == "cpu"


def test_t4_only_history_uses_supplied_rate():
    sessions = [
        {"environment": {"device": "cuda", "gpu": "Tesla T4"}},
        {"environment": {"device": "cuda:0", "gpu": "NVIDIA T4"}},
    ]
    result = t4_training_cost_proxy(7200, 1.07, sessions)
    assert result["t4_equivalent_units_at_user_rate"] == pytest.approx(2.14)
    assert "Every recorded session" in result["t4_cost_proxy_reason"]


@pytest.mark.parametrize("sessions", [
    [], [{}],
    [{"environment": {"device": "cpu", "gpu": None}}],
    [{"environment": {"device": "cpu", "gpu": None}},
     {"environment": {"device": "cuda", "gpu": "Tesla T4"}}],
    [{"environment": {"device": "cuda", "gpu": "Tesla T4"}},
     {"environment": {"device": "cuda", "gpu": "NVIDIA A100"}}],
    [{"environment": {"device": "cuda", "gpu": None}},
     {"environment": {"device": "cuda", "gpu": "Tesla T4"}}],
])
def test_non_t4_or_unattributed_history_has_no_cost_proxy(sessions):
    result = t4_training_cost_proxy(3600, 1.07, sessions)
    assert result["t4_equivalent_units_at_user_rate"] is None
    assert result["t4_cost_proxy_reason"].startswith("Unavailable:")


def test_environment_inventories_preserve_changes_between_sessions(tmp_path):
    first = save_environment_inventory(tmp_path, ["numpy==1.26.0", "torch==2.5.0"])
    original_bytes = (tmp_path / first["path"]).read_bytes()
    second = save_environment_inventory(tmp_path, ["numpy==2.0.0", "torch==2.5.0"])
    assert first["path"] != second["path"]
    assert (tmp_path / first["path"]).read_bytes() == original_bytes
    assert (tmp_path / second["path"]).read_bytes() != original_bytes
    for inventory in (first, second):
        assert hashlib.sha256((tmp_path / inventory["path"]).read_bytes()).hexdigest() == inventory["sha256"]
    assert (tmp_path / "environment-freeze.txt").read_bytes() == (tmp_path / second["path"]).read_bytes()
    # Inventory ordering and duplicate entries do not create another copy.
    repeated = save_environment_inventory(tmp_path, ["torch==2.5.0", "numpy==2.0.0", "torch==2.5.0"])
    assert repeated == second
    assert len(list((tmp_path / "environments").glob("*.txt"))) == 2


def test_corrupted_inventory_is_rejected_without_overwriting_it(tmp_path):
    inventory = save_environment_inventory(tmp_path, ["torch==2.5.0"])
    saved = tmp_path / inventory["path"]
    saved.write_text("changed\n")
    with pytest.raises(ValueError, match="content hash"):
        save_environment_inventory(tmp_path, ["torch==2.5.0"])
    assert saved.read_text() == "changed\n"


def test_legacy_session_inventory_is_explicitly_unavailable(tmp_path, tiny_config, monkeypatch):
    import distilled_rnns.experiment as experiment
    monkeypatch.setattr(experiment, "fit_model", lambda *args, **kwargs: {
        "stopped_reason": "time_limit", "completed_steps": 0, "elapsed_seconds": 0.01,
    })
    run_experiment(tiny_config, tmp_path, "cpu")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["sessions"][0]["package_inventory"]
    manifest_path.write_text(json.dumps(manifest))
    run_experiment(tiny_config, tmp_path, "cpu")
    sessions = json.loads(manifest_path.read_text())["sessions"]
    assert sessions[0]["package_inventory"]["status"] == "unavailable_legacy_session"
    assert "path" not in sessions[0]["package_inventory"]
    assert sessions[1]["package_inventory"]["status"] == "recorded"
