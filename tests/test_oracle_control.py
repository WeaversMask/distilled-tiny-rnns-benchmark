"""Independent numerical, leakage, selection, resume, and accounting checks."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distilled_rnns.config import ExperimentConfig
from distilled_rnns.evaluation import binary_kl, binary_nll, make_trial_masks
from distilled_rnns.models import BatchedStudentGRU, TeacherGRU
import distilled_rnns.oracle_control as control
import distilled_rnns.training as training


@pytest.fixture
def config():
    return control.OracleControlConfig(
        base=ExperimentConfig(n_subjects=6, n_trials=10, rare_fraction=1 / 3,
                              teacher_hidden=3, teacher_steps=3, student_steps=3,
                              seed=71, eval_every=1, max_seconds_per_phase=120),
        replication_seeds=(71, 81), max_seconds_per_invocation=120)


@pytest.fixture(autouse=True)
def skip_figures(monkeypatch):
    # Notebook smoke verification exercises actual figure creation separately.
    monkeypatch.setattr(control, "_plot_convergence", lambda *args: None)


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def predictions_from_state(config, data, state, teacher=False):
    model = (TeacherGRU(data["x"].shape[-1], config.teacher_hidden) if teacher else
             BatchedStudentGRU(config.n_subjects, data["x"].shape[-1], config.student_hidden))
    model.load_state_dict(state)
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(data["x"], dtype=torch.float32))).numpy()


def assert_nested_equal(left, right, *, atol=0):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=atol)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key], atol=atol)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b, atol=atol)
    else:
        assert left == right


def assert_checkpoints_equal(left, right, *, signature=True):
    a, b = checkpoint(left), checkpoint(right)
    keys = ["current_state", "best_state", "optimizer", "history", "completed_steps", "best_val", "best_step"]
    if signature:
        keys.append("signature")
    if "population_best_state" in a:
        keys += ["population_best_state", "population_best_val", "population_best_step"]
    for key in keys:
        assert_nested_equal(a[key], b[key])


@pytest.mark.parametrize("changes", [
    {"base": {}}, {"replication_seeds": None}, {"replication_seeds": ()},
    {"replication_seeds": (71, 71)}, {"replication_seeds": (71, 74)},
    {"replication_seeds": (True,)}, {"replication_seeds": (71.0,)},
    {"replication_seeds": (-1,)}, {"replication_seeds": (2 ** 32,)},
    {"replication_seeds": "71"}, {"solo_l2": True}, {"solo_l2": -1},
    {"solo_l2": float("nan")}, {"solo_l2": float("inf")}, {"solo_l2": "0.1"},
    {"max_seconds_per_invocation": True}, {"max_seconds_per_invocation": 0},
    {"max_seconds_per_invocation": float("inf")},
])
def test_invalid_protocol_values_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


@pytest.mark.parametrize("field,value", [("teacher_prop", 0), ("teacher_prop", .5), ("temperature", 2)])
def test_pure_targets_at_unit_temperature_required(config, field, value):
    with pytest.raises(ValueError, match="teacher_prop=1 and temperature=1"):
        replace(config, base=replace(config.base, **{field: value}))


def test_config_roundtrip_template_seed_and_nonoverlapping_offsets(tmp_path, config):
    assert config.base.seed in config.replication_seeds  # No calibration population.
    assert replace(config, replication_seeds=(75, 71)).replication_seeds == (75, 71)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert control.OracleControlConfig.from_file(path) == config
    defaults = control.OracleControlConfig()
    assert defaults.replication_seeds == (20261411, 20261421, 20261431, 20261441, 20261451)
    assert defaults.solo_l2 == .1
    assert (defaults.base.n_subjects, defaults.base.n_trials, defaults.base.rare_fraction) == (1000, 100, .02)
    assert (defaults.base.teacher_hidden, defaults.base.student_hidden) == (32, 2)
    assert (defaults.base.teacher_steps, defaults.base.student_steps, defaults.base.eval_every,
            defaults.base.patience, defaults.base.learning_rate) == (500, 4000, 10, 20, .01)


@pytest.mark.parametrize("value", [[], {"unknown": 1}, {"base": []}, {"base": {"unknown": 1}}, {"base": {"seed": True}}])
def test_strict_json_schema(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        control.OracleControlConfig.from_file(path)


def test_stable_training_only_logit_conversion():
    probabilities = np.array([[0., 1., .2, .7], [.25, .75, .3, .4]])
    mask = np.array([[True, True, False, False], [True, True, False, False]])
    actual = control.oracle_training_logits(probabilities, mask)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    # Independent float64 endpoint calculation, avoiding float32 1-epsilon=1.
    clipped = np.clip(probabilities[mask], 1e-12, 1 - 1e-12)
    expected = np.log(clipped / (1 - clipped)).astype(np.float32)
    np.testing.assert_array_equal(actual.numpy()[mask], expected)
    np.testing.assert_array_equal(actual.numpy()[~mask], np.zeros((~mask).sum()))
    changed = probabilities.copy()
    changed[~mask] = 1 - changed[~mask]
    torch.testing.assert_close(actual, control.oracle_training_logits(changed, mask), rtol=0, atol=0)


@pytest.mark.parametrize("probabilities,mask", [
    ([[.2, float("nan")]], [[True, False]]), ([[.2, float("inf")]], [[True, False]]),
    ([[.2, 1.1]], [[True, False]]), ([[.2, -.1]], [[True, False]]),
    ([[.2, .4]], [[False, False]]), ([[.2, .4]], [[1, 0]]),
    ([.2, .4], [True, False]), ([[.2, .4]], [[True]]),
])
def test_invalid_oracle_targets_rejected_even_off_train(probabilities, mask):
    with pytest.raises(ValueError):
        control.oracle_training_logits(probabilities, mask)


def test_real_smoke_freezes_before_training_and_predicts_from_learned_weights(tmp_path, config, monkeypatch):
    actual_fit = control.fit_model
    initial_states = {}

    def checked_fit(model, x, y, train, validation, **kwargs):
        folder = Path(kwargs["checkpoint_path"]).parent
        name = Path(kwargs["checkpoint_path"]).stem
        protocol = read_json(tmp_path / "protocol.json")
        frozen = read_json(folder / "selection.json")
        assert protocol["selection"] == frozen
        assert frozen["primary_mode"] == "population" and frozen["secondary_mode"] == "individual"
        if name != "teacher":
            previous = initial_states.setdefault(folder.name, copy.deepcopy(model.state_dict()))
            assert_nested_equal(previous, model.state_dict())
            assert kwargs["seed"] == int(folder.name.removeprefix("seed-")) + 3
            assert kwargs["track_population_selection"]
        if name in ("distilled", "oracle"):
            assert kwargs["teacher_prop"] == kwargs["temperature"] == 1
            assert kwargs["l2_strength"] == 0
            assert torch.count_nonzero(kwargs["teacher_logits"][~torch.as_tensor(train)]) == 0
        else:
            assert kwargs["teacher_prop"] == 0 and kwargs["teacher_logits"] is None
        return actual_fit(model, x, y, train, validation, **kwargs)

    monkeypatch.setattr(control, "fit_model", checked_fit)
    result = control.run_oracle_control(config, tmp_path, "cpu")
    assert result["status"] == "complete" and result["completed_seeds"] == [71, 81]
    assert result["pending_seeds"] == [] and "calibration" not in result
    assert not (tmp_path / "calibration").exists()
    protocol = read_json(tmp_path / "protocol.json")
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert digest == result["protocol_sha256"]
    assert result["primary_outcome"]["metric"] == "diagnostic_kl_gain"
    assert result["primary_outcome"]["group"] == "rare"
    assert result["timing"]["t4_equivalent_units_at_user_rate"] is None
    for seed in config.replication_seeds:
        folder = tmp_path / f"seed-{seed}"
        data, predictions = arrays(folder / "data.npz"), arrays(folder / "predictions.npz")
        candidates, targets = arrays(folder / "candidate_predictions.npz"), arrays(folder / "soft_training_logits.npz")
        report, phases = read_json(folder / "report.json"), read_json(folder / "training.json")
        assert report["status"] == "complete" and report["role"] == "replication"
        assert set(phases) == {"teacher", "solo", "distilled", "oracle"}
        assert phases["solo"]["l2_strength"] == .1
        masks = {key: data[key] for key in data if key.startswith("mask_")}
        assert set(masks) == {"mask_train", "mask_checkpoint", "mask_test"}
        assert all(mask.dtype == bool and np.all(mask.sum(1) > 0) for mask in masks.values())
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones(data["choice"].shape))
        expected_keys = {"teacher", "coinflip", "frequency"} | {f"{arm}_{mode}" for arm in ("solo", "distilled", "oracle") for mode in ("population", "individual")}
        assert set(candidates) == expected_keys
        assert set(predictions) == expected_keys | {"solo", "distilled", "oracle"}
        for arm in ("solo", "distilled", "oracle"):
            saved = checkpoint(folder / f"{arm}.pt")
            for mode, key in (("population", "population_best_state"), ("individual", "best_state")):
                expected = predictions_from_state(config.base, data, saved[key])
                np.testing.assert_array_equal(predictions[f"{arm}_{mode}"], expected)
                np.testing.assert_array_equal(candidates[f"{arm}_{mode}"], expected)
            np.testing.assert_array_equal(predictions[arm], predictions[f"{arm}_population"])
            per_person = (binary_nll(data["choice"], predictions[f"{arm}_individual"]) * data["mask_checkpoint"]).sum(1) / data["mask_checkpoint"].sum(1)
            # Stored BCE uses float32 logits; independent recomputation uses
            # float64 logs of rounded float32 probabilities (a few ulps differ).
            np.testing.assert_allclose(saved["best_val"].numpy(), per_person, rtol=0, atol=3e-7)
            population_nll = (binary_nll(data["choice"], predictions[f"{arm}_population"]) * data["mask_checkpoint"]).sum(1) / data["mask_checkpoint"].sum(1)
            assert saved["population_best_val"].item() == pytest.approx(population_nll.mean(), abs=3e-7)
            assert saved["population_best_step"] == phases[arm]["population_best_step"]
            current = predictions_from_state(config.base, data, saved["current_state"])
            current_nll = (binary_nll(data["choice"], current) * data["mask_checkpoint"]).sum(1) / data["mask_checkpoint"].sum(1)
            assert saved["history"][-1]["validation_nll"] == pytest.approx(current_nll.mean(), abs=3e-7)
        teacher = checkpoint(folder / "teacher.pt")
        np.testing.assert_array_equal(predictions["teacher"], predictions_from_state(config.base, data, teacher["best_state"], teacher=True))
        assert not np.allclose(predictions["oracle"], data["true_prob"])
        for logits in targets.values():
            np.testing.assert_array_equal(logits[~data["mask_train"]], np.zeros((~data["mask_train"]).sum()))
        oracle_p = torch.sigmoid(torch.from_numpy(targets["oracle"])).numpy()
        np.testing.assert_allclose(oracle_p[data["mask_train"]], data["true_prob"][data["mask_train"]], rtol=0, atol=1e-7)
        teacher_p = torch.sigmoid(torch.from_numpy(targets["distilled"])).numpy()
        np.testing.assert_array_equal(teacher_p[data["mask_train"]], predictions["teacher"][data["mask_train"]])
        for name, probability in predictions.items():
            raw_nll = (binary_nll(data["choice"], probability) * data["mask_checkpoint"]).sum(1) / data["mask_checkpoint"].sum(1)
            assert report["checkpoint_scores"][name] == pytest.approx(raw_nll.mean(), abs=1e-12)
        manifest = read_json(folder / "manifest.json")
        assert manifest["identity"]["protocol_sha256"] == digest
        assert manifest["identity"]["role"] == "replication"
        inventory = manifest["sessions"][0]["package_inventory"]
        assert hashlib.sha256((folder / inventory["path"]).read_bytes()).hexdigest() == inventory["sha256"]
        for row in [*report["per_subject"], *report["groups"].values()]:
            assert "likelihood_gain" not in row and "fidelity_loss" not in row


def test_heldout_oracle_changes_neither_fit_signature_checkpoints_nor_predictions(tmp_path, config, monkeypatch):
    baseline = control.run_oracle_control(config, tmp_path / "before", "cpu")
    actual_simulation = control.simulate_population

    def changed_heldout(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        train = make_trial_masks(n, t, seed + 1)["train"]
        data["true_prob"][~train] = 1 - data["true_prob"][~train]
        # Diagnostic and group data likewise have no role in fitting/selection.
        data["reference_prob"] = np.full_like(data["reference_prob"], .99)
        data["rare"] = ~data["rare"]
        return data

    monkeypatch.setattr(control, "simulate_population", changed_heldout)
    changed = control.run_oracle_control(config, tmp_path / "after", "cpu")
    assert baseline["groups"] != changed["groups"]
    assert baseline["selection"] == changed["selection"]
    for seed in config.replication_seeds:
        left, right = (tmp_path / root / f"seed-{seed}" for root in ("before", "after"))
        for arm in control.ARMS:
            assert_checkpoints_equal(left / f"{arm}.pt", right / f"{arm}.pt")
        for filename in ("predictions.npz", "soft_training_logits.npz"):
            before, after = arrays(left / filename), arrays(right / filename)
            for name in before:
                np.testing.assert_array_equal(before[name], after[name])
        assert read_json(left / "report.json")["checkpoint_scores"] == read_json(right / "report.json")["checkpoint_scores"]
    # The same held-out perturbation also permits resumption of the original
    # run folder: full checkpoint signatures are unaffected, not just gradients.
    resumed = control.run_oracle_control(config, tmp_path / "before", "cpu")
    assert resumed["status"] == "complete" and resumed["groups"] == changed["groups"]


def test_training_oracle_change_affects_only_oracle_fit(tmp_path, config, monkeypatch):
    control.run_oracle_control(config, tmp_path / "before", "cpu")
    actual_simulation = control.simulate_population

    def changed_training(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        mask = make_trial_masks(n, t, seed + 1)["train"]
        data["true_prob"][mask] = 1 - data["true_prob"][mask]
        return data

    monkeypatch.setattr(control, "simulate_population", changed_training)
    control.run_oracle_control(config, tmp_path / "after", "cpu")
    for seed in config.replication_seeds:
        left, right = (tmp_path / root / f"seed-{seed}" for root in ("before", "after"))
        for arm in ("teacher", "solo", "distilled"):
            assert_checkpoints_equal(left / f"{arm}.pt", right / f"{arm}.pt")
        before, after = checkpoint(left / "oracle.pt"), checkpoint(right / "oracle.pt")
        assert before["signature"] != after["signature"]
        assert any(not torch.equal(before["current_state"][key], after["current_state"][key]) for key in before["current_state"])
    with pytest.raises(ValueError, match="checkpoint does not match"):
        control.run_oracle_control(config, tmp_path / "before", "cpu")


def test_same_targets_and_initialization_give_same_soft_student_trajectory(tmp_path, config, monkeypatch):
    single = replace(config, replication_seeds=(71,))
    actual_fit = control.fit_model
    teacher_probabilities = {}

    def captured_teacher(*args, **kwargs):
        result = actual_fit(*args, **kwargs)
        if Path(kwargs["checkpoint_path"]).stem == "teacher":
            with torch.no_grad():
                # Double sigmoid/logit round-trip returns the teacher logits in
                # float32, unlike probability rounding before inversion.
                teacher_probabilities["p"] = torch.sigmoid(args[0](args[1]).double()).numpy()
        return result

    monkeypatch.setattr(control, "fit_model", captured_teacher)
    control.run_oracle_control(single, tmp_path / "reference", "cpu")
    actual_simulation = control.simulate_population

    def replace_train_targets(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        data["true_prob"] = data["true_prob"].astype(np.float64)
        mask = make_trial_masks(n, t, seed + 1)["train"]
        data["true_prob"][mask] = teacher_probabilities["p"][mask]
        return data

    monkeypatch.setattr(control, "simulate_population", replace_train_targets)
    control.run_oracle_control(single, tmp_path / "equal", "cpu")
    folder = tmp_path / "equal/seed-71"
    logits = arrays(folder / "soft_training_logits.npz")
    np.testing.assert_array_equal(logits["distilled"], logits["oracle"])
    assert_checkpoints_equal(folder / "distilled.pt", folder / "oracle.pt")
    saved_predictions = arrays(folder / "predictions.npz")
    for mode in ("population", "individual"):
        np.testing.assert_array_equal(saved_predictions[f"distilled_{mode}"], saved_predictions[f"oracle_{mode}"])


def test_observed_checkpoint_nll_selects_both_snapshots_not_true_policy_kl(tmp_path, config, monkeypatch):
    single = replace(config, base=replace(config.base, student_steps=8, learning_rate=.04), replication_seeds=(71,))
    actual_save, actual_simulation = training._atomic_save, control.simulate_population
    snapshots = {}

    def capture(payload, path):
        snapshots.setdefault(path.stem, {})[payload["completed_steps"]] = copy.deepcopy(payload["current_state"])
        return actual_save(payload, path)

    def opposed_checkpoint_policies(n, t, rare_fraction, seed):
        data = actual_simulation(n, t, rare_fraction, seed)
        masks = make_trial_masks(n, t, seed + 1)
        # Deliberately opposing criteria expose accidental true-policy selection.
        data["choice"][masks["val"]] = 0
        data["true_prob"][masks["val"]] = .999
        data["true_prob"][masks["train"]] = .01
        return data

    monkeypatch.setattr(training, "_atomic_save", capture)
    monkeypatch.setattr(control, "simulate_population", opposed_checkpoint_policies)
    control.run_oracle_control(single, tmp_path, "cpu")
    data = arrays(tmp_path / "seed-71/data.npz")
    mask = data["mask_checkpoint"]
    distinct_kl_selection = False
    for arm in ("solo", "distilled", "oracle"):
        steps = sorted(snapshots[arm])
        observed, oracle_kl = [], []
        for step in steps:
            predicted = predictions_from_state(single.base, data, snapshots[arm][step])
            observed.append((binary_nll(data["choice"], predicted) * mask).sum(1) / mask.sum(1))
            oracle_kl.append((binary_kl(data["true_prob"], predicted) * mask).sum(1) / mask.sum(1))
        observed, oracle_kl = np.asarray(observed), np.asarray(oracle_kl)
        saved = checkpoint(tmp_path / f"seed-71/{arm}.pt")
        expected_individual = np.asarray(steps)[observed.argmin(axis=0)]
        expected_population = steps[observed.mean(axis=1).argmin()]
        np.testing.assert_array_equal(saved["best_step"].numpy(), expected_individual)
        assert saved["population_best_step"] == expected_population
        distinct_kl_selection |= expected_population != steps[oracle_kl.mean(axis=1).argmin()]
        for subject, best_step in enumerate(expected_individual):
            for name in saved["best_state"]:
                torch.testing.assert_close(saved["best_state"][name][subject], snapshots[arm][best_step][name][subject], rtol=0, atol=0)
        assert_nested_equal(saved["population_best_state"], snapshots[arm][expected_population])
    assert distinct_kl_selection, "Fixture must distinguish observed-choice and true-policy selection"


def test_resume_identity_guards_and_idempotent_predictions(tmp_path, config, monkeypatch):
    original = control.run_oracle_control(config, tmp_path, "cpu")
    expected = {seed: arrays(tmp_path / f"seed-{seed}/predictions.npz") for seed in config.replication_seeds}
    resumed = control.run_oracle_control(config, tmp_path, "cpu")
    assert resumed["groups"] == original["groups"]
    for seed, before in expected.items():
        after = arrays(tmp_path / f"seed-{seed}/predictions.npz")
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    with pytest.raises(ValueError, match="configuration or source"):
        control.run_oracle_control(replace(config, solo_l2=0), tmp_path, "cpu")
    with monkeypatch.context() as patch:
        patch.setattr(control, "source_fingerprint", lambda: "changed")
        with pytest.raises(ValueError, match="configuration or source"):
            control.run_oracle_control(config, tmp_path, "cpu")
    manifest_path = tmp_path / "seed-71/manifest.json"
    manifest = read_json(manifest_path)
    manifest["identity"]["role"] = "calibration"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Dataset configuration or protocol"):
        control.run_oracle_control(config, tmp_path, "cpu")


def test_missing_provenance_and_tampered_selection_rejected(tmp_path, config):
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="without a protocol"):
        control.run_oracle_control(config, orphan, "cpu")
    control.run_oracle_control(config, tmp_path / "valid", "cpu")
    path = tmp_path / "valid/seed-71/selection.json"
    selected = read_json(path)
    selected["primary_mode"] = "individual"
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="Frozen selection differs"):
        control.run_oracle_control(config, tmp_path / "valid", "cpu")


def test_four_fits_per_seed_count_once_and_no_calibration(tmp_path, config, monkeypatch):
    actual_fit, calls = control.fit_model, []

    def counted_fit(*args, **kwargs):
        calls.append(Path(kwargs["checkpoint_path"]))
        result = actual_fit(*args, **kwargs)
        result["elapsed_seconds"] = float(len(calls))
        return result

    monkeypatch.setattr(control, "fit_model", counted_fit)
    result = control.run_oracle_control(config, tmp_path, "cpu")
    assert len(calls) == len(set(calls)) == 4 * len(config.replication_seeds)
    assert all(path.parent.name in ("seed-71", "seed-81") for path in calls)
    assert result["timing"]["cumulative_training_seconds"] == sum(range(1, len(calls) + 1))
    for seed in config.replication_seeds:
        phases = read_json(tmp_path / f"seed-{seed}/training.json")
        report = read_json(tmp_path / f"seed-{seed}/report.json")
        assert report["timing"]["cumulative_training_seconds"] == sum(phase["elapsed_seconds"] for phase in phases.values())


def test_interrupted_resume_restores_optimizer_and_both_selected_states(tmp_path, config, monkeypatch):
    actual_fit, actual_objective = control.fit_model, training.masked_objective
    active = {"oracle": False, "objective_calls": 0}
    clock = {"value": 0.0}

    def controlled_fit(*args, **kwargs):
        active["oracle"] = Path(kwargs["checkpoint_path"]).stem == "oracle"
        return actual_fit(*args, **kwargs)

    def interrupted_objective(*args, **kwargs):
        result = actual_objective(*args, **kwargs)
        if active["oracle"]:
            active["objective_calls"] += 1
            if active["objective_calls"] == 1:
                clock["value"] = 1000.0
        return result

    with monkeypatch.context() as patch:
        # time is one module object: the jump also exhausts the study budget,
        # only after the first full oracle optimizer step has been saved.
        patch.setattr(control, "fit_model", controlled_fit)
        patch.setattr(training, "masked_objective", interrupted_objective)
        patch.setattr(training.time, "perf_counter", lambda: clock["value"])
        partial = control.run_oracle_control(config, tmp_path / "resume", "cpu")
    assert partial["status"] == "incomplete" and "groups" not in partial
    saved = checkpoint(tmp_path / "resume/seed-71/oracle.pt")
    assert saved["completed_steps"] == 1 and saved["optimizer"]["state"]
    assert "population_best_state" in saved and "best_state" in saved
    assert not (tmp_path / "resume/seed-71/report.json").exists()
    resumed = control.run_oracle_control(config, tmp_path / "resume", "cpu")
    reference = control.run_oracle_control(config, tmp_path / "reference", "cpu")
    assert resumed["status"] == "complete" and resumed["groups"] == reference["groups"]
    for seed in config.replication_seeds:
        for arm in control.ARMS:
            assert_checkpoints_equal(tmp_path / f"resume/seed-{seed}/{arm}.pt", tmp_path / f"reference/seed-{seed}/{arm}.pt")


def test_no_aggregate_until_final_planned_replication_completes(tmp_path, config, monkeypatch):
    five = replace(config, replication_seeds=(71, 81, 91, 101, 111))
    actual_dataset = control._dataset

    def stop_fifth(study, base, folder, *args):
        if base.seed == 111:
            return {"status": "incomplete", "phase": "teacher"}
        return actual_dataset(study, base, folder, *args)

    monkeypatch.setattr(control, "_dataset", stop_fifth)
    result = control.run_oracle_control(five, tmp_path, "cpu")
    assert result["status"] == "incomplete" and "groups" not in result
    assert result["completed_seeds"] == [71, 81, 91, 101] and result["pending_seeds"] == [111]
    assert "groups" not in read_json(tmp_path / "summary.json")


def test_raw_metrics_equal_person_then_equal_seed_and_all_gains_favor_right(tmp_path, config):
    result = control.run_oracle_control(config, tmp_path, "cpu")
    metric_names = ("nll", "kl", "diagnostic_nll", "diagnostic_kl")
    for group_name, group in result["groups"].items():
        for row in group["per_seed"]:
            folder = tmp_path / f"seed-{row['seed']}"
            data, predictions = arrays(folder / "data.npz"), arrays(folder / "predictions.npz")
            test = data["mask_test"]
            diagnostic = test & (np.abs(data["true_prob"] - data["reference_prob"]) > config.base.diagnostic_threshold)
            group_mask = data["rare"] if group_name == "rare" else ~data["rare"]
            assert row["subject_count"] == group_mask.sum()
            assert row["diagnostic_count"] == diagnostic[group_mask].sum()
            assert row["diagnostic_subject_count"] == np.any(diagnostic[group_mask], axis=1).sum()
            for name, prediction in predictions.items():
                for loss_name, loss in (("nll", binary_nll(data["choice"], prediction)), ("kl", binary_kl(data["true_prob"], prediction))):
                    for prefix, mask in (("", test), ("diagnostic_", diagnostic)):
                        values = [loss[i, mask[i]].mean() for i in np.flatnonzero(group_mask) if mask[i].any()]
                        actual = row["models"][name][prefix + loss_name]
                        assert actual == (pytest.approx(np.mean(values), abs=1e-12) if values else None)
            pairs = {
                "target_comparisons": {mode: (f"distilled_{mode}", f"oracle_{mode}") for mode in ("population", "individual")},
                "contextual_comparisons": {"distilled_vs_solo": ("solo", "distilled"), "oracle_vs_solo": ("solo", "oracle"), "oracle_vs_teacher": ("teacher", "oracle")},
            }
            for section, comparisons in pairs.items():
                for label, (left, right) in comparisons.items():
                    for metric in metric_names:
                        a, b = row["models"][left][metric], row["models"][right][metric]
                        assert row[section][label][f"{metric}_gain"] == (None if a is None or b is None else a - b)
        for section in ("target_comparisons", "contextual_comparisons", "models"):
            for label, metrics in group[section].items():
                for metric, aggregate in metrics.items():
                    supported = [row[section][label][metric] for row in group["per_seed"] if row[section][label][metric] is not None]
                    assert aggregate["mean"] == (pytest.approx(np.mean(supported)) if supported else None)
                    assert aggregate["min"] == (min(supported) if supported else None)
                    assert aggregate["max"] == (max(supported) if supported else None)
                    assert aggregate["supported_seed_count"] == len(supported)
                    assert aggregate["planned_seed_count"] == len(config.replication_seeds)
                    if section != "models":
                        assert aggregate["positive_seed_count"] == sum(v > 0 for v in supported)
                        assert aggregate["negative_seed_count"] == sum(v < 0 for v in supported)
                        assert aggregate["tied_seed_count"] == sum(v == 0 for v in supported)


def test_null_diagnostic_support_and_aggregation_guard(tmp_path, config, monkeypatch):
    actual_simulation = control.simulate_population

    def no_diagnostics(*args):
        data = actual_simulation(*args)
        data["reference_prob"] = data["true_prob"].copy()
        return data

    monkeypatch.setattr(control, "simulate_population", no_diagnostics)
    result = control.run_oracle_control(config, tmp_path, "cpu")
    empty = {"mean": None, "min": None, "max": None, "supported_seed_count": 0, "planned_seed_count": 2}
    empty_gain = {**empty, "positive_seed_count": 0, "negative_seed_count": 0, "tied_seed_count": 0}
    for group in result["groups"].values():
        for model in group["models"].values():
            assert model["diagnostic_nll"] == empty and model["diagnostic_kl"] == empty
        for section in ("target_comparisons", "contextual_comparisons"):
            for comparison in group[section].values():
                assert comparison["diagnostic_nll_gain"] == empty_gain
                assert comparison["diagnostic_kl_gain"] == empty_gain
        assert all(row["diagnostic_count"] == row["diagnostic_subject_count"] == 0 for row in group["per_seed"])
    reports = [read_json(tmp_path / f"seed-{seed}/report.json") for seed in config.replication_seeds]
    for invalid in (reports[:1], list(reversed(reports)), []):
        with pytest.raises(ValueError, match="Every planned replication"):
            control.aggregate_oracle_control(invalid, config.replication_seeds)
    for key, value in (("status", "incomplete"), ("role", "calibration")):
        damaged = copy.deepcopy(reports)
        damaged[0][key] = value
        with pytest.raises(ValueError, match="Every planned replication"):
            control.aggregate_oracle_control(damaged, config.replication_seeds)
