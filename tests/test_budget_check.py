"""Bounded copied-checkpoint continuation, parent protection, and paired scores."""
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import distilled_rnns.budget_check as budget
import distilled_rnns.oracle_control as control
import distilled_rnns.training as training
from distilled_rnns.config import ExperimentConfig


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def assert_trajectory_equal(a, b):
    for key in ("signature", "current_state", "optimizer", "best_state", "best_val", "best_step",
                "population_best_state", "population_best_val", "population_best_step", "history",
                "rng_state", "cuda_rng_states", "completed_steps", "stale", "early_stopped"):
        assert_nested_equal(a[key], b[key])


@pytest.fixture(autouse=True)
def skip_figures(monkeypatch):
    monkeypatch.setattr(control, "_plot_convergence", lambda *args: None)
    monkeypatch.setattr(budget, "_plot_convergence", lambda *args: None)


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("budget-parent").resolve()
    parent_config = control.OracleControlConfig(base=ExperimentConfig(
        n_subjects=6, n_trials=10, rare_fraction=1 / 3, seed=71, teacher_hidden=3,
        teacher_steps=2, student_steps=4, eval_every=1, max_seconds_per_phase=120),
        replication_seeds=(71, 81), max_seconds_per_invocation=120)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(control, "_plot_convergence", lambda *args: None)
        assert control.run_oracle_control(parent_config, folder, "cpu")["status"] == "complete"
    config = budget.BudgetCheckConfig(parent_config, budget.source_fingerprint(), 8,
                                     max_seconds_per_phase=120, max_seconds_per_invocation=120)
    return folder, config


@pytest.fixture
def parent_copy(parent_run, tmp_path):
    parent, config = parent_run
    target = tmp_path.resolve() / "parent"
    shutil.copytree(parent, target)
    return target, config


def test_declared_production_and_smoke_configuration():
    root = Path(__file__).resolve().parents[1]
    production = budget.BudgetCheckConfig.from_file(root / "configs/budget_check.json")
    original = control.OracleControlConfig.from_file(root / "configs/oracle_control.json")
    assert production.parent_config == original
    assert production.expected_parent_source_sha256 == budget.ORIGINAL_SOURCE_SHA256
    assert (original.base.student_steps, production.student_steps) == (4000, 8000)
    assert original.replication_seeds == (20261411, 20261421, 20261431, 20261441, 20261451)
    assert production.max_seconds_per_phase == 600 and production.max_seconds_per_invocation == 1800
    smoke = budget.BudgetCheckConfig.from_file(root / "configs/budget_check_smoke.json")
    assert (smoke.parent_config.base.student_steps, smoke.student_steps) == (4, 8)
    assert smoke.parent_config.base.eval_every == 1
    assert budget.legacy_source_fingerprint() == budget.ORIGINAL_SOURCE_SHA256


@pytest.mark.parametrize("requested,selected", [("cpu", "cpu"), ("cuda", "cuda"), ("auto", "cuda")])
def test_parent_audit_uses_selected_run_device_before_any_child_write(parent_run, tmp_path, monkeypatch, requested, selected):
    """Even on a CPU host, GPU selection must reach the read-only audit."""
    parent, config = parent_run
    child = tmp_path.resolve() / "child"
    calls = []

    class AuditReached(RuntimeError):
        pass

    def resolved(value):
        assert value == requested
        return torch.device(selected)

    def audit(config_arg, parent_arg, *, device):
        assert config_arg is config and parent_arg == parent
        calls.append(device)
        raise AuditReached("stop before any continuation")

    monkeypatch.setattr(budget, "resolve_device", resolved)
    monkeypatch.setattr(budget, "validate_parent", audit)
    monkeypatch.setattr(budget, "fit_model", lambda *args, **kwargs: pytest.fail("audit gate trained"))
    with pytest.raises(AuditReached):
        budget.run_budget_check(config, parent, child, requested)
    assert calls == [selected]
    assert not child.exists()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA; exercises real GPU parent reconstruction in Colab"))])
def test_parent_reconstruction_device_preserves_cpu_identity_audit_and_rejects_changed_predictions(
        parent_run, tmp_path, monkeypatch, device):
    """Audit real device-produced artifacts without moving signature/Adam checks."""
    parent, config = parent_run
    target = tmp_path.resolve() / "parent"
    if device == "cuda":
        assert control.run_oracle_control(config.parent_config, target, device)["status"] == "complete"
    else:
        shutil.copytree(parent, target)
    before = budget._file_hashes(target)
    signatures, optimizers, forwards = [], [], []
    original_signature = budget._signature
    original_load_optimizer = torch.optim.Adam.load_state_dict

    def cpu_signature(model, settings, tensors):
        assert all(value.device.type == "cpu" for value in model.state_dict().values())
        assert all(value is None or value.device.type == "cpu" for value in tensors)
        signatures.append(type(model).__name__)
        return original_signature(model, settings, tensors)

    def cpu_optimizer(optimizer, saved):
        assert all(parameter.device.type == "cpu" for group in optimizer.param_groups for parameter in group["params"])
        assert all(value.device.type == "cpu" for state in saved["state"].values()
                   for value in state.values() if isinstance(value, torch.Tensor))
        result = original_load_optimizer(optimizer, saved)
        assert all(value.device.type == "cpu" for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor))
        optimizers.append(True)
        return result

    def forward_on_selected(original_forward):
        def checked(model, x):
            assert x.device.type == device
            assert all(value.device.type == device for value in model.state_dict().values())
            forwards.append(type(model).__name__)
            return original_forward(model, x)
        return checked

    monkeypatch.setattr(budget, "_signature", cpu_signature)
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", cpu_optimizer)
    for model_class in (budget.TeacherGRU, budget.BatchedStudentGRU):
        monkeypatch.setattr(model_class, "forward", forward_on_selected(model_class.forward))
    audit = budget.validate_parent(config, target, device=device)
    assert audit["file_sha256"] == before == budget._file_hashes(target)
    assert len(signatures) == len(optimizers) == 4 * len(config.parent_config.replication_seeds)
    assert set(forwards) == {"TeacherGRU", "BatchedStudentGRU"}

    # Device-aware reconstruction retains the original strict tolerance. Both
    # copies of the selected predictions must reject changed data before fit.
    assert budget.PROBABILITY_ATOL == 5e-6 and budget.NLL_ATOL == 5e-7
    monkeypatch.setattr(budget, "fit_model", lambda *args, **kwargs: pytest.fail("corrupt parent trained"))
    for filename in ("candidate_predictions.npz", "predictions.npz"):
        path = target / f"seed-{config.parent_config.replication_seeds[0]}" / filename
        original_bytes = path.read_bytes()
        arrays = budget._arrays(path)
        value = arrays["oracle_individual"][0, 0]
        arrays["oracle_individual"][0, 0] += -.05 if value > .5 else .05
        budget._save_npz(path, **arrays)
        with pytest.raises(ValueError, match="Nonreproducible selected predictions: oracle_individual"):
            budget.run_budget_check(config, target, tmp_path.resolve() / "child", device)
        assert not (tmp_path.resolve() / "child").exists()
        path.write_bytes(original_bytes)
    assert budget._file_hashes(target) == before


@pytest.mark.parametrize("changes", [
    {"parent_config": {}}, {"expected_parent_source_sha256": "x" * 64},
    {"student_steps": True}, {"student_steps": 4}, {"student_steps": 3}, {"student_steps": 8.0},
    {"max_seconds_per_phase": 0}, {"max_seconds_per_phase": True},
    {"max_seconds_per_invocation": float("inf")}, {"max_seconds_per_phase": float("nan")},
])
def test_invalid_budget_config(parent_run, changes):
    with pytest.raises(ValueError):
        replace(parent_run[1], **changes)


@pytest.mark.parametrize("value", [[], {}, {"unknown": 1}, {"parent_config": []},
                                  {"parent_config": {"unknown": 1}}, {"parent_config": {"base": []}}])
def test_strict_config_json(tmp_path, value):
    path = tmp_path / "config.json"
    write_json(path, value)
    with pytest.raises(ValueError):
        budget.BudgetCheckConfig.from_file(path)


def test_four_to_eight_matches_uninterrupted_with_inherited_selection_and_fixed_context(parent_run, tmp_path, monkeypatch):
    parent, config = parent_run
    before = budget._file_hashes(parent)
    output = tmp_path.resolve() / "child"
    calls = []
    actual_fit = budget.fit_model

    def checked_fit(*args, **kwargs):
        path = Path(kwargs["checkpoint_path"])
        calls.append((path.parent.name, path.stem))
        assert path.stem in budget.EXTENDED_ARMS
        inherited = budget._checkpoint(parent / path.parent.name / path.name)
        assert_trajectory_equal(budget._checkpoint(path), inherited)
        assert kwargs["steps"] == 8 and kwargs["teacher_prop"] == kwargs["temperature"] == 1
        assert kwargs["l2_strength"] == 0 and kwargs["track_population_selection"]
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(budget, "fit_model", checked_fit)
    result = budget.run_budget_check(config, parent, output, "cpu")
    assert result["status"] == "complete"
    assert calls == [(f"seed-{seed}", arm) for seed in config.parent_config.replication_seeds for arm in budget.EXTENDED_ARMS]
    assert result["selection"]["primary_mode"] == "population"
    assert result["selection"]["secondary_mode"] == "individual"
    assert result["timing"]["t4_equivalent_units_at_user_rate"] is None
    incremental = 0
    for seed in config.parent_config.replication_seeds:
        base = replace(config.parent_config.base, seed=seed)
        data = budget._arrays(parent / f"seed-{seed}/data.npz")
        soft = budget._arrays(parent / f"seed-{seed}/soft_training_logits.npz")
        old_predictions = budget._arrays(parent / f"seed-{seed}/predictions.npz")
        new_predictions = budget._arrays(output / f"seed-{seed}/predictions.npz")
        for name in ("teacher", "solo", "solo_population", "solo_individual", "coinflip", "frequency"):
            np.testing.assert_array_equal(old_predictions[name], new_predictions[name])
        for arm in budget.EXTENDED_ARMS:
            torch.manual_seed(seed + 3)
            model = budget._model(base, data["x"].shape[-1], arm)
            reference = tmp_path.resolve() / f"uninterrupted-{seed}-{arm}.pt"
            training.fit_model(model, data["x"], data["choice"], data["mask_train"], data["mask_checkpoint"],
                steps=8, lr=base.learning_rate, seed=seed + 3, eval_every=1, patience=base.patience,
                checkpoint_path=reference, teacher_logits=soft[arm], teacher_prop=1., temperature=1.,
                track_population_selection=True)
            saved = budget._checkpoint(output / f"seed-{seed}/{arm}.pt")
            inherited = budget._checkpoint(parent / f"seed-{seed}/{arm}.pt")
            assert_trajectory_equal(saved, budget._checkpoint(reference))
            assert saved["history"][:5] == inherited["history"]
            assert saved["population_best_val"] <= inherited["population_best_val"]
            assert torch.all(saved["best_val"] <= inherited["best_val"])
            incremental += saved["elapsed_seconds"] - inherited["elapsed_seconds"]
    assert result["timing"]["incremental_training_seconds"] == pytest.approx(incremental)
    assert budget._file_hashes(parent) == before
    child_before = budget._file_hashes(output)
    monkeypatch.setattr(budget, "fit_model", lambda *args, **kwargs: pytest.fail("completed arm retrained"))
    again = budget.run_budget_check(config, parent, output, "cpu")
    assert again == result
    assert budget._file_hashes(output) == child_before
    assert budget._file_hashes(parent) == before


def test_time_limit_resume_retains_progress_and_counts_time_once(parent_run, tmp_path, monkeypatch):
    parent, config = parent_run
    child, direct = (tmp_path.resolve() / name for name in ("resumed", "direct"))
    actual_fit = budget.fit_model
    first = True

    def interrupted_fit(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            ticks = itertools.count(1)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(training, "time", SimpleNamespace(perf_counter=lambda: next(ticks) * .01))
                kwargs["max_seconds"] = .025
                result = actual_fit(*args, **kwargs)
                assert result["stopped_reason"] == "time_limit" and result["completed_steps"] == 5
                return result
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(budget, "fit_model", interrupted_fit)
    stopped = budget.run_budget_check(config, parent, child, "cpu")
    assert stopped["status"] == "incomplete" and "groups" not in stopped and "timing" not in stopped
    checkpoint_path = child / "seed-71/distilled.pt"
    assert budget._checkpoint(checkpoint_path)["completed_steps"] == 5
    resumed = budget.run_budget_check(config, parent, child, "cpu")
    uninterrupted = budget.run_budget_check(config, parent, direct, "cpu")
    assert resumed["groups"] == uninterrupted["groups"]
    assert resumed["status"] == "complete"
    expected_seconds = 0
    for seed in config.parent_config.replication_seeds:
        for arm in budget.EXTENDED_ARMS:
            saved = budget._checkpoint(child / f"seed-{seed}/{arm}.pt")
            assert_trajectory_equal(saved, budget._checkpoint(direct / f"seed-{seed}/{arm}.pt"))
            expected_seconds += saved["elapsed_seconds"] - budget._checkpoint(parent / f"seed-{seed}/{arm}.pt")["elapsed_seconds"]
    assert resumed["timing"]["incremental_training_seconds"] == pytest.approx(expected_seconds)
    assert len(read_json(child / "seed-71/manifest.json")["sessions"]) == 2


def test_completed_first_arm_is_not_saved_or_retrained_when_second_arm_resumes(parent_run, tmp_path, monkeypatch):
    parent, config = parent_run
    child = tmp_path.resolve() / "child"
    actual_fit = budget.fit_model
    interrupted = False
    calls = []

    def stop_second_arm(*args, **kwargs):
        nonlocal interrupted
        path = Path(kwargs["checkpoint_path"])
        calls.append((path.parent.name, path.stem))
        if path.stem == "oracle" and not interrupted:
            interrupted = True
            ticks = itertools.count(1)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(training, "time", SimpleNamespace(perf_counter=lambda: next(ticks) * .01))
                kwargs["max_seconds"] = .025
                return actual_fit(*args, **kwargs)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(budget, "fit_model", stop_second_arm)
    assert budget.run_budget_check(config, parent, child, "cpu")["status"] == "incomplete"
    path = child / "seed-71/distilled.pt"
    before = path.read_bytes()
    assert budget._checkpoint(path)["completed_steps"] == 8
    calls.clear()
    assert budget.run_budget_check(config, parent, child, "cpu")["status"] == "complete"
    assert ("seed-71", "distilled") not in calls
    assert path.read_bytes() == before


@pytest.mark.parametrize("environment,is_t4", [({"device": "cuda", "gpu": "Tesla T4"}, True),
                                           ({"device": "cuda", "gpu": "unknown"}, False)])
def test_cost_proxy_uses_child_sessions_and_only_incremental_checkpoint_time(parent_run, tmp_path, monkeypatch, environment, is_t4):
    parent, config = parent_run
    # Attribution is simulated in this CPU-only accounting test; this is not
    # measured GPU performance. Parent sessions remain their genuine CPU run.
    monkeypatch.setattr(budget, "environment_info", lambda device: environment)
    result = budget.run_budget_check(config, parent, tmp_path.resolve() / "child", "cpu")
    timing = result["timing"]
    if is_t4:
        assert timing["t4_equivalent_units_at_user_rate"] == pytest.approx(
            timing["incremental_training_seconds"] / 3600 * config.parent_config.base.t4_units_per_hour)
    else:
        assert timing["t4_equivalent_units_at_user_rate"] is None


@pytest.mark.parametrize("where", ["snapshot", "second_student"])
def test_interrupted_copy_recovers_without_installing_partial_artifacts(parent_run, tmp_path, monkeypatch, where):
    parent, config = parent_run
    child = tmp_path.resolve() / "child"
    copyfile = shutil.copyfile
    triggered = False

    def broken_copy(source, target):
        nonlocal triggered
        matches = ("parent_snapshot" in str(target) and Path(source).name == "data.npz") if where == "snapshot" else (
            "parent_snapshot" not in str(target) and Path(source).name == "oracle.pt")
        if matches and not triggered:
            triggered = True
            Path(target).write_bytes(b"interrupted copy")
            raise OSError("simulated interrupted copy")
        return copyfile(source, target)

    monkeypatch.setattr(budget.shutil, "copyfile", broken_copy)
    with pytest.raises(OSError, match="simulated"):
        budget.run_budget_check(config, parent, child, "cpu")
    assert triggered and not list(child.rglob("*.copying"))
    if where == "second_student":
        assert not (child / "seed-71/oracle.pt").exists()
        assert (child / "seed-71/distilled.pt").read_bytes() == (parent / "seed-71/distilled.pt").read_bytes()
    assert budget.run_budget_check(config, parent, child, "cpu")["status"] == "complete"
    # Losing identity metadata must never cause a continued trajectory to be
    # silently replaced by its four-step parent.
    manifest = child / "seed-71/manifest.json"
    manifest.unlink()
    before = (child / "seed-71/distilled.pt").read_bytes()
    with pytest.raises(ValueError, match="Unidentified child checkpoint"):
        budget.run_budget_check(config, parent, child, "cpu")
    assert (child / "seed-71/distilled.pt").read_bytes() == before


@pytest.mark.parametrize("corruption", ["protocol_source", "protocol_config", "missing_seed", "summary",
    "report", "mask", "data", "target_mask", "signature", "steps", "weights", "nonfinite",
    "adam", "inventory", "training_metadata", "selection_history"])
def test_invalid_parent_rejected_before_fit(parent_copy, tmp_path, monkeypatch, corruption):
    parent, config = parent_copy
    seed = parent / "seed-71"
    if corruption in ("protocol_source", "protocol_config"):
        path = parent / "protocol.json"
        value = read_json(path)
        if corruption == "protocol_source":
            value["source_sha256"] = "0" * 64
        else:
            value["config"]["base"]["learning_rate"] *= 2
        write_json(path, value)
    elif corruption == "missing_seed":
        shutil.rmtree(parent / "seed-81")
    elif corruption in ("summary", "report", "training_metadata", "inventory"):
        if corruption == "inventory":
            inventory = read_json(seed / "manifest.json")["sessions"][0]["package_inventory"]["path"]
            (seed / inventory).write_text("changed==1\n")
        else:
            path = {"summary": parent / "summary.json", "report": seed / "report.json",
                    "training_metadata": seed / "training.json"}[corruption]
            value = read_json(path)
            if corruption == "summary":
                value["groups"]["rare"]["models"]["oracle"]["nll"]["mean"] += .1
            elif corruption == "report":
                value["groups"]["rare"]["models"]["oracle"]["nll"] += .1
            else:
                value["oracle"]["completed_steps"] -= 1
            write_json(path, value)
    elif corruption in ("mask", "data", "target_mask"):
        path = seed / ("soft_training_logits.npz" if corruption == "target_mask" else "data.npz")
        values = budget._arrays(path)
        if corruption == "mask":
            values["mask_train"][0, 0] ^= True
        elif corruption == "data":
            values["true_prob"][0, 0] += 1e-8
        else:
            mask = budget._arrays(seed / "data.npz")["mask_train"]
            values["oracle"][~mask] = 1
        budget._save_npz(path, **values)
    else:
        path = seed / "distilled.pt"
        saved = budget._checkpoint(path)
        if corruption == "signature":
            saved["signature"] = "0" * 64
        elif corruption == "steps":
            saved["completed_steps"] -= 1
        elif corruption in ("weights", "nonfinite"):
            next(iter(saved["current_state"].values())).fill_(0 if corruption == "weights" else float("nan"))
        elif corruption == "adam":
            saved["optimizer"]["param_groups"][0]["lr"] *= 2
        else:
            saved["population_best_step"] = 0 if saved["population_best_step"] else 1
        torch.save(saved, path)
    monkeypatch.setattr(budget, "fit_model", lambda *args, **kwargs: pytest.fail("invalid parent trained"))
    with pytest.raises(ValueError):
        budget.run_budget_check(config, parent, tmp_path.resolve() / "child", "cpu")


def test_changed_parent_and_changed_runtime_reject_resume(parent_copy, tmp_path, monkeypatch):
    parent, config = parent_copy
    child = tmp_path.resolve() / "child"
    budget.run_budget_check(config, parent, child, "cpu")
    (parent / "added.txt").write_text("parent tree changed")
    monkeypatch.setattr(budget, "fit_model", lambda *args, **kwargs: pytest.fail("changed parent trained"))
    with pytest.raises(ValueError, match="parent hashes changed"):
        budget.run_budget_check(config, parent, child, "cpu")
    monkeypatch.setattr(budget, "legacy_source_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError, match="numerical source changed"):
        budget.validate_parent(config, parent)


def test_float64_roundoff_accepted_but_integer_boolean_dtypes_exact(parent_copy, monkeypatch):
    parent, config = parent_copy
    original = budget.simulate_population

    def rounded(*args):
        values = original(*args)
        values["true_prob"] = np.nextafter(values["true_prob"], 1)
        return values

    monkeypatch.setattr(budget, "simulate_population", rounded)
    budget.validate_parent(config, parent)
    path = parent / "seed-71/data.npz"
    values = budget._arrays(path)
    values["mask_train"] = values["mask_train"].astype(np.uint8)
    budget._save_npz(path, **values)
    with pytest.raises(ValueError, match="data or masks"):
        budget.validate_parent(config, parent)


def test_overlap_and_symlink_tree_rejected(parent_copy, tmp_path):
    parent, config = parent_copy
    for target in (parent, parent / "child", parent.parent):
        with pytest.raises(ValueError, match="overlap"):
            budget.run_budget_check(config, parent, target, "cpu")
    link = tmp_path.resolve() / "parent-link"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        budget.run_budget_check(config, link, tmp_path.resolve() / "child", "cpu")
    (parent / "escape").symlink_to(tmp_path.resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        budget.validate_parent(config, parent)


def test_paired_aggregation_signs_nulls_and_complete_seed_requirement(parent_run):
    parent, config = parent_run
    reports = []
    for i, seed in enumerate(config.parent_config.replication_seeds):
        report = read_json(parent / f"seed-{seed}/report.json")
        report["baseline"] = copy.deepcopy(report)
        for group in ("rare", "typical"):
            for mode in control.MODES:
                old = report["baseline"]["groups"][group]["models"]
                new = report["groups"][group]["models"]
                old[f"distilled_{mode}"]["kl"] = 3 + i
                old[f"oracle_{mode}"]["kl"] = 1 + i
                new[f"distilled_{mode}"]["kl"] = 2 + i
                new[f"oracle_{mode}"]["kl"] = .5 + i
                for arm in budget.EXTENDED_ARMS:
                    old[f"{arm}_{mode}"]["diagnostic_kl"] = None
                    new[f"{arm}_{mode}"]["diagnostic_kl"] = None
        reports.append(report)
    result = budget.aggregate_budget_check(reports, config.parent_config.replication_seeds)
    for group in result.values():
        for mode in control.MODES:
            assert group["baseline_target_comparisons"][mode]["kl_gain"]["mean"] == 2
            assert group["extended_target_comparisons"][mode]["kl_gain"]["mean"] == 1.5
            change = group["paired_gain_change"][mode]["kl_gain"]
            assert change["mean"] == -.5 and change["negative_seed_count"] == 2
            assert group["arm_changes"][mode]["distilled"]["kl"]["mean"] == -1
            assert group["arm_changes"][mode]["oracle"]["kl"]["mean"] == -.5
            missing = group["paired_gain_change"][mode]["diagnostic_kl_gain"]
            assert missing["mean"] is None and missing["supported_seed_count"] == 0
            assert missing["positive_seed_count"] == missing["negative_seed_count"] == missing["tied_seed_count"] == 0
    for invalid in (reports[:1], reports[::-1], [reports[0], reports[0]]):
        with pytest.raises(ValueError, match="Every planned"):
            budget.aggregate_budget_check(invalid, config.parent_config.replication_seeds)
