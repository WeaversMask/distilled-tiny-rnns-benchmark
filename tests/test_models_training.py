import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from distilled_rnns.models import BatchedStudentGRU, TeacherGRU, copy_student_to_reference
from distilled_rnns.training import fit_model, masked_objective
import distilled_rnns.training as training_module


@pytest.fixture(autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("hidden_size", [1, 2, 5])
def test_batched_forward_and_gradients_match_pytorch_gru(hidden_size):
    torch.manual_seed(12)
    model = BatchedStudentGRU(3, 4, hidden_size).double()
    references = [copy_student_to_reference(model, i) for i in range(3)]
    x = torch.randn(3, 7, 4, dtype=torch.float64, requires_grad=True)
    inputs = [x[i:i+1].detach().clone().requires_grad_() for i in range(3)]
    batched_output = model(x)
    independent_outputs = [reference(inputs[i]) for i, reference in enumerate(references)]
    torch.testing.assert_close(batched_output, torch.cat(independent_outputs), rtol=1e-11, atol=1e-12)
    batched_output.square().sum().backward()
    for output in independent_outputs:
        output.square().sum().backward()
    torch.testing.assert_close(x.grad, torch.cat([item.grad for item in inputs]), rtol=1e-10, atol=1e-12)
    pairs = [
        ("weight_ih", "gru.weight_ih_l0"), ("weight_hh", "gru.weight_hh_l0"),
        ("bias_ih", "gru.bias_ih_l0"), ("bias_hh", "gru.bias_hh_l0"),
        ("readout_weight", "readout.weight"), ("readout_bias", "readout.bias"),
    ]
    for subject, reference in enumerate(references):
        ref_parameters = dict(reference.named_parameters())
        for batched_key, independent_key in pairs:
            actual = getattr(model, batched_key).grad[subject]
            expected = ref_parameters[independent_key].grad.reshape(actual.shape)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_subjects_have_no_cross_influence():
    torch.manual_seed(1)
    model = BatchedStudentGRU(3, 4, 2).double()
    x = torch.randn(3, 6, 4, dtype=torch.float64)
    before = model(x).detach()
    changed = x.clone()
    changed[0] += 100
    torch.testing.assert_close(model(changed)[1:], before[1:], rtol=0, atol=0)
    model(x)[0].sum().backward()
    for parameter in model.parameters():
        assert torch.count_nonzero(parameter.grad[1:]) == 0


def test_distillation_endpoints_and_target_masking():
    logits = torch.tensor([[0.2, -0.3, 0.7], [0.4, 0.1, -0.2]], requires_grad=True)
    targets = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    teacher = torch.tensor([[-0.5, 0.2, 0.9], [0.1, -0.7, 0.8]], requires_grad=True)
    hard = masked_objective(logits, targets, mask, teacher_logits=teacher, teacher_prop=0)
    torch.testing.assert_close(hard, F.binary_cross_entropy_with_logits(logits[mask], targets[mask]))
    soft = masked_objective(logits, targets, mask, teacher_logits=teacher, teacher_prop=1, temperature=2)
    expected = 4 * F.binary_cross_entropy_with_logits(logits[mask] / 2, torch.sigmoid(teacher[mask].detach() / 2))
    torch.testing.assert_close(soft, expected)
    mixture = masked_objective(logits, targets, mask, teacher_logits=teacher, teacher_prop=.3, temperature=2)
    torch.testing.assert_close(mixture, .7 * hard + .3 * soft)
    mixture.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    altered_targets = targets.clone()
    altered_teacher = teacher.detach().clone()
    altered_targets[~mask] = 1 - altered_targets[~mask]
    altered_teacher[~mask] = 99
    changed = masked_objective(logits, altered_targets, mask, teacher_logits=altered_teacher, teacher_prop=.3, temperature=2)
    torch.testing.assert_close(changed, mixture)


def test_independent_adam_step_matches_separate_optimizers():
    torch.manual_seed(8)
    model = BatchedStudentGRU(3, 4, 2).double()
    references = [copy_student_to_reference(model, i) for i in range(3)]
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    y = torch.randint(0, 2, (3, 5)).double()
    mask = torch.ones(3, 5, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    masked_objective(model(x), y, mask, independent_students=True).backward()
    optimizer.step()
    for subject, reference in enumerate(references):
        separate_optimizer = torch.optim.Adam(reference.parameters(), lr=.02)
        F.binary_cross_entropy_with_logits(reference(x[subject:subject+1]), y[subject:subject+1]).backward()
        separate_optimizer.step()
        torch.testing.assert_close(model(x)[subject], reference(x[subject:subject+1])[0], rtol=1e-10, atol=1e-12)


def training_data():
    generator = torch.Generator().manual_seed(22)
    x = torch.randn(3, 12, 4, generator=generator)
    y = torch.randint(0, 2, (3, 12), generator=generator).float()
    train = torch.zeros_like(y, dtype=torch.bool)
    train[:, :8] = True
    validation = ~train
    return x, y, train, validation


@pytest.mark.parametrize("kind", ["teacher", "students"])
@pytest.mark.parametrize("l2_strength", [0.0, 0.03])
def test_saved_resume_equals_uninterrupted_training(tmp_path, kind, l2_strength):
    torch.manual_seed(32)
    model = TeacherGRU(4, 3) if kind == "teacher" else BatchedStudentGRU(3, 4, 3)
    uninterrupted = copy.deepcopy(model)
    resumed = copy.deepcopy(model)
    data = training_data()
    kwargs = dict(lr=.02, seed=99, eval_every=2, patience=20, l2_strength=l2_strength)
    entire = fit_model(uninterrupted, *data, steps=8, checkpoint_path=tmp_path / "entire.pt", **kwargs)
    fit_model(resumed, *data, steps=4, checkpoint_path=tmp_path / "resumed.pt", **kwargs)
    # Replacement model deliberately has a different random initialization.
    resumed = TeacherGRU(4, 3) if kind == "teacher" else BatchedStudentGRU(3, 4, 3)
    continuation = fit_model(resumed, *data, steps=8, checkpoint_path=tmp_path / "resumed.pt", **kwargs)
    assert continuation["completed_steps"] == 8
    assert continuation["history"] == entire["history"]
    assert continuation["best_step"] == entire["best_step"]
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    left = torch.load(tmp_path / "entire.pt", weights_only=True)
    right = torch.load(tmp_path / "resumed.pt", weights_only=True)
    for key, value in left["current_state"].items():
        torch.testing.assert_close(value, right["current_state"][key], rtol=0, atol=0)
    assert not (tmp_path / "resumed.pt.tmp").exists()


def test_students_select_validation_weights_individually(tmp_path):
    model = BatchedStudentGRU(2, 1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.zeros(2, 6, 1)
    y = torch.ones(2, 6)
    y[1, 3:] = 0
    train = torch.zeros(2, 6, dtype=torch.bool)
    train[:, :3] = True
    result = fit_model(model, x, y, train, ~train, steps=6, lr=.1,
                       checkpoint_path=tmp_path / "per_person.pt", seed=0, eval_every=1)
    assert result["best_step"] == [6, 0]
    assert model.readout_bias[0] > 0
    assert model.readout_bias[1] == 0


def test_final_step_is_evaluated_between_regular_boundaries(tmp_path):
    model = BatchedStudentGRU(2, 1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.zeros(2, 6, 1)
    y = torch.ones(2, 6)
    train = torch.zeros(2, 6, dtype=torch.bool)
    train[:, :3] = True
    result = fit_model(model, x, y, train, ~train, steps=3, lr=.1,
                       checkpoint_path=tmp_path / "final.pt", seed=0, eval_every=2)
    assert [row["step"] for row in result["history"]] == [0, 2, 3]
    assert result["best_step"] == [3, 3]


def test_time_limit_between_evaluations_preserves_resume_schedule(tmp_path, monkeypatch):
    torch.manual_seed(2)
    model = TeacherGRU(4, 2)
    uninterrupted = copy.deepcopy(model)
    kwargs = dict(steps=3, lr=.02, seed=0, eval_every=2)
    entire = fit_model(uninterrupted, *training_data(), checkpoint_path=None, **kwargs)
    forward_calls = [0]
    original_forward = model.forward

    def counted_forward(x):
        result = original_forward(x)
        forward_calls[0] += 1
        return result

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", counted_forward)
        patch.setattr(training_module, "time", SimpleNamespace(perf_counter=lambda: 0. if forward_calls[0] < 2 else 2.))
        limited = fit_model(model, *training_data(), checkpoint_path=tmp_path / "interrupted.pt", max_seconds=1, **kwargs)
    assert limited["completed_steps"] == 1
    assert limited["stopped_reason"] == "time_limit"
    assert [row["step"] for row in limited["history"]] == [0]
    resumed = fit_model(model, *training_data(), checkpoint_path=tmp_path / "interrupted.pt", **kwargs)
    assert resumed["history"] == entire["history"]
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_checkpoint_rejects_changed_data_or_hyperparameters(tmp_path):
    model = TeacherGRU(4, 2)
    x, y, train, validation = training_data()
    kwargs = dict(steps=2, lr=.02, checkpoint_path=tmp_path / "saved.pt", seed=0, eval_every=1)
    fit_model(model, x, y, train, validation, **kwargs)
    altered = x.clone()
    altered[0, 0, 0] += 1
    with pytest.raises(ValueError, match="checkpoint does not match"):
        fit_model(model, altered, y, train, validation, **kwargs)
    with pytest.raises(ValueError, match="checkpoint does not match"):
        fit_model(model, x, y, train, validation, **{**kwargs, "lr": .03})


def test_time_limit_saves_resumable_state(tmp_path):
    model = TeacherGRU(4, 2)
    kwargs = dict(steps=2, lr=.02, checkpoint_path=tmp_path / "limited.pt", seed=0, eval_every=1)
    limited = fit_model(model, *training_data(), max_seconds=1e-12, **kwargs)
    assert limited["stopped_reason"] == "time_limit"
    assert limited["completed_steps"] == 0
    assert (tmp_path / "limited.pt").is_file()
    finished = fit_model(model, *training_data(), **kwargs)
    assert finished["completed_steps"] == 2
    assert finished["stopped_reason"] == "steps_completed"


def test_mask_and_distillation_validation():
    model = TeacherGRU(4, 2)
    x, y, train, validation = training_data()
    kwargs = dict(steps=1, lr=.02, checkpoint_path=None, seed=0)
    with pytest.raises(ValueError, match="disjoint"):
        fit_model(model, x, y, train, train, **kwargs)
    with pytest.raises(ValueError, match="boolean"):
        fit_model(model, x, y, train.float(), validation, **kwargs)
    with pytest.raises(ValueError, match="teacher_logits"):
        fit_model(model, x, y, train, validation, teacher_prop=1, **kwargs)
    with pytest.raises(ValueError, match="temperature"):
        fit_model(model, x, y, train, validation, temperature=0, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU; exercised by Colab preflight")
@pytest.mark.parametrize("l2_strength", [0.0, 0.01])
def test_cuda_checkpoint_resume_preserves_optimizer_devices(tmp_path, l2_strength):
    torch.manual_seed(12)
    model = BatchedStudentGRU(3, 4, 2).cuda()
    uninterrupted = copy.deepcopy(model)
    kwargs = dict(lr=.02, seed=9, eval_every=2, l2_strength=l2_strength)
    fit_model(uninterrupted, *training_data(), steps=4, checkpoint_path=None, **kwargs)
    path = tmp_path / "cuda.pt"
    fit_model(model, *training_data(), steps=2, checkpoint_path=path, **kwargs)
    fit_model(model, *training_data(), steps=4, checkpoint_path=path, **kwargs)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=1e-5, atol=1e-6)
    saved = torch.load(path, weights_only=True)
    for state in saved["optimizer"]["state"].values():
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].device.type == "cuda"


@pytest.mark.parametrize("l2_strength", [0.001, 0.1])
def test_regularized_fit_matches_separate_adam_optimizers(tmp_path, l2_strength):
    """The actual fitter must preserve independent gradients, including L2."""
    torch.manual_seed(73)
    model = BatchedStudentGRU(3, 4, 2).double()
    references = [copy_student_to_reference(model, i) for i in range(3)]
    x, y, train, validation = training_data()
    x, y = x.double(), y.double()
    path = tmp_path / "regularized.pt"
    fit_model(model, x, y, train, validation, steps=3, lr=.02,
              checkpoint_path=path, seed=99, eval_every=1, l2_strength=l2_strength)
    # Compare the last optimizer state; returned models select validation weights.
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["current_state"])
    for subject, reference in enumerate(references):
        optimizer = torch.optim.Adam(reference.parameters(), lr=.02)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = reference(x[subject:subject + 1])[0]
            objective = F.binary_cross_entropy_with_logits(
                logits[train[subject]], y[subject, train[subject]])
            objective = objective + (l2_strength / 2) * sum(
                parameter.square().sum() for parameter in reference.parameters())
            objective.backward()
            optimizer.step()
        actual = copy_student_to_reference(model, subject)
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("l2_strength", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_l2_strength_is_rejected(l2_strength):
    model = BatchedStudentGRU(3, 4, 2)
    with pytest.raises(ValueError, match="l2_strength"):
        fit_model(model, *training_data(), steps=1, lr=.02, checkpoint_path=None,
                  seed=0, l2_strength=l2_strength)


@pytest.mark.parametrize("initial, changed", [(0.0, 0.01), (0.01, 0.0), (0.01, 0.1)])
def test_checkpoint_rejects_changed_l2_strength(tmp_path, initial, changed):
    model = BatchedStudentGRU(3, 4, 2)
    kwargs = dict(steps=2, lr=.02, checkpoint_path=tmp_path / "l2.pt", seed=0, eval_every=1)
    fit_model(model, *training_data(), l2_strength=initial, **kwargs)
    with pytest.raises(ValueError, match="checkpoint does not match"):
        fit_model(model, *training_data(), l2_strength=changed, **kwargs)


def test_zero_l2_preserves_original_checkpoint_signature(tmp_path):
    """Old unregularized checkpoints can still be resumed directly."""
    model = BatchedStudentGRU(3, 4, 2)
    data = training_data()
    kwargs = dict(steps=1, lr=.02, checkpoint_path=tmp_path / "original.pt",
                  seed=0, eval_every=1, patience=20)
    fit_model(model, *data, l2_strength=0, **kwargs)
    saved = torch.load(kwargs["checkpoint_path"], weights_only=True)
    original_settings = {"lr": .02, "seed": 0, "eval_every": 1, "patience": 20,
                         "teacher_prop": 0.0, "temperature": 1.0}
    original_signature = training_module._signature(model, original_settings, [*data, None])
    assert saved["signature"] == original_signature


def test_population_and_individual_selection_can_choose_different_steps(tmp_path):
    model = BatchedStudentGRU(2, 1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    initial = copy.deepcopy(model.state_dict())
    x = torch.zeros(2, 6, 1)
    y = torch.ones(2, 6)
    y[1, 3:] = 0
    train = torch.zeros(2, 6, dtype=torch.bool)
    train[:, :3] = True
    path = tmp_path / "dual.pt"
    result = fit_model(model, x, y, train, ~train, steps=6, lr=.1,
                       checkpoint_path=path, seed=0, eval_every=1,
                       track_population_selection=True)
    assert result["best_step"] == [6, 0]
    assert result["population_best_step"] == 0
    assert result["population_best_validation_nll"] == pytest.approx(np.log(2))
    saved = torch.load(path, weights_only=True)
    for name, value in initial.items():
        torch.testing.assert_close(saved["population_best_state"][name], value, rtol=0, atol=0)
    # The normal returned model still uses independently selected slices.
    assert model.readout_bias[0] > 0 and model.readout_bias[1] == 0


def test_population_selection_matches_manual_adam_snapshot_search(tmp_path):
    torch.manual_seed(37)
    initial = BatchedStudentGRU(3, 4, 2).double()
    model, manual = copy.deepcopy(initial), copy.deepcopy(initial)
    x, y, train, validation = training_data()
    x, y = x.double(), y.double()
    # Different validation counts verify equal-person rather than trial weighting.
    train[0, 7] = False
    validation[0, 7] = True
    states, values = [], []
    optimizer = torch.optim.Adam(manual.parameters(), lr=.02)
    for step in range(9):
        if step % 2 == 0:
            with torch.no_grad():
                loss = F.binary_cross_entropy_with_logits(manual(x), y, reduction="none")
                values.append(float(((loss * validation).sum(1) / validation.sum(1)).mean()))
                states.append(copy.deepcopy(manual.state_dict()))
        if step == 8:
            break
        optimizer.zero_grad(set_to_none=True)
        objective = masked_objective(manual(x), y, train, independent_students=True)
        objective = objective + (.03 / 2) * sum(p.square().sum() for p in manual.parameters())
        objective.backward()
        optimizer.step()
    path = tmp_path / "manual_comparison.pt"
    result = fit_model(model, x, y, train, validation, steps=8, lr=.02,
                       checkpoint_path=path, seed=99, eval_every=2, l2_strength=.03,
                       track_population_selection=True)
    selected = min(range(len(values)), key=values.__getitem__)
    assert result["population_best_step"] == selected * 2
    assert result["population_best_validation_nll"] == pytest.approx(values[selected], abs=1e-12)
    saved = torch.load(path, weights_only=True)
    for name, value in states[selected].items():
        torch.testing.assert_close(saved["population_best_state"][name], value, rtol=0, atol=0)


def test_population_selection_exact_ties_keep_first_step(tmp_path):
    model = BatchedStudentGRU(2, 1, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.zeros(2, 4, 1)
    y = torch.tensor([[0., 1., 0., 1.], [0., 1., 0., 1.]])
    train = torch.tensor([[True, True, False, False]]).expand(2, -1)
    result = fit_model(model, x, y, train, ~train, steps=4, lr=.02,
                       checkpoint_path=tmp_path / "ties.pt", seed=0, eval_every=1,
                       track_population_selection=True)
    assert len({row["validation_nll"] for row in result["history"]}) == 1
    assert result["population_best_step"] == 0


def test_population_tracking_does_not_change_optimization_or_individual_selection(tmp_path):
    torch.manual_seed(53)
    initial = BatchedStudentGRU(3, 4, 2)
    ordinary, tracked = copy.deepcopy(initial), copy.deepcopy(initial)
    kwargs = dict(steps=6, lr=.02, seed=99, eval_every=2, l2_strength=.01)
    ordinary_result = fit_model(ordinary, *training_data(), checkpoint_path=tmp_path / "ordinary.pt", **kwargs)
    tracked_result = fit_model(tracked, *training_data(), checkpoint_path=tmp_path / "tracked.pt",
                               track_population_selection=True, **kwargs)
    for field in ("history", "best_step", "best_validation_nll", "completed_steps", "stopped_reason"):
        assert ordinary_result[field] == tracked_result[field]
    assert "population_best_step" not in ordinary_result
    left = torch.load(tmp_path / "ordinary.pt", weights_only=True)
    right = torch.load(tmp_path / "tracked.pt", weights_only=True)
    for state in ("current_state", "best_state"):
        for name, value in left[state].items():
            torch.testing.assert_close(value, right[state][name], rtol=0, atol=0)
    for name, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, tracked.state_dict()[name], rtol=0, atol=0)
    assert "population_best_state" not in left


@pytest.mark.parametrize("l2_strength", [0.0, .01])
def test_population_snapshot_resume_matches_uninterrupted(tmp_path, l2_strength):
    torch.manual_seed(61)
    initial = BatchedStudentGRU(3, 4, 2)
    full, interrupted = copy.deepcopy(initial), copy.deepcopy(initial)
    kwargs = dict(lr=.02, seed=99, eval_every=2, l2_strength=l2_strength,
                  track_population_selection=True)
    entire = fit_model(full, *training_data(), steps=8, checkpoint_path=tmp_path / "full.pt", **kwargs)
    fit_model(interrupted, *training_data(), steps=4, checkpoint_path=tmp_path / "interrupted.pt", **kwargs)
    resumed = BatchedStudentGRU(3, 4, 2)
    continued = fit_model(resumed, *training_data(), steps=8,
                          checkpoint_path=tmp_path / "interrupted.pt", **kwargs)
    for field in ("history", "best_step", "best_validation_nll", "population_best_step", "population_best_validation_nll"):
        assert continued[field] == entire[field]
    left = torch.load(tmp_path / "full.pt", weights_only=True)
    right = torch.load(tmp_path / "interrupted.pt", weights_only=True)
    for state in ("current_state", "best_state", "population_best_state"):
        for name, value in left[state].items():
            torch.testing.assert_close(value, right[state][name], rtol=0, atol=0)
    torch.testing.assert_close(left["population_best_val"], right["population_best_val"], rtol=0, atol=0)


@pytest.mark.parametrize("initial", [False, True])
def test_checkpoint_rejects_changed_population_tracking(tmp_path, initial):
    model = BatchedStudentGRU(3, 4, 2)
    kwargs = dict(steps=2, lr=.02, checkpoint_path=tmp_path / "tracking.pt", seed=99, eval_every=1)
    fit_model(model, *training_data(), track_population_selection=initial, **kwargs)
    with pytest.raises(ValueError, match="checkpoint does not match"):
        fit_model(model, *training_data(), track_population_selection=not initial, **kwargs)


@pytest.mark.parametrize("kind,path,setting", [
    ("teacher", "saved.pt", True), ("students", None, True),
    ("students", "saved.pt", 1), ("students", "saved.pt", "yes"),
])
def test_population_tracking_requires_students_path_and_boolean(tmp_path, kind, path, setting):
    model = TeacherGRU(4, 2) if kind == "teacher" else BatchedStudentGRU(3, 4, 2)
    with pytest.raises(ValueError, match="track_population_selection"):
        fit_model(model, *training_data(), steps=2, lr=.02, seed=99,
                  checkpoint_path=tmp_path / path if path else None,
                  track_population_selection=setting)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA; exercised by Colab preflight")
def test_cuda_population_selection_resume(tmp_path):
    torch.manual_seed(71)
    initial = BatchedStudentGRU(3, 4, 2).cuda()
    full, interrupted = copy.deepcopy(initial), copy.deepcopy(initial)
    kwargs = dict(lr=.02, seed=99, eval_every=2, l2_strength=.01,
                  track_population_selection=True)
    entire = fit_model(full, *training_data(), steps=4, checkpoint_path=tmp_path / "full.pt", **kwargs)
    fit_model(interrupted, *training_data(), steps=2, checkpoint_path=tmp_path / "resumed.pt", **kwargs)
    continued = fit_model(interrupted, *training_data(), steps=4, checkpoint_path=tmp_path / "resumed.pt", **kwargs)
    assert continued["population_best_step"] == entire["population_best_step"]
    assert continued["population_best_validation_nll"] == pytest.approx(entire["population_best_validation_nll"], abs=1e-6)
    left = torch.load(tmp_path / "full.pt", weights_only=True, map_location="cpu")
    right = torch.load(tmp_path / "resumed.pt", weights_only=True, map_location="cpu")
    for state in ("current_state", "best_state", "population_best_state"):
        for name, value in left[state].items():
            torch.testing.assert_close(value, right[state][name], rtol=1e-5, atol=1e-6)
