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
def test_saved_resume_equals_uninterrupted_training(tmp_path, kind):
    torch.manual_seed(32)
    model = TeacherGRU(4, 3) if kind == "teacher" else BatchedStudentGRU(3, 4, 3)
    uninterrupted = copy.deepcopy(model)
    resumed = copy.deepcopy(model)
    data = training_data()
    kwargs = dict(lr=.02, seed=99, eval_every=2, patience=20)
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
def test_cuda_checkpoint_resume_preserves_optimizer_devices(tmp_path):
    torch.manual_seed(12)
    model = BatchedStudentGRU(3, 4, 2).cuda()
    uninterrupted = copy.deepcopy(model)
    kwargs = dict(lr=.02, seed=9, eval_every=2)
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
