"""Bounded synthetic check of a pinned, externally supplied tinyRNN checkout.

No upstream code is vendored or edited. Only inspected neural/dataset modules
are imported; experiment launchers and human-data loading are never invoked.
The two standalone orchestration functions are compiled unchanged from their
AST nodes to avoid importing unrelated cluster/cognitive-model dependencies.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from distilled_rnns.models import BatchedStudentGRU, TeacherGRU
from distilled_rnns.simulation import simulate_population
from distilled_rnns.training import fit_model, masked_objective


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkout(checkout, pin):
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != pin["commit"]:
        raise ValueError(f"Expected authors revision {pin['commit']}; found {revision}")
    for name, expected in pin["file_sha256"].items():
        path = checkout / name
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Authors source mismatch: {name}")
    return checkout


def load_authors(checkout, scratch):
    """Import original source in isolated package namespaces, relocate outputs."""
    spec = importlib.util.spec_from_file_location("path_settings", checkout / "path_settings.py")
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    paths.MODEL_SAVE_PATH = scratch / "models"
    paths.ANA_SAVE_PATH = scratch / "analysis"
    sys.modules["path_settings"] = paths
    for alias, folder in (("alignment_authors_agents", "agents"),
                          ("alignment_authors_datasets", "datasets")):
        package = types.ModuleType(alias)
        package.__path__ = [str(checkout / folder)]
        sys.modules[alias] = package
    agent = importlib.import_module("alignment_authors_agents.RNNAgent").RNNAgent
    trainer = importlib.import_module("alignment_authors_agents.RNNAgentTrainer").RNNAgentTrainer
    dataset = importlib.import_module("alignment_authors_datasets.GillanHumanDataset").GillanHumanDataset
    import joblib
    namespace = dict(np=np, os=os, torch=torch, joblib=joblib,
                     ANA_SAVE_PATH=paths.ANA_SAVE_PATH, set_os_path_auto=nullcontext)
    training_file = checkout / "training_experiments/training.py"
    tree = ast.parse(training_file.read_text())
    names = {"split_over_trials", "_load_teacher_network_score"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Missing reviewed upstream orchestration functions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(training_file), "exec"), namespace)
    return agent, trainer, dataset, namespace, paths


def matched_binary_check(agent_class):
    """Compare the actual author's loss/forward against independent students."""
    torch.manual_seed(202611042)
    local = BatchedStudentGRU(1, 4, 2).double()
    native = agent_class(dict(rnn_type="GRU", input_dim=4, hidden_dim=2, output_dim=2,
                              seed=202611042, device="cpu", output_h0=False,
                              trainable_h0=False, readout_FC=True, distill_temp=2.0))
    with torch.no_grad():
        for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(native.model.rnn, key + "_l0").copy_(getattr(local, key)[0])
        native.model.lin.weight.zero_()
        native.model.lin.bias.zero_()
        native.model.lin.weight[1].copy_(local.readout_weight[0])
        native.model.lin.bias[1].copy_(local.readout_bias[0])
    data = simulate_population(2, 30, .5, 202611043)
    x = torch.tensor(data["x"][:1], dtype=torch.double)
    y = torch.tensor(data["choice"][:1], dtype=torch.double)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:, 1::3] = False
    teacher = torch.linspace(-2, 2, 30, dtype=torch.double)[None]
    result = {}
    for mode, proportion, temperature in (("hard", 0., 1.), ("soft", 1., 2.), ("mixed", .7, 2.)):
        local.zero_grad()
        native.model.zero_grad()
        logits = local(x)
        score = native.model(x.transpose(0, 1))
        discrepancy = (logits.T - (score[..., 1] - score[..., 0])).abs().max().item()
        loss_local = masked_objective(logits, y, mask, teacher_logits=teacher,
                                      teacher_prop=proportion, temperature=temperature)
        soft = torch.stack((1-torch.sigmoid(teacher.T/temperature),
                            torch.sigmoid(teacher.T/temperature)), -1)
        target = y.T.long() if mode == "hard" else soft
        if mode == "mixed":
            target = [("tch", proportion, soft), ("action", 1-proportion, y.T.long())]
        native.config["distill_temp"] = temperature
        loss_native = native._compare_to_target(x.transpose(0, 1), target, mask.T.double())["behav_loss"]
        loss_local.backward()
        loss_native.backward()
        gradient_errors = {
            key: (getattr(local, key).grad[0] - getattr(native.model.rnn, key+"_l0").grad).abs().max().item()
            for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        }
        gradient_errors["class1_readout_weight"] = (local.readout_weight.grad[0] - native.model.lin.weight.grad[1]).abs().max().item()
        gradient_errors["class1_readout_bias"] = abs(local.readout_bias.grad[0].item() - native.model.lin.bias.grad[1].item())
        # Held-out target perturbation cannot change the masked training loss.
        changed_y, changed_teacher = y.clone(), teacher.clone()
        changed_y[~mask] = 1-changed_y[~mask]
        changed_teacher[~mask] = 100
        changed = masked_objective(local(x), changed_y, mask, teacher_logits=changed_teacher,
                                   teacher_prop=proportion, temperature=temperature)
        result[mode] = dict(max_logit_error=discrepancy,
                            loss_error=abs(loss_local.item()-loss_native.item()),
                            gradient_errors=gradient_errors,
                            heldout_target_loss_change=abs(changed.item()-loss_local.item()))
        assert discrepancy < 1e-12
        assert result[mode]["loss_error"] < 1e-12
        assert max(gradient_errors.values()) < 1e-12
        assert result[mode]["heldout_target_loss_change"] == 0
    result["parameter_counts"] = dict(pilot=sum(p.numel() for p in local.parameters()),
        authors_binary=native.num_params)
    result["scope"] = "Mapped initial state, forward/loss/gradients; native two-logit optimizer trajectories are not asserted equivalent."
    return result


def native_dataset(dataset_class, population, scratch, augment):
    dataset = dataset_class(data_path=scratch, behav_data_spec=None)
    n = len(population["rare"])
    dataset.behav = {
        "stage_1_selected_stimulus": [row.copy()+1 for row in population["choice"]],
        "stage_2_state": [row.copy()+2 for row in population["state"]],
        "stage_2_selected_stimulus": [row.copy()+1 for row in population["second_choice"]],
        "reward": [row.copy() for row in population["reward"]],
        "sub_id": list(range(n)), "aug_block_number": list(range(n)),
    }
    dataset.behav["action"] = dataset.behav["stage_1_selected_stimulus"]
    if augment:
        dataset._augment_data(augment_level=1)
    dataset._detect_trial_type()
    dataset.behav_to(dict(behav_format="tensor", output_h0=True, device="cpu", include_embedding=False))
    return dataset


def check_native_encoding(dataset, population):
    """Decode the original stage inputs and action masks back into raw events."""
    features = dataset.torch_beahv_input.numpy().transpose(1, 0, 2)
    first, second = features[:, ::2], features[:, 1::2]
    choices = population["choice"]
    second_labels = 2 + 2*population["state"] + population["second_choice"]
    np.testing.assert_array_equal(first[..., :3], np.eye(3)[1+population["state"]])
    np.testing.assert_array_equal(first[..., 3:9], np.eye(6)[choices])
    np.testing.assert_array_equal(first[..., 9], np.zeros_like(choices))
    np.testing.assert_array_equal(second[..., :3], np.eye(3)[np.zeros_like(choices)])
    np.testing.assert_array_equal(second[..., 3:9], np.eye(6)[second_labels])
    np.testing.assert_array_equal(second[..., 9], population["reward"])
    labels = dataset.torch_beahv_target.numpy().T
    np.testing.assert_array_equal(labels[:, ::2], choices)
    np.testing.assert_array_equal(labels[:, 1::2], second_labels)
    valid = dataset.torch_beahv_mask[1].numpy().transpose(1, 0, 2) == 0
    first_valid = np.broadcast_to(np.arange(6) < 2, valid[:, ::2].shape)
    second_valid = (np.arange(6)//2 == (1+population["state"])[..., None])
    np.testing.assert_array_equal(valid[:, ::2], first_valid)
    np.testing.assert_array_equal(valid[:, 1::2], second_valid)


def pilot_companion(population, splits, settings, scratch):
    """Fit the project teacher/students to the identical first-stage splits."""
    choices = population["choice"]
    x = torch.from_numpy(population["x"])
    count = len(choices)
    train_mask, val_mask, test_mask = [
        np.broadcast_to(np.isin(np.arange(choices.shape[1]), indexes), choices.shape).copy()
        for indexes in splits
    ]
    seed = settings["initialization_seed"]
    options = dict(steps=settings["max_epoch_num"], lr=.01, seed=seed,
                   eval_every=10, patience=20)
    torch.manual_seed(seed)
    teacher = TeacherGRU(4, 32)
    teacher_fit = fit_model(teacher, x, choices, train_mask, val_mask, checkpoint_path=None, **options)
    with torch.no_grad():
        teacher_logits = teacher(x)
    def test_nll(logits):
        logits = logits.detach().double().numpy()
        return np.where(choices, np.logaddexp(0, -logits), np.logaddexp(0, logits))[test_mask].reshape(count, -1).mean(1)
    teacher_nll = test_nll(teacher_logits)
    records = {"teacher_h32": dict(parameters=sum(p.numel() for p in teacher.parameters()),
                                   selected_step=teacher_fit["best_step"],
                                   test_first_stage_nll=float(teacher_nll.mean()),
                                   per_subject_test_nll=teacher_nll.tolist())}
    for hidden in settings["student_hidden_sizes"]:
        torch.manual_seed(seed)
        model = BatchedStudentGRU(count, 4, hidden)
        checkpoint = scratch / f"pilot_h{hidden}.pt"
        fitted = fit_model(model, x, choices, train_mask, val_mask, checkpoint_path=checkpoint,
                           teacher_prop=1., teacher_logits=torch.where(torch.from_numpy(train_mask), teacher_logits, 0.),
                           track_population_selection=True, **options)
        assert fitted["completed_steps"] == options["steps"]
        with torch.no_grad():
            individual = test_nll(model(x))
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["population_best_state"])
            shared = test_nll(model(x))
        for group in ("rare", "typical"):
            who = int(np.flatnonzero(population["rare"] == (group == "rare"))[0])
            records[f"student_{group}_h{hidden}"] = dict(subject=who,
                parameters_per_subject=sum(p.numel() for p in model.parameters())//count,
                population_selected_step=fitted["population_best_step"],
                individual_selected_step=fitted["best_step"][who],
                population_test_first_stage_nll=float(shared[who]),
                individual_test_first_stage_nll=float(individual[who]))
    return dict(records=records, candidate_steps=list(range(0, options["steps"]+1, 10)),
                validation_target="observed first-stage choices", test_trials_per_person=int(test_mask[0].sum()))


def native_experiment(agent_class, trainer_class, dataset_class, functions, paths, scratch, settings):
    import joblib
    population = simulate_population(settings["n_subjects"], 200, .25, settings["data_seed"])
    original = native_dataset(dataset_class, population, scratch, False)
    check_native_encoding(original, population)
    augmented = native_dataset(dataset_class, population, scratch, True)
    splits = functions["split_over_trials"](200, [150, 25, 25], seed=0)
    assert len(np.unique(np.concatenate(splits))) == 200
    fmt = dict(behav_format="tensor", one_hot=False, output_h0=True)
    def get(dataset, people, split):
        return dataset.get_behav_data(people, fmt, selected_trial_indices=splits[split])
    n = settings["n_subjects"]
    data = dict(train=get(augmented, range(4*n), 0), val=get(original, range(n), 1),
                test=get(original, range(n), 2))
    for split, count in zip(("train", "val", "test"), (150, 25, 25)):
        mask = data[split]["mask"][0]
        assert torch.all(mask.sum(0) == 2*count)
        # Trial masks must include/exclude both stages together.
        assert torch.equal(mask[::2], mask[1::2])
        indices = splits[("train", "val", "test").index(split)]
        expected = np.isin(np.arange(200), indices)[:, None]
        np.testing.assert_array_equal(mask[::2].numpy().astype(bool), np.broadcast_to(expected, mask[::2].shape))
    base = dict(rnn_type="GRU", input_dim=10, output_dim=6, device="cpu",
                output_h0=True, trainable_h0=False, readout_FC=True,
                lr=.005, l1_weight=1e-5, weight_decay=0, penalized_weight="rec",
                max_epoch_num=settings["max_epoch_num"], early_stop_counter=200,
                batch_size=0, save_model_pass="none", training_diagnose=[],
                seed=settings["initialization_seed"], distill_temp=1)
    def fit(config, fit_data):
        trainer = trainer_class(agent_class(config))
        # Upstream logs every epoch and evaluates test for monitoring. Its
        # selection condition uses validation only; check argmin independently.
        trainer.train(fit_data, verbose_level=3)
        bp = trainer.best_model_pass
        expected_epoch = int(np.argmin(bp["val"]["loss_log"]))
        assert bp["epoch"] == expected_epoch
        validation = trainer._eval_1step(fit_data["val"]["input"], fit_data["val"]["target"], fit_data["val"]["mask"])["behav_loss"].item()
        assert abs(validation-min(bp["val"]["loss_log"])) < 1e-12
        return trainer
    teacher = fit(dict(base, hidden_dim=50, model_path="teacher"), data)
    # Create only our own synthetic teacher artifact, then call the author's
    # original loader to apply the native train AND validation target rules.
    with torch.no_grad():
        full_scores = teacher.model(augmented.torch_beahv_input).numpy()
    teacher_path = paths.ANA_SAVE_PATH / "teacher_outerfold0"
    teacher_path.mkdir(parents=True)
    joblib.dump(dict(scores=[full_scores[:, i] for i in range(4*n)],
                     mask=[(augmented.torch_beahv_mask[0][:, i].numpy(),
                            augmented.torch_beahv_mask[1][:, i].numpy()) for i in range(4*n)]),
                teacher_path / "total_scores.pkl")
    def describe(trainer, raw):
        bp = trainer.best_model_pass
        with torch.no_grad():
            scores = trainer.model(raw["test"]["input"])[:-1]-raw["test"]["mask"][1]
            probabilities = torch.softmax(scores, -1)[::2, :, 1].numpy()
            y = raw["test"]["target"][::2].numpy()
            mask = raw["test"]["mask"][0][::2].numpy().astype(bool)
        p = np.clip(probabilities, 1e-12, 1-1e-12)
        nll = -np.mean((y*np.log(p)+(1-y)*np.log1p(-p))[mask])
        return dict(parameters=trainer.num_params, selected_epoch=int(bp["epoch"]),
                    candidate_epochs=len(bp["val"]["loss_log"]),
                    validation_loss=bp["val"]["behav_loss"],
                    test_both_stage_nll=bp["test"]["behav_loss"],
                    test_first_stage_nll=float(nll),
                    train_loss_initial=bp["train"]["loss_log"][0],
                    train_loss_last_candidate=bp["train"]["loss_log"][-1])
    records = {"teacher_h50": describe(teacher, data)}
    for group in ("rare", "typical"):
        person = int(np.flatnonzero(population["rare"] == (group == "rare"))[0])
        indexes = augmented.get_after_augmented_block_number([person])
        for hidden in settings["student_hidden_sizes"]:
            raw = dict(train=get(augmented, indexes, 0), val=get(augmented, indexes, 1),
                       test=get(augmented, indexes[:1], 2))
            config = dict(base, hidden_dim=hidden, model_path=f"student_{group}_h{hidden}",
                          teacher_model_path="teacher_outerfold0")
            soft = functions["_load_teacher_network_score"](config, indexes, indexes, deepcopy(raw), 0)
            assert soft["train"]["target"].ndim == soft["val"]["target"].ndim == 3
            assert torch.equal(soft["test"]["target"], raw["test"]["target"])
            student = fit(config, soft)
            records[f"student_{group}_h{hidden}"] = dict(describe(student, raw), subject=person,
                                                       validation_target="teacher probabilities")
    # The initial native prediction uses h0; a current trial cannot affect it.
    altered = deepcopy(population)
    for key in ("choice", "state", "second_choice", "reward"):
        altered[key][:, 71:] = 1-altered[key][:, 71:]
    changed = native_dataset(dataset_class, altered, scratch, False)
    with torch.no_grad():
        before = teacher.model(original.torch_beahv_input)
        after = teacher.model(changed.torch_beahv_input)
    causal_error = (before[:2*71+1]-after[:2*71+1]).abs().max().item()
    assert causal_error == 0
    return dict(settings=settings, records=records, pilot_companion=pilot_companion(population, splits, settings, scratch),
                train_trials=splits[0].tolist(),
                validation_trials=splits[1].tolist(), test_trials=splits[2].tolist(),
                encoded_shape=list(original.torch_beahv_input.shape),
                augmented_subjects=augmented.batch_size, causal_prefix_error=causal_error,
                selection_check="All five restored models equal first minimum validation candidate.",
                update_count_note="Upstream evaluates epochs 0..100 then makes one final discarded update; selected candidates cover 0..100 updates.")


def main():
    if not __debug__:
        raise RuntimeError("Verification must run with Python assertions enabled.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin_path = ROOT / "configs/published_method_pin.json"
    pin = json.loads(pin_path.read_text())
    checkout = verify_checkout(args.authors_checkout, pin)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="tinyrnn-alignment-") as tmp:
        scratch = Path(tmp)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            agent, trainer, dataset, functions, paths = load_authors(checkout, scratch)
            binary = matched_binary_check(agent)
            experiment = native_experiment(agent, trainer, dataset, functions, paths, scratch, pin["fixture"])
    import joblib, tqdm
    report = dict(schema_version=1, status="passed", source=pin,
                  script_sha256=sha256(__file__), pin_sha256=sha256(pin_path),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   numpy=np.__version__, joblib=joblib.__version__, tqdm=tqdm.__version__,
                                   device="cpu", threads=1),
                  matched_binary=binary, native_synthetic=experiment,
                  limitation="Bounded synthetic implementation/recipe alignment check; not a published-result reproduction, converged fit, or comparison of predictive superiority.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed", authors_commit=pin["commit"], output=str(args.output))))


if __name__ == "__main__":
    main()
