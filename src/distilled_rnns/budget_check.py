"""One bounded continuation of the paired oracle-control student trajectories.

The parent is read-only. Current weights, Adam, RNG, history, and both selected
states are copied byte-for-byte before continuing. More steps also add checkpoint
candidates; this sensitivity check is not evidence of pure convergence.
"""
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .baseline_audit import _score, _test_report, reference_predictions
from .config import ExperimentConfig
from .evaluation import make_trial_masks
from .experiment import (atomic_json, environment_info, resolve_device, run_lock,
                         save_environment_inventory, source_fingerprint,
                         t4_training_cost_proxy, _save_npz)
from .followup import _range_summary, _plot_convergence, convergence_details
from .models import BatchedStudentGRU, TeacherGRU
from .oracle_control import (OracleControlConfig, MODES, METRICS, _selection,
                             _comparisons, _gain_summary, oracle_training_logits,
                             aggregate_oracle_control)
from .simulation import simulate_population
from .training import fit_model, _signature, masked_objective


ORIGINAL_SOURCE_SHA256 = "c68c9b2918c7199f0396a9a3bc28f98f1bd28f59469adf6524bfe6b67ac6c3d9"
EXTENDED_ARMS = ("distilled", "oracle")
# CPU reproduction of the original T4 artifact differed by <=3.16e-6 in
# probability and <=1.79e-7 in mean checkpoint NLL. These are fixed audit
# tolerances, never a threshold for deciding the scientific result.
PROBABILITY_ATOL = 5e-6
NLL_ATOL = 5e-7


@dataclass(frozen=True)
class BudgetCheckConfig:
    parent_config: OracleControlConfig
    expected_parent_source_sha256: str
    student_steps: int
    max_seconds_per_phase: float = 600.0
    max_seconds_per_invocation: float = 1800.0

    def __post_init__(self):
        if not isinstance(self.parent_config, OracleControlConfig):
            raise ValueError("parent_config must be an OracleControlConfig")
        if (not isinstance(self.expected_parent_source_sha256, str)
                or len(self.expected_parent_source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.expected_parent_source_sha256)):
            raise ValueError("expected_parent_source_sha256 must be a lowercase SHA256")
        if (isinstance(self.student_steps, bool) or not isinstance(self.student_steps, int)
                or self.student_steps <= self.parent_config.base.student_steps
                or self.parent_config.base.student_steps % self.parent_config.base.eval_every):
            raise ValueError("student_steps must extend a parent ending on its scheduled evaluation grid")
        for name in ("max_seconds_per_phase", "max_seconds_per_invocation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def to_dict(self):
        return {"parent_config": self.parent_config.to_dict(),
                "expected_parent_source_sha256": self.expected_parent_source_sha256,
                "student_steps": self.student_steps,
                "max_seconds_per_phase": self.max_seconds_per_phase,
                "max_seconds_per_invocation": self.max_seconds_per_invocation}

    @classmethod
    def from_file(cls, path):
        values = _json(path)
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown budget-check configuration settings")
        values = dict(values)
        parent = values.get("parent_config")
        if not isinstance(parent, dict) or set(parent) - {f.name for f in fields(OracleControlConfig)}:
            raise ValueError("Unknown or invalid parent configuration settings")
        parent = dict(parent)
        base = parent.get("base")
        if not isinstance(base, dict) or set(base) - {f.name for f in fields(ExperimentConfig)}:
            raise ValueError("Unknown or invalid parent base configuration settings")
        parent["base"] = ExperimentConfig(**base)
        values["parent_config"] = OracleControlConfig(**parent)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError("Missing budget-check configuration settings") from exc


def legacy_source_fingerprint():
    """The exact original numerical package, excluding only this new module."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name != "budget_check.py":
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink paths are not accepted for parent or child artifacts")
    if path.exists():
        for root, directories, files in os.walk(path, followlinks=False):
            if any((Path(root) / name).is_symlink() for name in directories + files):
                raise ValueError("Symlinks inside artifact directories are not accepted")


def _file_hashes(folder):
    _reject_symlinks(folder)
    return {p.relative_to(folder).as_posix(): _sha(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _finite(value, context):
    if isinstance(value, torch.Tensor):
        good = bool(torch.isfinite(value).all())
    elif isinstance(value, float):
        good = math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item, context)
        return
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item, context)
        return
    else:
        return
    if not good:
        raise ValueError(f"Nonfinite {context}")


def _close(actual, expected, context, atol=NLL_ATOL):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected).issubset(actual):
            raise ValueError(f"Invalid {context}")
        for key, value in expected.items():
            _close(actual[key], value, f"{context}.{key}", atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Invalid {context}")
        for a, b in zip(actual, expected):
            _close(a, b, context, atol)
    elif isinstance(expected, (float, np.floating)):
        if not isinstance(actual, (int, float, np.number)) or not math.isfinite(actual) or abs(actual - expected) > atol:
            raise ValueError(f"Nonreproducible {context}")
    elif actual != expected:
        raise ValueError(f"Invalid {context}")


def _probability_close(actual, expected, context):
    actual = np.asarray(actual)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=0, atol=PROBABILITY_ATOL)):
        raise ValueError(f"Nonreproducible selected predictions: {context}")


def _model(base, features, arm):
    return (TeacherGRU(features, base.teacher_hidden) if arm == "teacher" else
            BatchedStudentGRU(base.n_subjects, features, base.student_hidden))


def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_inventories(folder, sessions):
    for session in sessions:
        inventory = session.get("package_inventory", {})
        relative = Path(inventory.get("path", ""))
        digest = inventory.get("sha256")
        if (inventory.get("status") != "recorded" or relative.is_absolute()
                or ".." in relative.parts or len(relative.parts) != 2
                or relative.parts[0] != "environments"
                or relative.name != f"packages-{digest}.txt"
                or not (folder / relative).is_file() or _sha(folder / relative) != digest):
            raise ValueError("Parent package inventory is missing or differs from its hash")


def _validate_checkpoint(saved, model, x, y, masks, soft, base, arm, device="cpu"):
    teacher = arm == "teacher"
    completed = saved.get("completed_steps")
    limit = base.teacher_steps if teacher else base.student_steps
    if (isinstance(completed, bool) or not isinstance(completed, int) or completed < 1
            or completed > limit or (not teacher and completed != limit)):
        raise ValueError(f"Invalid parent completed steps for {arm}")
    if not teacher and saved.get("early_stopped") is not False:
        raise ValueError("Student parent must finish its fixed budget")
    settings = {"lr": base.learning_rate, "seed": base.seed + (2 if teacher else 3),
                "eval_every": base.eval_every, "patience": base.patience,
                "teacher_prop": 1.0 if soft is not None else 0.0, "temperature": 1.0}
    if arm == "solo":
        # Added by the caller because solo regularization is a study setting.
        settings["l2_strength"] = float(masks["solo_l2"])
        if not settings["l2_strength"]:
            settings.pop("l2_strength")
    if not teacher:
        settings["track_population_selection"] = True
    expected = _signature(model, settings, [x, y, torch.as_tensor(masks["train"]),
                                          torch.as_tensor(masks["checkpoint"]), soft])
    if saved.get("signature") != expected:
        raise ValueError(f"Parent checkpoint signature mismatch: {arm}")
    _finite(saved, f"checkpoint {arm}")
    if not isinstance(saved.get("elapsed_seconds"), (int, float)) or saved["elapsed_seconds"] < 0:
        raise ValueError("Invalid checkpoint elapsed time")
    history = saved.get("history", [])
    expected_steps = [0, *range(base.eval_every, completed + 1, base.eval_every)]
    if expected_steps[-1] != completed:
        expected_steps.append(completed)
    if [row.get("step") for row in history] != expected_steps:
        raise ValueError("Parent checkpoint history is incomplete")
    expected_state = model.state_dict()
    for key in ("current_state", "best_state") + (() if teacher else ("population_best_state",)):
        if (set(saved[key]) != set(expected_state) or any(
                not isinstance(saved[key][name], torch.Tensor)
                or saved[key][name].dtype != tensor.dtype or saved[key][name].shape != tensor.shape
                for name, tensor in expected_state.items())):
            raise ValueError(f"Invalid parent parameter shapes or dtypes: {arm}")
        model.load_state_dict(saved[key], strict=True)
    selected_steps = np.atleast_1d(saved["best_step"].numpy() if isinstance(saved["best_step"], torch.Tensor) else saved["best_step"])
    if selected_steps.size != (1 if teacher else base.n_subjects) or not np.isin(selected_steps, expected_steps).all():
        raise ValueError("Invalid individual selection steps")
    if not teacher and saved["population_best_step"] not in expected_steps:
        raise ValueError("Invalid population selection step")
    selected = min(history, key=lambda row: row["validation_nll"])
    if teacher:
        if saved["best_step"] != selected["step"]:
            raise ValueError("Teacher selected step differs from cumulative history")
        _close(saved["best_val"].item(), selected["validation_nll"], "teacher history best NLL")
    else:
        if saved["population_best_step"] != selected["step"]:
            raise ValueError("Population selected step differs from cumulative history")
        _close(saved["population_best_val"].item(), selected["validation_nll"], "population history best NLL")
    optimizer = torch.optim.Adam(model.parameters(), lr=base.learning_rate)
    optimizer.load_state_dict(saved["optimizer"])
    expected_settings = {"lr": base.learning_rate, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0, "amsgrad": False, "maximize": False, "foreach": None,
        "capturable": False, "differentiable": False, "fused": None}
    if (len(optimizer.param_groups) != 1 or any(optimizer.param_groups[0].get(key) != value
            for key, value in expected_settings.items())
            or optimizer.param_groups[0].get("decoupled_weight_decay", False)):
        raise ValueError("Parent Adam settings differ from original optimizer")
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Parent checkpoint has incomplete Adam state")
    for parameter, state in optimizer.state.items():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or state["step"].item() != completed
                or state["exp_avg"].shape != parameter.shape
                or state["exp_avg_sq"].shape != parameter.shape
                or state["exp_avg"].dtype != parameter.dtype
                or state["exp_avg_sq"].dtype != parameter.dtype):
            raise ValueError("Invalid parent Adam state")
    if saved["rng_state"].dtype != torch.uint8 or saved["rng_state"].shape != torch.get_rng_state().shape:
        raise ValueError("Invalid parent RNG state")
    # Keep checkpoint identity, structural and Adam checks on their saved CPU
    # representation. Reconstruct predictions on the requested run device:
    # a Colab CPU can differ from the original T4 beyond the fixed tolerance.
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    train_mask = torch.as_tensor(masks["train"], device=device)
    val_mask = torch.as_tensor(masks["checkpoint"], device=device)
    predictions = {}
    with torch.no_grad():
        model.load_state_dict(saved["current_state"])
        logits = model(x)
        losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        for name, mask in (("train_nll", train_mask), ("validation_nll", val_mask)):
            _close(history[-1][name], ((losses * mask).sum(1) / mask.sum(1)).mean().item(),
                   f"{arm} current-state {name}")
        for mode, key in (("individual", "best_state"),) if teacher else (("individual", "best_state"), ("population", "population_best_state")):
            model.load_state_dict(saved[key])
            logits = model(x)
            predictions[arm if teacher else f"{arm}_{mode}"] = torch.sigmoid(logits).cpu().numpy()
            losses = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * val_mask).sum(1) / val_mask.sum(1)
            if mode == "population":
                _close(saved["population_best_val"].item(), losses.mean().item(), f"{arm} population best NLL")
            elif teacher:
                _close(saved["best_val"].item(), losses.mean().item(), "teacher best NLL")
            elif not torch.allclose(saved["best_val"], losses.cpu(), rtol=0, atol=5e-6):
                raise ValueError(f"Nonreproducible {arm} individual checkpoint NLL")
    return predictions


def validate_parent(config, parent_dir, device="cpu"):
    """Audit saved CPU identity and reconstruct on device; never fit or write."""
    if not isinstance(config, BudgetCheckConfig):
        raise ValueError("config must be a BudgetCheckConfig")
    if legacy_source_fingerprint() != ORIGINAL_SOURCE_SHA256:
        raise ValueError("Original numerical source changed; continuation is incompatible")
    selected = resolve_device(device)
    parent = Path(parent_dir).absolute()
    _reject_symlinks(parent)
    if not parent.is_dir():
        raise ValueError("Parent directory does not exist")
    before = _file_hashes(parent)
    try:
        protocol, summary = _json(parent / "protocol.json"), _json(parent / "summary.json")
        expected_seeds = list(config.parent_config.replication_seeds)
        if (protocol.get("study") != "oracle_target_control" or protocol.get("schema_version") != 1
                or protocol.get("config") != config.parent_config.to_dict()
                or protocol.get("source_sha256") != config.expected_parent_source_sha256
                or protocol.get("selection") != _selection(config.parent_config)):
            raise ValueError("Parent protocol configuration or source differs from trusted specification")
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if (summary.get("status") != "complete" or summary.get("planned_seeds") != expected_seeds
                or summary.get("completed_seeds") != expected_seeds or summary.get("pending_seeds") != []
                or summary.get("protocol_sha256") != protocol_hash):
            raise ValueError("Every planned parent seed must be complete")
        if sorted(p.name for p in parent.glob("seed-*")) != sorted(f"seed-{s}" for s in expected_seeds):
            raise ValueError("Parent seed directories differ from expected seed set")
        parent_seconds, checkpoint_seconds, reports = 0.0, {}, []
        for seed in expected_seeds:
            folder = parent / f"seed-{seed}"
            base = replace(config.parent_config.base, seed=seed)
            manifest, report = _json(folder / "manifest.json"), _json(folder / "report.json")
            expected_identity = {"config": base.to_dict(), "selection": _selection(config.parent_config),
                                 "protocol_sha256": protocol_hash, "role": "replication", "third_split_role": "test"}
            if (manifest.get("identity") != expected_identity or not manifest.get("sessions")
                    or manifest["sessions"][-1].get("status") != "complete"
                    or report.get("status") != "complete" or report.get("seed") != seed
                    or report.get("role") != "replication"
                    or report.get("selection") != _selection(config.parent_config)
                    or _json(folder / "selection.json") != _selection(config.parent_config)):
                raise ValueError(f"Invalid or incomplete parent seed {seed}")
            _validate_inventories(folder, manifest["sessions"])
            if _json(folder / "training.json") != report["phases"]:
                raise ValueError("Parent training metadata differs from its report")
            generated = simulate_population(base.n_subjects, base.n_trials, base.rare_fraction, seed)
            original_masks = make_trial_masks(base.n_subjects, base.n_trials, seed + 1)
            masks = {"train": original_masks["train"], "checkpoint": original_masks["val"], "test": original_masks["test"]}
            data = _arrays(folder / "data.npz")
            expected_data = {**generated, **{f"mask_{k}": v for k, v in masks.items()}}
            # Platform libm differs at floating round-off in simulator float64
            # probabilities (observed maximum 4.45e-16). Histories, masks and
            # discrete observations remain exact; fit signatures use the
            # original saved bytes, never these regenerated audit values.
            if set(data) != set(expected_data) or any(
                    data[k].dtype != v.dtype or data[k].shape != v.shape or not np.isfinite(data[k]).all()
                    or not (np.allclose(data[k], v, rtol=0, atol=1e-12)
                            if v.dtype == np.float64 else np.array_equal(data[k], v))
                    for k, v in expected_data.items()):
                raise ValueError(f"Parent generated data or masks changed: seed {seed}")
            x, y = torch.as_tensor(data["x"], dtype=torch.float32), torch.as_tensor(data["choice"], dtype=torch.float32)
            soft = _arrays(folder / "soft_training_logits.npz")
            if set(soft) != set(EXTENDED_ARMS) or any(v.shape != y.shape or v.dtype != np.float32 or not np.isfinite(v).all() or np.count_nonzero(v[~masks["train"]]) for v in soft.values()):
                raise ValueError("Parent soft targets must be finite and exactly train-masked")
            if not np.array_equal(soft["oracle"], oracle_training_logits(data["true_prob"], masks["train"]).numpy()):
                raise ValueError("Parent oracle training targets changed")
            predictions, checkpoints = {}, {}
            for arm in ("teacher", "solo", *EXTENDED_ARMS):
                saved = checkpoints[arm] = _checkpoint(folder / f"{arm}.pt")
                predictions.update(_validate_checkpoint(saved, _model(base, x.shape[-1], arm), x, y,
                    {**masks, "solo_l2": config.parent_config.solo_l2},
                    torch.as_tensor(soft[arm]) if arm in soft else None, base, arm, device=selected))
                phase = report["phases"][arm]
                if phase["completed_steps"] != saved["completed_steps"] or phase["history"] != saved["history"]:
                    raise ValueError("Parent report and checkpoint history differ")
                _close(phase["best_step"], saved["best_step"].tolist() if isinstance(saved["best_step"], torch.Tensor)
                       else saved["best_step"], "parent report selected steps")
                _close(phase["best_validation_nll"], saved["best_val"].tolist(), "parent report best NLL")
                if arm != "teacher":
                    _close(phase["population_best_step"], saved["population_best_step"], "parent report population step")
                    _close(phase["population_best_validation_nll"], saved["population_best_val"].item(), "parent report population NLL")
                if not math.isfinite(phase["elapsed_seconds"]) or phase["elapsed_seconds"] < saved["elapsed_seconds"]:
                    raise ValueError("Parent report elapsed time precedes saved checkpoint")
                parent_seconds += saved["elapsed_seconds"]
            _probability_close(torch.sigmoid(torch.as_tensor(soft["distilled"]))[masks["train"]].numpy(),
                               predictions["teacher"][masks["train"]], "teacher training soft targets")
            predictions.update(reference_predictions(data["choice"], masks["train"]))
            candidates = _arrays(folder / "candidate_predictions.npz")
            if set(candidates) != set(predictions):
                raise ValueError("Parent candidate prediction keys differ")
            for name, expected in predictions.items():
                _probability_close(candidates[name], expected, name)
            predictions.update({arm: predictions[f"{arm}_population"] for arm in ("solo", *EXTENDED_ARMS)})
            stored_predictions = _arrays(folder / "predictions.npz")
            if set(stored_predictions) != set(predictions):
                raise ValueError("Parent selected prediction keys differ")
            for name, expected in predictions.items():
                _probability_close(stored_predictions[name], expected, name)
            # Re-score the preserved predictions exactly. The tolerance above
            # handles backend differences in weight-to-probability reproduction.
            reproduced = _report(data, stored_predictions, base)
            for key in ("per_subject", "groups", "checkpoint_scores"):
                _close(report[key], reproduced[key], f"parent report {key}")
            training_seconds = sum(phase["elapsed_seconds"] for phase in report["phases"].values())
            _close(report["timing"], {"cumulative_training_seconds": training_seconds,
                **t4_training_cost_proxy(training_seconds, base.t4_units_per_hour, manifest["sessions"])},
                "parent report timing")
            reports.append(report)
            checkpoint_seconds[str(seed)] = {arm: saved["elapsed_seconds"] for arm, saved in checkpoints.items()}
        _close(summary["groups"], aggregate_oracle_control(reports, expected_seeds), "parent aggregate groups")
        _close(summary["timing"]["cumulative_training_seconds"],
               sum(report["timing"]["cumulative_training_seconds"] for report in reports), "parent aggregate timing")
        proxies = [report["timing"]["t4_equivalent_units_at_user_rate"] for report in reports]
        _close(summary["timing"]["t4_equivalent_units_at_user_rate"],
               sum(proxies) if all(value is not None for value in proxies) else None, "parent aggregate T4 proxy")
        if _file_hashes(parent) != before:
            raise ValueError("Parent artifacts changed during validation")
        return {"parent_dir": str(parent), "file_sha256": before, "protocol_sha256": protocol_hash,
                "checkpoint_elapsed_seconds": checkpoint_seconds, "parent_training_seconds": parent_seconds}
    except (KeyError, FileNotFoundError, RuntimeError, TypeError, IndexError) as exc:
        raise ValueError(f"Invalid or incomplete parent artifacts: {exc}") from exc


def _report(data, predictions, base):
    result = _test_report(data, predictions, data["mask_test"], base.diagnostic_threshold)
    for row in [*result["per_subject"], *result["groups"].values()]:
        row.pop("likelihood_gain", None)
        row.pop("fidelity_loss", None)
        row.update(_comparisons(row["models"]))
    result["checkpoint_scores"] = _score(data["choice"], predictions, data["mask_checkpoint"])
    return result


def _copy_verified(source, target, expected_hash):
    """A failed copy never installs a partial artifact at the resumable path."""
    temporary = target.with_name(target.name + ".copying")
    try:
        shutil.copyfile(source, temporary)
        if _sha(temporary) != expected_hash:
            raise ValueError("Parent changed during snapshot copying")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot(parent, folder, audit):
    snapshot = folder / "parent_snapshot"
    snapshot.mkdir(exist_ok=True)
    complete = snapshot / "snapshot_manifest.json"
    if complete.exists() and _json(complete) != audit:
        raise ValueError("Parent snapshot identity changed")
    for relative, expected_hash in audit["file_sha256"].items():
        target = snapshot / relative
        if target.exists():
            if _sha(target) != expected_hash:
                raise ValueError(f"Parent snapshot changed: {relative}")
        elif complete.exists():
            raise ValueError(f"Parent snapshot file missing: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified(parent / relative, target, expected_hash)
    atomic_json(complete, audit) if not complete.exists() else None
    return snapshot


def _objective_snapshots(saved, model, x, y, mask, soft):
    result = {}
    with torch.no_grad():
        for key in ("current_state", "best_state", "population_best_state"):
            model.load_state_dict(saved[key])
            result[key] = float(masked_objective(model(x), y, mask, teacher_logits=soft,
                teacher_prop=1.0, temperature=1.0, independent_students=True).item() / x.shape[0])
    return result


def _seed(config, seed, folder, snapshot, protocol_hash, audit, device, deadline):
    base = replace(config.parent_config.base, seed=seed)
    parent = snapshot / f"seed-{seed}"
    folder.mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    identity = {"protocol_sha256": protocol_hash, "seed": seed}
    if manifest_path.exists():
        manifest = _json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError("Child seed resume identity differs")
        if manifest.get("status") == "complete":
            for name, expected in manifest["completed_file_sha256"].items():
                if not (folder / name).is_file() or _sha(folder / name) != expected:
                    raise ValueError("Completed child artifact changed")
            return _json(folder / "report.json")
        if not all((folder / f"{arm}.pt").is_file() for arm in EXTENDED_ARMS):
            raise ValueError("Initialized child checkpoint is missing; refusing to recopy parent")
    else:
        # Before the manifest exists, any partial copy must still match the
        # parent. A continued checkpoint is never overwritten on resume.
        for arm in EXTENDED_ARMS:
            target = folder / f"{arm}.pt"
            if target.exists():
                if _sha(target) != audit["file_sha256"][f"seed-{seed}/{arm}.pt"]:
                    raise ValueError("Unidentified child checkpoint differs from parent")
            else:
                _copy_verified(parent / target.name, target,
                               audit["file_sha256"][f"seed-{seed}/{arm}.pt"])
        manifest = {"identity": identity, "sessions": [], "status": "incomplete"}
        atomic_json(manifest_path, manifest)
    started = time.perf_counter()
    packages = {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(device),
               "package_inventory": save_environment_inventory(folder, packages)}
    manifest["sessions"].append(session)
    atomic_json(manifest_path, manifest)
    data, soft = _arrays(parent / "data.npz"), _arrays(parent / "soft_training_logits.npz")
    x, y = torch.as_tensor(data["x"], device=device), torch.as_tensor(data["choice"], dtype=torch.float32, device=device)
    train = torch.as_tensor(data["mask_train"], device=device)
    phases, objectives = {}, {}
    predictions = _arrays(parent / "predictions.npz")
    for arm in EXTENDED_ARMS:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        path = folder / f"{arm}.pt"
        saved = _checkpoint(path)
        inherited = _checkpoint(parent / path.name)
        if (saved.get("signature") != inherited["signature"]
                or not base.student_steps <= saved.get("completed_steps", -1) <= config.student_steps
                or saved.get("history", [])[:len(inherited["history"])] != inherited["history"]):
            raise ValueError("Child checkpoint lost its inherited trajectory or selection history")
        _finite(saved, "child checkpoint")
        torch.manual_seed(seed + 3)
        model = _model(base, x.shape[-1], arm).to(device)
        target = torch.as_tensor(soft[arm], device=device)
        # Completed fits are never re-saved: resume is idempotent and timing
        # cannot grow merely because a later arm needed another invocation.
        if saved["completed_steps"] < config.student_steps:
            print(f"\nSeed {seed}: continuing {arm}, {saved['completed_steps']} -> {config.student_steps}", flush=True)
            phase = fit_model(model, x, y, data["mask_train"], data["mask_checkpoint"],
                steps=config.student_steps, lr=base.learning_rate, seed=seed + 3,
                eval_every=base.eval_every, patience=base.patience, checkpoint_path=path,
                teacher_logits=target, teacher_prop=1.0, temperature=1.0, l2_strength=0.0,
                track_population_selection=True, max_seconds=min(config.max_seconds_per_phase, remaining))
            saved = _checkpoint(path)
            if phase["stopped_reason"] == "time_limit":
                break
        phases[arm] = {"history": saved["history"], "best_step": saved["best_step"].tolist(),
            "completed_steps": saved["completed_steps"], "stopped_reason": "steps_completed",
            "elapsed_seconds": saved["elapsed_seconds"], "best_validation_nll": saved["best_val"].tolist(),
            "population_best_validation_nll": saved["population_best_val"].item(),
            "population_best_step": saved["population_best_step"], "checkpoint_path": str(path), "l2_strength": 0.0}
        objectives[arm] = {"baseline": _objective_snapshots(inherited, model, x, y, train, target),
                           "extended": _objective_snapshots(saved, model, x, y, train, target)}
        with torch.no_grad():
            for mode, key in (("individual", "best_state"), ("population", "population_best_state")):
                model.load_state_dict(saved[key])
                predictions[f"{arm}_{mode}"] = torch.sigmoid(model(x)).cpu().numpy()
            predictions[arm] = predictions[f"{arm}_population"]
    session.update({"status": "complete" if len(phases) == 2 else "incomplete",
                    "wall_seconds": time.perf_counter() - started})
    if len(phases) != 2:
        atomic_json(manifest_path, manifest)
        return {"status": "incomplete", "seed": seed}
    report = _report(data, predictions, base)
    convergence = convergence_details(phases, replace(base, student_steps=config.student_steps))
    for arm in EXTENDED_ARMS:
        convergence[arm]["population_best_step"] = phases[arm]["population_best_step"]
        convergence[arm]["population_selected_at_limit"] = phases[arm]["population_best_step"] == config.student_steps
    incremental = sum(phases[arm]["elapsed_seconds"] - audit["checkpoint_elapsed_seconds"][str(seed)][arm]
                      for arm in EXTENDED_ARMS)
    report.update({"status": "complete", "role": "replication", "seed": seed,
        "selection": _selection(config.parent_config), "phases": phases, "convergence": convergence,
        "baseline": _json(parent / "report.json"), "soft_objective": objectives,
        "timing": {"incremental_training_seconds": incremental,
            "parent_training_seconds": sum(audit["checkpoint_elapsed_seconds"][str(seed)].values()),
            **t4_training_cost_proxy(incremental, base.t4_units_per_hour, manifest["sessions"])}})
    _save_npz(folder / "predictions.npz", **predictions)
    atomic_json(folder / "training.json", phases)
    _plot_convergence(phases, folder / "convergence.png")
    atomic_json(folder / "report.json", report)
    manifest.update({"status": "complete", "completed_file_sha256": {
        name: _sha(folder / name) for name in ("distilled.pt", "oracle.pt", "predictions.npz", "report.json", "training.json")}})
    atomic_json(manifest_path, manifest)
    return report


def _difference(extended, baseline):
    return None if extended is None or baseline is None else extended - baseline


def aggregate_budget_check(reports, expected_seeds):
    """Paired per-seed changes, equal-person then equal-seed descriptive means."""
    if (not expected_seeds or [r.get("seed") for r in reports] != list(expected_seeds)
            or any(r.get("status") != "complete" or r.get("role") != "replication" for r in reports)):
        raise ValueError("Every planned replication must complete before aggregation")
    groups = {}
    for group in ("rare", "typical"):
        rows = []
        for report in reports:
            old, new = report["baseline"]["groups"][group], report["groups"][group]
            old_gains, new_gains = _comparisons(old["models"])["target_comparisons"], _comparisons(new["models"])["target_comparisons"]
            rows.append({"seed": report["seed"], "subject_count": new["subject_count"],
                "test_count": new["test_count"], "diagnostic_count": new["diagnostic_count"],
                "diagnostic_subject_count": new["models"]["oracle"]["diagnostic_subject_count"],
                "baseline_models": old["models"], "extended_models": new["models"],
                "baseline_target_comparisons": old_gains, "extended_target_comparisons": new_gains,
                "paired_gain_change": {mode: {key: _difference(new_gains[mode][key], old_gains[mode][key])
                    for key in old_gains[mode]} for mode in MODES},
                "arm_changes": {mode: {arm: {metric: _difference(new["models"][f"{arm}_{mode}"][metric],
                    old["models"][f"{arm}_{mode}"][metric]) for metric in METRICS} for arm in EXTENDED_ARMS} for mode in MODES}})
        groups[group] = {"per_seed": rows,
            **{section: {model: {metric: _range_summary([row[section][model][metric] for row in rows])
                for metric in METRICS} for model in rows[0][section]} for section in ("baseline_models", "extended_models")},
            **{section: {mode: {f"{metric}_gain": _gain_summary([row[section][mode][f"{metric}_gain"] for row in rows])
                for metric in METRICS} for mode in MODES} for section in ("baseline_target_comparisons", "extended_target_comparisons", "paired_gain_change")},
            "arm_changes": {mode: {arm: {metric: _gain_summary([row["arm_changes"][mode][arm][metric] for row in rows])
                for metric in METRICS} for arm in EXTENDED_ARMS} for mode in MODES},
            "support": {key: _range_summary([row[key] for row in rows])
                for key in ("subject_count", "test_count", "diagnostic_count", "diagnostic_subject_count")}}
    return groups


def run_budget_check(config: BudgetCheckConfig, parent_dir, output_dir, device="auto"):
    if not isinstance(config, BudgetCheckConfig):
        raise ValueError("config must be a BudgetCheckConfig")
    parent, folder = Path(parent_dir).absolute(), Path(output_dir).absolute()
    _reject_symlinks(parent)
    _reject_symlinks(folder)
    parent, folder = parent.resolve(), folder.resolve()
    if parent == folder or parent in folder.parents or folder in parent.parents:
        raise ValueError("Parent and child directories must not overlap")
    selected = resolve_device(device)
    if selected.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with run_lock(folder):
        audit = validate_parent(config, parent, device=str(selected))
        folder.mkdir(parents=True, exist_ok=True)
        protocol = {"schema_version": 1, "study": "bounded_student_budget_check", "config": config.to_dict(),
            "source_sha256": source_fingerprint(), "original_numerical_source_sha256": legacy_source_fingerprint(),
            "parent": audit, "selection": _selection(config.parent_config),
            "primary_outcome": {"group": "rare", "mode": "population", "metric": "diagnostic_kl_gain"},
            "gain_sign": "Distilled minus oracle; positive favors oracle. Paired gain change is extended minus baseline.",
            "arm_change_sign": "Extended error minus baseline error; negative means improvement.",
            "scope": "One fixed extension of both existing soft-target trajectories on every original population. Teacher and solo remain fixed. More optimization steps and additional checkpoint candidates; no pure convergence, significance, or mechanism-recovery claim.",
            "stopping": "Stop at the configured total steps regardless of results; no threshold, new seeds, tuning, or preferred horizon selection."}
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        protocol_path = folder / "protocol.json"
        if protocol_path.exists():
            if _json(protocol_path) != protocol:
                raise ValueError("Child configuration, source, or parent hashes changed")
        elif any(folder.iterdir()):
            raise ValueError("Existing child output without a protocol cannot be resumed")
        else:
            atomic_json(protocol_path, protocol)
        snapshot = _snapshot(parent, folder, audit)
        deadline = time.perf_counter() + config.max_seconds_per_invocation
        reports = []
        for seed in config.parent_config.replication_seeds:
            if time.perf_counter() >= deadline:
                break
            report = _seed(config, seed, folder / f"seed-{seed}", snapshot, protocol_hash, audit, selected, deadline)
            if report["status"] != "complete":
                break
            reports.append(report)
        if _file_hashes(parent) != audit["file_sha256"]:
            raise ValueError("Read-only parent changed during continuation")
        complete = len(reports) == len(config.parent_config.replication_seeds)
        result = {"status": "complete" if complete else "incomplete", "output_dir": str(folder),
            "protocol_sha256": protocol_hash, "selection": protocol["selection"],
            "primary_outcome": protocol["primary_outcome"], "planned_seeds": list(config.parent_config.replication_seeds),
            "completed_seeds": [r["seed"] for r in reports],
            "pending_seeds": [seed for seed in config.parent_config.replication_seeds if seed not in [r["seed"] for r in reports]],
            "baseline_student_steps": config.parent_config.base.student_steps, "extended_student_steps": config.student_steps,
            "message": "Complete; stop at this predeclared budget." if complete else "Time budget reached. Rerun unchanged to resume. Aggregate unavailable until every seed completes."}
        if complete:
            proxies = [r["timing"]["t4_equivalent_units_at_user_rate"] for r in reports]
            result.update({"groups": aggregate_budget_check(reports, config.parent_config.replication_seeds),
                "convergence_warnings": [f"Seed {r['seed']}, {arm}: inspect boundary and observed-choice curves."
                    for r in reports for arm, c in r["convergence"].items()
                    if c["needs_longer_training_check"] or c["population_selected_at_limit"]],
                "timing": {"incremental_training_seconds": sum(r["timing"]["incremental_training_seconds"] for r in reports),
                    "parent_training_seconds": audit["parent_training_seconds"],
                    "t4_equivalent_units_at_user_rate": sum(proxies) if all(v is not None for v in proxies) else None,
                    "note": "Only two continued optimizer trajectories per seed; incremental checkpoint elapsed minus immutable parent checkpoint elapsed. Parent compute is separate. T4 conversion uses child sessions only; setup/evaluation/idle are additional."},
                "interpretation": protocol["scope"], "gain_sign": protocol["gain_sign"], "arm_change_sign": protocol["arm_change_sign"]})
        atomic_json(folder / "summary.json", result)
        return result
