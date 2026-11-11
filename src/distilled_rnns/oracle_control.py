"""Predeclared target replacement in learned two-unit students.

The teacher-target and oracle-target arms differ only in training soft targets.
Both select weights by observed-choice checkpoint NLL, with population selection
primary and per-person selection secondary. Generating probabilities outside the
training mask never enter either soft-target tensor or its checkpoint signature.
This remains a conditional first-stage feasibility study, not mechanism recovery.
"""
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .baseline_audit import _score, _test_report, reference_predictions
from .config import ExperimentConfig
from .evaluation import make_trial_masks
from .experiment import (atomic_json, environment_info, resolve_device, run_lock,
                         save_environment_inventory, source_fingerprint,
                         t4_training_cost_proxy, _save_npz)
from .followup import _range_summary, _plot_convergence, convergence_details
from .models import BatchedStudentGRU, TeacherGRU
from .simulation import simulate_population
from .training import fit_model


MODES = ("population", "individual")
ARMS = ("teacher", "solo", "distilled", "oracle")
METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl")
ORACLE_LOGIT_EPSILON = 1e-12


@dataclass(frozen=True)
class OracleControlConfig:
    base: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(
        seed=20261411, teacher_steps=500, student_steps=4000))
    replication_seeds: tuple = (20261411, 20261421, 20261431, 20261441, 20261451)
    solo_l2: float = 0.1
    max_seconds_per_invocation: float = 1800.0

    def __post_init__(self):
        if not isinstance(self.base, ExperimentConfig):
            raise ValueError("base must be an ExperimentConfig")
        if self.base.teacher_prop != 1 or self.base.temperature != 1:
            raise ValueError("Oracle control requires teacher_prop=1 and temperature=1")
        try:
            seeds = tuple(self.replication_seeds)
        except TypeError as exc:
            raise ValueError("replication_seeds must be a nonempty sequence of integers") from exc
        if not seeds:
            raise ValueError("replication_seeds must be nonempty")
        for seed in seeds:
            replace(self.base, seed=seed)  # Integer/bool/range checks, including offsets.
        if len(set(seeds)) != len(seeds):
            raise ValueError("replication_seeds must be unique")
        ordered = sorted(seeds)
        if any(right - left < 4 for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Replication seeds must be separated by at least four to avoid reusing random streams")
        # base.seed is a template value, not a calibration population.
        for name in ("solo_l2", "max_seconds_per_invocation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if self.solo_l2 < 0:
            raise ValueError("solo_l2 must be nonnegative")
        if self.max_seconds_per_invocation <= 0:
            raise ValueError("max_seconds_per_invocation must be positive")
        object.__setattr__(self, "replication_seeds", seeds)

    def to_dict(self):
        return {"base": self.base.to_dict(), "replication_seeds": list(self.replication_seeds),
                "solo_l2": self.solo_l2,
                "max_seconds_per_invocation": self.max_seconds_per_invocation}

    @classmethod
    def from_file(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown oracle-control configuration settings")
        values = dict(values)
        if "base" in values:
            base = values["base"]
            if not isinstance(base, dict) or set(base) - {f.name for f in fields(ExperimentConfig)}:
                raise ValueError("Unknown or invalid base configuration settings")
            values["base"] = ExperimentConfig(**base)
        return cls(**values)


def oracle_training_logits(probabilities, train_mask, device=None):
    """Validate probabilities and expose only training targets to ``fit_model``.

    Clip to [1e-12, 1-1e-12] in float64, compute log(p)-log1p(-p), then
    cast to float32. Clipping after float32 conversion would round 1-1e-12
    to one and produce infinity. Off-training entries are exactly zero logits
    (neutral probability .5), so held-out true_prob cannot change the signature.
    """
    probability = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(train_mask)
    if probability.ndim != 2 or mask.shape != probability.shape or mask.dtype != bool:
        raise ValueError("Probabilities and boolean train_mask must match [people, trials]")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Oracle probabilities must be finite and lie in [0, 1]")
    if (mask.sum(1) == 0).any():
        raise ValueError("Every person requires a training target")
    selected = np.clip(probability[mask], ORACLE_LOGIT_EPSILON, 1 - ORACLE_LOGIT_EPSILON)
    logits = np.zeros(probability.shape, dtype=np.float32)
    logits[mask] = (np.log(selected) - np.log1p(-selected)).astype(np.float32)
    return torch.as_tensor(logits, device=device)


def _selection(config):
    return {"solo_l2": config.solo_l2, "primary_mode": "population", "secondary_mode": "individual",
            "teacher_prop": 1.0, "temperature": 1.0, "soft_student_l2": 0.0,
            "criterion": "Observed-choice NLL on checkpoint split only; exact ties retain the earlier step.",
            "origin": "Predeclared settings inherited from the prior baseline audit; no new calibration or tuning."}


def _comparison(left, right):
    """Every gain is left minus right: positive always favors the right model."""
    return {f"{metric}_gain": None if left[metric] is None or right[metric] is None
            else left[metric] - right[metric] for metric in METRICS}


def _gain_summary(values):
    return {**_range_summary(values),
            "positive_seed_count": sum(value is not None and value > 0 for value in values),
            "negative_seed_count": sum(value is not None and value < 0 for value in values),
            "tied_seed_count": sum(value is not None and value == 0 for value in values)}


def _comparisons(models):
    return {"target_comparisons": {
                mode: _comparison(models[f"distilled_{mode}"], models[f"oracle_{mode}"])
                for mode in MODES},
            "contextual_comparisons": {
                "distilled_vs_solo": _comparison(models["solo"], models["distilled"]),
                "oracle_vs_solo": _comparison(models["solo"], models["oracle"]),
                "oracle_vs_teacher": _comparison(models["teacher"], models["oracle"])}}


def _incomplete(folder, manifest, session, started, phase):
    session.update({"status": "incomplete", "phase": phase, "wall_seconds": time.perf_counter() - started})
    atomic_json(folder / "manifest.json", manifest)
    return {"status": "incomplete", "phase": phase}


def _dataset(study, config, folder, device, protocol_hash, deadline):
    folder.mkdir(parents=True, exist_ok=True)
    selection = _selection(study)
    identity = {"config": config.to_dict(), "selection": selection,
                "protocol_sha256": protocol_hash, "role": "replication", "third_split_role": "test"}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Dataset configuration or protocol changed")
    else:
        if any(folder.iterdir()):
            raise ValueError("Existing dataset without a manifest cannot be resumed")
        manifest = {"identity": identity, "sessions": []}
    selection_path = folder / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen selection differs on resume")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    packages = {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(device),
               "package_inventory": save_environment_inventory(folder, packages)}
    manifest["sessions"].append(session)
    atomic_json(manifest_path, manifest)
    atomic_json(selection_path, selection)  # Frozen before simulation, fitting, or test scoring.
    data = simulate_population(config.n_subjects, config.n_trials, config.rare_fraction, config.seed)
    original_masks = make_trial_masks(config.n_subjects, config.n_trials, config.seed + 1)
    masks = {"train": original_masks["train"], "checkpoint": original_masks["val"], "test": original_masks["test"]}
    _save_npz(folder / "data.npz", **data, **{f"mask_{k}": v for k, v in masks.items()})
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["choice"], dtype=torch.float32, device=device)
    train = torch.as_tensor(masks["train"], device=device)
    # Prepare only the train-masked oracle tensor. Ground truth is otherwise
    # accessed by the reporting function after every learned prediction is saved.
    soft_logits = {"oracle": oracle_training_logits(data["true_prob"], masks["train"], device)}
    phases, candidates = {}, {}
    for name in ARMS:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return _incomplete(folder, manifest, session, started, name)
        is_teacher = name == "teacher"
        init_seed = config.seed + (2 if is_teacher else 3)
        torch.manual_seed(init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(init_seed)
        model = (TeacherGRU(x.shape[-1], config.teacher_hidden) if is_teacher else
                 BatchedStudentGRU(config.n_subjects, x.shape[-1], config.student_hidden)).to(device)
        strength = study.solo_l2 if name == "solo" else 0.0
        print(f"\nSeed {config.seed}: {name}, L2={strength}; observed-choice checkpoint selection", flush=True)
        path = folder / f"{name}.pt"
        phases[name] = fit_model(model, x, y, masks["train"], masks["checkpoint"],
            steps=config.teacher_steps if is_teacher else config.student_steps,
            lr=config.learning_rate, seed=init_seed, eval_every=config.eval_every,
            patience=config.patience, checkpoint_path=path,
            teacher_logits=soft_logits.get(name), teacher_prop=1.0 if name in soft_logits else 0.0,
            temperature=1.0, l2_strength=strength, track_population_selection=not is_teacher,
            max_seconds=min(config.max_seconds_per_phase, remaining))
        atomic_json(folder / "training.json", phases)
        if phases[name]["stopped_reason"] == "time_limit":
            return _incomplete(folder, manifest, session, started, name)
        with torch.no_grad():
            logits = model(x)
            candidates[name if is_teacher else f"{name}_individual"] = torch.sigmoid(logits).cpu().numpy()
            if is_teacher:
                soft_logits["distilled"] = torch.where(train, logits.detach(), torch.zeros_like(logits))
                _save_npz(folder / "soft_training_logits.npz", **{
                    key: value.cpu().numpy() for key, value in soft_logits.items()})
            else:
                saved = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(saved["population_best_state"])
                candidates[f"{name}_population"] = torch.sigmoid(model(x)).cpu().numpy()
                del saved
        del model
    candidates.update(reference_predictions(data["choice"], masks["train"]))
    predictions = {"teacher": candidates["teacher"],
                   **{name: candidates[f"{name}_population"] for name in ARMS if name != "teacher"},
                   **{f"{name}_{mode}": candidates[f"{name}_{mode}"]
                      for name in ARMS if name != "teacher" for mode in MODES},
                   "coinflip": candidates["coinflip"], "frequency": candidates["frequency"]}
    _save_npz(folder / "candidate_predictions.npz", **candidates)
    _save_npz(folder / "predictions.npz", **predictions)
    report = _test_report(data, predictions, masks["test"], config.diagnostic_threshold)
    # The legacy evaluator's fidelity_loss has the opposite sign. Exclude its
    # inherited comparison fields so every comparison in this study is a gain.
    for row in [*report["per_subject"], *report["groups"].values()]:
        row.pop("likelihood_gain", None)
        row.pop("fidelity_loss", None)
        row.update(_comparisons(row["models"]))
    report["metadata"].update({"comparison_sign": "All gains are left minus right; positive favors the right model.",
                               "oracle": "Learned student fitted to generating probabilities on training trials only; not perfect-oracle test predictions."})
    convergence = convergence_details(phases, config)
    for name in ARMS[1:]:
        convergence[name]["population_best_step"] = phases[name]["population_best_step"]
        convergence[name]["population_selected_at_limit"] = phases[name]["population_best_step"] == config.student_steps
    total_seconds = sum(phase["elapsed_seconds"] for phase in phases.values())
    report.update({"status": "complete", "role": "replication", "seed": config.seed,
                   "output_dir": str(folder), "selection": selection, "phases": phases,
                   "convergence": convergence, "checkpoint_scores": _score(data["choice"], predictions, masks["checkpoint"]),
                   "timing": {"cumulative_training_seconds": total_seconds,
                              **t4_training_cost_proxy(total_seconds, config.t4_units_per_hour, manifest["sessions"])}})
    _plot_convergence(phases, folder / "convergence.png")
    session.update({"status": "complete", "wall_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    atomic_json(folder / "report.json", report)
    atomic_json(manifest_path, manifest)
    return report


def aggregate_oracle_control(reports, expected_seeds):
    """Equal-person group scores, then equal-seed descriptive means and ranges."""
    if (not expected_seeds or [r.get("seed") for r in reports] != list(expected_seeds)
            or any(r.get("status") != "complete" or r.get("role") != "replication" for r in reports)):
        raise ValueError("Every planned replication must complete before aggregation")
    groups = {}
    for group in ("rare", "typical"):
        rows = []
        for report in reports:
            metrics = report["groups"][group]
            models = metrics["models"]
            rows.append({"seed": report["seed"], "subject_count": metrics["subject_count"],
                         "test_count": metrics["test_count"], "diagnostic_count": metrics["diagnostic_count"],
                         "diagnostic_subject_count": models["oracle"]["diagnostic_subject_count"],
                         "models": models, **_comparisons(models)})
        groups[group] = {"per_seed": rows,
            **{section: {name: {metric: _gain_summary([row[section][name][metric] for row in rows])
                         for metric in rows[0][section][name]} for name in rows[0][section]}
               for section in ("target_comparisons", "contextual_comparisons")},
            "models": {model: {metric: _range_summary([row["models"][model][metric] for row in rows])
                       for metric in METRICS} for model in rows[0]["models"]},
            "support": {metric: _range_summary([row[metric] for row in rows])
                        for metric in ("subject_count", "test_count", "diagnostic_count", "diagnostic_subject_count")}}
    return groups


def run_oracle_control(config: OracleControlConfig, output_dir, device="auto"):
    if not isinstance(config, OracleControlConfig):
        raise ValueError("config must be an OracleControlConfig")
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    if selected_device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with run_lock(folder):
        return _study(config, folder, selected_device)


def _study(config, folder, device):
    deadline = time.perf_counter() + config.max_seconds_per_invocation
    protocol = {"schema_version": 1, "study": "oracle_target_control", "config": config.to_dict(),
        "source_sha256": source_fingerprint(), "selection": _selection(config),
        "split_roles": "Every dataset is a fresh replication: 60% train, 20% checkpoint selection, 20% test; rounded for smoke configurations.",
        "oracle_training_only": "Only training true_prob entries form oracle soft targets. Both soft-target tensors have neutral zero logits elsewhere, including in fit signatures. Observed histories and observed-choice checkpoint targets are identical across arms.",
        "oracle_logits": {"clip_epsilon": ORACLE_LOGIT_EPSILON, "conversion": "Clip and log(p)-log1p(-p) in float64, then cast to float32; reject nonfinite or out-of-range probabilities."},
        "primary_comparison": "Population-selected teacher-target student (distilled) versus population-selected learned oracle-target student (oracle).",
        "primary_outcome": {"group": "rare", "comparison": "target_comparisons.population", "metric": "diagnostic_kl_gain",
                            "sign": "Distilled diagnostic KL minus oracle diagnostic KL; positive means oracle improvement."},
        "secondary_comparison": "Same target replacement with individual checkpoint selection, from the same two optimization trajectories.",
        "controls": "Fixed inherited solo L2; observed-choice teacher; 0.5 coinflip; Beta(1,1) smoothed training-only per-person choice frequency. No new calibration, grid, or test-driven choices.",
        "initialization": "Teacher seed+2; all three independent-student arms seed+3 with identical initialization and observed histories. Data seed; split seed+1.",
        "checkpoint_selection": "Observed-choice checkpoint NLL only. Population minimum is primary; individual minima are secondary. No oracle KL, rare labels, diagnostic masks, or test scores select weights.",
        "scope": "Descriptive conditional first-stage feasibility control. No mechanism recovery, significance, perfect-oracle prediction, or convergence claim.",
        "aggregation": "Complete planned seed set only; equal-person group means then equal-seed means/ranges and support counts. Unsupported diagnostics remain null.",
        "accounting": "Four fits per seed: teacher, solo, distilled, oracle. Both student checkpoint snapshots share each fit and each cumulative duration is counted once."}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Study configuration or source changed; choose a new output directory")
    elif any(folder.iterdir()):
        raise ValueError("Existing output without a protocol cannot be resumed")
    else:
        atomic_json(protocol_path, protocol)
    reports = []
    for seed in config.replication_seeds:
        if time.perf_counter() >= deadline:
            break
        report = _dataset(config, replace(config.base, seed=seed), folder / f"seed-{seed}",
                          device, protocol_hash, deadline)
        if report["status"] != "complete":
            break
        reports.append(report)
    complete = len(reports) == len(config.replication_seeds)
    result = {"status": "complete" if complete else "incomplete", "output_dir": str(folder),
        "protocol_sha256": protocol_hash, "selection": _selection(config),
        "primary_outcome": protocol["primary_outcome"], "planned_seeds": list(config.replication_seeds),
        "completed_seeds": [r["seed"] for r in reports],
        "pending_seeds": [seed for seed in config.replication_seeds if seed not in [r["seed"] for r in reports]],
        "message": "Complete." if complete else "Time budget reached. Rerun the same study cell and run name to resume. No aggregate is available yet."}
    if complete:
        result["groups"] = aggregate_oracle_control(reports, config.replication_seeds)
        result["convergence_warnings"] = [
            f"Seed {r['seed']}, {name}: inspect training boundary/validation curves."
            for r in reports for name, c in r["convergence"].items()
            if c["needs_longer_training_check"] or c.get("population_selected_at_limit", False)]
        proxies = [r["timing"]["t4_equivalent_units_at_user_rate"] for r in reports]
        result["timing"] = {"cumulative_training_seconds": sum(r["timing"]["cumulative_training_seconds"] for r in reports),
            "t4_equivalent_units_at_user_rate": sum(proxies) if all(p is not None for p in proxies) else None,
            "note": "Four optimizer trajectories per seed, counted once each. Setup/evaluation/idle are additional; mixed-device sessions have no T4 estimate."}
        result["interpretation"] = "Predeclared learned-student target replacement, with population selection primary and individual selection secondary. Every gain is left minus right; positive target gains favor oracle training. Descriptive means/ranges only; no mechanism-recovery, significance or convergence claim."
    atomic_json(folder / "summary.json", result)
    return result
