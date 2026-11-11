"""Separate checkpoint selection, pipeline tuning, and fresh-seed evaluation.

Calibration's third split tunes pipelines; it is not a test set. Individual and
population checkpoints share an identical optimization trajectory. Fresh-seed
test scores never choose L2 or selection modes.
"""
from dataclasses import dataclass, field, replace
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

from .config import ExperimentConfig
from .evaluation import binary_kl, binary_nll, evaluate_predictions, make_trial_masks
from .experiment import (atomic_json, environment_info, resolve_device, run_lock,
                         save_environment_inventory, source_fingerprint,
                         t4_training_cost_proxy, _save_npz)
from .followup import FollowupConfig, _range_summary, _plot_convergence, convergence_details
from .models import BatchedStudentGRU, TeacherGRU
from .training import fit_model
from .simulation import simulate_population


MODES = ("population", "individual")  # Declared exact-tie preference.
BASELINES = ("solo", "solo_legacy", "coinflip", "frequency")


@dataclass(frozen=True)
class BaselineAuditConfig:
    base: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(
        seed=20261201, teacher_steps=500, student_steps=1000))
    solo_l2_grid: tuple = (0.0, 0.001, 0.01, 0.1)
    replication_seeds: tuple = (20261211, 20261221, 20261231, 20261241, 20261251)
    legacy_solo_l2: float = 0.01
    max_seconds_per_invocation: float = 1800.0

    def __post_init__(self):
        checked = FollowupConfig(self.base, self.solo_l2_grid, self.replication_seeds,
                                 self.max_seconds_per_invocation)
        if isinstance(self.legacy_solo_l2, bool) or not isinstance(self.legacy_solo_l2, (int, float)) or not math.isfinite(self.legacy_solo_l2):
            raise ValueError("legacy_solo_l2 must be finite and numeric")
        if self.legacy_solo_l2 not in checked.solo_l2_grid:
            raise ValueError("legacy_solo_l2 must occur in the calibration grid")
        object.__setattr__(self, "solo_l2_grid", checked.solo_l2_grid)
        object.__setattr__(self, "replication_seeds", checked.replication_seeds)

    def to_dict(self):
        return {"base": self.base.to_dict(), "solo_l2_grid": list(self.solo_l2_grid),
                "replication_seeds": list(self.replication_seeds), "legacy_solo_l2": self.legacy_solo_l2,
                "max_seconds_per_invocation": self.max_seconds_per_invocation}

    @classmethod
    def from_file(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or set(values) - {
                "base", "solo_l2_grid", "replication_seeds", "legacy_solo_l2", "max_seconds_per_invocation"}:
            raise ValueError("Unknown baseline-audit configuration settings")
        values = dict(values)
        if "base" in values:
            values["base"] = ExperimentConfig(**values["base"])
        return cls(**values)


def reference_predictions(choice, train_mask):
    """Fixed fair-coin and Beta(1,1) training-only per-person choice frequencies."""
    choice, train_mask = np.asarray(choice), np.asarray(train_mask)
    if choice.ndim != 2 or train_mask.shape != choice.shape or train_mask.dtype != bool:
        raise ValueError("choice and boolean train_mask must match [people, trials]")
    if not np.isin(choice, (0, 1)).all() or (train_mask.sum(1) == 0).any():
        raise ValueError("Binary choices and nonempty training targets are required")
    probability = ((choice * train_mask).sum(1) + 1) / (train_mask.sum(1) + 2)
    return {"coinflip": np.full(choice.shape, .5),
            "frequency": np.broadcast_to(probability[:, None], choice.shape).copy()}


def _score(choice, predictions, mask):
    return {name: float(((binary_nll(choice, probability) * mask).sum(1) / mask.sum(1)).mean())
            for name, probability in predictions.items()}


def choose_pipeline(candidates):
    """Outer-tuning NLL only; exact ties favor lower L2, then population mode."""
    if not candidates or any(not math.isfinite(c["tuning_nll"]) or c["mode"] not in MODES
                             or not math.isfinite(c["l2"]) or c["l2"] < 0 for c in candidates):
        raise ValueError("Finite tuning scores, nonnegative L2, and known modes are required")
    return min(candidates, key=lambda c: (c["tuning_nll"], c["l2"], MODES.index(c["mode"])))


def _comparison(left, right):
    """Right is distilled: positive gain helps, positive fidelity loss hurts."""
    return {"likelihood_gain": left["nll"] - right["nll"],
            "fidelity_loss": right["kl"] - left["kl"],
            "diagnostic_likelihood_gain": None if left["diagnostic_nll"] is None else left["diagnostic_nll"] - right["diagnostic_nll"],
            "diagnostic_fidelity_loss": None if left["diagnostic_kl"] is None else right["diagnostic_kl"] - left["diagnostic_kl"]}


def _test_report(data, predictions, test_mask, threshold):
    report = evaluate_predictions(data, predictions, test_mask, threshold)
    diagnostic = test_mask & (np.abs(data["true_prob"] - data["reference_prob"]) > threshold)
    for name, probability in predictions.items():
        loss = binary_kl(data["true_prob"], probability)
        for i, row in enumerate(report["per_subject"]):
            row["models"][name]["diagnostic_kl"] = float(loss[i, diagnostic[i]].mean()) if diagnostic[i].any() else None
        for group, is_rare in (("rare", True), ("typical", False)):
            values = [row["models"][name]["diagnostic_kl"] for row in report["per_subject"]
                      if row["rare"] == is_rare and row["diagnostic_count"] > 0]
            report["groups"][group]["models"][name]["diagnostic_kl"] = float(np.mean(values)) if values else None
    return report


def _selection_plot(checkpoint_scores, heldout_scores, path, role, labels=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(14, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    names = list(checkpoint_scores)
    positions = np.arange(len(names))
    ax.bar(positions - .2, [checkpoint_scores[k] for k in names], .4, label="Checkpoint-selection split")
    ax.bar(positions + .2, [heldout_scores[k] for k in names], .4, label=role)
    ax.axhline(np.log(2), color="gray", linestyle="--", label="50/50 reference")
    ax.set(xticks=positions, xticklabels=[(labels or {}).get(name, name) for name in names], ylabel="Observed-choice NLL (lower is better)",
           title="Selection audit: selected weights, averaged across people")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.savefig(path, dpi=140)


def _dataset(audit, config, folder, device, strengths, frozen, protocol_hash, deadline):
    folder.mkdir(parents=True, exist_ok=True)
    calibration = frozen is None
    role = "tuning" if calibration else "test"
    identity = {"config": config.to_dict(), "strengths": list(strengths), "frozen_selection": frozen,
                "protocol_sha256": protocol_hash, "third_split_role": role}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Dataset configuration or protocol changed")
    else:
        if any(folder.iterdir()):
            raise ValueError("Existing dataset without a manifest cannot be resumed")
        manifest = {"identity": identity, "sessions": []}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    packages = {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment_info(device),
               "package_inventory": save_environment_inventory(folder, packages)}
    manifest["sessions"].append(session)
    atomic_json(manifest_path, manifest)
    data = simulate_population(config.n_subjects, config.n_trials, config.rare_fraction, config.seed)
    original_masks = make_trial_masks(config.n_subjects, config.n_trials, config.seed + 1)
    masks = {"train": original_masks["train"], "checkpoint": original_masks["val"], role: original_masks["test"]}
    _save_npz(folder / "data.npz", **data, **{f"mask_{k}": v for k, v in masks.items()})
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["choice"], dtype=torch.float32, device=device)
    phases, candidates, teacher_logits = {}, {}, None
    plan = [("teacher", 0.0)] + [(f"solo_{i}", v) for i, v in enumerate(strengths)] + [("distilled", 0.0)]
    for name, strength in plan:
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
        print(f"\nSeed {config.seed}: {name}, L2={strength}; tracking both student checkpoint rules", flush=True)
        path = folder / f"{name}.pt"
        phases[name] = fit_model(model, x, y, masks["train"], masks["checkpoint"],
            steps=config.teacher_steps if is_teacher else config.student_steps,
            lr=config.learning_rate, seed=init_seed, eval_every=config.eval_every,
            patience=config.patience, checkpoint_path=path,
            teacher_logits=teacher_logits if name == "distilled" else None,
            teacher_prop=config.teacher_prop if name == "distilled" else 0,
            temperature=config.temperature, l2_strength=strength,
            track_population_selection=not is_teacher,
            max_seconds=min(config.max_seconds_per_phase, remaining))
        atomic_json(folder / "training.json", phases)
        if phases[name]["stopped_reason"] == "time_limit":
            return _incomplete(folder, manifest, session, started, name)
        with torch.no_grad():
            logits = model(x)
            candidates[name if is_teacher else f"{name}_individual"] = torch.sigmoid(logits).cpu().numpy()
            if is_teacher:
                teacher_logits = logits.detach()
            else:
                saved = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(saved["population_best_state"])
                candidates[f"{name}_population"] = torch.sigmoid(model(x)).cpu().numpy()
                del saved
        del model
    candidates.update(reference_predictions(data["choice"], masks["train"]))
    checkpoint_scores = _score(data["choice"], candidates, masks["checkpoint"])
    if calibration:
        tuning_scores = _score(data["choice"], candidates, masks["tuning"])
        solo_options = [{"l2": v, "mode": mode, "tuning_nll": tuning_scores[f"solo_{i}_{mode}"]}
                        for i, v in enumerate(strengths) for mode in MODES]
        distilled_options = [{"l2": 0.0, "mode": mode, "tuning_nll": tuning_scores[f"distilled_{mode}"]} for mode in MODES]
        selected_solo, selected_distilled = choose_pipeline(solo_options), choose_pipeline(distilled_options)
        selection = {"protocol_sha256": protocol_hash, "solo_l2": selected_solo["l2"],
                     "solo_selection": selected_solo["mode"], "distilled_selection": selected_distilled["mode"],
                     "solo_candidates": solo_options, "distilled_candidates": distilled_options,
                     "criterion": "Mean observed-choice NLL on calibration's outer tuning split only; exact ties prefer lower L2 then population mode."}
    else:
        selection = frozen
    selection_path = folder / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen selection differs on resume; use the original environment and source")
    atomic_json(selection_path, selection)  # Durable before any fresh-seed test/oracle score.
    solo_index = strengths.index(selection["solo_l2"])
    legacy_index = strengths.index(audit.legacy_solo_l2)
    predictions = {"teacher": candidates["teacher"],
        "solo": candidates[f"solo_{solo_index}_{selection['solo_selection']}"],
        "distilled": candidates[f"distilled_{selection['distilled_selection']}"],
        "solo_legacy": candidates[f"solo_{legacy_index}_individual"],
        **{f"solo_{mode}": candidates[f"solo_{solo_index}_{mode}"] for mode in MODES},
        **{f"distilled_{mode}": candidates[f"distilled_{mode}"] for mode in MODES},
        "coinflip": candidates["coinflip"], "frequency": candidates["frequency"]}
    _save_npz(folder / "candidate_predictions.npz", **candidates)
    _save_npz(folder / "predictions.npz", **predictions)
    if calibration:
        report = {"role": "calibration", "tuning_scores": tuning_scores,
                  "note": "No independent calibration test score: its third split tunes the pipelines."}
        plot_check, plot_heldout = checkpoint_scores, tuning_scores
    else:
        report = _test_report(data, predictions, masks["test"], config.diagnostic_threshold)
        report["role"] = "replication"
        plot_check = _score(data["choice"], predictions, masks["checkpoint"])
        plot_heldout = _score(data["choice"], predictions, masks["test"])
    convergence = convergence_details(phases, config)
    for name, phase in phases.items():
        if name != "teacher":
            convergence[name]["population_best_step"] = phase["population_best_step"]
            convergence[name]["population_selected_at_limit"] = phase["population_best_step"] == config.student_steps
    total_seconds = sum(phase["elapsed_seconds"] for phase in phases.values())
    report.update({"status": "complete", "seed": config.seed, "output_dir": str(folder),
                   "selection": selection, "phases": phases, "convergence": convergence,
                   "checkpoint_scores": plot_check,
                   "timing": {"cumulative_training_seconds": total_seconds,
                              **t4_training_cost_proxy(total_seconds, config.t4_units_per_hour, manifest["sessions"])}})
    _plot_convergence(phases, folder / "convergence.png")
    plot_labels = {"teacher": "Teacher", "coinflip": "50/50", "frequency": "Choice frequency",
                   "solo": "Selected solo", "distilled": "Selected distilled", "solo_legacy": "Previous solo recipe"}
    for mode, label in (("individual", "per-person step"), ("population", "shared step")):
        plot_labels[f"solo_{mode}"] = f"Solo: {label}"
        plot_labels[f"distilled_{mode}"] = f"Distilled: {label}"
        for i, strength in enumerate(strengths):
            plot_labels[f"solo_{i}_{mode}"] = f"Solo L2={strength:g}: {label}"
    _selection_plot(plot_check, plot_heldout, folder / "selection_audit.png",
                    "Outer tuning split" if calibration else "Fresh-seed test split", plot_labels)
    session.update({"status": "complete", "wall_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    atomic_json(folder / "report.json", report)
    atomic_json(manifest_path, manifest)
    return report


def _incomplete(folder, manifest, session, started, phase):
    session.update({"status": "incomplete", "phase": phase, "wall_seconds": time.perf_counter() - started})
    atomic_json(folder / "manifest.json", manifest)
    return {"status": "incomplete", "phase": phase}


def aggregate_audit(reports, expected_seeds):
    if [r.get("seed") for r in reports] != list(expected_seeds) or any(r.get("status") != "complete" or r.get("role") != "replication" for r in reports):
        raise ValueError("Every planned replication must complete before aggregation")
    groups = {}
    for group in ("rare", "typical"):
        rows = []
        for report in reports:
            metrics = report["groups"][group]
            models = metrics["models"]
            rows.append({"seed": report["seed"], "subject_count": metrics["subject_count"],
                "diagnostic_count": metrics["diagnostic_count"],
                "diagnostic_subject_count": models["solo"]["diagnostic_subject_count"],
                "models": models,
                "comparisons": {name: _comparison(models[name], models["distilled"]) for name in BASELINES},
                "matched_comparisons": {mode: _comparison(models[f"solo_{mode}"], models[f"distilled_{mode}"]) for mode in MODES}})
        groups[group] = {"per_seed": rows,
            **{section: {name: {metric: _range_summary([row[section][name][metric] for row in rows])
                         for metric in rows[0][section][name]} for name in rows[0][section]}
               for section in ("comparisons", "matched_comparisons")},
            "models": {model: {metric: _range_summary([row["models"][model][metric] for row in rows])
                       for metric in ("nll", "kl", "diagnostic_nll", "diagnostic_kl")} for model in rows[0]["models"]}}
    return groups


def run_baseline_audit(config: BaselineAuditConfig, output_dir, device="auto"):
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
    protocol = {"schema_version": 1, "config": config.to_dict(), "source_sha256": source_fingerprint(),
        "split_roles": "Calibration: 60% train, 20% checkpoint selection, 20% outer tuning. Replications: 60% train, 20% checkpoint selection, 20% test. Rounded counts for small smoke configurations.",
        "selection": "Choose solo (L2, mode) and distilled mode independently on calibration outer NLL; freeze before fresh-seed testing. Exact ties: lower L2 then population before individual.",
        "primary_comparison": "Independently calibrated solo and distilled pipelines; this does not isolate distillation alone.",
        "matched_comparisons": "Individual versus individual and population versus population, using the frozen selected solo L2.",
        "controls": "Fixed individual L2=legacy_solo_l2; constant 0.5; per-person Beta(1,1) smoothed train-only choice frequency.",
        "population_selection": "Independent weights with one checkpoint step chosen by population mean validation NLL; dominated by typical people.",
        "scope": "Exploratory conditional first-stage parametric baseline audit; no mechanistic gate or significance test.",
        "aggregation": "Equal-weight seed means/ranges, calibration excluded, unsupported diagnostic values remain null; diagnostic NLL and true-policy KL reported separately.",
        "accounting": "Track both checkpoint rules on each single optimizer trajectory; count training time once per trajectory."}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Study configuration or source changed; choose a new output directory")
    elif any(folder.iterdir()):
        raise ValueError("Existing output without a protocol cannot be resumed")
    else:
        atomic_json(protocol_path, protocol)
    calibration = _dataset(config, config.base, folder / "calibration", device,
                           config.solo_l2_grid, None, protocol_hash, deadline)
    reports = []
    if calibration["status"] == "complete":
        selection = calibration["selection"]
        strengths = tuple(sorted({selection["solo_l2"], config.legacy_solo_l2}))
        for seed in config.replication_seeds:
            if time.perf_counter() >= deadline:
                break
            report = _dataset(config, replace(config.base, seed=seed), folder / f"seed-{seed}",
                              device, strengths, selection, protocol_hash, deadline)
            if report["status"] != "complete":
                break
            reports.append(report)
    complete = calibration["status"] == "complete" and len(reports) == len(config.replication_seeds)
    result = {"status": "complete" if complete else "incomplete", "output_dir": str(folder),
        "protocol_sha256": protocol_hash, "planned_seeds": list(config.replication_seeds),
        "completed_seeds": [r["seed"] for r in reports],
        "pending_seeds": [seed for seed in config.replication_seeds if seed not in [r["seed"] for r in reports]],
        "message": "Complete." if complete else "Time budget reached. Rerun the same study cell and run name to resume. No aggregate is available yet."}
    if calibration["status"] == "complete":
        result["calibration"] = {key: calibration["selection"][key] for key in ("solo_l2", "solo_selection", "distilled_selection")}
        result["calibration"]["lambda_at_grid_boundary"] = calibration["selection"]["solo_l2"] == max(config.solo_l2_grid) and max(config.solo_l2_grid) > 0
        result["calibration"]["note"] = "The third calibration split is tuning data; no independent calibration test result. All choices are population-selected, dominated by typical people."
    if complete:
        result["groups"] = aggregate_audit(reports, config.replication_seeds)
        all_reports = [calibration, *reports]
        warnings = [f"Seed {r['seed']}, {name}: inspect training boundary/validation curves."
                    for r in all_reports for name, c in r["convergence"].items()
                    if c["needs_longer_training_check"] or c.get("population_selected_at_limit", False)]
        if result["calibration"]["lambda_at_grid_boundary"]:
            warnings.append("Largest tested L2 selected; do not expand this grid using these test results.")
        result["convergence_warnings"] = warnings
        proxies = [r["timing"]["t4_equivalent_units_at_user_rate"] for r in all_reports]
        result["timing"] = {"cumulative_training_seconds": sum(r["timing"]["cumulative_training_seconds"] for r in all_reports),
            "t4_equivalent_units_at_user_rate": sum(proxies) if all(p is not None for p in proxies) else None,
            "note": "Every optimizer trajectory counted once, including calibration. Setup/evaluation/idle are additional; mixed-device sessions have no T4 estimate."}
        result["interpretation"] = "Descriptive comparison of calibrated pipelines plus predeclared matched-mode controls. No mechanism-recovery, significance or convergence claim. Positive gain favors distillation; positive fidelity loss means worse policy fidelity."
    atomic_json(folder / "summary.json", result)
    return result
