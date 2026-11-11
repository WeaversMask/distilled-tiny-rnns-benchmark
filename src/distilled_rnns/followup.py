"""Frozen validation calibration and descriptive checks on fresh simulation seeds.

Calibration is exploratory. Neither oracle policies, group labels nor test
scores select regularization. The fresh-seed aggregate excludes calibration.
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
from .evaluation import evaluate_predictions, make_trial_masks
from .experiment import (atomic_json, environment_info, resolve_device, run_lock,
                         save_environment_inventory, source_fingerprint,
                         t4_training_cost_proxy, _save_npz)
from .models import BatchedStudentGRU, TeacherGRU
from .simulation import simulate_population
from .training import fit_model


@dataclass(frozen=True)
class FollowupConfig:
    base: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(
        teacher_steps=500, student_steps=1000))
    solo_l2_grid: tuple = (0.0, 0.001, 0.01, 0.1)
    replication_seeds: tuple = (20261101, 20261111, 20261121, 20261131, 20261141)
    max_seconds_per_invocation: float = 1800.0

    def __post_init__(self):
        if not isinstance(self.base, ExperimentConfig):
            raise ValueError("base must be an ExperimentConfig")
        grid = tuple(self.solo_l2_grid)
        if not grid or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v < 0 for v in grid):
            raise ValueError("solo_l2_grid must contain finite nonnegative numbers")
        if grid != tuple(sorted(set(grid))) or grid[0] != 0:
            raise ValueError("solo_l2_grid must be ascending, unique, and include zero")
        seeds = tuple(self.replication_seeds)
        if not seeds or len(set(seeds)) != len(seeds) or self.base.seed in seeds:
            raise ValueError("replication_seeds must be unique and exclude calibration")
        for seed in seeds:
            replace(self.base, seed=seed)  # Reuse integer/range checks.
        # Separate each population's data/split/initialization seed offsets.
        all_seeds = sorted((self.base.seed, *seeds))
        if any(right - left < 4 for left, right in zip(all_seeds, all_seeds[1:])):
            raise ValueError("Population seeds must be separated by at least four to avoid reusing random streams")
        cap = self.max_seconds_per_invocation
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("max_seconds_per_invocation must be finite and positive")
        object.__setattr__(self, "solo_l2_grid", grid)
        object.__setattr__(self, "replication_seeds", seeds)

    def to_dict(self):
        return {"base": self.base.to_dict(), "solo_l2_grid": list(self.solo_l2_grid),
                "replication_seeds": list(self.replication_seeds),
                "max_seconds_per_invocation": self.max_seconds_per_invocation}

    @classmethod
    def from_file(cls, path):
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or set(values) - {
                "base", "solo_l2_grid", "replication_seeds", "max_seconds_per_invocation"}:
            raise ValueError("Unknown follow-up configuration settings")
        values = dict(values)
        if "base" in values:
            values["base"] = ExperimentConfig(**values["base"])
        return cls(**values)


def select_l2(validation_scores):
    """Choose one population-wide strength from validation scores alone.

    Scores are each candidate's mean selected per-person validation NLL.
    Exact ties choose the smaller strength; no subgroup or oracle access.
    """
    if not validation_scores or any(not math.isfinite(score) for score in validation_scores.values()):
        raise ValueError("All candidate validation scores must be finite")
    return min(validation_scores, key=lambda strength: (validation_scores[strength], strength))


def convergence_details(phases, config):
    result = {}
    for name, phase in phases.items():
        steps = config.teacher_steps if name == "teacher" else config.student_steps
        selected = np.atleast_1d(phase["best_step"])
        history = phase["history"]
        boundary = float(np.mean(selected == steps))
        recent = history[-min(6, len(history))]["validation_nll"] - history[-1]["validation_nll"]
        result[name] = {
            "completed_steps": phase["completed_steps"], "step_limit": steps,
            "best_step_min": int(selected.min()), "best_step_median": float(np.median(selected)),
            "best_step_max": int(selected.max()), "fraction_selected_at_limit": boundary,
            "recent_raw_validation_improvement": float(recent),
            "needs_longer_training_check": boundary > 0.1 or (
                phase["completed_steps"] == steps and recent > 0.0001),
            "note": "A flag requests inspection; absence of a flag does not prove convergence. Curves use current weights; scores use selected weights.",
        }
    return result


def _plot_convergence(phases, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(11, 3 * math.ceil(len(phases) / 2)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(math.ceil(len(phases) / 2), 2, squeeze=False).ravel()
    for ax, (name, phase) in zip(axes, phases.items()):
        history = phase["history"]
        for key, label in (("train_nll", "Train"), ("validation_nll", "Validation")):
            ax.plot([row["step"] for row in history], [row[key] for row in history], label=label)
        ax.set(title=name, xlabel="Optimization step", ylabel="Observed-choice NLL")
        ax.legend()
    for ax in axes[len(phases):]:
        ax.set_visible(False)
    figure.savefig(path, dpi=140)


def _dataset(config, folder, device, strengths, chosen_l2, protocol_hash, deadline):
    """Fit all candidates, freeze validation selection, then score test targets."""
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"config": config.to_dict(), "strengths": list(strengths),
                "frozen_l2": chosen_l2, "protocol_sha256": protocol_hash}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Dataset configuration or protocol changed")
    else:
        if any(folder.glob("*.pt")):
            raise ValueError("Checkpoints without a manifest cannot be resumed")
        manifest = {"identity": identity, "sessions": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    packages = {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}
    session = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "environment": environment_info(device),
               "package_inventory": save_environment_inventory(folder, packages)}
    manifest["sessions"].append(session)
    atomic_json(manifest_path, manifest)
    data = simulate_population(config.n_subjects, config.n_trials, config.rare_fraction, config.seed)
    masks = make_trial_masks(config.n_subjects, config.n_trials, config.seed + 1)
    _save_npz(folder / "data.npz", **data, **{f"mask_{k}": v for k, v in masks.items()})
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["choice"], dtype=torch.float32, device=device)
    phases, predictions = {}, {}
    teacher_logits = None
    plan = [("teacher", 0.0)] + [(f"solo_l2_{i}", v) for i, v in enumerate(strengths)] + [("distilled", 0.0)]
    for name, strength in plan:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return _incomplete_dataset(folder, manifest, session, started, name)
        is_teacher = name == "teacher"
        init_seed = config.seed + (2 if is_teacher else 3)
        torch.manual_seed(init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(init_seed)
        model = (TeacherGRU(x.shape[-1], config.teacher_hidden) if is_teacher else
                 BatchedStudentGRU(config.n_subjects, x.shape[-1], config.student_hidden)).to(device)
        print(f"\nSeed {config.seed}: {name}, L2={strength}", flush=True)
        phases[name] = fit_model(
            model, x, y, masks["train"], masks["val"],
            steps=config.teacher_steps if is_teacher else config.student_steps,
            lr=config.learning_rate, seed=init_seed, eval_every=config.eval_every,
            patience=config.patience, checkpoint_path=folder / f"{name}.pt",
            teacher_logits=teacher_logits if name == "distilled" else None,
            teacher_prop=config.teacher_prop if name == "distilled" else 0,
            temperature=config.temperature, l2_strength=strength,
            max_seconds=min(config.max_seconds_per_phase, remaining))
        atomic_json(folder / "training.json", phases)
        if phases[name]["stopped_reason"] == "time_limit":
            return _incomplete_dataset(folder, manifest, session, started, name)
        with torch.no_grad():
            logits = model(x)
            predictions[name] = torch.sigmoid(logits).cpu().numpy()
            if is_teacher:
                teacher_logits = logits.detach()
        del model
    validation_scores = {value: float(np.mean(phases[f"solo_l2_{i}"]["best_validation_nll"]))
                         for i, value in enumerate(strengths)}
    selected = select_l2(validation_scores) if chosen_l2 is None else chosen_l2
    selection = {
        "protocol_sha256": protocol_hash, "selected_l2": selected,
        "selection_kind": "calibration_validation" if chosen_l2 is None else "frozen_from_calibration",
        "candidate_validation_nll": [{"l2": k, "validation_nll": v} for k, v in validation_scores.items()],
        "criterion": "Population mean of per-person selected observed-choice validation NLL; exact ties choose smaller L2.",
    }
    selection_path = folder / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen selection differs on resume; use the original environment and source")
    atomic_json(selection_path, selection)  # Durable BEFORE any test or oracle scoring.
    chosen_name = f"solo_l2_{strengths.index(selected)}"
    zero_name = f"solo_l2_{strengths.index(0.0)}"
    scored = {"teacher": predictions["teacher"], "distilled": predictions["distilled"],
              "solo": predictions[chosen_name], "solo_unregularized": predictions[zero_name]}
    _save_npz(folder / "predictions.npz", **scored)
    _save_npz(folder / "candidate_predictions.npz", **predictions)
    report = evaluate_predictions(data, scored, masks["test"], config.diagnostic_threshold)
    session.update({"status": "complete", "wall_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    total_seconds = sum(phase["elapsed_seconds"] for phase in phases.values())
    report.update({"status": "complete", "seed": config.seed, "output_dir": str(folder),
                   "selection": selection, "phases": phases,
                   "model_labels": {"solo": "population-selected L2 solo", "solo_unregularized": "longer unregularized solo"},
                   "convergence": convergence_details(phases, config),
                   "timing": {"cumulative_training_seconds": total_seconds,
                              **t4_training_cost_proxy(total_seconds, config.t4_units_per_hour, manifest["sessions"])}})
    _plot_convergence(phases, folder / "convergence.png")
    atomic_json(folder / "report.json", report)
    atomic_json(manifest_path, manifest)
    return report


def _incomplete_dataset(folder, manifest, session, started, phase):
    session.update({"status": "incomplete", "phase": phase, "wall_seconds": time.perf_counter() - started})
    atomic_json(folder / "manifest.json", manifest)
    return {"status": "incomplete", "phase": phase}


def _range_summary(values):
    supported = [v for v in values if v is not None]
    return {"mean": float(np.mean(supported)) if supported else None,
            "min": min(supported) if supported else None,
            "max": max(supported) if supported else None,
            "supported_seed_count": len(supported), "planned_seed_count": len(values)}


def aggregate_replicates(reports, expected_seeds):
    """Paired seed-level descriptive summaries; never pool people across seeds."""
    if [r.get("seed") for r in reports] != list(expected_seeds) or any(r.get("status") != "complete" for r in reports):
        raise ValueError("Every planned seed must complete before aggregation")
    groups = {}
    for group in ("rare", "typical"):
        rows = []
        for report in reports:
            metrics = report["groups"][group]
            models = metrics["models"]
            comparisons = {}
            for baseline in ("solo", "solo_unregularized"):
                solo, distilled = models[baseline], models["distilled"]
                comparisons[baseline] = {
                    "likelihood_gain": solo["nll"] - distilled["nll"],
                    "fidelity_loss": distilled["kl"] - solo["kl"],
                    "diagnostic_likelihood_gain": None if solo["diagnostic_nll"] is None else solo["diagnostic_nll"] - distilled["diagnostic_nll"],
                }
            rows.append({"seed": report["seed"], "subject_count": metrics["subject_count"],
                         "diagnostic_count": metrics["diagnostic_count"],
                         "diagnostic_subject_count": models["solo"]["diagnostic_subject_count"],
                         "comparisons": comparisons, "models": models})
        groups[group] = {"per_seed": rows,
            "comparisons": {baseline: {metric: _range_summary([row["comparisons"][baseline][metric] for row in rows])
                for metric in ("likelihood_gain", "fidelity_loss", "diagnostic_likelihood_gain")}
                for baseline in ("solo", "solo_unregularized")},
            "models": {model: {metric: _range_summary([row["models"][model][metric] for row in rows])
                for metric in ("nll", "kl", "diagnostic_nll")} for model in rows[0]["models"]}}
    return groups


def run_followup(config: FollowupConfig, output_dir, device="auto"):
    """Run/resume a fixed calibration and all planned fresh seeds.

    The invocation cap is soft: setup, complete steps, checkpoint writes and
    evaluation can add time. Never write one output folder from two runtimes.
    """
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
                "selection": "One global solo L2 minimizes calibration mean selected validation NLL; ties choose smaller strength. Freeze across replication seeds.",
                "scope": "Exploratory conditional first-stage parametric study; no mechanistic gate or hypothesis test.",
                "l2_convention": "(lambda/2) * sum of all squared parameters including biases, added to summed per-person training means.",
                "aggregation": "Equal-weight seed summaries; calibration excluded; unsupported diagnostics remain null."}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Study configuration or source changed; choose a new output directory")
    elif any(folder.iterdir()):
        raise ValueError("Existing output without a protocol cannot be resumed")
    else:
        atomic_json(protocol_path, protocol)
    calibration = _dataset(config.base, folder / "calibration", device,
                           config.solo_l2_grid, None, protocol_hash, deadline)
    reports = []
    if calibration["status"] == "complete":
        chosen = calibration["selection"]["selected_l2"]
        for seed in config.replication_seeds:
            if time.perf_counter() >= deadline:
                break
            report = _dataset(replace(config.base, seed=seed), folder / f"seed-{seed}", device,
                              tuple(sorted({0.0, chosen})), chosen, protocol_hash, deadline)
            if report["status"] != "complete":
                break
            reports.append(report)
    complete = calibration["status"] == "complete" and len(reports) == len(config.replication_seeds)
    result = {"status": "complete" if complete else "incomplete", "output_dir": str(folder),
              "protocol_sha256": protocol_hash, "planned_seeds": list(config.replication_seeds),
              "completed_seeds": [report["seed"] for report in reports],
              "pending_seeds": [seed for seed in config.replication_seeds if seed not in [r["seed"] for r in reports]],
              "message": "Complete." if complete else "Time budget reached. Rerun the same cell with the same configuration and run name to resume. No aggregate is available yet."}
    if calibration["status"] == "complete":
        chosen = calibration["selection"]["selected_l2"]
        result["calibration"] = {"selected_l2": chosen, "report": str(folder / "calibration/report.json"),
                                 "lambda_at_grid_boundary": chosen == max(config.solo_l2_grid) and chosen > 0,
                                 "note": "Already inspected seed; exploratory and excluded from repeat summaries. Selection is dominated by the typical majority."}
    if complete:
        result["groups"] = aggregate_replicates(reports, config.replication_seeds)
        all_reports = [calibration, *reports]
        warnings = [f"Seed {r['seed']}, {name}: inspect the longer-training flag."
                    for r in all_reports for name, detail in r["convergence"].items() if detail["needs_longer_training_check"]]
        if result["calibration"]["lambda_at_grid_boundary"]:
            warnings.append("Calibration chose the largest tested L2; inspect a wider grid in a separate future protocol.")
        result["convergence_warnings"] = warnings
        result["interpretation"] = "Descriptive seed means/ranges only. No significance, mechanism recovery, or convergence claim. Positive gain favors distillation; positive fidelity loss means worse policy fidelity."
        seconds = sum(r["timing"]["cumulative_training_seconds"] for r in all_reports)
        proxies = [r["timing"]["t4_equivalent_units_at_user_rate"] for r in all_reports]
        result["timing"] = {"cumulative_training_seconds": seconds,
                            "t4_equivalent_units_at_user_rate": sum(proxies) if all(p is not None for p in proxies) else None,
                            "note": "Includes all calibration candidates and replication fits; excludes setup, evaluation and idle billing. Mixed-device histories have no cost estimate."}
    atomic_json(folder / "summary.json", result)
    return result
