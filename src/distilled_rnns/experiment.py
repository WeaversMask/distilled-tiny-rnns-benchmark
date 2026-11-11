"""Resumable three-model pilot and provenance recording."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import torch

from .config import ExperimentConfig
from .evaluation import evaluate_predictions, make_trial_masks, save_quadrant_plot
from .models import BatchedStudentGRU, TeacherGRU
from .simulation import simulate_population
from .training import fit_model


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def save_environment_inventory(folder, packages):
    """Save a content-addressed package inventory and a convenience latest copy.

    Session manifests reference the immutable inventory, never the latest copy.
    Existing inventory contents must match their content-addressed filename.
    """
    folder = Path(folder)
    contents = "\n".join(sorted(set(packages))) + "\n"
    sha256 = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    relative_path = Path("environments") / f"packages-{sha256}.txt"
    inventory_path = folder / relative_path
    if inventory_path.exists():
        if inventory_path.read_text(encoding="utf-8") != contents:
            raise ValueError("Saved environment inventory does not match its content hash")
    else:
        _atomic_text(inventory_path, contents)
    _atomic_text(folder / "environment-freeze.txt", contents)
    return {"status": "recorded", "path": relative_path.as_posix(), "sha256": sha256}


def t4_training_cost_proxy(total_seconds, units_per_hour, sessions):
    """Convert cumulative time only when every session is attributed to a T4.

    Checkpoint times lack per-device attribution. A mixed or unknown history
    therefore cannot safely be converted, even if the current device is a T4.
    This conservative check includes sessions which only reload completed fits.
    """
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise ValueError("Training duration must be finite and nonnegative")
    if not math.isfinite(units_per_hour) or units_per_hour <= 0:
        raise ValueError("The T4 unit rate must be finite and positive")
    if not sessions or any(not isinstance(s.get("environment"), dict) for s in sessions):
        return {"t4_equivalent_units_at_user_rate": None,
                "t4_cost_proxy_reason": "Unavailable: one or more sessions have no device attribution."}
    for session in sessions:
        environment = session["environment"]
        device = environment.get("device")
        gpu = environment.get("gpu")
        if not isinstance(device, str) or device.split(":", 1)[0] != "cuda" or not isinstance(gpu, str) or "T4" not in gpu:
            return {"t4_equivalent_units_at_user_rate": None,
                    "t4_cost_proxy_reason": "Unavailable: cumulative training includes a non-T4 or unknown device history."}
    return {"t4_equivalent_units_at_user_rate": total_seconds / 3600 * units_per_hour,
            "t4_cost_proxy_reason": "Every recorded session used a T4; conversion covers cumulative training time only."}


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_device(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this run. Select a GPU runtime in Colab.")
    return torch.device(device)


def environment_info(device):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "matplotlib")},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
    }


@contextmanager
def run_lock(folder):
    """Local-runtime lock released by the OS on interruption.

    Store it on the runtime disk: Drive mounts need not support file locks.
    Do not write to one Drive run folder from multiple Colab runtimes.
    """
    import fcntl  # Local verification is macOS; Colab is Linux.
    key = hashlib.sha256(str(folder).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"distilled-rnns-{key}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output folder is already being trained by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_experiment(config: ExperimentConfig, output_dir, device="auto"):
    """Run or resume one configuration. Test metrics never select weights.

    Outputs contain synthetic observations only. Reusing a folder after changing
    configuration or source is rejected; use a new directory for a new run.
    """
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    selected = resolve_device(device)
    if selected.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with run_lock(folder):
        return _run(config, folder, selected)


def _run(config, folder, device):
    started = time.perf_counter()
    source_hash = source_fingerprint()
    manifest_path = folder / "manifest.json"
    identity = {"config": config.to_dict(), "source_sha256": source_hash, "schema_version": 1}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Run configuration or source changed; choose a new output directory")
        manifest = previous
    else:
        # A partial/copy without provenance must not silently reuse checkpoints.
        if any(folder.glob("*.pt")):
            raise ValueError("Checkpoints without a manifest cannot be resumed")
        manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(), "sessions": []}
    for previous_session in manifest["sessions"]:
        if "package_inventory" not in previous_session:
            previous_session["package_inventory"] = {
                "status": "unavailable_legacy_session",
                "note": "No immutable package inventory was recorded for this earlier session; its complete environment cannot be reconstructed.",
            }
    # Record installed distributions directly; launching a package installer is
    # unnecessary for provenance and can block an otherwise valid training run.
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
    train = torch.as_tensor(masks["train"], dtype=torch.bool, device=device)
    val = torch.as_tensor(masks["val"], dtype=torch.bool, device=device)
    predictions, phases = {}, {}
    teacher_logits = None
    for name in ("teacher", "solo", "distilled"):
        # Solo and distilled use exactly the same initial student weights.
        init_seed = config.seed + (2 if name == "teacher" else 3)
        torch.manual_seed(init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(init_seed)
        model = (TeacherGRU(x.shape[-1], config.teacher_hidden) if name == "teacher" else
                 BatchedStudentGRU(config.n_subjects, x.shape[-1], config.student_hidden)).to(device)
        steps = config.teacher_steps if name == "teacher" else config.student_steps
        print(f"\n{name}: up to {steps} training steps on {device}", flush=True)
        phases[name] = fit_model(
            model, x, y, train, val, steps=steps, lr=config.learning_rate,
            checkpoint_path=folder / f"{name}.pt", seed=init_seed,
            eval_every=config.eval_every, patience=config.patience,
            teacher_logits=teacher_logits if name == "distilled" else None,
            teacher_prop=config.teacher_prop if name == "distilled" else 0.0,
            temperature=config.temperature,
            max_seconds=config.max_seconds_per_phase,
        )
        atomic_json(folder / "training.json", phases)
        # Early stopping is an intentional completion; a time cap is resumable.
        reason = phases[name]["stopped_reason"]
        if reason == "time_limit":
            session["wall_seconds"] = time.perf_counter() - started
            session["status"] = "incomplete"
            atomic_json(manifest_path, manifest)
            return {"status": "incomplete", "phase": name, "output_dir": str(folder),
                    "message": "Time budget reached. Rerun with the same configuration and output folder to resume."}
        model.eval()
        with torch.no_grad():
            logits = model(x)
            predictions[name] = torch.sigmoid(logits).cpu().numpy()
            if name == "teacher":
                teacher_logits = logits.detach()
        del model

    _save_npz(folder / "predictions.npz", **predictions)
    report = evaluate_predictions(data, predictions, masks["test"], config.diagnostic_threshold)
    session["wall_seconds"] = time.perf_counter() - started
    session["status"] = "complete"
    session["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    total_seconds = sum(p["elapsed_seconds"] for p in phases.values())
    report.update({
        "status": "complete", "output_dir": str(folder), "config": config.to_dict(),
        "source_sha256": source_hash,
        "scope": "Preliminary synthetic first-stage parametric pilot; not a mechanistic gate or published-method replication.",
        "timing": {
            "cumulative_training_seconds": total_seconds,
            "this_invocation_seconds": session["wall_seconds"],
            **t4_training_cost_proxy(total_seconds, config.t4_units_per_hour, manifest["sessions"]),
            "note": "Training wall time only; billed GPU sessions also include setup, evaluation, and idle time. CPU time is not a GPU cost estimate.",
        },
        "phases": phases,
    })
    atomic_json(folder / "report.json", report)
    save_quadrant_plot(report, folder / "quadrant.png")
    atomic_json(manifest_path, manifest)
    print(f"\nPilot complete. Results: {folder / 'report.json'}", flush=True)
    return report
