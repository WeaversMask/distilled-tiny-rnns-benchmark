"""Masked training, validation selection, and resumable local checkpoints.

Held-out observations may appear in *later* causal history inputs, as appropriate
for one-step-ahead prediction with interspersed held-out target losses. Neither
their hard targets nor their teacher targets enter the optimization objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .models import BatchedStudentGRU


def masked_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None = None,
    teacher_prop: float = 0.0,
    temperature: float = 1.0,
    independent_students: bool = False,
) -> torch.Tensor:
    """Binary distillation loss, reduced separately for each independent student.

At alpha=1 the soft cross-entropy has the same student gradients as KL, with
the customary T**2 factor. At alpha=0 it is ordinary observed-choice BCE.
"""
    if not math.isfinite(teacher_prop) or not 0 <= teacher_prop <= 1:
        raise ValueError("teacher_prop must lie in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    hard = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = hard
    if teacher_prop:
        if teacher_logits is None:
            raise ValueError("teacher_logits are required when teacher_prop > 0")
        soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
        soft = F.binary_cross_entropy_with_logits(logits / temperature, soft_targets, reduction="none")
        loss = (1 - teacher_prop) * hard + teacher_prop * (temperature**2) * soft
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("every subject must have at least one training target")
    if independent_students:
        # Summing per-person means preserves each student's gradient scale when
        # more independent students are added. No cross-person clipping is used.
        return ((loss * mask).sum(dim=1) / counts).sum()
    return (loss * mask).sum() / counts.sum()


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signature(model: nn.Module, settings: dict[str, Any], tensors: list[torch.Tensor | None]) -> str:
    digest = hashlib.sha256()
    description = {
        "format": 1,
        "model": type(model).__name__,
        "parameters": {name: [list(value.shape), str(value.dtype)] for name, value in model.state_dict().items()},
        **settings,
    }
    digest.update(json.dumps(description, sort_keys=True).encode())
    for tensor in tensors:
        if tensor is None:
            digest.update(b"none")
        else:
            cpu = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
            digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def fit_model(
    model: nn.Module,
    x,
    targets,
    train_mask,
    val_mask,
    *,
    steps: int,
    lr: float,
    checkpoint_path: str | Path | None,
    seed: int,
    eval_every: int = 10,
    patience: int = 20,
    teacher_logits=None,
    teacher_prop: float = 0.0,
    temperature: float = 1.0,
    max_seconds: float | None = None,
    l2_strength: float = 0.0,
) -> dict[str, Any]:
    """Fit a model, resuming a matching checkpoint when it exists.

``steps`` is the total desired number of optimization steps, including those
already saved. Seed the model initialization *before constructing it* as well.
Students run a fixed number of steps and select a best validation state for
each person; ``patience`` applies only to the shared teacher. A time limit is
per invocation and checked between complete steps. Checkpoints hold the latest
optimizer state as well as selected weights; the returned model uses selected
weights, while a resumed run continues from the latest optimization state.

The optional L2 penalty is ``(l2_strength / 2) * sum(parameter**2)`` over
all parameters, including biases. For independent students this is added to
the sum of per-person mean losses, so each person receives the same penalty
as a separate fit, regardless of population size. Validation selection uses
observed-choice NLL without the penalty. Zero preserves the original loss
and checkpoint signature.
"""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if eval_every < 1 or patience < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("eval_every, patience, and lr must be positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    if not math.isfinite(teacher_prop) or not 0 <= teacher_prop <= 1:
        raise ValueError("teacher_prop must lie in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if (isinstance(l2_strength, bool) or not isinstance(l2_strength, (int, float))
            or not math.isfinite(l2_strength) or l2_strength < 0):
        raise ValueError("l2_strength must be finite, numeric, and nonnegative")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    x = torch.as_tensor(x, dtype=dtype, device=device)
    targets = torch.as_tensor(targets, dtype=dtype, device=device)
    train_mask = torch.as_tensor(train_mask, device=device)
    val_mask = torch.as_tensor(val_mask, device=device)
    if train_mask.dtype != torch.bool or val_mask.dtype != torch.bool:
        raise ValueError("masks must have boolean dtype")
    if x.ndim != 3 or targets.shape != x.shape[:2] or train_mask.shape != targets.shape or val_mask.shape != targets.shape:
        raise ValueError("x must be [subjects, trials, features]; targets and masks must match [subjects, trials]")
    if not torch.isfinite(x).all() or not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("inputs must be finite and observed targets must be binary")
    if (train_mask & val_mask).any() or (train_mask.sum(1) == 0).any() or (val_mask.sum(1) == 0).any():
        raise ValueError("train and validation masks must be disjoint and nonempty for every subject")
    if teacher_logits is not None:
        teacher_logits = torch.as_tensor(teacher_logits, dtype=dtype, device=device).detach()
        if teacher_logits.shape != targets.shape or not torch.isfinite(teacher_logits).all():
            raise ValueError("teacher_logits must be finite and match targets")
    if teacher_prop > 0 and teacher_logits is None:
        raise ValueError("teacher_logits are required when teacher_prop > 0")
    independent = isinstance(model, BatchedStudentGRU)
    signature_settings = {
        "lr": lr, "seed": seed, "eval_every": eval_every, "patience": patience,
        "teacher_prop": teacher_prop, "temperature": temperature,
    }
    # Leave the original unregularized signature unchanged for old checkpoints.
    if l2_strength:
        signature_settings["l2_strength"] = float(l2_strength)
    signature = _signature(model, signature_settings,
                           [x, targets, train_mask, val_mask, teacher_logits])
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    history: list[dict[str, Any]] = []
    completed = 0
    elapsed_before = 0.0
    stale = 0
    early_stopped = False
    best_state = _state_copy(model)
    best_val = torch.full((x.shape[0],) if independent else (), float("inf"), dtype=dtype, device=device)
    best_step = torch.zeros(x.shape[0], dtype=torch.int64, device=device) if independent else 0
    if path is not None and path.exists():
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get("signature") != signature:
            raise ValueError("checkpoint does not match this model, data, masks, or training settings; use a new run directory")
        if saved["completed_steps"] > steps:
            raise ValueError("requested steps are fewer than those already saved")
        model.load_state_dict(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        # map_location=device also moves Adam's scalar step counters. Ordinary
        # (non-capturable, non-fused) Adam keeps these on CPU; the moments are
        # already moved to the parameter device by load_state_dict.
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].cpu()
        best_state, best_val, best_step = saved["best_state"], saved["best_val"], saved["best_step"]
        history, completed = saved["history"], saved["completed_steps"]
        elapsed_before, stale = saved["elapsed_seconds"], saved["stale"]
        early_stopped = saved["early_stopped"]
        torch.set_rng_state(saved["rng_state"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_states"]])

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()

    def evaluate(step: int) -> None:
        nonlocal best_val, best_state, best_step, stale
        model.eval()
        with torch.no_grad():
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            validation = (bce * val_mask).sum(1) / val_mask.sum(1)
            value = validation if independent else (bce * val_mask).sum() / val_mask.sum()
            if not torch.isfinite(value).all():
                raise FloatingPointError("non-finite validation loss")
            improved = value < best_val
            if independent:
                for key, current in model.state_dict().items():
                    selector = improved.reshape((-1,) + (1,) * (current.ndim - 1))
                    best_state[key] = torch.where(selector, current.detach(), best_state[key])
                best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
            elif bool(improved):
                best_state = _state_copy(model)
                best_step = step
            best_val = torch.minimum(best_val, value)
            stale = 0 if bool(improved.any()) else stale + 1
            train_bce = ((bce * train_mask).sum() / train_mask.sum()).item()
            history.append({"step": step, "train_nll": train_bce, "validation_nll": validation.mean().item()})
            print(f"{type(model).__name__}: step {step}/{steps}, train NLL {train_bce:.4f}, validation NLL {validation.mean().item():.4f}", flush=True)
        model.train()

    def save() -> None:
        synchronize()
        if path is not None:
            _atomic_save({
                "signature": signature, "current_state": _state_copy(model),
                "best_state": best_state, "best_val": best_val, "best_step": best_step,
                "optimizer": optimizer.state_dict(), "history": history,
                "completed_steps": completed, "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "stale": stale, "early_stopped": early_stopped,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }, path)

    if not history:
        evaluate(0)
        save()
    stopped_reason = "early_stopping" if early_stopped else "steps_completed"
    model.train()
    for step in range(completed + 1, steps + 1):
        if early_stopped:
            break
        synchronize()
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            stopped_reason = "time_limit"
            break
        optimizer.zero_grad(set_to_none=True)
        loss = masked_objective(model(x), targets, train_mask,
                                teacher_logits=teacher_logits, teacher_prop=teacher_prop,
                                temperature=temperature, independent_students=independent)
        if l2_strength:
            loss = loss + (l2_strength / 2) * sum(
                parameter.square().sum() for parameter in model.parameters()
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if not torch.stack([torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]).all():
            raise FloatingPointError("non-finite training gradient")
        optimizer.step()
        completed = step
        if step % eval_every == 0 or step == steps:
            evaluate(step)
            if not independent and stale >= patience:
                early_stopped = True
                stopped_reason = "early_stopping"
            save()
    # A time-limited interruption between scheduled evaluations does not add a
    # validation checkpoint. Only the requested final step gets an extra one.
    save()
    synchronize()
    elapsed = elapsed_before + time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    return {
        "history": history, "best_step": best_step.cpu().tolist() if independent else best_step,
        "completed_steps": completed, "stopped_reason": stopped_reason,
        "elapsed_seconds": elapsed,
        "best_validation_nll": best_val.detach().cpu().tolist(),
        "l2_strength": float(l2_strength),
        "checkpoint_path": str(path) if path is not None else None,
    }
