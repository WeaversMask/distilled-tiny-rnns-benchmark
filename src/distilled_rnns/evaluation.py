"""Conditional first-stage prediction metrics for the preliminary pilot.

Interspersed held-out trials exclude their targets from the fitting objective.
Observed *past* choices and outcomes remain legitimate causal model inputs.
These scores describe conditional prediction on observed histories, not free
rollouts or a test of recovery of the generating cognitive mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PROBABILITY_EPSILON = 1e-12


def _probability_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return array


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def binary_nll(choice: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Elementwise Bernoulli NLL in nats; probabilities are clipped at 1e-12.

    ``probability`` is P(choice=1). Shapes must match exactly; broadcasting is
    deliberately disallowed to expose accidental subject/trial misalignment.
    """
    choice = np.asarray(choice)
    probability = _probability_array(probability, "probability")
    if choice.shape != probability.shape:
        raise ValueError("choice and probability must have identical shapes")
    if not np.all((choice == 0) | (choice == 1)):
        raise ValueError("choice must contain only 0 or 1")
    q = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return np.where(choice == 1, -np.log(q), -np.log1p(-q))


def binary_kl(true_probability: np.ndarray, model_probability: np.ndarray) -> np.ndarray:
    """Elementwise KL(Bernoulli(true) || Bernoulli(model)), in nats.

    Both distributions are clipped to [1e-12, 1-1e-12] for finite scoring.
    Equal inputs give exactly zero, including endpoints; tiny negative values
    from floating-point roundoff are clamped to zero.
    """
    true_probability = _probability_array(true_probability, "true_probability")
    model_probability = _probability_array(model_probability, "model_probability")
    if true_probability.shape != model_probability.shape:
        raise ValueError("true and model probabilities must have identical shapes")
    p = np.clip(true_probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    q = np.clip(model_probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    value = p * (np.log(p) - np.log(q)) + (1 - p) * (np.log1p(-p) - np.log1p(-q))
    return np.maximum(value, 0.0)


def make_trial_masks(n_subjects: int, n_trials: int, seed: int) -> dict[str, np.ndarray]:
    """Create independent, interspersed ~60/20/20 splits for each subject.

    Every subject has at least one trial in each split. Splits partition all
    trial positions, including the first prediction from the initial state.
    This is not a reproduction of the original paper's exact split scheme.
    """
    n_subjects = _positive_integer(n_subjects, "n_subjects")
    n_trials = _positive_integer(n_trials, "n_trials", minimum=3)
    seed = _positive_integer(seed, "seed", minimum=0)
    n_val = max(1, round(0.2 * n_trials))
    n_test = max(1, round(0.2 * n_trials))
    masks = {name: np.zeros((n_subjects, n_trials), dtype=bool) for name in ("train", "val", "test")}
    for subject, sequence in enumerate(np.random.SeedSequence(seed).spawn(n_subjects)):
        order = np.random.default_rng(sequence).permutation(n_trials)
        masks["val"][subject, order[:n_val]] = True
        masks["test"][subject, order[n_val:n_val + n_test]] = True
        masks["train"][subject, order[n_val + n_test:]] = True
    return masks


def _mean_or_none(values: list[float | None]) -> float | None:
    supported = [value for value in values if value is not None]
    return float(np.mean(supported)) if supported else None


def evaluate_predictions(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    test_mask: np.ndarray,
    diagnostic_threshold: float = 0.2,
) -> dict[str, Any]:
    """Summarize held-out NLL and generating-policy KL per subject and group.

    Required predictions are teacher, distilled and solo, each P(first action
    = 1) with shape [subjects, trials]. Extra named models are also supported.
    Diagnostic trials have |true_prob-reference_prob| > threshold. The fixed
    reference is a majority-like policy on the same histories, not a measured
    population mean. Group metrics are unweighted means over subject metrics;
    diagnostic means include only subjects with diagnostic test support.
    """
    choice = np.asarray(data["choice"])
    if choice.ndim != 2 or min(choice.shape) < 1:
        raise ValueError("choice must have nonempty shape [subjects, trials]")
    if not np.all((choice == 0) | (choice == 1)):
        raise ValueError("choice must contain only 0 or 1")
    shape = choice.shape
    true_probability = _probability_array(data["true_prob"], "true_prob")
    reference_probability = _probability_array(data["reference_prob"], "reference_prob")
    if true_probability.shape != shape or reference_probability.shape != shape:
        raise ValueError("true_prob and reference_prob must match choice shape")
    rare = np.asarray(data["rare"])
    if rare.shape != (shape[0],) or rare.dtype != np.dtype(bool):
        raise ValueError("rare must be a boolean vector with one entry per subject")
    test_mask = np.asarray(test_mask)
    if test_mask.shape != shape or test_mask.dtype != np.dtype(bool):
        raise ValueError("test_mask must be boolean and match choice shape")
    if np.any(test_mask.sum(axis=1) == 0):
        raise ValueError("every subject must have at least one held-out test trial")
    if not np.isscalar(diagnostic_threshold) or not np.isfinite(diagnostic_threshold) or not 0 <= diagnostic_threshold <= 1:
        raise ValueError("diagnostic_threshold must be finite and lie in [0, 1]")
    if not {"teacher", "distilled", "solo"}.issubset(predictions):
        raise ValueError("predictions must include teacher, distilled, and solo")
    losses = {}
    for name, probability in predictions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("prediction names must be nonempty strings")
        probability = _probability_array(probability, f"predictions[{name}]")
        if probability.shape != shape:
            raise ValueError(f"predictions[{name}] must match choice shape")
        losses[name] = {"nll": binary_nll(choice, probability), "kl": binary_kl(true_probability, probability)}
    diagnostic_mask = (np.abs(true_probability - reference_probability) > diagnostic_threshold) & test_mask
    subjects = []
    for subject in range(shape[0]):
        mask = test_mask[subject]
        diagnostic = diagnostic_mask[subject]
        diagnostic_count = int(diagnostic.sum())
        row = {
            "subject_id": subject,
            "rare": bool(rare[subject]),
            "test_count": int(mask.sum()),
            "diagnostic_count": diagnostic_count,
            "models": {},
        }
        for name, loss in losses.items():
            row["models"][name] = {
                "nll": float(loss["nll"][subject, mask].mean()),
                "kl": float(loss["kl"][subject, mask].mean()),
                "diagnostic_nll": float(loss["nll"][subject, diagnostic].mean()) if diagnostic_count else None,
            }
        row["likelihood_gain"] = row["models"]["solo"]["nll"] - row["models"]["distilled"]["nll"]
        row["fidelity_loss"] = row["models"]["distilled"]["kl"] - row["models"]["solo"]["kl"]
        subjects.append(row)
    groups = {}
    for group, is_rare in (("rare", True), ("typical", False)):
        rows = [row for row in subjects if row["rare"] == is_rare]
        groups[group] = {
            "subject_count": len(rows),
            "test_count": sum(row["test_count"] for row in rows),
            "diagnostic_count": sum(row["diagnostic_count"] for row in rows),
            "models": {
                name: {
                    "nll": _mean_or_none([row["models"][name]["nll"] for row in rows]),
                    "kl": _mean_or_none([row["models"][name]["kl"] for row in rows]),
                    "diagnostic_nll": _mean_or_none([row["models"][name]["diagnostic_nll"] for row in rows]),
                    "diagnostic_subject_count": sum(row["diagnostic_count"] > 0 for row in rows),
                }
                for name in losses
            },
            "likelihood_gain": _mean_or_none([row["likelihood_gain"] for row in rows]),
            "fidelity_loss": _mean_or_none([row["fidelity_loss"] for row in rows]),
        }
    return {
        "metadata": {
            "scope": "preliminary conditional first-stage prediction pilot",
            "units": "nats per trial",
            "group_aggregation": "unweighted mean of subject metrics; diagnostic means omit unsupported subjects",
            "diagnostic_threshold": float(diagnostic_threshold),
            "diagnostic_reference": "fixed majority-like reference policy on the same observed histories",
            "probability_epsilon": PROBABILITY_EPSILON,
            "statistical_inference": "none; no hypothesis acceptance or rejection from this pilot",
        },
        "per_subject": subjects,
        "groups": groups,
    }


def save_quadrant_plot(report: dict[str, Any], path: str | Path) -> Path:
    """Save a preliminary descriptive quadrant plot without GUI dependencies."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(7, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    for is_rare, name, color in ((False, "Typical", "#457b9d"), (True, "Rare", "#d1495b")):
        rows = [row for row in report["per_subject"] if row["rare"] == is_rare]
        if rows:
            axes.scatter(
                [row["likelihood_gain"] for row in rows],
                [row["fidelity_loss"] for row in rows],
                label=f"{name} (n={len(rows)})", c=color, alpha=0.65, s=25,
            )
    axes.axhline(0, color="#555555", linewidth=0.8)
    axes.axvline(0, color="#555555", linewidth=0.8)
    axes.set_xlabel("Prediction gain: solo NLL − distilled NLL (nats/trial)")
    axes.set_ylabel("Fidelity loss: distilled KL − solo KL (nats/trial)")
    axes.set_title("Preliminary pilot: conditional first-stage predictions")
    axes.text(0.99, 0.99, "Upper right: better prediction, worse policy fidelity", transform=axes.transAxes, ha="right", va="top", fontsize=8)
    axes.legend()
    figure.savefig(path, dpi=160)
    return path
