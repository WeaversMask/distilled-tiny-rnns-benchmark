"""Training-only scientific baselines and the declared behavioral evaluation."""
from __future__ import annotations

from itertools import product

import numpy as np

from distilled_rnns.baseline_audit import reference_predictions
from distilled_rnns.evaluation import binary_kl, binary_nll
from distilled_rnns.simulation import replay_policy
from foundation.classical_baselines import candidate_probabilities, fit_baselines
from foundation.reward_cue import cue_probabilities, fit_family_mixture
from foundation.strategy_probes import SIGNATURE_ORDER, rollout_stay_signature, two_trial_probe_bank

from .simulation import HISTORY_KEYS


SCALAR_METRICS = ("nll", "kl", "diagnostic_nll", "diagnostic_kl", "signature_rms_error", "probe_kl", "pull")


def _probe_history():
    histories, _ = two_trial_probe_bank()
    # The final row is a placeholder: trial-three prediction precedes that row.
    return {
        name: np.column_stack((histories[:, :, i], np.zeros(256, dtype=np.int64)))
        for i, name in enumerate(HISTORY_KEYS)
    }


def baseline_predictions(config, data, train_mask):
    """Fit from observations and training targets; freeze weights in the assay."""
    settings = config.design["baseline_recipe"]
    grid = np.array(
        list(
            product(
                settings["hybrid_grid_weight"], settings["hybrid_grid_alpha"], settings["hybrid_grid_beta"]
            )
        ),
        dtype=np.float64,
    )
    cue_grid = tuple(settings["cue_grid"])
    observed = {key: data[key] for key in HISTORY_KEYS}
    hybrid = fit_baselines(
        observed,
        train_mask,
        grid,
        pseudocount=settings["hybrid_population_pseudocount_per_point"],
        max_iterations=settings["em_maximum_iterations"],
        tolerance=settings["em_relative_objective_tolerance"],
    )
    # This already-reviewed two-family estimator has these exact frozen values.
    expected = (0.1, 2000, 1e-10)
    actual = (
        settings["between_family_pseudocount_per_family"],
        settings["em_maximum_iterations"],
        settings["em_relative_objective_tolerance"],
    )
    if actual != expected:
        raise ValueError("Family mixture settings differ from the reviewed implementation")
    mixture = fit_family_mixture(observed, train_mask, grid, cue_grid)
    if not hybrid.converged or not mixture.converged:
        raise RuntimeError("Classical baseline did not converge at its frozen tolerance")
    fixed = reference_predictions(data["choice"], train_mask)
    frequency = fixed["frequency"][:, 0]
    predictions = {
        "individual_grid_ml": hybrid.individual_probability,
        "pooled_grid_ml": hybrid.pooled_probability,
        "hierarchical_hybrid": hybrid.hierarchical_probability,
        "hierarchical_hybrid_cue_mixture": mixture.probability,
        "coinflip": fixed["coinflip"],
        "training_frequency": fixed["frequency"],
    }
    probe_history = _probe_history()
    hybrid_probes = candidate_probabilities(probe_history, grid).transpose(1, 0, 2)
    cue_probes = cue_probabilities(probe_history, cue_grid).transpose(1, 0, 2)
    mixture_hybrid = np.einsum("ng,gbt->nbt", mixture.hybrid_posterior, hybrid_probes)
    mixture_cue = np.einsum("ng,gbt->nbt", mixture.cue_posterior, cue_probes)
    probes = {
        "individual_grid_ml": hybrid_probes[hybrid.individual_index],
        "pooled_grid_ml": np.broadcast_to(
            hybrid_probes[hybrid.pooled_index], (len(frequency), 256, 3)
        ).copy(),
        "hierarchical_hybrid": np.einsum("ng,gbt->nbt", hybrid.posterior_weights, hybrid_probes),
        "hierarchical_hybrid_cue_mixture": (
            mixture.family_posterior[:, :1, None] * mixture_hybrid
            + mixture.family_posterior[:, 1:, None] * mixture_cue
        ),
        "coinflip": np.full((len(frequency), 256, 3), 0.5),
        "training_frequency": np.broadcast_to(frequency[:, None, None], (len(frequency), 256, 3)).copy(),
    }
    metadata = {
        "hybrid_grid": grid.tolist(),
        "cue_grid": list(cue_grid),
        "individual_grid_indices": hybrid.individual_index.tolist(),
        "pooled_grid_index": hybrid.pooled_index,
        "hybrid_prior_weights": hybrid.prior_weights.tolist(),
        "hybrid_posterior_weights": hybrid.posterior_weights.tolist(),
        "family_prior_hybrid_cue": mixture.family_prior.tolist(),
        "family_posterior_hybrid_cue": mixture.family_posterior.tolist(),
        "family_mixture_hybrid_posterior": mixture.hybrid_posterior.tolist(),
        "family_mixture_cue_posterior": mixture.cue_posterior.tolist(),
        "hybrid_em_iterations": len(hybrid.marginal_objective) - 1,
        "family_em_iterations": mixture.iterations,
        "converged": True,
        "assay_posterior": "Training posterior remains fixed at every reached probe history.",
        "training_frequency_prior": "Beta(1,1), matching the pilot training-only reference.",
    }
    return predictions, probes, metadata


def true_probe_predictions(data):
    """Evaluate each true generating policy on the same canonical histories."""
    histories = _probe_history()
    cache = {}
    out = []
    for person in range(len(data["rare"])):
        if data["is_cue"][person]:
            key = ("cue", float(data["cue_reliability"][person]))
            if key not in cache:
                cache[key] = cue_probabilities(histories, [key[1]])[:, 0]
        else:
            key = ("hybrid", *(float(data[name][person]) for name in ("model_based_weight", "alpha", "beta")))
            if key not in cache:
                cache[key] = replay_policy(histories, *key[1:])
        out.append(cache[key])
    return np.stack(out)


def _probability(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{label} must have shape {shape} and finite probabilities")
    return array


def _mean_supported(values):
    supported = [value for value in values if value is not None]
    return float(np.mean(supported)) if supported else None


def evaluate(config, data, predictions, probe_predictions, test_mask, baseline_metadata=None):
    """Return equal-person scores and primary coordinates for both selections."""
    choice = np.asarray(data["choice"])
    if choice.ndim != 2 or min(choice.shape) < 1:
        raise ValueError("Observed choices must have shape [people, trials]")
    mask = np.asarray(test_mask)
    if mask.shape != choice.shape or mask.dtype != bool or not mask.any(1).all():
        raise ValueError("Every person needs nonempty boolean test support")
    rare = np.asarray(data["rare"])
    if rare.shape != (len(choice),) or rare.dtype != bool or not rare.any() or rare.all():
        raise ValueError("Both rare and typical groups must be represented")
    truth = _probability(data["true_prob"], choice.shape, "true_prob")
    reference = _probability(data["reference_prob"], choice.shape, "reference_prob")
    if predictions.keys() != probe_predictions.keys():
        raise ValueError("Observed-history and probe models must match")
    if not {"solo_population", "distilled_population"}.issubset(predictions):
        raise ValueError("Primary solo and distilled population predictions are required")
    diagnostic = mask & (
        np.abs(truth - reference) > config.design["secondary_outcomes"]["diagnostic_threshold"]
    )
    truth_probes = true_probe_predictions(data)
    truth_signatures = np.array([rollout_stay_signature(p) for p in truth_probes])
    majority_signature = truth_signatures[~rare].mean(0)
    pull_directions = majority_signature - truth_signatures
    pull_denominators = np.square(pull_directions).sum(1)
    subjects = [
        {
            "subject_id": i,
            "rare": bool(rare[i]),
            "test_count": int(mask[i].sum()),
            "diagnostic_count": int(diagnostic[i].sum()),
            "true_signature": truth_signatures[i].tolist(),
            "models": {},
            "primary": {},
        }
        for i in range(len(choice))
    ]
    for name, probability in predictions.items():
        p = _probability(probability, choice.shape, name)
        probes = _probability(probe_predictions[name], (len(choice), 256, 3), name + " probes")
        nll, kl = binary_nll(choice, p), binary_kl(truth, p)
        for i, row in enumerate(subjects):
            signature = rollout_stay_signature(probes[i])
            error = signature - truth_signatures[i]
            diagnostic_exists = diagnostic[i].any()
            row["models"][name] = {
                "nll": float(nll[i, mask[i]].mean()),
                "kl": float(kl[i, mask[i]].mean()),
                "diagnostic_nll": float(nll[i, diagnostic[i]].mean()) if diagnostic_exists else None,
                "diagnostic_kl": float(kl[i, diagnostic[i]].mean()) if diagnostic_exists else None,
                "signature": signature.tolist(),
                "signature_rms_error": float(np.sqrt(np.square(error).mean())),
                "probe_kl": float(binary_kl(truth_probes[i, :, 2], probes[i, :, 2]).mean()),
                "pull": float(np.dot(error, pull_directions[i]) / pull_denominators[i])
                if pull_denominators[i] > 1e-12
                else None,
            }
    for row in subjects:
        for mode in ("population", "individual"):
            if not {"solo_" + mode, "distilled_" + mode}.issubset(predictions):
                continue
            solo, distilled = (row["models"][arm + "_" + mode] for arm in ("solo", "distilled"))
            gain = solo["nll"] - distilled["nll"]
            distortion = distilled["signature_rms_error"] - solo["signature_rms_error"]
            row["primary"][mode] = {
                "prediction_gain": gain,
                "behavioral_distortion": distortion,
                "quadrant_fraction": float(gain > 0 and distortion > 0),
            }
    groups = {}
    for group_name, selector in (("rare", rare), ("typical", ~rare)):
        rows = [row for row in subjects if row["rare"] == (group_name == "rare")]
        models = {}
        for name in predictions:
            scores = {
                metric: _mean_supported([row["models"][name][metric] for row in rows])
                for metric in SCALAR_METRICS
            }
            scores["signature"] = np.mean([row["models"][name]["signature"] for row in rows], axis=0).tolist()
            scores["pull_subject_count"] = sum(row["models"][name]["pull"] is not None for row in rows)
            scores["diagnostic_subject_count"] = sum(row["diagnostic_count"] > 0 for row in rows)
            models[name] = scores
        primary = {
            mode: {
                metric: float(np.mean([row["primary"][mode][metric] for row in rows]))
                for metric in subjects[0]["primary"][mode]
            }
            for mode in subjects[0]["primary"]
        }
        groups[group_name] = {
            "subject_count": len(rows),
            "test_count": int(mask[selector].sum()),
            "diagnostic_count": int(diagnostic[selector].sum()),
            "diagnostic_subject_count": int(diagnostic[selector].any(1).sum()),
            "models": models,
            "primary": primary,
        }
    return {
        "metadata": {
            "signature_order": list(SIGNATURE_ORDER),
            "primary_selection": "population",
            "secondary_selection": "individual",
            "gain_sign": "solo NLL minus distilled NLL; positive favors distilled prediction",
            "distortion_sign": "distilled RMS minus solo RMS; positive means more behavioral error",
            "assay_environment": "Three first-stage choices, 70/30 transitions, fair second actions and rewards",
        },
        "true_majority_signature": majority_signature.tolist(),
        "per_subject": subjects,
        "groups": groups,
        "baseline_metadata": baseline_metadata,
    }
