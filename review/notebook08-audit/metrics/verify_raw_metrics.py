"""Read-only Notebook 08, 2% baseline and smoke saved-array audit.

Run with Python -B and the project dependencies installed. This wrapper
uses the pinned checkout's Notebook 07 audit for independently reconstructed
metric formulas and its numerical runtime to refit the classical controls.
It also refits the classical models with separate replay and EM formulas.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import traceback

import numpy as np

MODELS = {
    "teacher",
    *(f"{arm}_{selection}" for arm in ("solo", "distilled", "oracle")
      for selection in ("population", "individual")),
    "individual_grid_ml", "pooled_grid_ml", "hierarchical_hybrid",
    "hierarchical_hybrid_cue_mixture", "coinflip", "training_frequency",
}
TOL = 1e-12


def replay_hybrid(events, weight, alpha, beta):
    """Replay before updates using a direct first-stage value contrast."""
    choices, states, second, rewards = (events[key] for key in prior.KEYS)
    n_people, n_trials = choices.shape
    q1 = np.full((n_people, 2), 0.5)
    q2 = np.full((n_people, 2, 2), 0.5)
    probabilities = np.empty(choices.shape)
    row = np.arange(n_people)
    for t in range(n_trials):
        model_free = q1[:, 1] - q1[:, 0]
        best_second = q2.max(axis=2)
        model_based = 0.4 * (best_second[:, 1] - best_second[:, 0])
        contrast = (1 - weight) * model_free + weight * model_based
        probabilities[:, t] = 1 / (1 + np.exp(-beta * contrast))
        first_idx = (row, choices[:, t])
        second_idx = (row, states[:, t], second[:, t])
        old_second = q2[second_idx].copy()
        second_error = rewards[:, t] - old_second
        q1[first_idx] += alpha * (old_second - q1[first_idx] + 0.5 * second_error)
        q2[second_idx] += alpha * second_error
    return probabilities


def replay_cue(events, rho):
    result = np.full(events["choice"].shape, 0.5)
    result[:, 1:] = np.where(events["state"][:, :-1] == events["reward"][:, :-1], rho, 1 - rho)
    return result


def training_likelihood(candidates, choices, mask):
    p = np.clip(candidates, TOL, 1 - TOL)
    y = choices[:, None, :]
    return np.where(mask[:, None, :], np.where(y == 1, np.log(p), np.log1p(-p)), 0).sum(axis=2)


def fit_prior(log_likelihood, pseudocount, settings):
    # A person's additive likelihood shift leaves posterior and EM updates
    # unchanged, but contributes to the relative convergence criterion.
    shift = log_likelihood.max(axis=1, keepdims=True)
    relative_likelihood = np.exp(log_likelihood - shift)
    n_people, n_candidates = log_likelihood.shape
    weights = np.full(n_candidates, 1 / n_candidates)
    last_objective = None
    for step in range(settings["em_maximum_iterations"] + 1):
        unnormalized = relative_likelihood * weights
        totals = unnormalized.sum(axis=1, keepdims=True)
        posterior = unnormalized / totals
        objective = (np.log(totals) + shift).sum() + pseudocount * np.log(weights).sum()
        if last_objective is not None and abs(objective - last_objective) <= (
            settings["em_relative_objective_tolerance"] * (1 + abs(last_objective))
        ):
            return weights, posterior, step
        weights = (posterior.sum(axis=0) + pseudocount) / (n_people + n_candidates * pseudocount)
        last_objective = objective
    raise ValueError("Independent EM failed to converge")


def family_evidence(log_likelihood):
    peak = log_likelihood.max(axis=1, keepdims=True)
    relative = np.exp(log_likelihood - peak)
    return peak[:, 0] + np.log(relative.mean(axis=1)), relative / relative.sum(axis=1, keepdims=True)


def independent_refit(audit, config, data, metadata, label):
    settings = config.design["baseline_recipe"]
    grid = list(product(settings["hybrid_grid_weight"], settings["hybrid_grid_alpha"], settings["hybrid_grid_beta"]))
    cue_grid = settings["cue_grid"]
    histories = {key: data[key] for key in prior.KEYS}
    probe_bank = {key: np.column_stack((prior.HISTORIES[:, :, i], np.zeros(256, dtype=np.int64)))
                  for i, key in enumerate(prior.KEYS)}
    hybrid = np.stack([replay_hybrid(histories, *params) for params in grid], axis=1)
    cue = np.stack([replay_cue(histories, rho) for rho in cue_grid], axis=1)
    hybrid_probe = np.stack([replay_hybrid(probe_bank, *params) for params in grid])
    cue_probe = np.stack([replay_cue(probe_bank, rho) for rho in cue_grid])
    mask = data["mask_train"]
    y = histories["choice"]
    hy_ll = training_likelihood(hybrid, y, mask)
    cue_ll = training_likelihood(cue, y, mask)
    indices = np.asarray(metadata["individual_grid_indices"])
    pooled = metadata["pooled_grid_index"]
    rows = np.arange(len(y))
    # Independently replayed probabilities can reorder near-exact ties. Verify
    # that the exact indices reproduced by the pinned refit are maximizers to
    # numerical tolerance, then use those indices to check predicted outputs.
    audit.array_match(hy_ll[rows, indices], hy_ll.max(axis=1), label + ".individual_ML_likelihood",
                      "independent_ml_log_likelihood", TOL)
    total_ll = hy_ll.sum(axis=0)
    audit.array_match(total_ll[pooled], total_ll.max(), label + ".pooled_ML_likelihood",
                      "independent_ml_log_likelihood", TOL)
    audit.independent_ml_argmax_differences += int((hy_ll.argmax(axis=1) != indices).sum())
    audit.independent_pooled_argmax_differences += int(total_ll.argmax() != pooled)
    prior_w, posterior, hybrid_steps = fit_prior(hy_ll, settings["hybrid_population_pseudocount_per_point"], settings)
    hy_ev, within_hybrid = family_evidence(hy_ll)
    cue_ev, within_cue = family_evidence(cue_ll)
    family_prior, families, family_steps = fit_prior(
        np.column_stack((hy_ev, cue_ev)), settings["between_family_pseudocount_per_family"], settings
    )
    independent_meta = {
        "hybrid_prior_weights": prior_w,
        "hybrid_posterior_weights": posterior,
        "family_prior_hybrid_cue": family_prior,
        "family_posterior_hybrid_cue": families,
        "family_mixture_hybrid_posterior": within_hybrid,
        "family_mixture_cue_posterior": within_cue,
        "hybrid_em_iterations": hybrid_steps,
        "family_em_iterations": family_steps,
    }
    for key, value in independent_meta.items():
        audit.array_match(value, metadata[key], label + "." + key, "independent_baseline_metadata", TOL)
    # Labels and true/reference probabilities enter only here, after fitting.
    truth_reference = replay_hybrid(histories, 0.75, data["alpha"], data["beta"])
    audit.array_match(truth_reference, data["reference_prob"], label + ".reference",
                      "independent_reference_probability", TOL)
    expected_test = data["mask_test"] & (np.abs(data["true_prob"] - truth_reference) > 0.2)
    observed_test = data["mask_test"] & (np.abs(data["true_prob"] - data["reference_prob"]) > 0.2)
    audit.array_match(expected_test, observed_test, label + ".diagnostic_support",
                      "independent_diagnostic_mask", 0)
    frequency = ((y * mask).sum(axis=1) + 1) / (mask.sum(axis=1) + 2)
    mixture_weights = np.column_stack((families[:, :1] * within_hybrid, families[:, 1:] * within_cue))
    probability = {
        "individual_grid_ml": hybrid[rows, indices],
        "pooled_grid_ml": hybrid[:, pooled],
        "hierarchical_hybrid": (posterior[:, :, None] * hybrid).sum(axis=1),
        "hierarchical_hybrid_cue_mixture": (
            (mixture_weights[:, :, None] * np.concatenate((hybrid, cue), axis=1)).sum(axis=1)
        ),
        "coinflip": np.full(y.shape, 0.5),
        "training_frequency": np.broadcast_to(frequency[:, None], y.shape),
    }
    # Marginalize conditional probabilities at each history using fixed weights
    # from the observed training fit, before integrating the assay trajectories.
    probe_shape = (len(y), 256, 3)
    probes = {
        "individual_grid_ml": hybrid_probe[indices],
        "pooled_grid_ml": np.broadcast_to(hybrid_probe[pooled], probe_shape),
        "hierarchical_hybrid": np.tensordot(posterior, hybrid_probe, axes=1),
        "hierarchical_hybrid_cue_mixture": np.tensordot(
            mixture_weights, np.concatenate((hybrid_probe, cue_probe)), axes=1
        ),
        "coinflip": np.full(probe_shape, 0.5),
        "training_frequency": np.broadcast_to(frequency[:, None, None], probe_shape),
    }
    return probability, probes


def family_summary(metadata, rare, kind):
    p_cue = np.asarray(metadata["family_posterior_hybrid_cue"])[:, 1]
    result = {"estimated_family_prior": metadata["family_prior_hybrid_cue"]}
    for group, mask in (("rare", rare), ("typical", ~rare)):
        p = p_cue[mask]
        clipped = np.clip(p, TOL, 1 - TOL)
        entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log1p(-clipped))
        expected = p if group == "rare" and kind == "reward_cue" else 1 - p
        result[group] = {
            "subject_count": int(mask.sum()),
            "cue_posterior_mean": float(p.mean()), "cue_posterior_min": float(p.min()),
            "cue_posterior_max": float(p.max()), "cue_count": int((p > 0.5).sum()),
            "hybrid_count": int((p < 0.5).sum()), "tie_count": int((p == 0.5).sum()),
            "uncertain_count": int(((p >= 0.25) & (p <= 0.75)).sum()),
            "entropy_mean": float(entropy.mean()), "true_family_posterior_mean": float(expected.mean()),
            "true_family_correct_count": int((expected > 0.5).sum()),
        }
    return result


def primary_quadrants(groups, data, prediction, probes):
    true_signature = prior.signatures(prior.true_probe(data))
    test = data["mask_test"]
    rare = data["rare"]
    output = {}
    for mode in ("population", "individual"):
        solo, distilled = f"solo_{mode}", f"distilled_{mode}"
        solo_nll = prior.nll(data["choice"], prediction[solo])
        distilled_nll = prior.nll(data["choice"], prediction[distilled])
        gains = np.array([s[t].mean() - d[t].mean() for s, d, t in zip(solo_nll, distilled_nll, test)])
        rms = {key: np.sqrt(np.mean((prior.signatures(probes[key]) - true_signature) ** 2, axis=1))
               for key in (solo, distilled)}
        changes = rms[distilled] - rms[solo]
        output[mode] = {}
        for group, selected in (("rare", rare), ("typical", ~rare)):
            gain, change = gains[selected], changes[selected]
            n = groups[group]["subject_count"]
            conditions = {
                "gain_positive_error_positive": (gain > 0) & (change > 0),
                "gain_positive_error_negative": (gain > 0) & (change < 0),
                "gain_negative_error_positive": (gain < 0) & (change > 0),
                "gain_negative_error_negative": (gain < 0) & (change < 0),
                "ties": (gain == 0) | (change == 0),
            }
            counts = {name: int(value.sum()) for name, value in conditions.items()}
            if sum(counts.values()) != n:
                raise ValueError("Primary quadrants do not cover this group")
            output[mode][group] = {"counts": counts, "fractions": {name: v / n for name, v in counts.items()}}
    return output


def audit_population(audit, config, folder, item):
    data = prior.arrays(folder / "data.npz")
    prediction = prior.arrays(folder / "predictions.npz")
    probes = prior.arrays(folder / "probe_predictions.npz")
    report = prior.load_json(folder / "report.json")
    label = str(folder)
    audit.check(prediction.keys() == probes.keys() == MODELS, label + ".exact_model_coverage")
    audit.check(report["kind"] == item["kind"] and report["seed"] == item["seed"], label + ".identity")
    audit.check(len(report["per_subject"]) == config.n_subjects, label + ".report_subject_count")
    audit.check(all(row["models"].keys() == MODELS for row in report["per_subject"]), label + ".report_model_coverage")
    # Capture the reviewed refit performed by the prior auditor. This avoids a
    # second invocation and keeps the family summaries tied to the refit.
    refit = prior.baseline_predictions
    captured = {}

    def capture(*args):
        result = refit(*args)
        captured["metadata"] = result[2]
        return result

    prior.baseline_predictions = capture
    try:
        row = prior.audit_population(audit, folder, config)
    finally:
        prior.baseline_predictions = refit
    metadata = captured["metadata"]
    p, probe = independent_refit(audit, config, data, metadata, label)
    for model in p:
        audit.array_match(p[model], prediction[model], label + "." + model,
                          "independent_baseline_prediction", TOL)
        audit.array_match(probe[model], probes[model], label + "." + model,
                          "independent_baseline_probe", TOL)
    row["family_posterior_summary"] = family_summary(metadata, data["rare"], item["kind"])
    row["quadrants"] = primary_quadrants(row["groups"], data, prediction, probes)
    for mode, groups in row["quadrants"].items():
        for group, quad in groups.items():
            audit.match(quad["fractions"]["gain_positive_error_positive"],
                        row["groups"][group]["primary"][mode]["quadrant_fraction"],
                        label + f".{mode}.{group}.quadrant_fraction", "quadrants", TOL)
    return row


def audit_cell(folder, config, label, embedded=None):
    audit = prior.Audit()
    audit.independent_ml_argmax_differences = 0
    audit.independent_pooled_argmax_differences = 0
    summary = prior.load_json(folder / "summary.json")
    if embedded is not None:
        audit.check(summary == embedded, str(folder) + ".outer_summary_exact")
    audit.check(summary.get("status") == "complete", str(folder) + ".complete_status")
    audit.check(summary.get("config") == config.to_dict(), str(folder) + ".config")
    for key in ("planned_populations", "completed_populations"):
        audit.check(summary.get(key) == config.plan, str(folder) + "." + key)
    audit.check(summary.get("pending_populations") == [], str(folder) + ".pending_populations")
    actual = {str(p.relative_to(folder)) for p in folder.glob("*/seed-*") if p.is_dir()}
    planned = {f"{item['kind']}/seed-{item['seed']}" for item in config.plan}
    audit.check(actual == planned, str(folder) + ".population_directories")
    populations = []
    for item in config.plan:
        path = folder / item["kind"] / f"seed-{item['seed']}"
        before_issues = len(audit.issues)
        try:
            populations.append(audit_population(audit, config, path, item))
        except (ValueError, OSError, TypeError, RuntimeError, KeyError, IndexError) as exc:
            audit.issues.append(f"{path}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        print(f"{label} {item['seed']} {len(audit.issues) - before_issues} issue(s)", flush=True)
    groups_by_kind = {}
    for kind in config.design["population"]["rare_kinds"]:
        rows = [row for row in populations if row["kind"] == kind]
        seeds = [item["seed"] for item in config.plan if item["kind"] == kind]
        saved = summary["kinds"][kind]
        audit.check(saved.get("planned_seeds") == saved.get("completed_seeds") == seeds,
                    str(folder) + "." + kind + ".seeds")
        if [row["seed"] for row in rows] == seeds:
            groups = prior.aggregate([row["groups"] for row in rows])
            groups_by_kind[kind] = groups
            audit.match(saved["groups"], groups, kind + ".aggregate", "summary", TOL)
            audit.match(saved["per_seed"], [{"seed": row["seed"], "groups": row["groups"]} for row in rows],
                        kind + ".per_seed", "summary", TOL)
    return {
        "label": label, "folder": str(folder), "rare_fraction": config.rare_fraction, "mode": config.mode,
        "status": "passed" if not audit.issues else "failed", "check_count": audit.check_count,
        "maximum_errors": audit.maximum_errors, "issues": audit.issues, "groups_by_kind": groups_by_kind,
        "per_population": populations,
        "independent_ml_argmax_differences": audit.independent_ml_argmax_differences,
        "independent_pooled_argmax_differences": audit.independent_pooled_argmax_differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new", dest="production", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--only", choices=("production", "baseline", "smoke", "all"), default="all")
    args = parser.parse_args()
    modes = ("baseline", "production", "smoke") if args.only == "all" else (args.only,)
    for mode in modes:
        if getattr(args, mode) is None:
            parser.error(f"Specify {'--new' if mode == 'production' else '--' + mode} for the selected mode")
    source = args.source.resolve()
    roots = (source, args.production, args.baseline, args.smoke)
    if any(args.output.resolve().is_relative_to(root.resolve()) for root in roots if root is not None):
        parser.error("Output must be outside the source checkout and input trees")

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(source / "src"), str(source / "scripts")]
    global prior
    from grid_study.config import GridTrancheConfig
    from main_study.config import MainStudyConfig
    import verify_main_study_metrics as prior

    cells = []
    for label in modes:
        root = getattr(args, label)
        if label == "baseline":
            baseline_config = MainStudyConfig.from_file(source / "configs/main_study_first_tranche.json")
            cells.append(audit_cell(root, baseline_config, "prior 2%"))
            continue
        filename = "rarity_tranche.json" if label == "production" else "rarity_smoke.json"
        config = GridTrancheConfig.from_file(source / "configs" / filename)
        outer = prior.load_json(root / "summary.json")
        if (outer["config"] != config.to_dict() or outer["planned_cells"] != config.plan
                or outer["completed_cells"] != config.plan or outer["pending_cells"] or outer["status"] != "complete"
                or set(outer["cells"]) != set(config.plan)):
            raise ValueError(f"Outer summary diverges from pinned plan at {root}")
        for cell in config.cell_configs:
            cells.append(audit_cell(root / "cells" / cell.cell_id, cell, f"{label} {cell.rare_fraction:.0%}",
                                    outer["cells"][cell.cell_id]))
    result = {
        "scope": "Independent per-person saved-array metric recomputation, train-only control refits, and cell/outer-summary comparisons",
        "numpy_version": np.__version__, "absolute_metric_tolerance": TOL,
        "probe_probability_prefix_and_mass_tolerance": 5e-6,
        "ml_argmax_policy": "Verify pinned earliest index exactly against saved; independent replay verifies maximal likelihood within 1e-12 and counts argmax disagreements due to numerically tied values.",
        "wrapper_sha256": prior.sha(Path(__file__)),
        "prior_metric_audit_sha256": prior.sha(Path(prior.__file__)),
        "signature_order": ["unrewarded rare", "unrewarded common", "rewarded rare", "rewarded common"],
        "family_uncertainty": "Cue posterior between 0.25 and 0.75 inclusive; classified cue strictly above 0.5.",
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for cell in cells:
        print(cell["label"], cell["status"], cell["check_count"], cell["maximum_errors"], cell["issues"][:8])
    raise SystemExit(1 if any(cell["issues"] for cell in cells) else 0)


if __name__ == "__main__":
    main()
