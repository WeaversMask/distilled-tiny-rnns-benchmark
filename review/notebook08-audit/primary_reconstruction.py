"""Independent primary comparison from saved trials and probes. No project imports."""

from itertools import product
from pathlib import Path
import argparse
import json

import numpy as np


KINDS = ("parametric_model_free", "reward_cue")
SELECTIONS = ("population", "individual")
HISTORIES = np.array(list(product(range(2), repeat=8))).reshape(256, 2, 4)
CHOICE, STATE, ACTION2, REWARD = HISTORIES.transpose(2, 0, 1)
COMMON = STATE == CHOICE


def frozen_population_paths(downloads):
    for index, rarity in enumerate((2, 5, 10, 25)):
        if rarity == 2:
            cell = downloads / "main-study-first-tranche-v1"
        else:
            cell = downloads / "main-study-rarity-v1" / "cells" / f"rarity-{rarity:02d}_history-100"
        for strategy, kind in enumerate(KINDS):
            for replicate in range(3):
                seed = 202611401 + 1000 * index + 10000 * strategy + 10 * replicate
                yield rarity, kind, seed, cell / kind / f"seed-{seed}"


def true_policy(data):
    """Calculate the probability of choice 1 at each of the three decisions."""
    n = len(data["rare"])
    probability = np.empty((n, 256, 3))
    for i in range(n):
        if data["is_cue"][i]:
            preferred_one = np.where(REWARD == 1, STATE, 1 - STATE)
            reliability = data["cue_reliability"][i]
            probability[i, :, 0] = 0.5
            probability[i, :, 1:] = np.where(preferred_one == 1, reliability, 1 - reliability)
            continue
        w, alpha, beta = (data[key][i] for key in ("model_based_weight", "alpha", "beta"))
        q1 = np.full((256, 2), 0.5)
        q2 = np.full((256, 2, 2), 0.5)
        history_ids = np.arange(256)
        for t in range(3):
            v = q2.max(axis=2)
            mb_values = v @ np.array([[0.7, 0.3], [0.3, 0.7]])
            values = (1 - w) * q1 + w * mb_values
            probability[i, :, t] = 1 / (1 + np.exp(beta * (values[:, 0] - values[:, 1])))
            if t == 2:
                break
            first = history_ids, CHOICE[:, t]
            second = history_ids, STATE[:, t], ACTION2[:, t]
            td_second = REWARD[:, t] - q2[second]
            td_first = q2[second] - q1[first]
            q1[first] += alpha * td_first + 0.5 * alpha * td_second
            q2[second] += alpha * td_second
    return probability


def assay(probability):
    p = probability.astype(np.float64)
    first_choice = np.where(CHOICE[None, :, 0], p[:, :, 0], 1 - p[:, :, 0])
    second_choice = np.where(CHOICE[None, :, 1], p[:, :, 1], 1 - p[:, :, 1])
    transitions = np.where(COMMON, 0.7, 0.3)
    path_probability = first_choice * transitions[:, 0] / 4 * second_choice * transitions[:, 1] / 4
    np.testing.assert_allclose(path_probability.sum(axis=1), 1, atol=5e-6, rtol=0)
    stay = np.where(CHOICE[None, :, 1], p[:, :, 2], 1 - p[:, :, 2])
    result = []
    for rewarded, common in product(range(2), repeat=2):
        cell = (REWARD[:, 1] == rewarded) & (COMMON[:, 1] == common)
        mass = path_probability[:, cell]
        result.append((mass * stay[:, cell]).sum(axis=1) / mass.sum(axis=1))
    return np.stack(result, axis=1)


def subject_nll(choice, mask, p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    loss = -np.log(np.where(choice == 1, p, 1 - p))
    return np.where(mask, loss, 0).sum(axis=1) / mask.sum(axis=1)


def classify(gain, change):
    zero_gain, zero_change = gain == 0, change == 0
    gain_positive, change_positive = gain > 0, change > 0
    gain_negative, change_negative = gain < 0, change < 0
    return {
        "+/+": int(np.sum(gain_positive & change_positive)),
        "+/-": int(np.sum(gain_positive & change_negative)),
        "-/+": int(np.sum(gain_negative & change_positive)),
        "-/-": int(np.sum(gain_negative & change_negative)),
        "any_tie": int(np.sum(zero_gain | zero_change)),
        "gain_only_tie": int(np.sum(zero_gain & ~zero_change)),
        "assay_only_tie": int(np.sum(zero_change & ~zero_gain)),
        "both_tied": int(np.sum(zero_gain & zero_change)),
    }


def primary(downloads):
    rows = []
    maxima = {}
    issues = []
    for rarity, kind, seed, folder in frozen_population_paths(downloads):
        with np.load(folder / "data.npz", allow_pickle=False) as data:
            true_signature = assay(true_policy(data))
            choice = data["choice"]
            test = data["mask_test"]
            rare = data["rare"]
        assert int(rare.sum()) == 10 * rarity and len(rare) == 1000
        assert np.all(test.sum(axis=1) == 20)
        report = json.loads((folder / "report.json").read_text())
        maxima["true_signature"] = max(maxima.get("true_signature", 0), float(np.max(
            np.abs(true_signature - [s["true_signature"] for s in report["per_subject"]])
        )))
        row = {"rarity": rarity, "kind": kind, "seed": seed, "group": {}}
        with (np.load(folder / "predictions.npz", allow_pickle=False) as predictions,
              np.load(folder / "probe_predictions.npz", allow_pickle=False) as probes):
            scores = {}
            for model in ("teacher", *(f"{m}_{s}" for m in ("solo", "distilled", "oracle") for s in SELECTIONS)):
                nll = subject_nll(choice, test, predictions[model])
                rms = np.sqrt(((assay(probes[model]) - true_signature) ** 2).mean(axis=1))
                scores[model] = (nll, rms)
        for group, mask in (("rare", rare), ("typical", ~rare)):
            grp = row["group"][group] = {"people": int(mask.sum()), "selection": {}}
            grp["models"] = {m: {"nll": float(x[mask].mean()), "rms": float(y[mask].mean())}
                             for m, (x, y) in scores.items()}
            for selection in SELECTIONS:
                solo_nll, solo_rms = scores[f"solo_{selection}"]
                distilled_nll, distilled_rms = scores[f"distilled_{selection}"]
                gain = (solo_nll - distilled_nll)[mask]
                change = (distilled_rms - solo_rms)[mask]
                people_quadrants = classify(gain, change)
                assert sum(people_quadrants[k] for k in ("+/+", "+/-", "-/+", "-/-", "any_tie")) == grp["people"]
                outcome = grp["selection"][selection] = {
                    "gain": float(gain.mean()), "change": float(change.mean()),
                    "person_quadrant_counts": people_quadrants,
                    "person_quadrant_fractions": {k: n / grp["people"] for k, n in people_quadrants.items()},
                    "near_zero_nonzero_gain": int(np.sum((np.abs(gain) <= 1e-12) & (gain != 0))),
                    "near_zero_nonzero_change": int(np.sum((np.abs(change) <= 1e-12) & (change != 0))),
                }
                saved = report["groups"][group]["primary"][selection]
                for name, value in (("prediction_gain", outcome["gain"]), ("behavioral_distortion", outcome["change"]),
                                    ("quadrant_fraction", outcome["person_quadrant_fractions"]["+/+"])):
                    discrepancy = abs(saved[name] - value)
                    maxima[name] = max(maxima.get(name, 0), discrepancy)
                    if discrepancy > 1e-12:
                        issues.append(f"{seed} {group} {selection} {name}: {discrepancy:.12g}")
        rows.append(row)
    if maxima["true_signature"] > 1e-12:
        issues.append(f"true signature maximum difference {maxima['true_signature']}")
    return {"populations": rows, "maximum_saved_discrepancies": maxima, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True,
                        help="Parent of the extracted baseline and rarity run folders")
    parser.add_argument("--output", type=Path, help="Write the full JSON here instead of stdout")
    args = parser.parse_args()
    reference = Path(__file__).resolve().with_name("primary-reconstruction.json")
    if args.output is not None and args.output.resolve() == reference:
        parser.error("choose an output path separate from the checked-in reconstruction")

    results = primary(args.downloads)
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
        primary_pairs = [row["group"]["rare"]["selection"]["population"] for row in results["populations"]]
        print(json.dumps({
            "populations": len(primary_pairs),
            "rare_groups_with_prediction_gain": sum(pair["gain"] > 0 for pair in primary_pairs),
            "rare_groups_with_lower_assay_error": sum(pair["change"] < 0 for pair in primary_pairs),
            "maximum_saved_discrepancies": results["maximum_saved_discrepancies"],
            "issues": results["issues"],
        }, indent=2))
    return int(bool(results["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
