"""Compare the two independent raw-array primary analyses for 24 populations."""
import argparse
import json
from pathlib import Path
from statistics import mean

AUDIT_DIR = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-12
QUADRANTS = {
    "gain_positive_error_positive": "+/+",
    "gain_positive_error_negative": "+/-",
    "gain_negative_error_positive": "-/+",
    "gain_negative_error_negative": "-/-",
    "ties": "any_tie",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=AUDIT_DIR / "metrics/raw-metrics.json")
    parser.add_argument("--primary", type=Path, default=AUDIT_DIR / "primary-reconstruction.json")
    args = parser.parse_args()
    full = json.loads(args.metrics.read_text())
    primary = json.loads(args.primary.read_text())
    by_seed = {row["seed"]: row for row in primary["populations"]}
    maximums = {"per_seed_gain": 0.0, "per_seed_change": 0.0, "cell_mean_gain": 0.0, "cell_mean_change": 0.0}
    issue_count = 0
    compared = set()
    n_cells = 0
    for cell in full["cells"]:
        if cell["mode"] == "smoke":
            continue
        for kind, summaries in cell["groups_by_kind"].items():
            rows = [row for row in cell["per_population"] if row["kind"] == kind]
            n_cells += 1
            for group in ("rare", "typical"):
                for selection in ("population", "individual"):
                    g_list, c_list = [], []
                    for row in rows:
                        seed = row["seed"]
                        compared.add(seed)
                        a = by_seed[seed]["group"][group]["selection"][selection]
                        b = row["groups"][group]["primary"][selection]
                        g, c = a["gain"], a["change"]
                        bg, bc = b["prediction_gain"], b["behavioral_distortion"]
                        g_list.append(g)
                        c_list.append(c)
                        maximums["per_seed_gain"] = max(maximums["per_seed_gain"], abs(bg - g))
                        maximums["per_seed_change"] = max(maximums["per_seed_change"], abs(bc - c))
                        counts = row["quadrants"][selection][group]["counts"]
                        for k, label in QUADRANTS.items():
                            issue_count += counts[k] != a["person_quadrant_counts"][label]
                        issue_count += ((bg > 0) - (bg < 0)) != ((g > 0) - (g < 0))
                        issue_count += ((bc > 0) - (bc < 0)) != ((c > 0) - (c < 0))
                    aggregate = summaries[group]["primary"][selection]
                    maximums["cell_mean_gain"] = max(maximums["cell_mean_gain"],
                                                    abs(mean(g_list) - aggregate["prediction_gain"]["mean"]))
                    maximums["cell_mean_change"] = max(maximums["cell_mean_change"],
                                                      abs(mean(c_list) - aggregate["behavioral_distortion"]["mean"]))
    issue_count += compared != set(by_seed) or len(compared) != 24
    issue_count += n_cells != 8
    issue_count += sum(value > TOLERANCE for value in maximums.values())
    print(json.dumps({
        "status": "passed" if issue_count == 0 else "failed",
        "populations": len(compared),
        "rarity_strategy_cells": n_cells,
        "groups": ["rare", "typical"],
        "selection_rules": ["population", "individual"],
        "signs_and_person_quadrant_counts_require_exact_agreement": True,
        "absolute_tolerance": TOLERANCE,
        "maximum_absolute_differences": maximums,
        "issues": issue_count,
    }, indent=2))
    raise SystemExit(1 if issue_count else 0)


if __name__ == "__main__":
    main()
