"""Render Notebook 08 figures and supporting tables from reconstructed results."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

AUDIT = Path(__file__).resolve().parent
KINDS = {"parametric_model_free": "Model-free", "reward_cue": "Reward cue"}
SELECTIONS = ("population", "individual")
GROUPS = ("rare", "typical")
QUADRANTS = ("+/+", "+/-", "-/+", "-/-", "any_tie")
DIRECTIONS = {"+-": "Both improve", "++": "Tradeoff", "--": "Only assay improves", "-+": "Both worsen"}
MODELS = {
    "teacher": "Teacher",
    "solo_population": "Solo (population)",
    "distilled_population": "Distilled (population)",
    "oracle_population": "Oracle (population)",
    "solo_individual": "Solo (individual)",
    "distilled_individual": "Distilled (individual)",
    "oracle_individual": "Oracle (individual)",
    "individual_grid_ml": "Individual hybrid-grid ML",
    "pooled_grid_ml": "Pooled hybrid-grid ML",
    "hierarchical_hybrid": "Hierarchical hybrid",
    "hierarchical_hybrid_cue_mixture": "Hybrid/cue mixture",
    "coinflip": "Coinflip",
    "training_frequency": "Training frequency",
}


def md_table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(v) for v in values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def fmt(value, signed=False):
    if value is None:
        return "null"
    return f"{value:+.6f}" if signed and value != 0 else f"{value:.6f}"


def span(values):
    return f"{fmt(mean(values), True)} [{fmt(min(values), True)}, {fmt(max(values), True)}]"


def cell_title(rarity):
    return "2% (prior)" if rarity == 2 else f"{rarity}%"


def cell_rows(populations):
    for rarity in (2, 5, 10, 25):
        for kind in KINDS:
            yield rarity, kind, [row for row in populations if row["rarity"] == rarity and row["kind"] == kind]


def pairs(rows, group, selection):
    return [row["group"][group]["selection"][selection] for row in rows]


def direction(outcome):
    return "".join("+" if outcome[k] > 0 else "-" if outcome[k] < 0 else "0" for k in ("gain", "change"))


def primary_tables(populations):
    rows, seed_rows = [], []
    for rarity, kind, seeds in cell_rows(populations):
        outcomes = pairs(seeds, "rare", "population")
        gains, changes = [[x[key] for x in outcomes] for key in ("gain", "change")]
        counts = Counter(map(direction, outcomes))
        rows.append([cell_title(rarity), KINDS[kind], span(gains), span(changes),
                     *(counts[k] for k in DIRECTIONS), sum(n for k, n in counts.items() if "0" in k)])
        for seed in seeds:
            p, i = (seed["group"]["rare"]["selection"][s] for s in SELECTIONS)
            seed_rows.append([cell_title(rarity), KINDS[kind], seed["seed"], fmt(p["gain"], True),
                              fmt(p["change"], True), DIRECTIONS.get(direction(p), "Tie"),
                              fmt(i["gain"], True), fmt(i["change"], True)])
    header = ("Rarity", "Rare strategy", "NLL gain, mean [range]", "RMS change, mean [range]", *DIRECTIONS.values(), "Ties")
    primary = md_table(header, rows)
    seeds = md_table(("Rarity", "Rare strategy", "Seed", "Population gain", "Population RMS change",
                      "Population direction", "Individual gain", "Individual RMS change"), seed_rows)
    return primary, seeds


def write_quadrants(populations, output_dir):
    parts = [
        "# Person-level quadrants and secondary checkpoint selection",
        "Signs refer to NLL gain and RMS change, in that order. +/+ means better prediction and worse assay behavior. "
        "+/- means improvement on both. Ties include any exact zero. "
        "Each cell reports count and fraction of people, with fractions written as percentages. "
        "Both checkpoint rules use the same fitted trajectories.",
    ]
    for group in GROUPS:
        for selection in SELECTIONS:
            parts.append(f"## {group.title()} people, {selection} checkpoint")
            rows, seed_rows = [], []
            for rarity, kind, seeds in cell_rows(populations):
                outcomes = pairs(seeds, group, selection)
                count = Counter()
                people = sum(r["group"][group]["people"] for r in seeds)
                for seed, outcome in zip(seeds, outcomes):
                    counts = outcome["person_quadrant_counts"]
                    count.update(counts)
                    n = seed["group"][group]["people"]
                    seed_rows.append([cell_title(rarity), KINDS[kind], seed["seed"], n,
                                      fmt(outcome["gain"], True), fmt(outcome["change"], True),
                                      *(f"{counts[key]} ({counts[key] / n:.2%})" for key in QUADRANTS)])
                rows.append([cell_title(rarity), KINDS[kind], people,
                             span([r["gain"] for r in outcomes]), span([r["change"] for r in outcomes]),
                             *(f"{count[key]} ({count[key] / people:.2%})" for key in QUADRANTS)])
            header = ["Rarity", "Rare strategy", "People", "Gain, mean [range]", "RMS change, mean [range]", *QUADRANTS]
            parts.append(md_table(header, rows))
            parts.append("### By seed")
            header = ["Rarity", "Rare strategy", "Seed", "People", "Gain", "RMS change", *QUADRANTS]
            parts.append(md_table(header, seed_rows))
    (output_dir / "person-quadrants.md").write_text("\n\n".join(parts) + "\n")


def draw(populations, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.patches import Rectangle

    colors = {2: "#666F7C", 5: "#176CA4", 10: "#B75B12", 25: "#7056A6"}
    markers = {"parametric_model_free": "o", "reward_cue": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, group in zip(axes, GROUPS):
        limits = (-0.009, 0.105, -0.113, 0.017) if group == "rare" else (-0.002, 0.026, -0.008, 0.0095)
        ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="Held-out NLL gain  →  better prediction")
        ax.set_title(f"{group.title()} people", loc="left", pad=10, weight="bold", fontsize=12)
        ax.add_patch(Rectangle((0, 0), limits[1], limits[3], color="#F4DAD1", alpha=0.47, lw=0))
        ax.text(limits[1] * 0.98, limits[3] * 0.58, "Proposed tradeoff", ha="right", va="center",
                color="#7B3426", fontsize=9)
        ax.axvline(0, color="#777777", lw=0.8, zorder=2)
        ax.axhline(0, color="#777777", lw=0.8, zorder=2)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(0.02 if group == "rare" else 0.005))
        ax.yaxis.set_major_locator(MultipleLocator(0.02 if group == "rare" else 0.004))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{0 if abs(value) < 0.0005 else value:.3f}"))
        for rarity, kind, seeds in cell_rows(populations):
            for p in pairs(seeds, group, "population"):
                ax.scatter(p["gain"], p["change"], marker=markers[kind], edgecolor=colors[rarity],
                           facecolor=colors[rarity] if rarity != 2 else "none", s=53, linewidth=1.25,
                           alpha=0.9, zorder=4)
    axes[0].set_ylabel("Change in assay RMS  →  higher error")
    handles = [
        Line2D([0], [0], linestyle="none", marker="s", color=c, markerfacecolor=c if r != 2 else "none",
               label=cell_title(r)) for r, c in colors.items()
    ] + [
        Line2D([0], [0], linestyle="none", marker=m, color="#333333", label=KINDS[k]) for k, m in markers.items()
    ]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.suptitle("Distilled versus solo across 24 simulated populations", x=0.08, ha="left", fontsize=17, weight="bold")
    fig.subplots_adjust(left=0.095, right=0.98, top=0.82, bottom=0.19, wspace=0.22)
    fig.savefig(output_dir / "primary-comparison.png", dpi=190, facecolor="white")
    plt.close(fig)


def mean_available(values):
    support = [x for x in values if x is not None]
    return mean(support) if support else None


def write_metrics(raw, output_dir):
    cells = [x for x in raw["cells"] if x["mode"] == "production"]
    metrics = ("nll", "kl", "diagnostic_nll", "diagnostic_kl", "signature_rms_error", "probe_kl", "pull")
    parts = [
        "# Absolute model scores, diagnostic support and mixture recovery",
        "Scores weight people equally within group and then populations equally. Diagnostic scores use supported people "
        "and supported populations. Pull uses the same support convention. Null denotes absent support. "
        "Assay stay probabilities are ordered unrewarded rare, unrewarded common, rewarded rare, rewarded common. "
        "KL on the third probe choice weights all 256 histories equally. Population-selected neural results are primary. "
        "Detailed per-seed model values and support appear in [raw-metrics.json](metrics/raw-metrics.json).",
    ]
    for cell in sorted(cells, key=lambda c: c["rare_fraction"]):
        rarity = round(cell["rare_fraction"] * 100)
        for kind in KINDS:
            runs = [r for r in cell["per_population"] if r["kind"] == kind]
            parts.append(f"## {cell_title(rarity)} · {KINDS[kind]}")
            parts.append("### Diagnostic support by seed")
            supports = []
            for r in runs:
                for group in GROUPS:
                    g = r["groups"][group]
                    supports.append([r["seed"], group, g["diagnostic_count"], g["test_count"],
                                     g["diagnostic_subject_count"], g["subject_count"],
                                     g["models"]["teacher"]["pull_subject_count"]])
            parts.append(md_table(("Seed", "Group", "Diagnostic trials", "All test trials", "Diagnostic people",
                                   "All people", "People with defined pull"), supports))
            for group in GROUPS:
                gs = [r["groups"][group] for r in runs]
                parts.append(f"### {group.title()} group")
                scores, assays = [], []
                for key, name in MODELS.items():
                    ms = [g["models"][key] for g in gs]
                    scores.append([name, *(fmt(mean_available([m[k] for m in ms])) for k in metrics)])
                    assays.append([name, *(fmt(mean(m["signature"][i] for m in ms)) for i in range(4))])
                parts.append(md_table(("Model", "Test NLL", "Test KL", "Diag. NLL", "Diag. KL", "Assay RMS",
                                       "Probe KL", "Pull"), scores))
                parts.append(md_table(("Model", "Unrewarded rare", "Unrewarded common", "Rewarded rare", "Rewarded common"), assays))
            parts.append("### Fitted hybrid/cue mixture")
            family = []
            for r in runs:
                summary = r["family_posterior_summary"]
                for group in GROUPS:
                    g = summary[group]
                    family.append([r["seed"], group, fmt(summary["estimated_family_prior"][1]), fmt(g["cue_posterior_mean"]),
                                   fmt(g["cue_posterior_min"]), fmt(g["cue_posterior_max"]), g["cue_count"],
                                   g["hybrid_count"], g["tie_count"], g["uncertain_count"], g["subject_count"],
                                   fmt(g["entropy_mean"])])
            parts.append(md_table(("Seed", "Group", "Fitted cue prior", "Mean cue posterior", "Min", "Max", "Cue > 0.5",
                                   "Hybrid > 0.5", "Ties", "Posterior in [0.25, 0.75]", "People", "Mean entropy"), family))
    (output_dir / "model-and-diagnostic-tables.md").write_text("\n\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=AUDIT / "primary-reconstruction.json")
    parser.add_argument("--metrics", type=Path, default=AUDIT / "metrics/raw-metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.primary.read_text())
    metrics = json.loads(args.metrics.read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if data["issues"]:
        raise ValueError(data["issues"])
    primary, seeds = primary_tables(data["populations"])
    (output_dir / "primary-tables.md").write_text("# Rare-group primary comparison\n\n" + primary
                                            + "\n\n## All seeds\n\n" + seeds + "\n")
    write_quadrants(data["populations"], output_dir)
    draw(data["populations"], output_dir)
    write_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()
