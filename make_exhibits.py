#!/usr/bin/env python3
"""Rebuild the paper's tables and two figures from saved per-seed results.

This script does not fit a defense, select attacks, or run a simulator. The saved
result keys retain their original identifiers: M5 is the paper's proposed M4;
the saved M4 is the separate full-catalogue reference. Only display labels change.

Run ``python make_exhibits.py --check`` to also compare every displayed numeric
table cell with the corresponding frozen table in paper_exhibits/tables/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
COLORS = ["#7c8791", "#b08b48", "#cf6b42", "#40836b", "#315b91"]
METRICS = ["R_max", "R_mean", "R_unsafe_cat", "C_clean"]
METHODS = [("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M5")]


def stat(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Statistics require at least two finite per-seed values")
    return float(a.mean()), float(a.std(ddof=1))


def pm(values, digits=3, spaces=True):
    mean, sd = stat(values)
    separator = r" \pm " if spaces else r"\pm"
    return f"${mean:.{digits}f}{separator}{sd:.{digits}f}$"


def paired(a, b):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, sd = stat(delta)
    half = float(t.ppf(.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
    return mean, mean - half, mean + half


def tabular(columns, headings, rows):
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return ("\n".join(lines) + "\n").replace("-0.000", "0.000")


def panel(title, content):
    return "{" + title + r"}\par\smallskip" + "\n" + content


def make_tables(main, strategic):
    def values(section, key):
        return np.asarray([s[section][key] for s in main], dtype=float)

    def strategic_values(name, key):
        return np.asarray([s["generators"][name][key] for s in strategic], dtype=float)

    tables = {}
    rows = [[f"Top-{k} recall", pm(values("exp1", f"top{k}_gap_recall")),
             pm(values("exp1", f"top{k}_pred_recall"))] for k in [3, 5, 10]]
    gap_rank = values("exp1", "worst_cat_rank_gap")
    tv_rank = values("exp1", "worst_cat_rank_pred")
    rows += [[r"Rank of most unsafe attack (mean $\pm$ SD)",
              pm(gap_rank, 2), pm(tv_rank, 2)],
             ["Rank of most unsafe attack (median)",
              f"${np.median(gap_rank):.1f}$", f"${np.median(tv_rank):.1f}$"]]
    tables["ranking"] = tabular("@{}lrr@{}", ["Ranking measure", "Utility gap", "Posterior TV"], rows)

    def performance(section, methods):
        rows = [[label, *[pm(values(section, f"{key}_{m}")) for m in METRICS]]
                for label, key in methods]
        return tabular("@{}lrrrr@{}", ["Method", "Worst gap", "Mean gap", "CatUnsafe", "Clean cost"], rows)

    def differences(section, comparators):
        rows = []
        labels = [r"$R_{\max}$", "Mean gap", "CatUnsafe", r"$C_{\rm clean}$"]
        for other in comparators:
            for metric, label in zip(METRICS, labels):
                mean, lower, upper = paired(values(section, f"M5_{metric}"),
                                             values(section, f"{other}_{metric}"))
                rows.append([f"M4 $-$ {other}", label, f"${mean:.3f}$",
                             rf"$[{lower:.3f},\ {upper:.3f}]$"])
        return tabular("@{}llrr@{}", ["Contrast", "Metric", "Mean difference", r"95\% paired CI"], rows)

    tables["defense_and_uncertainty"] = (
        panel("(a) Test performance", performance("exp2", METHODS)) + "\n\\medskip" +
        panel("(b) Paired differences on the same seeds", differences("exp2", ["M1", "M2"])))

    rows = []
    for n in [1, 2, 5]:
        records = [s["exp3"][f"n{n}"] for s in main]
        fields = ["unif_mean_unsafe", "unif_worst_unsafe", "agn_mean_unsafe",
                  "agn_worst_unsafe", "agn_sup_size"]
        rows.append([str(n), *[pm([s[k] for s in records]) for k in fields]])
    tables["catalogue_composition"] = tabular(
        "@{}crrrrr@{}", [r"$n_{\rm cat}$", "M2 mean", "M2 worst", "M4 mean", "M4 worst", "$|E|$"], rows)

    final = final_support_points(main)
    rows = [["Gap search (final)", pm(final[:, 0], 2),
             f"${final[:, 1].mean():.3f}$", f"${final[:, 2].mean():.3f}$",
             f"${(final[:, 2] - final[:, 1]).mean():.3f}$"]]
    for kind, label, selected in [("pred", "Posterior TV", [5, 8]), ("random", "Random", [20, 50])]:
        for k, v, w in support_curves(main, kind):
            if k in selected:
                rows.append([label, str(int(k)), f"${v:.3f}$", f"${w:.3f}$", f"${w-v:.3f}$"])
    tables["support_certificate"] = tabular(
        "@{}lrrrr@{}", ["Attack selection", "Count", "Lower bound", "Upper bound", "Residual"], rows)

    rows = [["M4", *[pm(values("exp7", f"M5_{m}"), spaces=False)
                       for m in ["H_max", "F_max", "S_H", "S_F"]]]]
    family = tabular("@{}lrrrr@{}", ["Method", r"$H_{\max}$", r"$F_{\max}$", "$S_H$", "$S_F$"], rows)
    tables["conflicting_results"] = (
        panel("(a) Full-catalogue test performance", performance("exp7", [METHODS[0], METHODS[1], METHODS[3]])) +
        "\n\\medskip" + panel("(b) M4 results for hide and fake attacks separately", family) +
        "\n\\medskip" + panel("(c) Paired differences on the same seeds", differences("exp7", ["M2"])))

    rows = []
    for d in strategic[0]["diagnostic"]:
        label = r"$\tau_H$" if d["generator"] == "tau_H" else r"$\tau_F$"
        rows.append([label, d["family"], r"$\checkmark$" if d["is_global_best"] else "---",
                     *[f"${d[k]:.2f}$" for k in ["true_gap", "perceived_gap", "cost", "stealth", "attacker_utility"]]])
    tables["attacker_utility"] = tabular(
        "@{}llcrrrrr@{}", ["Model", "Family", "Chosen", "True gap", "Perceived gap", "Cost", "Stealth", "Payoff $U_A$"], rows)

    names = [("H only", "tau_H"), ("F only", "tau_F"), ("Joint Level-1", "joint"), ("Level-0 reference", "L0")]
    rows = []
    for label, name in names:
        weight_cells = ["---", "---"] if name == "L0" else [pm(strategic_values(name, k), spaces=False) for k in ["h", "f"]]
        rows.append([label, *weight_cells, *[pm(strategic_values(name, k), 2, spaces=False) for k in ["sz", "positive_support_size"]]])
    first = tabular("@{}lrrrr@{}", ["Attack selection", "LF weight on H", "LF weight on F", "$|E|$", r"$s_\pi$"], rows)
    rows = []
    for label, name in names:
        response = "---" if name == "L0" else pm(strategic_values(name, "Rstrat"), spaces=False)
        rows.append([label, response, *[pm(strategic_values(name, k), spaces=False) for k in ["Rfull", "Cclean"]]])
    second = tabular("@{}lrrr@{}", ["Attack selection", "Response gap", "Worst catalogue gap", "Clean cost"], rows)
    tables["strategic_results"] = (
        panel("(a) LF weights and numbers of collected attacks", first) + "\n\\medskip" +
        panel("(b) Defense performance on independent test samples", second))

    rows = []
    for label, key in [("Clean (M1)", "M1"), ("Uniform (M2)", "M2"), ("Agnostic (M4)", "M5")]:
        rows.append([label, *[pm([s["heldout_attacks"][key][m] for s in main], spaces=False)
                             for m in ["R_max", "R_mean", "R_unsafe_cat", "R_unsafe_cat_worst"]]])
    tables["out_of_catalogue"] = tabular(
        "@{}lrrrr@{}", ["Method", "Worst gap", "Mean gap", "Mean unsafe", "Worst unsafe"], rows)
    return tables


def final_support_points(main):
    points = []
    for s in main:
        d = s["exp4"]
        valid = [i for i, k in enumerate(d["gp_K"])
                 if k > 0 and d["gp_V"][i] is not None and np.isfinite(d["gp_V"][i])]
        if not valid:
            raise ValueError(f"No final support point for seed {s['seed']}")
        i = valid[-1]
        points.append([d["gp_K"][i], d["gp_V"][i], d["gp_W"][i]])
    return np.asarray(points, dtype=float)


def support_curves(main, kind):
    ks = sorted({k for s in main for k in s["exp4"][kind + "_K"] if k > 0})
    result = []
    for k in ks:
        points = []
        for s in main:
            d = s["exp4"]
            if k in d[kind + "_K"]:
                i = d[kind + "_K"].index(k)
                v, w = d[kind + "_V"][i], d[kind + "_W"][i]
                if v is not None and np.isfinite(v):
                    points.append((v, w))
        if points:
            values = np.asarray(points)
            result.append((k, float(values[:, 0].mean()), float(values[:, 1].mean())))
    return result


def make_figures(main, destination):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 160,
                         "savefig.dpi": 220, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for kind, label, color, marker in [("gap", "Utility gap", COLORS[4], "o"),
                                      ("pred", "Posterior TV", COLORS[2], "s")]:
        points = [stat([s["exp1"][f"top{k}_{kind}_recall"] for s in main]) for k in [3, 5, 10]]
        ax.errorbar([3, 5, 10], [x[0] for x in points], yerr=[x[1] for x in points],
                    marker=marker, color=color, capsize=4, label=label)
    ax.set(xlabel="Number of top-ranked attacks K", ylabel="Recall of calibrated catastrophic attacks",
           xticks=[3, 5, 10], ylim=(-.12, 1.12))
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "attack_ranking.png", bbox_inches="tight")
    plt.close(fig)

    final = final_support_points(main)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for kind, label, color in [("pred", "Posterior TV", COLORS[2]), ("random", "Random", COLORS[0])]:
        points = np.asarray(support_curves(main, kind))
        x, lower, upper = points.T
        axes[0].plot(x, lower, marker="o", color=color, label=label + " lower")
        axes[0].plot(x, upper, marker="o", mfc="white", ls="--", color=color, label=label + " upper")
        axes[1].plot(x, upper - lower, marker="o", color=color, label=label)
    axes[0].axhline(np.mean([s["full_minimax_train"] for s in main]), ls=":", color=COLORS[3], label="Full-catalogue minimax")
    axes[0].scatter([final[:, 0].mean()] * 2, [final[:, 1].mean(), final[:, 2].mean()],
                    marker="D", s=50, color=COLORS[4], label="Gap search (final)")
    axes[1].scatter(final[:, 0], final[:, 2] - final[:, 1], marker="D", s=22,
                    color=COLORS[4], label="Gap search (all runs)")
    axes[1].axhline(.1, ls=":", color="gray", label="Residual tolerance 0.1")
    for ax in axes:
        ax.set(xscale="log", xlabel="Number of selected attacks $K$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Training utility gap")
    axes[1].set_ylabel("Difference between bounds $W_K-V_K$")
    fig.tight_layout()
    fig.savefig(destination / "support_certificate.png", bbox_inches="tight")
    plt.close(fig)


def strip_revision_commands(text):
    r"""Remove balanced \rev{...} wrappers while preserving their contents."""
    while r"\rev{" in text:
        start = text.index(r"\rev{")
        cursor, depth = start + len(r"\rev{"), 1
        end = cursor
        while depth and end < len(text):
            if text[end] in "{}" and (end == 0 or text[end - 1] != "\\"):
                depth += 1 if text[end] == "{" else -1
            end += 1
        if depth:
            raise ValueError("Unbalanced revision command in reference table")
        text = text[:start] + text[cursor:end - 1] + text[end:]
    return text


def numeric_body_rows(text):
    """Extract formatted row numbers, excluding table headings and LaTeX layout."""
    text = strip_revision_commands(text)
    rows = []
    for body in re.findall(r"\\midrule(.*?)\\bottomrule", text, flags=re.S):
        for row in body.split(r"\\"):
            if "&" not in row:
                continue
            numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", row)
            rows.append(numbers)
    return rows


def check_tables(tables, reference):
    checked = []
    for name, content in tables.items():
        path = reference / f"{name}.tex"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen paper table: {path}")
        actual, expected = numeric_body_rows(content), numeric_body_rows(path.read_text())
        if actual != expected:
            raise AssertionError(f"Displayed numeric rows differ for {name}:\n{actual}\n{expected}")
        checked.append(name)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="also verify every numeric table row against the frozen paper tables")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root; defaults to this script's directory")
    args = parser.parse_args()
    root = args.root.resolve()
    main_path = root / "results/main/all_seeds.json"
    strategic_path = root / "results/strategic/all_seeds.json"
    main_results = json.loads(main_path.read_text())
    strategic_document = json.loads(strategic_path.read_text())
    strategic_results = strategic_document["all_seeds"]
    if len(main_results) != 20 or len(strategic_results) != 20:
        raise ValueError("Paper exhibits require the complete 20-seed results in both suites")
    if [s["seed"] for s in main_results] != [s["seed"] for s in strategic_results]:
        raise ValueError("Main and strategic seed identifiers do not match")
    table_dir, figure_dir = root / "generated/tables", root / "generated/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = make_tables(main_results, strategic_results)
    for name, content in tables.items():
        (table_dir / f"{name}.tex").write_text(content)
    make_figures(main_results, figure_dir)
    checked = check_tables(tables, root / "paper_exhibits/tables") if args.check else []
    manifest = {"sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [main_path, strategic_path]},
                "seed_count": len(main_results), "sd_ddof": 1,
                "paired_interval": "two-sided paired t, 95%, df=19; unadjusted",
                "table_numeric_checks": checked, "tables": sorted(tables),
                "figures": ["attack_ranking.png", "support_certificate.png"],
                "scientific_models_recomputed": False}
    (root / "generated/exhibit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(tables)} tables and 2 figures from unchanged saved results.")
    if checked:
        print(f"PASS: all displayed numeric rows match {len(checked)} frozen paper tables.")


if __name__ == "__main__":
    main()
