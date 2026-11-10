#!/usr/bin/env python3
"""Rebuild the paper's tables and figures from archived results.

This script does not fit a defense, select attacks, or run a simulator. The saved
result keys retain their original identifiers: M5 is the paper's proposed M4;
the saved M4 is the separate full-catalog reference. Only display labels change.

Generated tables and figures are written to the local, Git-ignored generated/
directory. Numerical consistency is checked separately by verify_results.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COLORS = ["#7c8791", "#b08b48", "#cf6b42", "#40836b", "#315b91"]
METRICS = ["R_max", "R_mean", "R_unsafe_cat", "C_clean"]
METHODS = [("M1", "M1"), ("M2", "M2"), ("M3", "M3"), ("M4", "M5")]
METHOD_COLORS = {"M1": COLORS[0], "M2": COLORS[1], "M3": COLORS[2], "M4": COLORS[4]}
METHOD_NAMES = {"M1": "Clean Bayes", "M2": "Uniform prior", "M3": "Posterior TV", "M4": "Proposed"}
METRIC_TITLES = ["(a) Worst gap", "(b) Mean gap", "(c) CatUnsafe (%)", "(d) Clean cost"]
POLICY_CONFIGURATIONS = [("original36", "36"), ("cautious120", "120"), ("balanced144", "144")]
POLICY_METRICS = ["original_worst_gap", "original_mean_gap", "CatUnsafe", "original_clean_cost"]


def stat(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Statistics require at least two finite per-seed values")
    return float(a.mean()), float(a.std(ddof=1))


def pm(values, digits=3, spaces=True):
    mean, sd = stat(values)
    separator = r" \pm " if spaces else r"\pm"
    return f"${mean:.{digits}f}{separator}{sd:.{digits}f}$"


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
    tables["attack_ranking"] = tabular("@{}lrr@{}", ["Ranking measure", "Utility gap", "Posterior TV"], rows)

    def performance(section, methods):
        rows = [[label, *[pm(values(section, f"{key}_{m}")) for m in METRICS]]
                for label, key in methods]
        return tabular("@{}lrrrr@{}", ["Method", "Worst gap", "Mean gap", "CatUnsafe", "Clean cost"], rows)

    tables["defense_comparison"] = performance("exp2", METHODS)

    rows = []
    for n in [1, 2, 5]:
        records = [s["exp3"][f"n{n}"] for s in main]
        fields = ["unif_mean_unsafe", "unif_worst_unsafe", "agn_mean_unsafe",
                  "agn_worst_unsafe", "agn_sup_size"]
        rows.append([str(n), *[pm([s[k] for s in records]) for k in fields]])
    tables["catalog_composition"] = tabular(
        "@{}crrrrr@{}", [r"$n_{\rm cat}$", "M2 mean", "M2 worst", "M4 mean", "M4 worst", "$|E|$"], rows)

    final = final_support_points(main)
    rows = [["M4 (final)", pm(final[:, 0], 2),
             f"${final[:, 1].mean():.3f}$", f"${final[:, 2].mean():.3f}$",
             f"${(final[:, 2] - final[:, 1]).mean():.3f}$"]]
    for kind, label, selected in [("pred", "Posterior TV", [5, 8]), ("random", "Random", [20, 50])]:
        for k, v, w in support_curves(main, kind):
            if k in selected:
                rows.append([label, str(int(k)), f"${v:.3f}$", f"${w:.3f}$", f"${w-v:.3f}$"])
    tables["support_certificate"] = tabular(
        "@{}lrrrr@{}", ["Attack selection", r"\shortstack{Number of\\selected attacks}",
                         "Lower bound", "Upper bound", "Residual"], rows)

    tables["conflicting_threats"] = performance("exp7", METHODS)

    rows = []
    for d in strategic[0]["diagnostic"]:
        label = r"$\tau_H$" if d["generator"] == "tau_H" else r"$\tau_F$"
        rows.append([label, d["family"], r"$\checkmark$" if d["is_global_best"] else "---",
                     *[f"${d[k]:.2f}$" for k in ["true_gap", "perceived_gap", "cost", "stealth", "attacker_utility"]]])
    tables["attacker_utility"] = tabular(
        "@{}llcrrrrr@{}", ["Model", "Family", "Chosen", "Defender gap", "Perceived gap", "Cost", "Stealth", "Utility $U_A^{(1)}$"], rows)

    names = [("Collision model only", "tau_H"), ("Disruption model only", "tau_F"),
             ("Both models", "joint"), ("Level-0 reference", "L0")]
    rows = []
    for label, name in names:
        weight_cells = ["---", "---"] if name == "L0" else [pm(strategic_values(name, k), spaces=False) for k in ["h", "f"]]
        rows.append([label, *weight_cells, *[pm(strategic_values(name, k), 2, spaces=False) for k in ["sz", "positive_support_size"]]])
    first = tabular("@{}lrrrr@{}", ["Models considered", "LF weight on H", "LF weight on F", "$|E|$", r"$s_\pi$"], rows)
    rows = []
    for label, name in names:
        response = "---" if name == "L0" else pm(strategic_values(name, "Rstrat"), spaces=False)
        rows.append([label, response, *[pm(strategic_values(name, k), spaces=False) for k in ["Rfull", "Cclean"]]])
    second = tabular("@{}lrrr@{}", ["Models considered", "Response gap", "Worst catalog gap", "Clean cost"], rows)
    tables["strategic_responses"] = (
        panel("(a) LF weights and numbers of collected attacks", first) + "\n\\medskip" +
        panel("(b) Defense performance on independent test samples", second))

    rows = []
    for label, key in [("Clean (M1)", "M1"), ("Uniform (M2)", "M2"), ("Agnostic (M4)", "M5")]:
        rows.append([label, *[pm([s["heldout_attacks"][key][m] for s in main], spaces=False)
                             for m in ["R_max", "R_mean", "R_unsafe_cat", "R_unsafe_cat_worst"]]])
    tables["out_of_catalog"] = tabular(
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


def save_comparison(fig, destination, name, title):
    paths = []
    for extension in ["pdf", "png"]:
        path = destination / f"{name}.{extension}"
        kwargs = {"metadata": {"Title": title}} if extension == "pdf" else {"dpi": 250}
        fig.savefig(path, **kwargs)
        paths.append(path.name)
    plt.close(fig)
    return paths


def comparison_axes():
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.3))
    fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.29, wspace=.60)
    for ax, title in zip(axes, METRIC_TITLES):
        ax.set_title(title, loc="left", fontsize=9)
        ax.tick_params(labelsize=8, width=.6, length=3)
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=.65)
    return fig, axes


def comparison_legend(fig, markers=None):
    markers = markers or ["s"] * len(METHODS)
    handles = [plt.Line2D([], [], color=METHOD_COLORS[label], marker=marker, linestyle="none",
                         markersize=5, label=f"{label}  {METHOD_NAMES[label]}")
               for (label, _), marker in zip(METHODS, markers)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4,
               frameon=False, fontsize=8, handlelength=1, handletextpad=.4, columnspacing=1.2)


def performance_figure(main, section, destination, name, title):
    fig, axes = comparison_axes()
    labels = [label for label, _ in METHODS]
    for ax, metric in zip(axes, METRICS):
        scale = 100 if metric == "R_unsafe_cat" else 1
        points = [stat([s[section][f"{key}_{metric}"] for s in main]) for _, key in METHODS]
        means, sds = np.asarray(points).T * scale
        ax.bar(range(len(METHODS)), means, width=.6,
               color=[METHOD_COLORS[label] for label in labels], yerr=sds, capsize=2,
               error_kw={"ecolor": "#222222", "elinewidth": .8, "capthick": .8})
        ax.set_xticks(range(len(METHODS)), labels)
        ax.set_xlim(-.55, 3.55)
        ax.set_ylim(0, max(float((means + sds).max()) * 1.12, .01))
    comparison_legend(fig)
    return save_comparison(fig, destination, name, title)


def policy_library_figure(summary, destination):
    """Plot the three separately fitted libraries as categorical comparisons."""
    expected = {(configuration, label) for configuration, _ in POLICY_CONFIGURATIONS
                for label, _ in METHODS}
    records = {(row["configuration"], row["method"]): row for row in summary}
    if len(summary) != len(expected) or records.keys() != expected:
        raise ValueError("Policy-library summary must contain each library/method pair exactly once")
    if any(row["n"] != 20 for row in records.values()):
        raise ValueError("Policy-library exhibits require all 20 additional runs")
    fig, axes = comparison_axes()
    fig.subplots_adjust(bottom=.34)
    markers = ["o", "s", "^", "D"]
    for ax, metric in zip(axes, POLICY_METRICS):
        scale = 100 if metric == "CatUnsafe" else 1
        lower, upper = [], []
        for i, (label, _) in enumerate(METHODS):
            points = np.asarray([
                [records[(configuration, label)]["metrics"][metric][field] for field in ["mean", "sd"]]
                for configuration, _ in POLICY_CONFIGURATIONS], dtype=float)
            if not np.all(np.isfinite(points)) or np.any(points[:, 1] < 0):
                raise ValueError(f"Invalid policy-library summary for {label}/{metric}")
            means, sds = points.T * scale
            x = np.arange(len(POLICY_CONFIGURATIONS)) + (i - 1.5) * .14
            # These libraries are categorical: the two extensions are not nested.
            ax.errorbar(x, means, yerr=sds, fmt=markers[i], linestyle="none",
                        markersize=3.8, markeredgewidth=.5, color=METHOD_COLORS[label],
                        elinewidth=.8, capsize=1.5, capthick=.7)
            lower.extend(means - sds)
            upper.extend(means + sds)
        low, high = min(lower), max(upper)
        margin = max(high - low, .01) * .10
        ax.set_ylim(max(0, low - margin), high + margin)
        ax.set_xticks(range(len(POLICY_CONFIGURATIONS)), [label for _, label in POLICY_CONFIGURATIONS])
        ax.set_xlim(-.46, 2.46)
    fig.text(.525, .20, "Policy library", ha="center", fontsize=8)
    comparison_legend(fig, markers)
    return save_comparison(fig, destination, "policy_library_comparison", "Policy-library comparison")


def make_figures(main, policy_summary, destination):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 160,
                         "savefig.dpi": 220, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    for kind, label, color, marker in [("gap", "Utility gap", COLORS[4], "o"),
                                      ("pred", "Posterior TV", COLORS[2], "s")]:
        points = [stat([s["exp1"][f"top{k}_{kind}_recall"] for s in main]) for k in [3, 5, 10]]
        ax.errorbar([3, 5, 10], [x[0] for x in points], yerr=[x[1] for x in points],
                    marker=marker, color=color, capsize=4, label=label)
    ax.set(xlabel="Number of top-ranked attacks K", ylabel="Recall of designated severe attacks",
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
    axes[0].axhline(np.mean([s["full_minimax_train"] for s in main]), ls=":", color=COLORS[3], label="Full-catalog minimax")
    axes[0].scatter([final[:, 0].mean()] * 2, [final[:, 1].mean(), final[:, 2].mean()],
                    marker="D", s=50, color=COLORS[4], label="Algorithm 1 (final)")
    axes[1].scatter(final[:, 0], final[:, 2] - final[:, 1], marker="D", s=22,
                    color=COLORS[4], label="Algorithm 1 (all runs)")
    axes[1].axhline(.1, ls=":", color="gray", label="Residual tolerance 0.1")
    for ax in axes:
        ax.set(xscale="log", xlabel="Number of selected attacks $K$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Training utility gap")
    axes[1].set_ylabel("Difference between bounds $W_K-V_K$")
    fig.tight_layout()
    fig.savefig(destination / "support_certificate.png", bbox_inches="tight")
    plt.close(fig)
    outputs = ["attack_ranking.png", "support_certificate.png"]
    outputs += performance_figure(main, "exp2", destination, "defense_comparison", "Defense comparison")
    outputs += performance_figure(main, "exp7", destination, "conflicting_threats_comparison",
                                  "Conflicting hide and fake threats")
    outputs += policy_library_figure(policy_summary, destination)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root; defaults to this script's directory")
    args = parser.parse_args()
    root = args.root.resolve()
    main_path = root / "results/main/all_seeds.json"
    strategic_path = root / "results/strategic/all_seeds.json"
    policy_path = root / "results/policy_library/summary.json"
    main_results = json.loads(main_path.read_text())
    strategic_document = json.loads(strategic_path.read_text())
    strategic_results = strategic_document["all_seeds"]
    policy_summary = json.loads(policy_path.read_text())
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
    figures = make_figures(main_results, policy_summary, figure_dir)
    manifest = {"sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [main_path, strategic_path, policy_path]},
                "seed_count": len(main_results), "policy_library_run_count": 20, "sd_ddof": 1,
                "error_bars": "Sample standard deviations across runs",
                "tables": sorted(tables),
                "figures": figures,
                "scientific_models_recomputed": False}
    (root / "generated/exhibit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(tables)} tables and {len(figures)} figure files from unchanged saved results.")


if __name__ == "__main__":
    main()
