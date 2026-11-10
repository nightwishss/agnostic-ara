#!/usr/bin/env python3
"""Validate the policy-library archive; optionally replay one complete run.

    python verify_policy_library.py
    python verify_policy_library.py --rerun
    python verify_policy_library.py --rerun --start-run 27182800

Archive verification recomputes summaries and checks probability weights, library
membership, metric ranges, and saved minimax bounds. --rerun also recomputes the
specified calibration/training/test experiment and compares the full record.
This script does not overwrite the archived results.
"""

import argparse
import json
from pathlib import Path

import policy_library as experiment
import numpy as np


ATOL = 1e-8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_tree(actual, expected, location="result"):
    """Compare nested scientific records; return largest numeric discrepancy."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                f"{location}: different object keys")
        return max((compare_tree(actual[k], expected[k], f"{location}.{k}")
                    for k in expected), default=0.0)
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected),
                f"{location}: different list lengths")
        return max((compare_tree(a, b, f"{location}[{i}]")
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (float, int)) and np.isfinite(actual),
                f"{location}: expected a finite number")
        delta = abs(actual - expected)
        require(delta <= ATOL + 1e-10 * abs(expected),
                f"{location}: discrepancy {delta:g}")
        return float(delta)
    require(actual == expected, f"{location}: different values")
    return 0.0


def probability_vector(values, size, location):
    values = np.asarray(values, dtype=float)
    require(values.shape == (size,) and np.isfinite(values).all(), f"{location}: invalid shape")
    require(values.min() >= -ATOL and values.max() <= 1 + ATOL,
            f"{location}: invalid weights")
    require(abs(values.sum() - 1) <= ATOL, f"{location}: weights do not sum to one")
    return values


def verify_archive(data_dir):
    runs = json.loads((data_dir / "all_runs.json").read_text())
    metadata = json.loads((data_dir / "metadata.json").read_text())
    summary = json.loads((data_dir / "summary.json").read_text())
    require(len(runs) > 0, "No archived runs")
    ids = [r["run"] for r in runs]
    require(len(set(ids)) == len(ids), "Run identifiers must be unique")
    compare_tree(metadata, experiment.metadata(ids), "metadata")
    max_summary_error = compare_tree(summary, experiment.summarize(runs), "summary")
    policies = experiment.build_policies()
    library_ids = {name: set(experiment.library_indices(policies, name))
                   for name in experiment.CONFIGURATIONS}
    expected_comparisons = {(name, method) for name in experiment.CONFIGURATIONS
                            for method in experiment.METHODS}
    max_residual = 0.0
    for run in runs:
        prefix = f"run {run['run']}"
        require(set(run) == {"run", "configs", "catalog", "test_crossing_count"},
                f"{prefix}: unexpected record fields")
        attacks = run["catalog"]
        require(len(attacks) == 100 and len({a['name'] for a in attacks}) == 100,
                f"{prefix}: expected 100 distinct attacks")
        require(sum(a['category'] == 'catastrophic' for a in attacks) == 5,
                f"{prefix}: expected five designated severe attacks")
        perturbations = np.array([a["delta"] for a in attacks])
        require(perturbations.shape == (100, 4) and np.isfinite(perturbations).all()
                and np.max(np.abs(perturbations)) <= experiment.sim.MAX_DELTA + ATOL,
                f"{prefix}: invalid attack perturbations")
        require(0 < run["test_crossing_count"] < experiment.SAMPLE_SIZES["test"],
                f"{prefix}: invalid number of crossing observations")
        entries = {(e["configuration"], e["method"]): e for e in run["configs"]}
        require(len(entries) == len(run["configs"]) == 12 and set(entries) == expected_comparisons,
                f"{prefix}: expected all three libraries and all four methods")
        for (name, method), entry in entries.items():
            label = f"{prefix}/{name}/{method}"
            require(entry["policy_count"] == len(library_ids[name]), f"{label}: wrong library size")
            rho = probability_vector(entry["rho"], len(policies), label)
            require(all(abs(rho[i]) <= ATOL for i in range(len(policies)) if i not in library_ids[name]),
                    f"{label}: policy outside the requested library")
            if method in ("M1", "M2"):
                require(np.count_nonzero(rho > 1e-8) == 1, f"{label}: expected deterministic policy")
            positive = [{"id": i, "name": policies[i]["name"], "weight": float(w)}
                        for i, w in enumerate(rho) if w > 1e-9]
            compare_tree(entry["positive_policies"], positive, f"{label}/positive_policies")
            for split in ("training", "test"):
                values = entry[split]
                require(set(values) == set(experiment.METRICS), f"{label}/{split}: metric keys")
                require(all(np.isfinite(v) and v >= -ATOL for v in values.values()),
                        f"{label}/{split}: negative or nonfinite gap/rate")
                require(values["original_mean_gap"] <= values["original_worst_gap"] + ATOL,
                        f"{label}/{split}: average gap exceeds worst gap")
                require(0 <= values["CatUnsafe"] <= values["worst_severe_unsafe"] + ATOL <= 1 + 2*ATOL,
                        f"{label}/{split}: invalid conditional unsafe rates")
                require(values["mean_clear_false_emergency"] <= 1 + ATOL,
                        f"{label}/{split}: invalid false-emergency rate")
            if method == "M4":
                support = entry["support_names"]
                require(len(support) == len(set(support)) > 0 and
                        set(support) <= {a["name"] for a in attacks}, f"{label}: invalid support")
                probability_vector(entry["LF_weights"], len(support), f"{label}/prior")
                lower, full, upper = entry["lower"], entry["full_value"], entry["upper"]
                require(lower - ATOL <= full <= upper + ATOL, f"{label}: invalid minimax bounds")
                require(upper - lower <= .1 + ATOL, f"{label}: residual above tolerance")
                require(abs((upper-lower) - entry["certificate_residual"]) <= ATOL,
                        f"{label}: inconsistent residual")
                require(abs(upper - entry["training"]["original_worst_gap"]) <= ATOL,
                        f"{label}: upper bound must equal training worst gap")
                max_residual = max(max_residual, entry["certificate_residual"])
        for name in experiment.CONFIGURATIONS:
            for split in ("training", "test"):
                compare_tree(entries[name, "M1"][split], entries["original36", "M1"][split],
                             f"{prefix}: clean Bayes must be unchanged across libraries")
            require(abs(entries[name, "M1"]["training"]["original_clean_cost"]) <= ATOL,
                    f"{prefix}: clean Bayes clean gap should vanish")
            require(entries[name, "M4"]["full_value"] <= entries["original36", "M4"]["full_value"] + ATOL,
                    f"{prefix}: expanded full-game optimum exceeds original optimum")
            uniform = entries[name, "M2"]["training"]["original_mean_gap"]
            for method in experiment.METHODS:
                require(uniform <= entries[name, method]["training"]["original_mean_gap"] + ATOL,
                        f"{prefix}: M2 must minimize uniform training risk")
                require(entries[name, "M4"]["upper"] <=
                        entries[name, method]["training"]["original_worst_gap"] + .1 + ATOL,
                        f"{prefix}: M4's certified training bound is inconsistent")
    return runs, {"run_count": len(runs), "configuration_count": 3, "methods_per_configuration": 4,
                  "summary_max_abs_error": max_summary_error,
                  "max_training_certificate_residual": max_residual,
                  "archive_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=experiment.ROOT / "results" / "policy_library")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--start-run", type=int, default=experiment.BASE_RUN)
    args = parser.parse_args()
    runs, report = verify_archive(args.data_dir)
    if args.rerun:
        expected = next((r for r in runs if r["run"] == args.start_run), None)
        require(expected is not None, "Requested run is not present in the archive")
        replay = experiment.run_once(args.start_run)
        report["replayed_run"] = args.start_run
        report["replay_max_abs_error"] = compare_tree(replay, expected, "complete replay")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
