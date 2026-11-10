#!/usr/bin/env python3
"""Reproduce the policy-library comparison accompanying Experiment 2.

Examples:
    python policy_library.py --runs 20
    python policy_library.py --runs 1 --start-run 27182800 --output-dir generated/policy_library

The three libraries share every run's attacks and sample banks. Only candidate
policies change: fitting and evaluation always use the original defender utility.
The output directory receives all_runs.json, summary.json, and metadata.json.
"""

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import simulation as sim


ROOT = Path(__file__).resolve().parent
BASE_RUN = 27182800
RUN_STEP = 100
SAMPLE_SIZES = {"calibration": 2000, "train": 5000, "test": 20000}
CONFIGURATIONS = {
    "original36": [[1, 2, 4], [1, 2], [1, 2, 4], [1, 2]],
    "balanced144": [[1, 2, 4, 8], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4]],
    "cautious120": [[1, 2, 4, 8, 16], [1, 2, 4, 8], [1, 2, 4], [1, 2]],
}
FACTOR_KEYS = ("collision", "residual", "false_alarm", "far")
METHODS = ("M1", "M2", "M3", "M4")
METRICS = ("CatUnsafe", "worst_severe_unsafe", "original_worst_gap",
           "original_mean_gap", "original_clean_cost", "mean_clear_false_emergency")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_policy(c, r, f, d):
    utility = sim.U_D.copy()
    utility[0, 2] *= c
    utility[1, 2] *= r
    utility[2, 1] *= c
    utility[3, 1] *= c
    utility[2, 0] *= f
    utility[3, 0] *= f
    utility[0, 1] *= d
    utility[1, 1] *= d
    return {"U": utility, "collision": c, "residual": r, "false_alarm": f,
            "far": d, "name": f"c{c}_r{r}_f{f}_d{d}"}


def build_policies():
    """Stable common ordering: original36, remaining cautious120, then extras."""
    policies = sim.build_policy_library()
    known = {tuple(p[key] for key in FACTOR_KEYS) for p in policies}
    for factors in itertools.product(*CONFIGURATIONS["cautious120"]):
        if factors not in known:
            policies.append(make_policy(*factors))
            known.add(factors)
    extra = set(itertools.product(*CONFIGURATIONS["balanced144"])) - known
    policies.extend(make_policy(*factors) for factors in sorted(extra))
    require(len(policies) == 192, "Expected 192 policies in the union of the three grids")
    return policies


def library_indices(policies, name):
    grids = CONFIGURATIONS[name]
    ids = np.array([i for i, policy in enumerate(policies)
                    if all(policy[key] in levels for key, levels in zip(FACTOR_KEYS, grids))])
    require(len(ids) == int(np.prod([len(g) for g in grids])), "Incorrect policy count")
    require(set(range(36)) <= set(ids), "Each expanded library must retain original36")
    return ids


def evaluate_policies(policies, attacks, observations, posterior, states):
    """Compute true-utility gaps and event rates for each policy and attack."""
    expected_utility = posterior @ sim.U_D.T
    clean_optimum = expected_utility.max(axis=1)
    rows = np.arange(len(observations))
    crossing, clear = states == 2, states == 0
    require(crossing.any() and clear.any(), "Both crossing and clear cases are required")
    shape = (len(policies), len(attacks))
    gaps = np.empty(shape)
    unsafe = np.empty(shape)
    emergency = np.empty(shape)
    clean_gaps = np.empty(len(policies))
    for j in range(len(attacks) + 1):
        attacked_posterior = (sim.posterior_3state(observations + attacks[j]["delta"])
                              if j < len(attacks) else posterior)
        for i, policy in enumerate(policies):
            actions = np.argmax(attacked_posterior @ policy["U"].T, axis=1)
            gap = float(np.mean(clean_optimum - expected_utility[rows, actions]))
            if j < len(attacks):
                gaps[i, j] = gap
                unsafe[i, j] = np.mean(actions[crossing] < 2)
                emergency[i, j] = np.mean(actions[clear] == 3)
            else:
                clean_gaps[i] = gap
    return {"G": gaps, "unsafe": unsafe, "emergency": emergency, "clean_G": clean_gaps}


def metrics(rho, data, severe_ids):
    gaps = rho @ data["G"]
    unsafe = rho @ data["unsafe"][:, severe_ids]
    return {"CatUnsafe": float(unsafe.mean()),
            "worst_severe_unsafe": float(unsafe.max()),
            "original_worst_gap": float(gaps.max()),
            "original_mean_gap": float(gaps.mean()),
            "original_clean_cost": float(rho @ data["clean_G"]),
            "mean_clear_false_emergency": float(np.mean(rho @ data["emergency"]))}


def onehot(index, size):
    result = np.zeros(size)
    result[index] = 1.0
    return result


def run_once(run_id):
    policies = build_policies()
    calibration, _, calibration_states = sim.generate_sample_bank(
        SAMPLE_SIZES["calibration"], np.random.RandomState(run_id + 1))
    train_z, train_post, train_states = sim.generate_sample_bank(
        SAMPLE_SIZES["train"], np.random.RandomState(run_id))
    test_z, test_post, test_states = sim.generate_sample_bank(
        SAMPLE_SIZES["test"], np.random.RandomState(run_id + 2))
    attacks = sim.build_catalogue_100(np.random.RandomState(run_id + 3),
                                    calibration, calibration_states)
    severe_ids = [i for i, attack in enumerate(attacks) if attack["category"] == "catastrophic"]
    require(len(attacks) == 100 and len(severe_ids) == 5, "Unexpected attack catalog")
    require(max(np.max(np.abs(a["delta"])) for a in attacks) <= sim.MAX_DELTA + 1e-12,
            "Attack exceeds the per-sensor perturbation limit")
    training = evaluate_policies(policies, attacks, train_z, train_post, train_states)
    tv = sim.compute_prediction_loss(attacks, train_z, train_post)
    top_names = set(sorted(tv, key=tv.get, reverse=True)[:5])
    top_ids = [i for i, attack in enumerate(attacks) if attack["name"] in top_names]
    fits = []
    for name in CONFIGURATIONS:
        ids = library_indices(policies, name)
        gaps = training["G"][ids]
        rhos = {"M1": onehot(int(np.argmin(training["clean_G"][ids])), len(ids)),
                "M2": onehot(int(np.argmin(gaps.mean(axis=1))), len(ids)),
                "M3": sim.solve_lp_primal(gaps[:, top_ids])[0]}
        rhos["M4"], support, prior, _ = sim.method_agnostic_support(
            gaps, [policies[i] for i in ids], max_iter=50, eps=0.1)
        full_value = sim.solve_lp_primal(gaps)[1]
        lower = sim.solve_lp_primal(gaps[:, support])[1]
        upper = float((rhos["M4"] @ gaps).max())
        require(lower - 1e-7 <= full_value <= upper + 1e-7, "Invalid minimax bounds")
        require(upper - lower <= .1000001, "Search residual exceeds tolerance")
        for method in METHODS:
            rho = np.zeros(len(policies))
            rho[ids] = rhos[method]
            entry = {"configuration": name, "policy_count": len(ids), "method": method,
                     "rho": rho, "training": metrics(rho, training, severe_ids)}
            if method == "M4":
                entry.update({"support_names": [attacks[j]["name"] for j in support],
                              "LF_weights": prior.tolist(), "full_value": full_value,
                              "lower": lower, "upper": upper,
                              "certificate_residual": upper - lower})
            fits.append(entry)
    # Test outcomes cannot affect policies, TV selection, or least-favorable weights.
    active = sorted({i for fit in fits for i, weight in enumerate(fit["rho"]) if weight > 1e-12})
    test = evaluate_policies([policies[i] for i in active], attacks,
                             test_z, test_post, test_states)
    for entry in fits:
        rho = entry["rho"]
        entry["test"] = metrics(rho[active], test, severe_ids)
        entry["positive_policies"] = [
            {"id": i, "name": policies[i]["name"], "weight": float(weight)}
            for i, weight in enumerate(rho) if weight > 1e-9]
        entry["rho"] = rho.tolist()
    return {"run": run_id, "configs": fits,
            "catalog": [{**attack, "delta": attack["delta"].tolist()} for attack in attacks],
            "test_crossing_count": int(np.sum(test_states == 2))}


def summarize(runs):
    require(len(runs) > 0, "At least one run is required")
    summary = []
    for name in CONFIGURATIONS:
        for method in METHODS:
            entries = [next(e for e in run["configs"]
                            if (e["configuration"], e["method"]) == (name, method))
                       for run in runs]
            statistics = {}
            for metric in METRICS:
                values = np.array([e["test"][metric] for e in entries])
                statistics[metric] = {"mean": float(values.mean()),
                                      "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
            summary.append({"configuration": name, "method": method,
                            "n": len(runs), "metrics": statistics})
    return summary


def metadata(run_ids):
    policies = build_policies()
    return {
        "experiment": "Experiment 2: policy-library comparison",
        "configurations": CONFIGURATIONS,
        "factor_order": list(FACTOR_KEYS),
        "construction": "Complete factorial grids; each contains the original36 library. The 120- and 144-policy grids are not nested in one another.",
        "run_ids": list(run_ids), "run_step": RUN_STEP, "sample_sizes": SAMPLE_SIZES,
        "random_generator": "numpy.random.RandomState",
        "random_stream_offsets": {"train": 0, "calibration": 1, "test": 2, "attacks": 3},
        "shared_within_run": ["calibration samples", "training samples", "test samples", "attack catalog", "evaluation utility"],
        "environment": {"state_prior": sim.PRIOR.tolist(), "sensor_means": sim.MU.tolist(),
                        "sensor_standard_deviation": sim.SIGMA_S,
                        "state_order": list(sim.STATE_NAMES), "action_order": list(sim.ACTION_NAMES),
                        "sensor_order": list(sim.SENSOR_NAMES)},
        "true_utility": sim.U_D.tolist(),
        "policy_factor_entries": {
            "collision": [[0, 2], [2, 1], [3, 1]], "residual": [[1, 2]],
            "false_alarm": [[2, 0], [3, 0]], "far": [[0, 1], [1, 1]]},
        "policy_note": "Factors change candidate policy utilities only; all fitting and evaluation gaps use true_utility. Entry indices are zero-based [action,state].",
        "policies": [{"id": i, "name": p["name"],
                      **{key: p[key] for key in FACTOR_KEYS}} for i, p in enumerate(policies)],
        "rho_indexing": "All rho vectors index the common policies list; entries outside the selected library are zero.",
        "attack_catalog": {"size": 100, "severe_attack_count": 5,
                           "per_sensor_limit": sim.MAX_DELTA,
                           "generator": "simulation.build_catalogue_100",
                           "calibration": "The original severe-attack calibration under the clean Bayes policy, using only the calibration sample bank."},
        "methods": {"M1": "Clean Bayes policy: smallest clean training gap",
                    "M2": "One policy minimizing the uniform average training gap",
                    "M3": "Mixed minimax over the five highest training posterior-TV attacks",
                    "M4": "Algorithm 1: iteratively add a full-catalog worst-gap attack and solve the catalog game"},
        "M4_settings": {"initial_policy": "original clean Bayes", "max_iterations": 50,
                        "residual_tolerance": .1,
                        "stop": "Worst catalog attack is already collected and residual is within tolerance"},
        "metric_definitions": {
            "original_worst_gap": "Largest utility gap over all 100 attacks",
            "original_mean_gap": "Uniform average utility gap over all 100 attacks",
            "original_clean_cost": "Utility gap with no sensor manipulation",
            "CatUnsafe": "Mean rate of go or slow given true crossing, equally averaged over the five designated severe attacks",
            "worst_severe_unsafe": "Maximum such conditional unsafe rate over the five designated severe attacks",
            "mean_clear_false_emergency": "Mean emergency-braking rate given a clear road, equally averaged over all 100 attacks"},
        "uncertainty": "Summary uses arithmetic means and sample SD across runs; SD is null for one run. Probability metrics are fractions, not percentages.",
        "design": "Grids are fixed before these additional runs; test samples do not select rules, attacks, or weights.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--start-run", type=int, default=BASE_RUN)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "policy_library")
    args = parser.parse_args()
    require(args.runs >= 1, "--runs must be positive")
    require(0 <= args.start_run and args.start_run + RUN_STEP * (args.runs - 1) + 3 < 2**32,
            "Random stream identifiers must fit the RandomState seed range")
    run_ids = [args.start_run + RUN_STEP * i for i in range(args.runs)]
    runs = []
    for run_id in run_ids:
        runs.append(run_once(run_id))
        print(f"Completed run {run_id} ({len(runs)}/{len(run_ids)})", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "all_runs.json", runs)
    write_json(args.output_dir / "summary.json", summarize(runs))
    write_json(args.output_dir / "metadata.json", metadata(run_ids))


if __name__ == "__main__":
    main()
