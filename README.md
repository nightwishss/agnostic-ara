# Agnostic ARA

Reproduction code for *Agnostic adversarial risk analysis for Bayesian decision systems under unknown sensor attacks*.

The experiments compare attacks by their consequences for decisions, compute a minimax defense and least-favorable attack weights without an externally specified attack prior, and incorporate responses generated from an attacker's own utility.

## Run

Use Python 3.11 and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This builds tables and figures from the saved simulation results and checks their numerical consistency. To regenerate all results before building the outputs:

```bash
python reproduce.py --rerun
```

The full rerun replaces the saved results using the same random streams and experiment settings. SciPy's HiGHS interface solves the linear programs; small floating-point differences between environments are possible. Outputs and verification reports go to `generated/`, which is ignored by Git. A LaTeX installation is not required. The repository contains code, configurations, and numerical results; manuscript files and pre-rendered exhibits are not included.

## Experiments

| Experiment | What is compared | Code and saved results |
| --- | --- | --- |
| 1. Identifying consequential attacks | Utility gap versus posterior total variation (TV) for ordering the same attacks | `simulation.py`; `results/main/`, key `exp1` |
| 2. Computing a defense | M1–M4 on the same catalog, with a catalog-composition sensitivity check | `simulation.py`; keys `exp2`, `exp3` |
| 2. Larger policy libraries | Matched comparisons using 36, 120, and 144 policies | `policy_library.py`; `results/policy_library/` |
| 3. Defense from a small attack set | Algorithm 1 versus TV and random attack selection; a hide/fake extension includes all four methods | `simulation.py`; keys `exp4`, `exp7`, `certificates` |
| 4. Attacker objectives | Responses from a collision-oriented model, a disruption-oriented model, or both; Level-0 provides a reference | `strategic.py`; `results/strategic/` |
| Attack-coverage diagnostic | Frozen defenses tested on independently generated attacks | `simulation.py`; key `heldout_attacks` |

The main-result files retain their established storage keys. The following mapping is applied when generating the paper's tables and figures:

| Paper method | Main-result key | Definition |
| --- | --- | --- |
| M1 | M1 | Clean Bayes policy |
| M2 | M2 | Defense minimizing average gap under equal attack weights |
| M3 | M3 | Minimax defense over the five attacks with the largest posterior TV |
| M4 | M5 | Algorithm 1: iteratively add attacks with the largest defender gap |
| Full-catalog reference | M4 | Minimax solution using all catalog attacks |

The policy-library results use M1–M4 directly. In the strategic results, `tau_H` and `tau_F` denote the two attacker models; the stored key `joint` means using the union of their responses, not a three-player game. Least-favorable weights are computed over the collected attacks.

## Settings and measures

- The simulator has three environmental states, four sensor channels, and four driving actions. Perturbations are added to sensor readings, with an absolute limit of 1.5 per channel.
- Main and strategic experiments each use 20 runs, identified by `20260721 + 100*i` for `i = 0,...,19`. Each run has 2,000 calibration, 5,000 training, and 20,000 test observations from separate random streams.
- The first catalog has 100 attacks, including five calibrated against the clean policy's unsafe-action rate. The hide/fake catalog has three severe attacks of each kind, calibrated toward a clean-policy utility gap of 22 within the same perturbation limit. Some single-channel attacks reach the limit before that target. The strategic comparison uses the same construction with a separate background-attack stream.
- Training data determine the defense, selected attacks, and least-favorable weights. Test observations evaluate the fitted defense. Experiment 1 uses test observations only for its descriptive attack comparison. Reported variability is the sample standard deviation across runs.
- The policy-library comparison uses 20 additional matched runs, identified by `27182800 + 100*i`. All methods and libraries in each run share samples, attacks, and the true evaluation utility. The 120-policy grid adds more cautious rules; the 144-policy grid expands the four preference dimensions more evenly. Both contain the original 36 policies. Exact grids and random-stream offsets are saved in `results/policy_library/metadata.json`.

Worst gap is the largest expected utility loss over the evaluated attacks; mean gap averages those losses. CatUnsafe averages the probability of choosing `go` or `slow` in an imminent-crossing state over the designated severe attacks. It measures unsafe actions, not simulated collisions. Clean cost is the utility loss without an attack relative to the clean Bayes policy, not a computational cost. All policies are evaluated using the same true utility matrix, including those constructed with different preference multipliers.

Level-0 checks all attacks in the specified finite catalog. Its upper/lower-bound difference measures optimization error for that catalog and policy library. Level-1 instead generates attacks from modeled opponent utilities and checks that their current best responses have been collected. Worst-gap optimization can trade off clean performance and individual safety indicators. The payoff matrices are precomputed; a small selected attack set is not an end-to-end runtime claim.

## Code and checks

```text
simulation.py            Sensor model, policies, minimax solver, main experiments
strategic.py             Attacker utilities and response-generation loop
policy_library.py        Matched policy-library comparison
reproduce.py             Reproduction entry point
make_exhibits.py          Tables and figures from saved results
verify_results.py        Core algorithms and main/strategic archive checks
verify_policy_library.py Policy-library definitions and archive checks
requirements.txt         Pinned numerical and plotting dependencies
results/                 Numerical records and experiment settings
```

For core checks without saved results, run `python verify_results.py --unit-only`. The default `python reproduce.py` also validates saved summaries, bounds, mixture weights, sample sizes, and policy-library consistency. These checks complement a full simulation rerun; they do not replace it.
