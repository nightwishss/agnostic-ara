# Agnostic ARA

Reproducible code and archived results for **Agnostic adversarial risk analysis for Bayesian decision systems under unknown sensor attacks**.

The experiments compare sensor attacks through decision-level utility loss, construct a minimax defense without specifying attack frequencies, recover least-favorable attack weights, and incorporate attacks chosen using an opponent's own utility.

## Reproduce the paper

The archived simulation environment used Python 3.11.15. Install the pinned dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Rebuild the paper's tables and figures from the included 20-seed results, compare their numerical contents with the paper exhibits, and run the numerical checks:

```bash
python reproduce.py
```

To rerun both simulation suites from scratch before rebuilding the exhibits:

```bash
python reproduce.py --rerun
```

The second command replaces the archived result files with freshly calculated results. It uses the same model, seeds, calibration, training, and test protocol. Linear programs are solved with SciPy's HiGHS interface; small floating-point differences between environments are possible.

For core numerical checks without reading the archived experiment results:

```bash
python verify_results.py --unit-only
```

Generated tables, figures, summaries, and verification records are written to `generated/`. No LaTeX installation is needed to reproduce the numerical exhibits.

The included checks in `checks/` record the archived-result verification and a fresh rerun of the first seed in both simulation suites: all 969 compared numerical values matched the archived records exactly. This first-seed rerun is separate from the consistency checks over all 20 archived seeds.

## Experiments and files

| Paper experiment | Question | Implementation | Archived results |
| --- | --- | --- | --- |
| 1. Decision-critical attack ranking | Does utility gap identify consequential attacks more reliably than posterior total variation? | `simulation.py` | `results/main/all_seeds.json`: `exp1` |
| 2. Protection without attack-frequency information | How do clean, uniform-weight, TV-guided, and agnostic defenses compare? How does catalogue composition affect protection? | `simulation.py` | `exp2`, `exp3` |
| 3. A few attacks and the full catalogue | Can selected attacks recover the finite full-catalogue game? What happens when hide and fake threats conflict? | `simulation.py` | `exp4`, `exp7`, `certificates` |
| 4. Attacker objectives and joint defense | How do different attacker utilities change best responses and the joint least-favorable defense? | `strategic.py` | `results/strategic/all_seeds.json` |
| Appendix: attack-catalogue coverage | How do frozen defenses perform on separately generated attacks? | `simulation.py` | `heldout_attacks` |

The stored JSON keys retain the original experiment-batch schema so that their numerical records remain unchanged. Paper numbering follows the table above.

### Method labels

The paper uses four methods. The archived simulation schema also stores a full-catalogue reference:

| Paper label | Stored main-result key | Meaning |
| --- | --- | --- |
| M1 | M1 | Clean Bayes policy |
| M2 | M2 | Uniform-weight gap defense |
| M3 | M3 | Minimax defense over five attacks selected by posterior TV |
| M4 | M5 | Proposed iterative utility-gap attack search |
| Full-catalogue reference, unnumbered | M4 | Minimax solution using every catalogue attack |

Generated paper tables use the paper's M1-M4 labels. The reference is not an additional competing method.

## Experimental protocol

- Three environmental states, four sensor channels, four actions, and a fixed library of 36 defender policies.
- Twenty seeds: `20260721 + 100*i`, for `i = 0,...,19`.
- Per seed: 2,000 calibration observations, 5,000 training observations, and 20,000 independent test observations. Random streams use `seed+1`, `seed`, and `seed+2`, respectively.
- The main attack generator starts at `seed+3`. It constructs 100 attacks, including five attacks calibrated against the clean rule's unsafe-action rate.
- The conflicting-threat construction contains three hide and three fake attacks, calibrated toward a clean-rule utility gap of 22, subject to the per-sensor limit of 1.5. Some single-sensor attacks reach that limit before reaching the target.
- The strategic suite uses the same hide/fake construction but begins its attack RNG independently. Its ordinary background attacks therefore differ from those in the main conflicting-threat experiment. Both suites retain their original generation order.
- Policies, attack selection, and least-favorable weights are fitted on training observations. Evaluation uses independent test observations. Tables report means and sample standard deviations across seeds; selected comparisons use paired t intervals.

The unsafe-action rate is the probability of choosing `go` or `slow` conditional on the true imminent-danger state. It is an action indicator, not a simulated collision rate. The main CatUnsafe measure averages over the designated severe attacks; the appendix averages over a different, separately generated attack set.

The appendix sampler uses normalized Gaussian directions, an L2 multiplier uniform on `[0.05, 2.5]`, coordinate clipping at 1.5, and a minimum L2 distance of 0.10 from the training catalogue. These attacks are evaluated without refitting the defense.

## Repository layout

```text
simulation.py                 Main simulation and finite minimax algorithms
strategic.py                  Attacker utilities and updated best-response loop
reproduce.py                  Single entry point for reproduction
make_exhibits.py               Tables and figures calculated from saved results
verify_results.py             Numerical, protocol, and archived-result checks
requirements.txt              Pinned dependencies
results/main/                 Original 20-seed main results and metadata
results/strategic/             Original 20-seed strategic results
paper_exhibits/                Current paper tables and figures used as references
code_provenance.json           Computational-function and result-file integrity record
generated/                    Rebuilt outputs; ignored by Git
```

## Interpretation

Level-0 searches the finite catalogue for the largest true defender gap. Level-1 generates candidate attacks from modeled attacker utilities; the defender still evaluates their consequences using its own utility. Response-set stability checks that the modeled current best responses have already been collected. It is distinct from a full-catalogue minimax certificate.

Small attack support preserves the specified finite game's protection; it does not establish protection against every unsearched perturbation. Worst utility gap also need not move in the same direction as every event-specific safety rate. The simulations use a one-shot synthetic crossing model and a fixed policy library. Payoff matrices are precomputed, and support size is not a measured end-to-end runtime saving.

This repository contains the experiments reported in the paper and its catalogue-coverage appendix. Separate exploratory experiments and earlier manuscript drafts are not part of this artifact.
