#!/usr/bin/env python3
"""Check the simulation core and the archived paper results without rerunning them.

Run ``python verify_results.py --unit-only`` for checks that need no result files.
The default also validates every archived seed, recomputes the saved summaries,
and writes ``generated/verification.json``. These checks validate implementation
and archive consistency; they do not constitute an independent experiment rerun.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import simulation as sim
import strategic as ara


ROOT = Path(__file__).resolve().parent
ATOL = 1e-8


def require(condition, message):
    """Use an explicit exception, so checks also run under ``python -O``."""
    if not condition:
        raise AssertionError(message)


def check_close(actual, expected, message, atol=ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"{message}: unequal shapes")
    if not np.allclose(actual, expected, atol=atol, rtol=1e-10):
        error = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{message}: maximum discrepancy {error:g}")


def compare_tree(actual, expected, label):
    """Compare nested JSON summaries, including counts and null fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label}: expected an object")
        require(set(actual) == set(expected), f"{label}: unequal summary keys")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{label}.{key}")
    elif expected is None:
        require(actual is None, f"{label}: expected null")
    elif isinstance(expected, (list, tuple, int, float, np.number)):
        check_close(actual, expected, label)
    else:
        require(actual == expected, f"{label}: unequal values")


def reference_gap(policies, pairs, z, posterior, utility):
    """Direct policy-by-policy definition, separate from the optimized routine."""
    expected_utility = posterior @ utility.T
    clean_optimum = expected_utility.max(axis=1)
    rows = np.arange(len(z))
    return np.array([
        [float((clean_optimum - expected_utility[rows, policy['fn'](z + delta)]).mean())
         for _, delta in pairs]
        for policy in policies
    ])


def check_unit_cases():
    checks = []
    policies = sim.build_policy_library()
    require(len(policies) == 36, "Expected the paper's 36-policy library")
    z, posterior, _ = sim.generate_sample_bank(97, np.random.RandomState(400))
    pairs = [('none', np.zeros(4)), ('hide', np.array([-.8, -.2, 0., 0.])),
             ('fake', np.array([.1, .4, 1., 0.]))]
    for utility in (sim.U_D, ara.perceived_U('H'), ara.perceived_U('F')):
        check_close(sim.compute_gap_matrix(policies, pairs, z, posterior, utility),
                    reference_gap(policies, pairs, z, posterior, utility),
                    'Gap matrix versus direct definition', atol=1e-10)
    checks.append('All 36 policies: optimized and direct gaps agree for true and both perceived utilities.')

    game = np.array([[2., 0.], [0., 2.]])
    rho, value = sim.solve_lp_primal(game)
    prior, dual = sim.solve_lp(game)
    check_close(rho, [.5, .5], 'Known mixed defense')
    check_close(prior, [.5, .5], 'Known least-favorable prior')
    check_close([value, dual], [1., 1.], 'Known primal and dual values')
    rho, support, prior, history = sim.method_agnostic_support(game, [{}, {}])
    check_close((game.T @ rho).max(), 1., 'Gap-search mixed game value')
    require(len(support) == 2, 'The known game needs both attacks')
    require(history['W'][-1] - history['v'][-1] <= .1 + ATOL,
            'Gap-search final certificate')
    checks.append('Known game without a pure saddle: primal, dual and gap search recover the mixed value.')

    # A stop based on the pre-update best reply would miss attack 1 here.
    rho, support, prior, history, replies = ara.strategic_loop(
        game, [game, game], np.zeros(2), np.zeros(2))
    require(len(support) == 2 and len(history) == 2, 'Fresh replies must add the second attack')
    require(history[-1]['stable'] and set(replies) == {0, 1},
            'All tied final replies must be retained')
    check_close(rho, [.5, .5], 'Fresh-response mixed defense')
    tied, per_model = ara.response_set([np.ones((2, 3))], rho, np.zeros(3), np.zeros(3))
    require(tied == [0, 1, 2] and per_model == [[0, 1, 2]], 'Best-response tie handling')
    checks.append('Strategic regression: recomputed post-update replies and all tied best replies are retained.')

    # Distinct generator streams reproduce their own banks and do not reuse rows.
    seed = 20260721
    banks = [sim.generate_sample_bank(64, np.random.RandomState(seed + offset))
             for offset in (0, 1, 2)]
    repeated = sim.generate_sample_bank(64, np.random.RandomState(seed))
    for actual, expected in zip(banks[0], repeated):
        check_close(actual, expected, 'Deterministic sample-bank replay', atol=0)
    for i in range(len(banks)):
        for j in range(i + 1, len(banks)):
            require(not np.array_equal(banks[i][0], banks[j][0]), 'Different sample-bank seeds')
            rows_i = {tuple(row) for row in banks[i][0]}
            require(not any(tuple(row) in rows_i for row in banks[j][0]),
                    'No identical sampled observation rows across the small bank checks')
    checks.append('Sample-bank generation is reproducible and separate seed streams yield distinct observations.')

    zc, _, tc = sim.generate_sample_bank(2000, np.random.RandomState(seed + 1))
    attacks = sim.build_catalogue_100(np.random.RandomState(seed + 3), zc, tc)
    require(len(attacks) == 100, 'Original catalogue size')
    unseen = sim.build_unseen_attacks(seed, 100, attacks)
    deltas = np.array([attack['delta'] for attack in unseen])
    training_deltas = np.array([attack['delta'] for attack in attacks])
    distances = np.linalg.norm(deltas[:, None, :] - training_deltas[None, :, :], axis=2)
    require(len(unseen) == 100 and len({a['name'] for a in unseen}) == 100,
            'Held-out attack count and identifiers')
    require(float(np.max(np.abs(deltas))) <= sim.MAX_DELTA + 1e-12,
            'Held-out per-coordinate perturbation budget')
    require(float(distances.min()) >= .10 - 1e-12,
            'Held-out minimum distance from every training attack')
    replay = sim.build_unseen_attacks(seed, 100, attacks)
    check_close(deltas, [a['delta'] for a in replay], 'Held-out sampler replay', atol=0)
    checks.append('Held-out sampler: 100 reproducible attacks, coordinate budget and 0.10 minimum training distance.')
    return checks


def probability(value, label):
    require(np.isfinite(value) and -ATOL <= value <= 1 + ATOL, f'{label}: invalid probability')


def check_archives():
    main_dir = ROOT / 'results' / 'main'
    main = json.loads((main_dir / 'all_seeds.json').read_text())
    metadata = json.loads((main_dir / 'metadata.json').read_text())
    summary = json.loads((main_dir / 'aggregated.json').read_text())
    strategic = json.loads((ROOT / 'results' / 'strategic' / 'all_seeds.json').read_text())
    rows = strategic['all_seeds']
    checks = []
    sizes = {'train': 5000, 'test': 20000, 'calibration': 2000}
    require(len(main) == len(rows) == 20, 'The archived paper results must contain 20 paired runs')
    seeds = [record['seed'] for record in main]
    require(len(set(seeds)) == 20 and seeds == metadata['seeds'], 'Main seed identifiers')
    require(seeds == [record['seed'] for record in rows], 'Paired strategic seed identifiers')
    require(metadata['sample_sizes'] == sizes, 'Main sample sizes')
    require(metadata['exp3_selection_bank'] == 'train', 'Catalogue selection uses training data')
    require(metadata['exp1_ranking_bank'].startswith('test for both'), 'Descriptive ranking-bank metadata')
    require(strategic['metadata']['sample_sizes'] == sizes, 'Strategic sample sizes')
    require(strategic['metadata']['response_generation_bank'] == 'train', 'Strategic replies use training data')
    checks.append('All 20 paired seed IDs and archived training, calibration, descriptive-ranking and test contracts agree.')

    for record in main:
        seed = record['seed']
        for name, certificate in record['certificates'].items():
            prefix = f'{seed}/{name}'
            require(abs(certificate['primal_dual_gap']) < 1e-7, f'{prefix}: primal-dual gap')
            require(-ATOL <= certificate['pricing_gap'] <= .1 + 1e-7, f'{prefix}: stopping certificate')
            check_close(certificate['W_train'] - certificate['V_train'], certificate['pricing_gap'],
                        f'{prefix}: saved bound difference')
            require(0 < certificate['positive_prior_size'] <= certificate['discovered_size'] <= 100,
                    f'{prefix}: prior support accounting')
            require(0 < certificate['positive_defender_size'] <= 36, f'{prefix}: defender support accounting')
        for section in ('exp2', 'exp7'):
            methods = ('M1', 'M2', 'M3', 'M4', 'M5') if section == 'exp2' else ('M1', 'M2', 'M4', 'M5')
            for method in methods:
                mean = record[section][f'{method}_R_unsafe_cat']
                worst = record[section][f'{method}_R_unsafe_cat_worst']
                probability(mean, f'{seed}/{section}/{method}: mean unsafe')
                probability(worst, f'{seed}/{section}/{method}: worst unsafe')
                require(mean <= worst + ATOL, f'{seed}/{section}/{method}: mean exceeds maximum')
                require(record[section][f'{method}_R_mean'] <= record[section][f'{method}_R_max'] + ATOL,
                        f'{seed}/{section}/{method}: mean gap exceeds maximum')
        require(record['catalogue_train_selection']['rank_bank'] == 'train', f'{seed}: catalogue ranking bank')
        ranking = record['exp1']
        require(ranking['worst_cat_selection_metric'] == 'ground-truth conditional unsafe-action rate on test bank',
                f'{seed}: independent ranking target')
        check_close(ranking['worst_cat_unsafe_rate'], max(ranking['catastrophic_unsafe_rates'].values()),
                    f'{seed}: most-unsafe ranking target')
        check_close(ranking['worst_cat_unsafe_rate'],
                    ranking['catastrophic_unsafe_rates'][ranking['worst_cat_name']], f'{seed}: named target')
        for attack in record['calibration'] + record['twin_calibration']:
            require(np.max(np.abs(attack['delta'])) <= sim.MAX_DELTA + ATOL, f'{seed}: calibrated attack budget')
        history = record['exp4']
        check_close(history['gp_W'][-1] - history['gp_V'][-1], record['certificates']['main']['pricing_gap'],
                    f'{seed}: final gap-search curve and certificate')
        for method, metrics in record['heldout_attacks'].items():
            probability(metrics['R_unsafe_cat'], f'{seed}/{method}: unseen mean unsafe')
            probability(metrics['R_unsafe_cat_worst'], f'{seed}/{method}: unseen worst unsafe')
            require(metrics['R_unsafe_cat'] <= metrics['R_unsafe_cat_worst'] + ATOL,
                    f'{seed}/{method}: unseen mean exceeds maximum')
    checks.append('Every main seed: valid LP bounds, stopping residuals, support counts, calibration budgets and safety probabilities.')

    # The initial curve point has no restricted-game value. In memory this is
    # NaN; strict JSON stores it as null. Restore only that sentinel for the
    # original aggregation function, without changing the archived records.
    aggregation_rows = [dict(record, exp4=dict(record['exp4'], gp_V=[
        np.nan if value is None else value for value in record['exp4']['gp_V']
    ])) for record in main]
    recomputed = sim.json_safe(sim.aggregate(aggregation_rows))
    compare_tree(recomputed, summary, 'main summary')
    # Explicit independent mean/SD calculations also check the aggregation convention.
    for section, methods in [('exp2', ('M1', 'M2', 'M3', 'M4', 'M5')),
                             ('exp7', ('M1', 'M2', 'M4', 'M5'))]:
        for method in methods:
            for metric in ('R_max', 'R_mean', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'C_clean'):
                values = np.array([row[section][f'{method}_{metric}'] for row in main])
                check_close(summary[f'{section}_{method}_{metric}_mean'], values.mean(), 'Archived mean')
                check_close(summary[f'{section}_{method}_{metric}_std'], values.std(ddof=1), 'Archived sample SD')
    checks.append(f'All {len(summary)} archived main summary entries recompute from the 20 raw runs; means and sample SDs agree.')

    main_by_seed = {record['seed']: record for record in main}
    for record in rows:
        seed = record['seed']
        catalogue = record['catalogue']
        require(len(catalogue) == 100, f'{seed}: strategic catalogue size')
        require(np.max(np.abs([attack['delta'] for attack in catalogue])) <= sim.MAX_DELTA + ATOL,
                f'{seed}: strategic attack budget')
        for name, result in record['generators'].items():
            prefix = f'{seed}/{name}'
            support = result['support']
            prior, rho = np.array(result['prior']), np.array(result['rho'])
            require(len(set(support)) == len(support) == result['sz'] == len(prior), f'{prefix}: unique support')
            require(all(0 <= j < len(catalogue) for j in support), f'{prefix}: support indices')
            require(len(rho) == 36 and min(prior) >= -ATOL and min(rho) >= -ATOL, f'{prefix}: mixture weights')
            check_close([prior.sum(), rho.sum()], [1., 1.], f'{prefix}: normalized mixtures')
            positive = [j for j, weight in zip(support, prior) if weight > ara.POS_TOL]
            require(positive == result['positive_support'], f'{prefix}: positive support list')
            require(len(positive) == result['positive_support_size'], f'{prefix}: positive support count')
            for family, key in [('H', 'h'), ('F', 'f')]:
                weight = sum(p for j, p in zip(support, prior) if catalogue[j]['family'] == family)
                check_close(result[key], weight, f'{prefix}: {family} prior weight')
            require(bool(result['both']) == (result['h'] > ara.POS_TOL and result['f'] > ara.POS_TOL),
                    f'{prefix}: both-family indicator')
            require(result['R_E_test'] <= result['Rfull'] + ATOL, f'{prefix}: restricted/full test bound')
            if name != 'L0':
                final = result['history'][-1]
                require(result['stable'] and final['stable'], f'{prefix}: response-set stability')
                require(set(result['final_response_set']).issubset(support), f'{prefix}: final reply coverage')
                require(final['response_set'] == result['final_response_set'], f'{prefix}: fresh final replies')
                require(len(result['history']) == result['iters'], f'{prefix}: iteration accounting')
                for step in result['history']:
                    require(step['primal_dual_gap'] < 1e-7, f'{prefix}: intermediate LP duality')
                check_close(final['R_strategic_train'], result['R_strategic_train'], f'{prefix}: final training response risk')
                require(result['Rstrat'] <= result['Rfull'] + ATOL, f'{prefix}: final-response/full test bound')
        # The background catalogues use separate streams. Equality below is an
        # empirical property of these saved runs, not an assumed shared catalogue.
        l0 = record['generators']['L0']
        matching = main_by_seed[seed]['exp7']
        check_close(l0['Rfull'], matching['M5_R_max'], f'{seed}: empirical main/strategic test equivalence')
        check_close(l0['Cclean'], matching['M5_C_clean'], f'{seed}: empirical clean-cost equivalence')
    compare_tree(ara.aggregate(rows), strategic['comparison_20seed'], 'strategic summary')
    compare_tree(ara.stats([row['generators']['joint']['iters'] for row in rows]),
                 strategic['joint_iters_20seed'], 'joint iteration summary')
    predicates = {
        'L0_H': lambda row: row['initial_choice_families']['L0'] == 'H',
        'L1H_H': lambda row: row['initial_choice_families']['tau_H'] == 'H',
        'L1F_H': lambda row: row['initial_choice_families']['tau_F'] == 'H',
        'L0_vs_L1H_div': lambda row: row['initial_choices']['L0'] != row['initial_choices']['tau_H'],
        'L0_vs_L1F_div': lambda row: row['initial_choices']['L0'] != row['initial_choices']['tau_F'],
        'L1H_vs_L1F_div': lambda row: row['initial_choices']['tau_H'] != row['initial_choices']['tau_F'],
    }
    divergence = {key: [sum(predicate(row) for row in rows),
                        sum(predicate(row) for row in rows) / len(rows)]
                  for key, predicate in predicates.items()}
    compare_tree(divergence, strategic['divergence_20seed'], 'initial-choice counts')
    checks.append('Every strategic seed: normalized priors/defenses, family weights, fresh final-reply coverage and LP consistency.')
    checks.append('Strategic means, sample SDs, initial-choice counts and iteration summaries recompute from all raw runs.')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit-only', action='store_true', help='Skip archived-result checks')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated' / 'verification.json',
                        help='Verification JSON output path')
    args = parser.parse_args()
    started = time.perf_counter()
    checks = check_unit_cases()
    if not args.unit_only:
        checks.extend(check_archives())
    result = {'status': 'passed', 'mode': 'unit-only' if args.unit_only else 'core-and-archives',
              'check_count': len(checks), 'checks': checks,
              'scope': 'Core regression tests and archived-result consistency; no full experiment rerun.',
              'runtime_seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for check in checks:
        print('PASS:', check)
    print(f"{len(checks)} checks passed in {result['runtime_seconds']:.2f}s")
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
