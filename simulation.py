#!/usr/bin/env python3
"""Agnostic ARA: decision-level sensor-attack experiments.

Runs attack ranking, defense comparison, attack selection, hide/fake comparison,
and an independent held-out attack diagnostic.
Uses separate calibration, training, and testing banks; both descriptive rankings
use identical test data, while defense fitting and attack selection use train only.
Policies are evaluated as the mixed saddle-point strategies returned by the LP.
The finite catalog is fully precomputed; sparsity is not a query-cost claim.
"""
import numpy as np
from numpy import exp, log
from scipy.optimize import linprog
import warnings, time, os, csv, json
from collections import defaultdict
warnings.filterwarnings('ignore')

# Domain constants
N_STATES, N_SENSORS, N_ACTIONS = 3, 4, 4
STATE_NAMES  = ["clear", "ped_far", "imminent"]
SENSOR_NAMES = ["Camera", "LiDAR", "Radar", "GPS/Map"]
ACTION_NAMES = ["go", "slow", "brake", "emergency"]

PRIOR  = np.array([0.60, 0.25, 0.15])
SIGMA_S = 0.35
MAX_DELTA = 1.5

MU = np.array([
    [0.1, 0.1, 0.1, 0.2],  # clear
    [0.7, 0.5, 0.3, 0.7],  # ped_far
    [0.9, 0.9, 0.8, 0.8],  # imminent
])

U_D = np.array([
    [ 10,   0, -200],   # go
    [  0,   8, -100],   # slow
    [-20,   5,   15],   # brake
    [-30, -10,   20],   # emergency
], dtype=float)


# ============================================================================
# Core functions
# ============================================================================
def posterior_3state(z, sigma=SIGMA_S):
    z = np.atleast_2d(z)
    diff = z[:, None, :] - MU[None, :, :]
    log_lik = -0.5 * np.sum(diff**2, axis=2) / sigma**2
    log_post = log_lik + log(PRIOR)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    post = exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post

def best_action_3state(z, U=U_D, sigma=SIGMA_S):
    post = posterior_3state(z, sigma)
    return np.argmax(post @ U.T, axis=1)

def generate_sample_bank(N, rng):
    theta = rng.choice(N_STATES, size=N, p=PRIOR)
    z_clean = rng.normal(MU[theta], SIGMA_S)
    post = posterior_3state(z_clean)
    return z_clean, post, theta

def build_policy_library():
    c_grid = [1.0, 2.0, 4.0]
    r_grid = [1.0, 2.0]
    f_grid = [1.0, 2.0, 4.0]
    d_grid = [1.0, 2.0]
    policies = []
    def bayes_fn(z): return best_action_3state(z, U_D)
    policies.append({'fn': bayes_fn, 'name': 'bayes', 'collision': 1.0,
                     'residual': 1.0, 'false_alarm': 1.0, 'far': 1.0, 'U': U_D.copy()})
    for c in c_grid:
        for r in r_grid:
            for f in f_grid:
                for d in d_grid:
                    if c == 1.0 and r == 1.0 and f == 1.0 and d == 1.0:
                        continue
                    Up = U_D.copy()
                    Up[0, 2] *= c; Up[1, 2] *= r
                    Up[2, 1] *= c; Up[3, 1] *= c
                    Up[2, 0] *= f; Up[3, 0] *= f
                    Up[0, 1] *= d; Up[1, 1] *= d
                    def mk(Up_): return lambda z, U=Up_: best_action_3state(z, U)
                    policies.append({'fn': mk(Up), 'name': f"σ(c={c},r={r},f={f},d={d})",
                                     'collision': c, 'residual': r,
                                     'false_alarm': f, 'far': d, 'U': Up.copy()})
    return policies

def get_attack_templates():
    return [
        ("f_cam_blur",    np.array([-1.3,  0.0,  0.0,  0.0]), 'hide'),
        ("f_lidar_drift", np.array([ 0.0, -1.3,  0.0,  0.0]), 'hide'),
        ("f_radar_jam",   np.array([ 0.0,  0.0, -1.0,  0.0]), 'hide'),
        ("f_gps_spoof",   np.array([ 0.0,  0.0,  0.0,  1.2]), 'fake'),
        ("f_cam_lidar_hide",  np.array([-1.0, -1.0,  0.0,  0.0]), 'hide'),
        ("f_radar_gps_hide",  np.array([ 0.0,  0.0, -1.0, -1.0]), 'hide'),
        ("f_triple_hide",     np.array([-1.0, -1.0, -1.0,  0.0]), 'hide'),
        ("f_full_hide",       np.array([-1.2, -1.2, -1.2, -1.2]), 'hide'),
        ("f_cam_lidar_fake",  np.array([ 1.0,  1.0,  0.0,  0.0]), 'fake'),
        ("f_radar_gps_fake",  np.array([ 0.0,  0.0,  1.0,  1.0]), 'fake'),
        ("f_triple_fake",     np.array([ 1.0,  1.0,  1.0,  0.0]), 'fake'),
        ("f_full_fake",       np.array([ 1.2,  1.2,  1.2,  1.2]), 'fake'),
    ]


# ============================================================================
# Attack catalogues
# ============================================================================
def build_catalogue_100(rng, z_calib, theta_calib):
    templates = get_attack_templates()
    attacks = []
    # Nuisance (40)
    for i in range(40):
        direction = rng.normal(0, 1, N_SENSORS)
        direction /= np.linalg.norm(direction) + 1e-10
        scale = rng.uniform(0.05, 0.35)
        delta = np.clip(direction * scale, -MAX_DELTA, MAX_DELTA)
        attacks.append({'name': f"nuisance_{i:03d}", 'delta': delta, 'category': 'nuisance'})
    # False-alarm (20)
    fake_t = [t for t in templates if t[2] == 'fake']
    for i in range(20):
        t_name, t_delta, t_dir = fake_t[rng.choice(len(fake_t))]
        scale = rng.uniform(0.15, 0.45)
        noise = rng.normal(0, 0.05, N_SENSORS)
        delta = np.clip(t_delta * scale + noise, -MAX_DELTA, MAX_DELTA)
        attacks.append({'name': f"false_alarm_{i:03d}", 'delta': delta, 'category': 'false_alarm'})
    # Moderate hiding (35)
    hide_t = [t for t in templates if t[2] == 'hide']
    for i in range(35):
        t_name, t_delta, t_dir = hide_t[rng.choice(len(hide_t))]
        scale = rng.uniform(0.2, 0.5)
        noise = rng.normal(0, 0.05, N_SENSORS)
        delta = np.clip(t_delta * scale + noise, -MAX_DELTA, MAX_DELTA)
        attacks.append({'name': f"moderate_hide_{i:03d}", 'delta': delta, 'category': 'moderate'})
    # Catastrophic (5): severity-matched by go-rate target
    z_cal_imm = z_calib[theta_calib == 2]
    if len(z_cal_imm) < 50: z_cal_imm = z_calib
    cata_specs = [
        ("f_cata_cam",       np.array([-1.0,  0.0,  0.0,  0.0]), 0.55),
        ("f_cata_lidar",     np.array([ 0.0, -1.0,  0.0,  0.0]), 0.60),
        ("f_cata_dual_cl",   np.array([-1.0, -1.0,  0.0,  0.0]), 0.65),
        ("f_cata_dual_rg",   np.array([ 0.0,  0.0, -1.0, -1.0]), 0.65),
        ("f_cata_triple",    np.array([-1.0, -1.0, -1.0,  0.0]), 0.75),
    ]
    for c_name, c_dir, target_go_rate in cata_specs:
        lo, hi = 0.1, MAX_DELTA / max(abs(c_dir))
        best_scale = hi * 0.5
        for _ in range(25):
            mid = (lo + hi) / 2
            test_delta = np.clip(c_dir * mid, -MAX_DELTA, MAX_DELTA)
            actions = best_action_3state(z_cal_imm + test_delta, U_D)
            go_rate = np.mean((actions == 0) | (actions == 1))
            if go_rate < target_go_rate: lo = mid
            else: hi = mid
            best_scale = mid
        c_delta = np.clip(c_dir * best_scale, -MAX_DELTA, MAX_DELTA)
        achieved_actions = best_action_3state(z_cal_imm + c_delta, U_D)
        achieved = float(np.mean(achieved_actions < 2))
        attacks.append({'name': c_name, 'delta': c_delta, 'category': 'catastrophic',
                        'calibration_target_unsafe': target_go_rate,
                        'calibration_unsafe': achieved,
                        'calibration_go_only': float(np.mean(achieved_actions == 0)),
                        'calibration_n_imminent': len(z_cal_imm),
                        'at_budget': bool(np.max(np.abs(c_delta)) >= MAX_DELTA - 1e-5)})
    return attacks[:100]


def build_extra_ordinary(rng, n_extra):
    """Generate n_extra ordinary (non-catastrophic) attacks to keep a catalogue fixed at 100."""
    templates = get_attack_templates()
    fake_t = [t for t in templates if t[2] == 'fake']
    hide_t = [t for t in templates if t[2] == 'hide']
    extra = []
    for i in range(n_extra):
        kind = rng.choice(['nuisance', 'false_alarm', 'moderate'])
        if kind == 'nuisance':
            direction = rng.normal(0, 1, N_SENSORS)
            direction /= np.linalg.norm(direction) + 1e-10
            delta = np.clip(direction * rng.uniform(0.05, 0.35), -MAX_DELTA, MAX_DELTA)
        elif kind == 'false_alarm':
            t_delta = fake_t[rng.choice(len(fake_t))][1]
            delta = np.clip(t_delta * rng.uniform(0.15, 0.45) + rng.normal(0, 0.05, N_SENSORS), -MAX_DELTA, MAX_DELTA)
        else:
            t_delta = hide_t[rng.choice(len(hide_t))][1]
            delta = np.clip(t_delta * rng.uniform(0.2, 0.5) + rng.normal(0, 0.05, N_SENSORS), -MAX_DELTA, MAX_DELTA)
        extra.append({'name': f"extra_{i:03d}", 'delta': delta, 'category': kind})
    return extra


def build_twin_catalogue(rng, policies, z_calib, theta_calib, zt, pt, target_gap=22.0):
    """Hide/fake catalog calibrated on the training sample bank.
    Main five-attack calibration remains on its independent calibration bank.
    """
    zc_imm = z_calib[theta_calib == 2]
    zc_clr = z_calib[theta_calib == 0]
    if len(zc_imm) < 50: zc_imm = z_calib
    if len(zc_clr) < 50: zc_clr = z_calib
    
    bg = []
    # Nuisance (30)
    for i in range(30):
        d = rng.normal(0, 1, N_SENSORS)
        d /= np.linalg.norm(d) + 1e-10
        delta = np.clip(d * rng.uniform(0.05, 0.35), -MAX_DELTA, MAX_DELTA)
        bg.append({'name': f"nuisance_{i:03d}", 'delta': delta, 'category': 'nuisance', 'family': None})
    
    FAKE_T = [np.array([0.9,0,0,0]), np.array([0,0.9,0,0]), np.array([0.9,0.9,0,0]),
              np.array([0,0,0.9,0.9]), np.array([0.9,0.9,0.9,0]), np.array([0.9,0.9,0.9,0.9])]
    for i in range(30):
        t_d = FAKE_T[rng.choice(len(FAKE_T))]
        scale = rng.uniform(0.15, 0.45)
        delta = np.clip(t_d * scale + rng.normal(0, 0.05, N_SENSORS), -MAX_DELTA, MAX_DELTA)
        bg.append({'name': f"false_alarm_{i:03d}", 'delta': delta, 'category': 'false_alarm', 'family': None})
    
    HIDE_T = [np.array([-0.9,0,0,0]), np.array([0,-0.9,0,0]), np.array([-0.9,-0.9,0,0]),
              np.array([0,0,-0.9,-0.9]), np.array([-0.9,-0.9,-0.9,0]), np.array([-0.9,-0.9,-0.9,-0.9])]
    for i in range(34):
        t_d = HIDE_T[rng.choice(len(HIDE_T))]
        scale = rng.uniform(0.2, 0.5)
        delta = np.clip(t_d * scale + rng.normal(0, 0.05, N_SENSORS), -MAX_DELTA, MAX_DELTA)
        bg.append({'name': f"mod_hide_{i:03d}", 'delta': delta, 'category': 'moderate', 'family': None})
    
    # H-family (hide, 3 attacks)
    h_dirs = [np.array([-1.0,0,0,0]), np.array([-1.0,-1.0,0,0]), np.array([-1.0,-1.0,-1.0,0])]
    h_names = ["f_H_cam_hide", "f_H_dual_hide", "f_H_triple_hide"]
    h_attacks = []
    for h_dir, h_name in zip(h_dirs, h_names):
        lo, hi = 0.5, 2.0
        best = hi * 0.7
        for _ in range(20):
            mid = (lo + hi) / 2
            delta = np.clip(h_dir * mid, -MAX_DELTA, MAX_DELTA)
            gaps = compute_gap_vector_single(policies[0]['fn'], delta, zt, pt)
            if gaps < target_gap: lo = mid
            else: hi = mid
            best = mid
        delta = np.clip(h_dir * best, -MAX_DELTA, MAX_DELTA)
        h_attacks.append({'name': h_name, 'delta': delta, 'category': 'catastrophic', 'family': 'H'})
    
    # F-family (fake, 3 attacks)
    f_dirs = [np.array([1.5,0,0,0]), np.array([1.5,1.5,0,0]), np.array([1.5,1.5,1.5,0])]
    f_names = ["f_F_cam_fake", "f_F_dual_fake", "f_F_triple_fake"]
    f_attacks = []
    for f_dir, f_name in zip(f_dirs, f_names):
        lo, hi = 0.3, 1.5
        best = hi * 0.5
        for _ in range(20):
            mid = (lo + hi) / 2
            delta = np.clip(f_dir * mid, -MAX_DELTA, MAX_DELTA)
            gaps = compute_gap_vector_single(policies[0]['fn'], delta, zt, pt)
            if gaps < target_gap: lo = mid
            else: hi = mid
            best = mid
        delta = np.clip(f_dir * best, -MAX_DELTA, MAX_DELTA)
        f_attacks.append({'name': f_name, 'delta': delta, 'category': 'catastrophic', 'family': 'F'})
    
    return bg + h_attacks + f_attacks


def compute_gap_vector_single(policy_fn, delta, z, post):
    """Gap for a single attack."""
    N = len(z)
    eu = post @ U_D.T
    opt_val = eu.max(axis=1)
    a_corr = policy_fn(z + delta)
    return float((opt_val - eu[np.arange(N), a_corr]).mean())


# ============================================================================
# Gap and metric computation
# ============================================================================
def compute_gap_matrix(policies, atk_pairs, z, post, utility=None, posterior_fn=None):
    """Full empirical gap matrix; posterior computations shared across policies.

    This is deliberately full precomputation, not an oracle-query saving claim.
    Optional domain arguments are used by the unified five-state sensitivity run.
    """
    utility = U_D if utility is None else utility
    posterior_fn = posterior_3state if posterior_fn is None else posterior_fn
    G = np.zeros((len(policies), len(atk_pairs)))
    eu = post @ utility.T
    opt_val = eu.max(axis=1)
    rows = np.arange(len(z))
    for j, (_, delta) in enumerate(atk_pairs):
        attacked_post = posterior_fn(z + delta)
        for i, policy in enumerate(policies):
            a_corr = np.argmax(attacked_post @ policy['U'].T, axis=1)
            G[i, j] = float((opt_val - eu[rows, a_corr]).mean())
    return G


def compute_prediction_loss(attacks, z, post):
    losses = {}
    post_z = posterior_3state(z) if post is None else post
    for atk in attacks:
        post_c = posterior_3state(z + atk['delta'])
        tv = np.abs(post_z - post_c).sum(axis=1).mean() / 2
        losses[atk['name']] = float(tv)
    return losses

def compute_unsafe_rate(pol_fn, attacks, z_imminent):
    results = {}
    for atk in attacks:
        z_corr = z_imminent + atk['delta']
        actions = pol_fn(z_corr)
        unsafe = np.mean((actions == 0) | (actions == 1))
        results[atk['name']] = float(unsafe)
    return results

def compute_clean_cost(pol_fn, z, post):
    eu = post @ U_D.T
    opt = eu.max(axis=1)
    a = pol_fn(z)
    return float((opt - eu[np.arange(len(z)), a]).mean())

def compute_false_emergency_rate(pol_fn, z_clear):
    actions = pol_fn(z_clear)
    return float(np.mean(actions == 3))


# ============================================================================
# LP solver
# ============================================================================
def solve_lp(G):
    nP, nC = G.shape
    if nC == 0: return np.array([]), 0.0
    c = np.zeros(nC + 1); c[0] = -1.0
    A_ub = np.hstack([np.ones((nP, 1)), -G])
    A_eq = np.zeros((1, nC + 1)); A_eq[0, 1:] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(nP), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(None, None)] + [(0, 1)] * nC, method='highs')
    if res.success:
        pi = np.maximum(res.x[1:], 0); s = pi.sum()
        return (pi / s if s > 0 else np.ones(nC) / nC), -res.fun
    raise RuntimeError(f"LF-prior LP failed: {res.message}")


def solve_lp_primal(G):
    """Solve the primal minimax LP: min_rho max_j (G^T rho)_j.
    Returns the mixed defender strategy rho* and the game value."""
    nP, nC = G.shape
    if nC == 0:
        return np.ones(nP) / nP, 0.0
    # variables x = [v, rho_0, ..., rho_{nP-1}]; minimize v
    c = np.zeros(nP + 1); c[0] = 1.0
    # constraint per attack j: sum_r G[r,j] rho_r - v <= 0
    A_ub = np.hstack([-np.ones((nC, 1)), G.T])
    b_ub = np.zeros(nC)
    # sum rho = 1
    A_eq = np.zeros((1, nP + 1)); A_eq[0, 1:] = 1.0
    b_eq = [1.0]
    bounds = [(None, None)] + [(0, 1)] * nP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if res.success:
        rho = np.maximum(res.x[1:], 0); s = rho.sum()
        rho = rho / s if s > 0 else np.ones(nP) / nP
        return rho, float(res.fun)
    raise RuntimeError(f"Primal LP failed: {res.message}")


# ============================================================================
# Methods
# ============================================================================
def method_clean(policies, z, post):
    best_i, best_c = 0, float('inf')
    for i, p in enumerate(policies):
        c = compute_clean_cost(p['fn'], z, post)
        if c < best_c: best_i, best_c = i, c
    return best_i

def method_uniform(G_train):
    return int(np.argmin(G_train.mean(axis=1)))

def method_prediction_topk(policies, attacks, atk_pairs, z, post, G_train, pred_losses, K=5):
    """Prediction-loss-guided support selection + mixed minimax over the selected support (M3)."""
    sorted_atks = sorted(pred_losses.items(), key=lambda x: x[1], reverse=True)
    top_names = set(n for n, _ in sorted_atks[:K])
    top_indices = [j for j, a in enumerate(attacks) if a['name'] in top_names]
    if len(top_indices) == 0:
        rho = np.zeros(len(policies)); rho[0] = 1.0
        return rho, []
    G_sub = G_train[:, top_indices]
    rho, val = solve_lp_primal(G_sub)  # mixed minimax over the prediction-loss top-K support
    return rho, top_indices

def method_full_minimax(G_train):
    """Mixed-strategy full-catalogue minimax (M4). Returns (rho*, value)."""
    return solve_lp_primal(G_train)

def method_agnostic_support(G_train, policies, max_iter=50, eps=0.1):
    """
    Mixed-strategy-consistent agnostic support generation (M5).
    Returns the saddle-point mixed strategy rho* over the discovered support.
    Records V(E_t) (primal LP value, lower bound) and W_t (worst on full catalogue, upper bound).
    """
    nP = G_train.shape[0]
    discovered = []
    rho = np.zeros(nP); rho[0] = 1.0  # start from the clean (pure) policy
    history = {'iter': [], 'size': [], 'v': [], 'W': []}
    prev_discovered = set()

    for t in range(max_iter):
        gap_vec = G_train.T @ rho                 # gap of mixed rho against each attack
        worst_j = int(np.argmax(gap_vec))
        worst_gap = float(gap_vec[worst_j])       # worst over FULL catalogue

        # Convergence: worst attack already in support AND gap near LP value
        if worst_j in discovered and len(discovered) > 0:
            G = G_train[:, discovered]
            _, v = solve_lp_primal(G)
            if worst_gap <= v + eps:
                break

        if worst_j not in discovered:
            discovered.append(worst_j)

        G = G_train[:, discovered]
        rho, v = solve_lp_primal(G)              # mixed strategy over support (primal)
        pi_star, v_dual = solve_lp(G)            # LF prior (dual); v == v_dual by LP duality

        history['iter'].append(t)
        history['size'].append(len(discovered))
        history['v'].append(v)                     # V(E_t) = mixed minimax value on E_t
        history['W'].append(float((G_train.T @ rho).max()))  # W_t on full catalogue

        curr_discovered = set(discovered)
        if curr_discovered == prev_discovered and t > 0:
            break
        prev_discovered = curr_discovered

    if len(discovered) > 0:
        Gf = G_train[:, discovered]
        rho, v = solve_lp_primal(Gf)
        pi_star, _ = solve_lp(Gf)
    else:
        rho = np.zeros(nP); rho[0] = 1.0
        pi_star = np.array([])

    if len(discovered) and float((G_train.T @ rho).max()) > v + eps + 1e-8:
        raise RuntimeError('Support generation stopped without its pricing-gap certificate')
    return rho, discovered, pi_star, history


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_method(pol_idx, policies, G_test, attacks, cat_indices, z_test, post_test, theta_test):
    pol_fn = policies[pol_idx]['fn']
    cat_gaps = G_test[pol_idx]
    
    R_max = float(cat_gaps.max())
    R_mean = float(cat_gaps.mean())
    nA = len(attacks)
    R_cvar95 = float(np.sort(cat_gaps)[-max(1, int(0.05 * nA)):].mean())
    
    R_cat_max = float(cat_gaps[cat_indices].max()) if len(cat_indices) else 0.0
    R_cat_mean = float(cat_gaps[cat_indices].mean()) if len(cat_indices) else 0.0
    
    z_imm = z_test[theta_test == 2]
    if len(z_imm) > 0:
        unsafe = compute_unsafe_rate(pol_fn, [attacks[j] for j in cat_indices], z_imm)
        R_unsafe_cat = np.mean(list(unsafe.values())) if unsafe else 0.0
        R_unsafe_cat_worst = max(unsafe.values()) if unsafe else 0.0
    else:
        R_unsafe_cat = 0.0
        R_unsafe_cat_worst = 0.0
    
    z_clear = z_test[theta_test == 0]
    R_false_emergency = compute_false_emergency_rate(pol_fn, z_clear) if len(z_clear) > 0 else 0.0
    
    C_clean = compute_clean_cost(pol_fn, z_test, post_test)
    
    return {
        'policy': policies[pol_idx]['name'],
        'R_max': R_max, 'R_mean': R_mean, 'R_cvar95': R_cvar95,
        'R_cat_max': R_cat_max, 'R_cat_mean': R_cat_mean,
        'R_unsafe_cat': R_unsafe_cat, 'R_unsafe_cat_worst': R_unsafe_cat_worst, 'R_false_emergency': R_false_emergency,
        'C_clean': C_clean
    }


def evaluate_mixed(rho, policies, G_test, attacks, cat_indices, z_test, post_test, theta_test):
    """Evaluate a mixed defender strategy rho over the policy library."""
    gap_vec = G_test.T @ rho
    R_max = float(gap_vec.max())
    R_mean = float(gap_vec.mean())
    nA = len(attacks)
    R_cvar95 = float(np.sort(gap_vec)[-max(1, int(0.05 * nA)):].mean())
    R_cat_max = float(gap_vec[cat_indices].max()) if len(cat_indices) else 0.0
    R_cat_mean = float(gap_vec[cat_indices].mean()) if len(cat_indices) else 0.0

    z_imm = z_test[theta_test == 2]
    R_unsafe_cat = 0.0
    unsafe_by_attack = np.zeros(len(cat_indices))
    if len(z_imm) > 0:
        for r, pr in enumerate(rho):
            if pr <= 1e-9: continue
            unsafe = compute_unsafe_rate(policies[r]['fn'], [attacks[j] for j in cat_indices], z_imm)
            unsafe_by_attack += pr * np.array(list(unsafe.values()))
    R_unsafe_cat = float(unsafe_by_attack.mean()) if len(unsafe_by_attack) else 0.0
    R_unsafe_cat_worst = float(unsafe_by_attack.max()) if len(unsafe_by_attack) else 0.0

    z_clear = z_test[theta_test == 0]
    R_false_emergency = 0.0
    if len(z_clear) > 0:
        for r, pr in enumerate(rho):
            if pr <= 1e-9: continue
            R_false_emergency += pr * compute_false_emergency_rate(policies[r]['fn'], z_clear)

    C_clean = 0.0
    for r, pr in enumerate(rho):
        if pr <= 1e-9: continue
        C_clean += pr * compute_clean_cost(policies[r]['fn'], z_test, post_test)

    return {
        'R_max': R_max, 'R_mean': R_mean, 'R_cvar95': R_cvar95,
        'R_cat_max': R_cat_max, 'R_cat_mean': R_cat_mean,
        'R_unsafe_cat': R_unsafe_cat, 'R_unsafe_cat_worst': R_unsafe_cat_worst, 'R_false_emergency': R_false_emergency,
        'C_clean': C_clean, 'rho': rho
    }


# ============================================================================
# Single-seed experiment runner
# ============================================================================
def compute_exp1_results(policies, attacks, z_test, post_test, theta_test, clean_gaps=None):
    """Rank metrics on one test bank; select the severe target by independent safety.

    The target attack is the catastrophic-label member with greatest empirical
    P(go OR slow | imminent), not the member favored by either ranking metric.
    Equal unsafe rates are resolved by the fixed catalogue order.
    """
    if clean_gaps is None:
        clean_gaps = compute_gap_matrix(policies[:1], [(a['name'], a['delta']) for a in attacks], z_test, post_test)[0]
    predictions = compute_prediction_loss(attacks, z_test, post_test)
    tv = np.array([predictions[a['name']] for a in attacks])
    cat_indices = [j for j,a in enumerate(attacks) if a['category']=='catastrophic']
    cat_set = set(cat_indices)
    unsafe = compute_unsafe_rate(policies[0]['fn'], [attacks[j] for j in cat_indices], z_test[theta_test==2])
    exp1 = {}
    for k in [3,5,10]:
        exp1[f'top{k}_gap_recall'] = len(set(np.argsort(clean_gaps)[-k:]) & cat_set) / max(len(cat_set),1)
        exp1[f'top{k}_pred_recall'] = len(set(np.argsort(tv)[-k:]) & cat_set) / max(len(cat_set),1)
    worst_cat = max(cat_indices, key=lambda j: unsafe[attacks[j]['name']])
    ranked_gap = np.argsort(np.argsort(clean_gaps))
    ranked_pred = np.argsort(np.argsort(tv))
    exp1['worst_cat_rank_gap'] = int(len(attacks)-ranked_gap[worst_cat])
    exp1['worst_cat_rank_pred'] = int(len(attacks)-ranked_pred[worst_cat])
    exp1['worst_cat_name'] = attacks[worst_cat]['name']
    exp1['worst_cat_unsafe_rate'] = unsafe[attacks[worst_cat]['name']]
    exp1['worst_cat_selection_metric'] = 'ground-truth conditional unsafe-action rate on test bank'
    exp1['catastrophic_unsafe_rates'] = unsafe
    return exp1


def run_single_seed(seed):
    """Run all experiments for a single seed. Returns dict of results."""
    rng = np.random.RandomState(seed)
    cr = np.random.RandomState(seed + 1)
    tr = np.random.RandomState(seed + 2)
    ar = np.random.RandomState(seed + 3)
    
    NT, NE, NC = 5000, 20000, 2000
    
    zc, pc, tc = generate_sample_bank(NC, cr)
    zt, pt, tt = generate_sample_bank(NT, rng)
    ze, pe, te = generate_sample_bank(NE, tr)
    
    policies = build_policy_library()
    
    # Build 100-attack catalogue
    attacks = build_catalogue_100(ar, zc, tc)
    atk_pairs = [(a['name'], a['delta']) for a in attacks]
    cat_names = [a['name'] for a in attacks if a['category'] == 'catastrophic']
    
    # Precompute gaps
    G_train = compute_gap_matrix(policies, atk_pairs, zt, pt)
    G_test = compute_gap_matrix(policies, atk_pairs, ze, pe)
    pred_losses = compute_prediction_loss(attacks, zt, pt)
    
    # Full minimax on TRAIN (mixed-strategy game value; used for Exp 4 comparisons)
    _, full_minimax_train_val = solve_lp_primal(G_train)
    
    cat_labels = np.array([1 if a['category'] == 'catastrophic' else 0 for a in attacks])
    cat_indices = np.where(cat_labels == 1)[0].tolist()
    
    results = {'seed': seed, 'full_minimax_train': full_minimax_train_val}
    
    # --- Exp 1: Metric validity; target selected by independent safety metric ---
    results['exp1'] = compute_exp1_results(policies, attacks, ze, pe, te, clean_gaps=G_test[0])

    # --- Exp 2: 5-method comparison ---
    m1_idx = method_clean(policies, zt, pt)
    m2_idx = method_uniform(G_train)
    m3_rho, m3_top = method_prediction_topk(policies, attacks, atk_pairs, zt, pt, G_train, pred_losses, K=5)
    m4_rho, m4_val = method_full_minimax(G_train)
    m5_rho, m5_sup, m5_pi, m5_hist = method_agnostic_support(G_train, policies)
    
    exp2 = {}
    # M1/M2 are pure policies; M3/M4/M5 are mixed minimax over their selected support
    for lb, ix in [('M1', m1_idx), ('M2', m2_idx)]:
        ev = evaluate_method(ix, policies, G_test, attacks, cat_indices, ze, pe, te)
        for k in ['R_max', 'R_mean', 'R_cvar95', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'R_false_emergency', 'C_clean']:
            exp2[f'{lb}_{k}'] = ev[k]
        exp2[f'{lb}_policy'] = ev['policy']
    for lb, rho in [('M3', m3_rho), ('M4', m4_rho), ('M5', m5_rho)]:
        ev = evaluate_mixed(rho, policies, G_test, attacks, cat_indices, ze, pe, te)
        for k in ['R_max', 'R_mean', 'R_cvar95', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'R_false_emergency', 'C_clean']:
            exp2[f'{lb}_{k}'] = ev[k]
        exp2[f'{lb}_policy'] = 'mixed'
    
    exp2['M5_support_size'] = len(m5_sup)
    exp2['M5_support_cat_recall'] = len(set(m5_sup) & set(cat_indices)) / max(len(cat_indices), 1)
    results['exp2'] = exp2
    
    # --- Exp 3: Rarity sweep (vary TRAINING catalogue, evaluate on FULL 5-cat set) ---
    # Sort catastrophic attacks by gap against clean policy → severity ordering (nested)
    cat_gaps_sorted = sorted([(j, G_train[0, j]) for j in cat_indices], key=lambda x: -x[1])
    cat_ordered = [j for j, _ in cat_gaps_sorted]
    non_cat_indices = [j for j in range(len(attacks)) if j not in cat_indices]
    
    # Helper: evaluate a policy against full 5-cat test set (mean + worst-case unsafe rate)
    def eval_full_cat_unsafe(pol_idx):
        pol_fn = policies[pol_idx]['fn']
        z_imm = ze[te == 2]
        if len(z_imm) == 0:
            return {'mean': 0.0, 'worst': 0.0, 'per_attack': []}
        per_cat = []
        for j in cat_indices:
            z_corr = z_imm + attacks[j]['delta']
            actions = pol_fn(z_corr)
            unsafe = float(np.mean((actions == 0) | (actions == 1)))
            per_cat.append(unsafe)
        return {'mean': float(np.mean(per_cat)), 'worst': float(np.max(per_cat)),
                'per_attack': per_cat}
    
    def eval_full_cat_gap(pol_idx):
        """Worst-case gap and mean gap against full 5-cat set (G_test)."""
        gaps = [float(G_test[pol_idx, j]) for j in cat_indices]
        return {'mean': float(np.mean(gaps)), 'worst': float(np.max(gaps))}
    
    # Extra ordinary attacks so the training catalogue always has exactly 100 attacks
    extra_attacks = build_extra_ordinary(ar, 4)
    G_extra = compute_gap_matrix(policies, [(a['name'], a['delta']) for a in extra_attacks], zt, pt)
    G_train_ext = np.hstack([G_train, G_extra])  # 100 original + 4 extra = 104 columns

    # Mixed-strategy evaluation helpers against the full 5-cat test set
    def eval_full_cat_unsafe_mixed(rho):
        z_imm = ze[te == 2]
        if len(z_imm) == 0:
            return {'mean': 0.0, 'worst': 0.0}
        per_cat = []
        for j in cat_indices:
            u_j = 0.0
            for r, pr in enumerate(rho):
                if pr <= 1e-9: continue
                actions = policies[r]['fn'](z_imm + attacks[j]['delta'])
                u_j += pr * float(np.mean((actions == 0) | (actions == 1)))
            per_cat.append(u_j)
        return {'mean': float(np.mean(per_cat)), 'worst': float(np.max(per_cat))}

    def eval_full_cat_gap_mixed(rho):
        gap_vec = G_test.T @ rho
        gaps = [float(gap_vec[j]) for j in cat_indices]
        return {'mean': float(np.mean(gaps)), 'worst': float(np.max(gaps))}

    exp3 = {}
    for n_cat in [1, 2, 5]:
        cat_train = cat_ordered[:n_cat]  # nested catastrophic subset
        n_fill = 5 - n_cat
        fill_indices = [len(attacks) + i for i in range(n_fill)]  # extra ordinary attacks
        train_indices = non_cat_indices + cat_train + fill_indices  # exactly 100 attacks
        G_sub = G_train_ext[:, train_indices]

        # Uniform (trained on fixed-100 catalogue)
        u_idx = int(np.argmin(G_sub.mean(axis=1)))
        u_eval = eval_full_cat_unsafe(u_idx)
        u_gap = eval_full_cat_gap(u_idx)

        # Agnostic (trained on fixed-100 catalogue; mixed strategy)
        a_rho, a_sup_sub, a_pi, a_hist = method_agnostic_support(G_sub, policies)
        a_eval = eval_full_cat_unsafe_mixed(a_rho)
        a_gap = eval_full_cat_gap_mixed(a_rho)

        exp3[f'n{n_cat}'] = {
            'unif_mean_unsafe': u_eval['mean'],
            'unif_worst_unsafe': u_eval['worst'],
            'unif_mean_gap': u_gap['mean'],
            'unif_worst_gap': u_gap['worst'],
            'agn_mean_unsafe': a_eval['mean'],
            'agn_worst_unsafe': a_eval['worst'],
            'agn_mean_gap': a_gap['mean'],
            'agn_worst_gap': a_gap['worst'],
            'agn_sup_size': len(a_sup_sub),
        }
    results['exp3'] = exp3
    
    # --- Exp 4: Support efficiency (V(E_K) lower bound + W_K upper bound) ---
    exp4 = {'full_minimax_train': full_minimax_train_val}
    
    # Helper: for a support E, compute V(E) (mixed minimax value, lower bound) and W (mixed worst on full catalogue, upper bound)
    def support_bounds(train_indices):
        """Returns dict with V (LP value on support) and W (worst on full catalogue)."""
        G_sub = G_train[:, train_indices]
        rho_K, V = solve_lp_primal(G_sub)         # V(E_K) = mixed minimax value on support
        W = float((G_train.T @ rho_K).max())       # W_K = worst over FULL catalogue under rho_K
        return {'V': V, 'W': W, 'rho': rho_K}
    
    # Gap-pricing curve: per-iteration V(E_t) and W_t
    gp_K = [0] + m5_hist['size']  # support sizes (0 = empty)
    # At K=0: clean policy. V(E_0) undefined (empty support). W_0 = clean worst.
    gp_V = [float('nan')] + m5_hist['v']   # V(E_t)
    gp_W = [G_train[0].max()] + m5_hist['W']  # W_t
    exp4['gp_K'] = gp_K
    exp4['gp_V'] = gp_V
    exp4['gp_W'] = gp_W
    
    # Random baseline: for each K, compute both bounds
    rng_rand = np.random.RandomState(seed + 42)
    exp4['random_K'], exp4['random_V'], exp4['random_W'] = [], [], []
    for K in [1, 2, 3, 5, 8, 12, 20, 50, 100]:
        indices = list(rng_rand.choice(len(attacks), min(K, len(attacks)), replace=False))
        b = support_bounds(indices)
        exp4['random_K'].append(K)
        exp4['random_V'].append(b['V'])
        exp4['random_W'].append(b['W'])
    
    # Prediction-loss baseline
    sorted_pred = np.argsort([pred_losses[a['name']] for a in attacks])[::-1]
    exp4['pred_K'], exp4['pred_V'], exp4['pred_W'] = [], [], []
    for K in [1, 2, 3, 5, 8, 12, 20]:
        indices = list(sorted_pred[:K])
        b = support_bounds(indices)
        exp4['pred_K'].append(K)
        exp4['pred_V'].append(b['V'])
        exp4['pred_W'].append(b['W'])
    
    results['exp4'] = exp4
    
    # --- Exp 7: Twin-threat ---
    twin_attacks = build_twin_catalogue(ar, policies, zc, tc, zt, pt)
    twin_pairs = [(a['name'], a['delta']) for a in twin_attacks]
    h_names = [a['name'] for a in twin_attacks if a.get('family') == 'H']
    f_names = [a['name'] for a in twin_attacks if a.get('family') == 'F']
    twin_cat = [j for j, a in enumerate(twin_attacks) if a.get('family') in ('H', 'F')]
    
    Gt_train = compute_gap_matrix(policies, twin_pairs, zt, pt)
    Gt_test = compute_gap_matrix(policies, twin_pairs, ze, pe)
    
    h_ix = [j for j, a in enumerate(twin_attacks) if a.get('family') == 'H']
    f_ix = [j for j, a in enumerate(twin_attacks) if a.get('family') == 'F']
    h_attacks = [twin_attacks[j] for j in h_ix]
    f_attacks = [twin_attacks[j] for j in f_ix]

    H_per_pol = Gt_train[:, h_ix].max(axis=1)
    F_per_pol = Gt_train[:, f_ix].max(axis=1)
    h_best = int(np.argmin(H_per_pol))
    f_best = int(np.argmin(F_per_pol))

    tm1 = method_clean(policies, zt, pt)
    tm2 = method_uniform(Gt_train)
    twin_pred_losses = compute_prediction_loss(twin_attacks, zt, pt)
    tm3_rho, tm3_top = method_prediction_topk(
        policies, twin_attacks, twin_pairs, zt, pt, Gt_train, twin_pred_losses, K=5)
    tm4_rho, tm4_val = method_full_minimax(Gt_train)
    tm5_rho, tm5_sup, tm5_pi, tm5_hist = method_agnostic_support(Gt_train, policies)

    # Family-specific safety metrics: S_H, S_F
    def family_safety(rho):
        z_imm = ze[te == 2]
        z_clr = ze[te == 0]
        S_H = 0.0; S_F = 0.0
        for r, pr in enumerate(rho):
            if pr <= 1e-9: continue
            fn = policies[r]['fn']
            if len(z_imm) > 0:
                h_unsafe = [float(np.mean((fn(z_imm + a['delta']) == 0) | (fn(z_imm + a['delta']) == 1))) for a in h_attacks]
                S_H += pr * np.mean(h_unsafe)
            if len(z_clr) > 0:
                f_emer = [float(np.mean(fn(z_clr + a['delta']) == 3)) for a in f_attacks]
                S_F += pr * np.mean(f_emer)
        return S_H, S_F

    exp7 = {}
    for lb, ix in [('M1', tm1), ('M2', tm2)]:
        ev = evaluate_method(ix, policies, Gt_test, twin_attacks, twin_cat, ze, pe, te)
        for k in ['R_max', 'R_mean', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'C_clean']:
            exp7[f'{lb}_{k}'] = ev[k]
        exp7[f'{lb}_policy'] = ev['policy']
    for lb, rho in [('M3', tm3_rho), ('M4', tm4_rho), ('M5', tm5_rho)]:
        ev = evaluate_mixed(rho, policies, Gt_test, twin_attacks, twin_cat, ze, pe, te)
        for k in ['R_max', 'R_mean', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'C_clean']:
            exp7[f'{lb}_{k}'] = ev[k]
        exp7[f'{lb}_policy'] = 'mixed'
        sH, sF = family_safety(rho)
        exp7[f'{lb}_S_H'] = sH
        exp7[f'{lb}_S_F'] = sF
        gv = Gt_test.T @ rho
        exp7[f'{lb}_H_max'] = float(gv[h_ix].max()) if h_ix else 0.0
        exp7[f'{lb}_F_max'] = float(gv[f_ix].max()) if f_ix else 0.0

    exp7['M3_selected_attacks'] = [twin_attacks[j]['name'] for j in tm3_top]
    exp7['M5_support_size'] = len(tm5_sup)
    h_in = sum(1 for j in tm5_sup if twin_attacks[j].get('family') == 'H')
    f_in = sum(1 for j in tm5_sup if twin_attacks[j].get('family') == 'F')
    exp7['M5_H_in_support'] = h_in
    exp7['M5_F_in_support'] = f_in
    # LF prior family weights (pi* aggregated by family over the discovered support)
    pi_H = sum(pi for j, pi in enumerate(tm5_pi) if twin_attacks[tm5_sup[j]].get('family') == 'H')
    pi_F = sum(pi for j, pi in enumerate(tm5_pi) if twin_attacks[tm5_sup[j]].get('family') == 'F')
    exp7['M5_pi_H'] = float(pi_H)
    exp7['M5_pi_F'] = float(pi_F)
    exp7['M5_both_families_discovered'] = bool(h_in > 0 and f_in > 0)
    exp7['M5_both_families'] = bool(pi_H > 1e-8 and pi_F > 1e-8)
    exp7['M5_positive_support_size'] = int(np.count_nonzero(tm5_pi > 1e-8))
    exp7['h_best'] = (h_best != f_best)
    exp7['H_degradation'] = H_per_pol[f_best] - H_per_pol[h_best]
    exp7['F_degradation'] = F_per_pol[h_best] - F_per_pol[f_best]
    exp7['M5_eq_M4'] = abs(exp7['M5_R_max'] - exp7['M4_R_max']) < 0.5

    results['exp7'] = exp7
    results['calibration'] = [{k: v for k, v in a.items() if k != 'fn'} for a in attacks if a['category'] == 'catastrophic']
    results['twin_calibration'] = [dict(name=a['name'], delta=a['delta'], target_gap=22.0, achieved_gap=compute_gap_vector_single(policies[0]['fn'],a['delta'],zt,pt), at_budget=bool(np.max(np.abs(a['delta']))>=MAX_DELTA-1e-5)) for a in twin_attacks if a.get('family') in ('H','F')]
    results['catalogue_train_selection'] = {'catastrophic_order': [attacks[j]['name'] for j in cat_ordered], 'rank_bank': 'train'}
    results['certificates'] = {}
    for label, gm, rho, support, prior in [('main', G_train, m5_rho, m5_sup, m5_pi), ('twin', Gt_train, tm5_rho, tm5_sup, tm5_pi)]:
        _, vp = solve_lp_primal(gm[:, support])
        _, vd = solve_lp(gm[:, support])
        full = float((gm.T @ rho).max())
        results['certificates'][label] = dict(V_train=vp, W_train=full, pricing_gap=full-vp, primal_dual_gap=abs(vp-vd), discovered_size=len(support), positive_prior_size=int(np.count_nonzero(prior>1e-8)), positive_defender_size=int(np.count_nonzero(rho>1e-8)))
    # Held-out attack directions, generated from an independent RNG and never used to fit.
    unseen = build_unseen_attacks(seed, 100, attacks)
    Gu = compute_gap_matrix(policies, [(a['name'], a['delta']) for a in unseen], ze, pe)
    results['heldout_attacks'] = {}
    for lb, rho in [('M1', np.eye(len(policies))[m1_idx]), ('M2', np.eye(len(policies))[m2_idx]), ('M4', m4_rho), ('M5', m5_rho)]:
        ev = evaluate_mixed(rho, policies, Gu, unseen, list(range(len(unseen))), ze, pe, te)
        results['heldout_attacks'][lb] = {k: ev[k] for k in ['R_max', 'R_mean', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'C_clean']}
    
    return results


# ============================================================================
# Multi-seed aggregator
# ============================================================================
def run_multi_seed(n_seeds=20, base_seed=20260721):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    all_results = []
    # Isolate each run so native solver state and allocations do not accumulate.
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for s in range(n_seeds):
            seed = base_seed + s * 100
            t0 = time.time()
            res = pool.submit(run_single_seed, seed).result()
            dt = time.time() - t0
            print(f"  seed={seed:3d}  t={dt:.1f}s  M5_sup={res['exp2']['M5_support_size']}  "
                  f"M5_CatUnsafe={res['exp2']['M5_R_unsafe_cat']:.3f}  "
                  f"M3_CatUnsafe={res['exp2']['M3_R_unsafe_cat']:.3f}", flush=True)
            all_results.append(res)
    return all_results


def aggregate(all_results):
    """Compute mean ± std across seeds."""
    n = len(all_results)
    
    def stats_across_seeds(key_path):
        """Get mean ± std for a nested key like 'exp2.M5_R_unsafe_cat'."""
        parts = key_path.split('.')
        vals = []
        for res in all_results:
            v = res
            for p in parts:
                v = v.get(p, {}) if isinstance(v, dict) else v
            if isinstance(v, (int, float)):
                vals.append(v)
        if len(vals) == 0: return 0, 0
        return np.mean(vals), (np.std(vals, ddof=1) if len(vals)>1 else 0.)
    
    agg = {}
    
    # Exp 2: method comparison
    for method in ['M1', 'M2', 'M3', 'M4', 'M5']:
        for metric in ['R_max', 'R_mean', 'R_cvar95', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'R_false_emergency', 'C_clean']:
            m, s = stats_across_seeds(f'exp2.{method}_{metric}')
            agg[f'exp2_{method}_{metric}_mean'] = m
            agg[f'exp2_{method}_{metric}_std'] = s
    
    m, s = stats_across_seeds('exp2.M5_support_size')
    agg['exp2_M5_support_mean'] = m
    agg['exp2_M5_support_std'] = s
    
    # Exp 1
    for k in ['top3', 'top5', 'top10']:
        for mname in ['gap_recall', 'pred_recall']:
            m, s = stats_across_seeds(f'exp1.{k}_{mname}')
            agg[f'exp1_{k}_{mname}_mean'] = m
            agg[f'exp1_{k}_{mname}_std'] = s
    m, s = stats_across_seeds('exp1.worst_cat_rank_gap')
    agg['exp1_worst_cat_rank_gap_mean'] = m
    m, s = stats_across_seeds('exp1.worst_cat_rank_pred')
    agg['exp1_worst_cat_rank_pred_mean'] = m
    
    # Exp 3: rarity sweep (all policies evaluated on full 5-cat set)
    for n_cat in [1, 2, 5]:
        for k in ['unif_mean_unsafe', 'unif_worst_unsafe', 'unif_mean_gap', 'unif_worst_gap',
                   'agn_mean_unsafe', 'agn_worst_unsafe', 'agn_mean_gap', 'agn_worst_gap', 'agn_sup_size']:
            m, s = stats_across_seeds(f'exp3.n{n_cat}.{k}')
            agg[f'exp3_n{n_cat}_{k}_mean'] = m
            agg[f'exp3_n{n_cat}_{k}_std'] = s
    
    # Exp 4: full minimax train
    m, s = stats_across_seeds('full_minimax_train')
    agg['exp4_full_minimax_train_mean'] = m
    agg['exp4_full_minimax_train_std'] = s
    
    # Exp 4: pricing gap convergence — aggregate across seeds
    # For each seed, compute Δ_K = W_K - V(E_K) at the final (converged) support
    # Also save the gap-pricing K and final Δ
    gp_final_Ks, gp_final_Deltas = [], []
    for res in all_results:
        e4 = res['exp4']
        gp_final_Ks.append(e4['gp_K'][-1])
        # Final pricing gap
        if len(e4['gp_V']) > 1:
            # Last valid V, W pair
            valid_V = [v for v in e4['gp_V'] if not np.isnan(v)]
            valid_W = e4['gp_W'][-len(valid_V):] if len(valid_V) > 0 else []
            if len(valid_V) > 0:
                gp_final_Deltas.append(valid_W[-1] - valid_V[-1])
    agg['exp4_gp_K_mean'] = np.mean(gp_final_Ks) if gp_final_Ks else 0
    agg['exp4_gp_K_std'] = np.std(gp_final_Ks, ddof=1) if len(gp_final_Ks)>1 else 0
    agg['exp4_gp_final_Delta_mean'] = np.mean(gp_final_Deltas) if gp_final_Deltas else 0
    agg['exp4_gp_final_Delta_std'] = np.std(gp_final_Deltas, ddof=1) if len(gp_final_Deltas)>1 else 0
    
    # Exp 7
    for method in ['M1', 'M2', 'M3', 'M4', 'M5']:
        for metric in ['R_max', 'R_mean', 'R_unsafe_cat', 'R_unsafe_cat_worst', 'C_clean']:
            m, s = stats_across_seeds(f'exp7.{method}_{metric}')
            agg[f'exp7_{method}_{metric}_mean'] = m
            agg[f'exp7_{method}_{metric}_std'] = s

    # Exp 7: family-specific safety metrics
    for method in ['M3', 'M4', 'M5']:
        for metric in ['S_H', 'S_F', 'H_max', 'F_max']:
            m, s = stats_across_seeds(f'exp7.{method}_{metric}')
            agg[f'exp7_{method}_{metric}_mean'] = m
            agg[f'exp7_{method}_{metric}_std'] = s
    
    for k in ['M5_support_size', 'M5_H_in_support', 'M5_F_in_support', 'M5_pi_H', 'M5_pi_F', 'M5_positive_support_size']:
        m, s = stats_across_seeds(f'exp7.{k}')
        agg[f'exp7_{k}_mean'] = m
        agg[f'exp7_{k}_std'] = s
    
    m, s = stats_across_seeds('exp7.M5_both_families')
    agg['exp7_both_families_rate'] = m  # fraction of seeds achieving both
    
    m, s = stats_across_seeds('exp7.M5_eq_M4')
    agg['exp7_M5_eq_M4_rate'] = m
    
    m, s = stats_across_seeds('exp7.h_best')
    agg['exp7_conflict_exists_rate'] = m  # fraction where H-best ≠ F-best
    
    return agg


def build_unseen_attacks(seed, n=100, training_attacks=None):
    """Independent isotropic directions; includes stronger jointly perturbing attacks.

    L2 amplitudes U(.05, 2.5); each coordinate is clipped to the same 1.5 budget.
    This is a distribution shift diagnostic, not a certified unseen-space bound.
    """
    rng = np.random.RandomState(seed + 12345)
    out = []
    train_deltas = np.array([a['delta'] for a in training_attacks]) if training_attacks else None
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 100000: raise RuntimeError('Held-out attack rejection sampler exhausted')
        j = len(out)
        direction = rng.normal(size=N_SENSORS)
        direction /= np.linalg.norm(direction)
        delta = np.clip(direction * rng.uniform(.05, 2.5), -MAX_DELTA, MAX_DELTA)
        if train_deltas is not None and np.min(np.linalg.norm(train_deltas - delta, axis=1)) < .10:
            continue
        out.append(dict(name=f'unseen_{j:03d}', delta=delta, category='heldout'))
    return out


def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main():
    import argparse, platform, scipy
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=20)
    args = ap.parse_args()
    t0 = time.perf_counter()
    all_results = run_multi_seed(args.seeds)
    agg = aggregate(all_results)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'main')
    os.makedirs(out, exist_ok=True)
    elapsed = time.perf_counter()-t0
    meta = dict(experiment_suite="main", runtime_seconds=elapsed, seeds=[r['seed'] for r in all_results],
                sample_sizes=dict(train=5000, test=20000, calibration=2000),
                statistics='Mean and sample standard deviation (ddof=1) over seeds; paired inference uses the seed as the unit.',
                main_calibration='Target P(go or slow | imminent), independent calibration bank.',
                twin_calibration='Target clean Bayes utility gap 22 on training bank; independent test evaluation.',
                exp1_ranking_bank='test for both gap and TV; M3 and Exp4 TV rankings use train only',
                exp1_worst_target='Highest ground-truth conditional unsafe-action rate among labeled catastrophic attacks; ties use catalog order',
                exp3_selection_bank='train',
                heldout_attack_diagnostic=dict(n_attacks=100, direction='isotropic Gaussian normalized to unit L2', amplitude='uniform [0.05,2.5]', coordinate_budget=1.5, minimum_L2_distance_from_training_attack=0.10, random_seed_offset=12345),
                grids=dict(c=[1,2,4], r=[1,2], f=[1,2,4], d=[1,2]),
                policy_scaling={'c':['go/imminent','brake/ped_far','emergency/ped_far'], 'r':['slow/imminent'], 'f':['brake/clear','emergency/clear'], 'd':['go/ped_far (zero remains zero)','slow/ped_far']},
                policy_count=36, positive_weight_tolerance=1e-8,
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                thread_limits={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']})
    calibrations = [a for r in all_results for a in r['calibration']]
    meta['calibration_summary'] = dict(n=len(calibrations), at_budget=sum(a['at_budget'] for a in calibrations), max_absolute_target_error=max(abs(a['calibration_unsafe']-a['calibration_target_unsafe']) for a in calibrations), below_target_by_more_than_002=sum(a['calibration_unsafe']<a['calibration_target_unsafe']-.02 for a in calibrations))
    twin_calibrations = [a for r in all_results for a in r['twin_calibration']]
    meta['twin_calibration_summary'] = dict(n=len(twin_calibrations), at_budget=sum(a['at_budget'] for a in twin_calibrations), max_absolute_target_error=max(abs(a['achieved_gap']-a['target_gap']) for a in twin_calibrations), target_error_above_025=sum(abs(a['achieved_gap']-a['target_gap'])>.25 for a in twin_calibrations), explanation='Single-sensor directions hit the 1.5 coordinate budget below target; dual/triple directions are approximately gap-balanced on training data.')
    for filename, data in [('all_seeds.json',all_results),('aggregated.json',agg),('metadata.json',meta)]:
        with open(os.path.join(out,filename),'w') as f: json.dump(json_safe(data),f,indent=2,allow_nan=False)
    print(f'Saved {len(all_results)} seeds to {out}; runtime {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
