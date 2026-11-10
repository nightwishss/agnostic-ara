#!/usr/bin/env python3
"""Attacker-perspective response generation with held-out evaluation.

All fitting, attack calibration, and response-set stopping precede test evaluation.
The final response set is recomputed after every update, including the final update.
R_strategic(test) evaluates the frozen final training-BR set on independent data;
V_E(train), R_E(test), and a test-recomputed-BR diagnostic are separately reported.
"""
import argparse
import json
import os
import time
import numpy as np
import simulation as sim

LAM_H, LAM_C, LAM_S = 1., .5, .5
POS_TOL, BR_TOL = 1e-8, 1e-9


def perceived_U(kind):
    u = sim.U_D.copy()
    if kind == 'H': u[0, 2] *= 1.5; u[1, 2] *= 1.5
    elif kind == 'F': u[2, 0] *= 3.; u[3, 0] *= 3.
    else: raise ValueError(kind)
    return u


def cost_stealth(attacks):
    return (np.array([np.count_nonzero(abs(a['delta']) > 1e-6) for a in attacks], dtype=float),
            -np.array([abs(a['delta']).sum() for a in attacks]))


def attacker_utility(gper, rho, cost, stealth):
    return LAM_H * (gper.T @ rho) - LAM_C * cost + LAM_S * stealth


def response_set(matrices, rho, cost, stealth):
    per_model = []
    for gper in matrices:
        u = attacker_utility(gper, rho, cost, stealth)
        per_model.append(np.flatnonzero(u >= u.max()-BR_TOL).tolist())
    return sorted(set(j for indices in per_model for j in indices)), per_model


def strategic_loop(gtrue, perceived_matrices, cost, stealth, max_iter=100):
    n_p = gtrue.shape[0]
    rho = np.eye(n_p)[0]
    support, history = [], []
    pi = np.array([])
    for iteration in range(max_iter):
        rational, _ = response_set(perceived_matrices, rho, cost, stealth)
        unseen = [j for j in rational if j not in support]
        if not unseen:
            return rho, support, pi, history, rational
        gaps = gtrue.T @ rho
        support.append(max(unseen, key=lambda j: gaps[j]))
        rho, value = sim.solve_lp_primal(gtrue[:, support])
        pi, dual = sim.solve_lp(gtrue[:, support])
        # Fresh best responses to the UPDATED defense, never stale responses.
        updated_rational, _ = response_set(perceived_matrices, rho, cost, stealth)
        updated_gaps = gtrue.T @ rho
        stable = set(updated_rational).issubset(support)
        history.append(dict(iteration=iteration+1, size=len(support), V_E_train=value,
                            R_strategic_train=float(updated_gaps[updated_rational].max()),
                            R_full_train=float(updated_gaps.max()), response_set=updated_rational,
                            stable=stable, primal_dual_gap=abs(value-dual)))
        if stable:
            return rho, support, pi, history, updated_rational
    raise RuntimeError('Strategic loop did not reach response-set stability')


def build(seed):
    zc, pc, tc = sim.generate_sample_bank(2000, np.random.RandomState(seed+1))
    zt, pt, tt = sim.generate_sample_bank(5000, np.random.RandomState(seed))
    ze, pe, te = sim.generate_sample_bank(20000, np.random.RandomState(seed+2))
    policies = sim.build_policy_library()
    ar = np.random.RandomState(seed+3)
    # Use a separate background-attack stream for the strategic comparison.
    attacks = sim.build_twin_catalogue(ar, policies, zc, tc, zt, pt)
    pairs = [(a['name'], a['delta']) for a in attacks]
    gtrain = sim.compute_gap_matrix(policies, pairs, zt, pt)
    gtest = sim.compute_gap_matrix(policies, pairs, ze, pe)
    perceived_train = [sim.compute_gap_matrix(policies, pairs, zt, pt, utility=perceived_U(k)) for k in ['H', 'F']]
    cost, stealth = cost_stealth(attacks)
    return policies, attacks, pairs, (zc, pc, tc), (zt, pt, tt), (ze, pe, te), gtrain, gtest, perceived_train, cost, stealth


def mixture_summary(support, pi, attacks):
    h = float(sum(w for j,w in zip(support,pi) if attacks[j].get('family')=='H'))
    f = float(sum(w for j,w in zip(support,pi) if attacks[j].get('family')=='F'))
    return dict(sz=len(support), positive_support_size=int(np.count_nonzero(pi>POS_TOL)),
                h=h, f=f, both=float(h>POS_TOL and f>POS_TOL))


def evaluate(seed):
    pols, atks, pairs, calib, train, test, gt, ge, gp, cost, stealth = build(seed)
    zt,pt,tt=train; ze,pe,te=test
    n_p=len(pols); rho0=np.eye(n_p)[0]
    generators={}; representative_diagnostic=[]
    baseline_j=int(np.argmax(gt[0]))
    initial_choices={'L0':baseline_j}
    for name,gper in zip(['tau_H','tau_F'],gp):
        ua=attacker_utility(gper,rho0,cost,stealth)
        j=int(np.argmax(ua)); initial_choices[name]=j
        for family in ['H','F']:
            indices=[k for k,a in enumerate(atks) if a.get('family')==family]
            best=max(indices,key=lambda k:ua[k])
            representative_diagnostic.append(dict(generator=name,family=family,attack=atks[best]['name'],
                true_gap=float(gt[0,best]),perceived_gap=float(gper[0,best]),cost=float(cost[best]),
                stealth=float(stealth[best]),attacker_utility=float(ua[best]),is_global_best=bool(best==j)))
    rho_l0,sup_l0,pi_l0,_=sim.method_agnostic_support(gt,pols)
    specifications=[('L0',None),('tau_H',[gp[0]]),('tau_F',[gp[1]]),('joint',gp)]
    for name,matrices in specifications:
        if name=='L0':
            rho,support,pi=rho_l0,sup_l0,pi_l0
            history=[]; rational=list(range(len(atks))); stable=True
        else:
            rho,support,pi,history,rational=strategic_loop(gt,matrices,cost,stealth)
            fresh,_=response_set(matrices,rho,cost,stealth)
            stable=set(fresh).issubset(support)
            assert fresh==rational and stable
        train_gaps=gt.T@rho; test_gaps=ge.T@rho
        _,value=sim.solve_lp_primal(gt[:,support])
        clean=float(sum(rho[k]*sim.compute_clean_cost(p['fn'],ze,pe) for k,p in enumerate(pols) if rho[k]>POS_TOL))
        metrics=mixture_summary(support,pi,atks)
        metrics.update(Rstrat=float(test_gaps[rational].max()) if name!='L0' else None,
            Rfull=float(test_gaps.max()),Cclean=clean,V_E_train=value,
            R_strategic_train=float(train_gaps[rational].max()) if name!='L0' else None,
            R_full_train=float(train_gaps.max()),R_E_test=float(test_gaps[support].max()),
            stable=stable,iters=len(history),support=support,positive_support=[j for j,w in zip(support,pi) if w>POS_TOL],
            prior=pi.tolist(),rho=rho.tolist(),final_response_set=rational if name!='L0' else None,
            history=history)
        # Test re-estimated BR is a retrospective diagnostic; it never changes rho/support.
        if name!='L0':
            kind_list=['H'] if name=='tau_H' else ['F'] if name=='tau_F' else ['H','F']
            test_rational=[]
            for kind in kind_list:
                eu=pe@perceived_U(kind).T; optimal=eu.max(axis=1)
                mixed_perceived=np.zeros(len(atks))
                for r,pr in enumerate(rho):
                    if pr<=POS_TOL: continue
                    for j,a in enumerate(atks):
                        actions=pols[r]['fn'](ze+a['delta'])
                        mixed_perceived[j]+=pr*float((optimal-eu[np.arange(len(ze)),actions]).mean())
                ua=LAM_H*mixed_perceived-LAM_C*cost+LAM_S*stealth
                test_rational.extend(np.flatnonzero(ua>=ua.max()-BR_TOL).tolist())
            metrics['R_strategic_test_reestimated_BR']=float(test_gaps[sorted(set(test_rational))].max())
            metrics['test_reestimated_BR_set']=sorted(set(test_rational))
        generators[name]=metrics
    return dict(seed=seed,generators=generators,initial_choices=initial_choices,
                initial_choice_families={k:atks[j].get('family') for k,j in initial_choices.items()},
                diagnostic=representative_diagnostic,
                catalogue=[dict(name=a['name'],delta=a['delta'].tolist(),family=a.get('family')) for a in atks])


def stats(xs):
    x=np.array(xs,float)
    return [float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.]


def aggregate(all_seeds):
    out={}
    for name in ['L0','tau_H','tau_F','joint']:
        rows=[r['generators'][name] for r in all_seeds]
        out[name]={k:stats([r[k] for r in rows]) for k in ['sz','positive_support_size','Rfull','Cclean','V_E_train','R_full_train','R_E_test']}
        out[name].update(pi_H=stats([r['h'] for r in rows]),pi_F=stats([r['f'] for r in rows]),
            both_frac=float(np.mean([r['both'] for r in rows])),stable_frac=float(np.mean([r['stable'] for r in rows])),
            Rstrat=stats([r['Rstrat'] for r in rows]) if name!='L0' else None,
            R_strategic_train=stats([r['R_strategic_train'] for r in rows]) if name!='L0' else None)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=20); args=ap.parse_args()
    started=time.perf_counter(); all_seeds=[]
    for i in range(args.seeds):
        seed=20260721+100*i; t0=time.perf_counter(); row=evaluate(seed); all_seeds.append(row)
        joint=row['generators']['joint']
        print(f"Strategic seed={seed} t={time.perf_counter()-t0:.1f}s both-models E={joint['sz']} Rstrat_test={joint['Rstrat']:.3f} Rfull_test={joint['Rfull']:.3f} stable={joint['stable']}",flush=True)
    n=len(all_seeds)
    div={}
    predicates={'L0_H':lambda r:r['initial_choice_families']['L0']=='H',
        'L1H_H':lambda r:r['initial_choice_families']['tau_H']=='H',
        'L1F_H':lambda r:r['initial_choice_families']['tau_F']=='H',
        'L0_vs_L1H_div':lambda r:r['initial_choices']['L0']!=r['initial_choices']['tau_H'],
        'L0_vs_L1F_div':lambda r:r['initial_choices']['L0']!=r['initial_choices']['tau_F'],
        'L1H_vs_L1F_div':lambda r:r['initial_choices']['tau_H']!=r['initial_choices']['tau_F']}
    for name,fn in predicates.items():
        count=sum(fn(r) for r in all_seeds); div[name]=[count,count/n]
    result=dict(divergence_20seed=div,comparison_20seed=aggregate(all_seeds),
        joint_iters_20seed=stats([r['generators']['joint']['iters'] for r in all_seeds]),all_seeds=all_seeds,
        metadata=dict(runtime_seconds=time.perf_counter()-started,sample_sizes={'train':5000,'test':20000,'calibration':2000},
            seed_count=n,statistic='Mean and sample SD (ddof=1)',catalogue='Hide/fake catalog; same six severe attacks as the non-recursive comparison, independent background-attack RNG stream',
            calibration_bank='Hide/fake calibration on training bank; independent evaluation bank',response_generation_bank='train',stopping='fresh final training BR set is a subset of discovered E',
            Rstrat='Held-out test gap, maximized over final frozen training-BR set A(rho)',
            V_E_train='Restricted-game training LP value, distinct from Rstrat and R_E_test',
            R_strategic_test_reestimated_BR='Retrospective diagnostic only; does not modify training solution',positive_weight_tolerance=POS_TOL,best_response_tie_tolerance=BR_TOL))
    output=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','strategic','all_seeds.json')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output,'w') as f: json.dump(sim.json_safe(result),f,indent=2,allow_nan=False)
    print(f"Saved {output}; runtime {result['metadata']['runtime_seconds']:.1f}s",flush=True)


if __name__=='__main__': main()
