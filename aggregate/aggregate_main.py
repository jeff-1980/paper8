"""Recompute every main-text table / figure value of the MST manuscript from the per-seed
result files in results/ (no training, no data needed).

    python aggregate/aggregate_main.py            # from the repository root
    python aggregate/aggregate_main.py > aggregate/expected_output.txt

Conventions (manuscript section 3.4):
  * mean +- SD over seeds, SD with ddof = 1;
  * seed-paired differences a - b, percentile bootstrap 95% interval of the mean paired
    difference, B = 20,000 resamples; a fresh numpy Generator, np.random.default_rng(12345),
    is created for every contrast (resample index matrix rng.integers(0, n, (B, n)));
  * exact Wilcoxon signed-rank test by full enumeration of sign flips (zeros dropped, mid-ranks for
    ties; identical to scipy's exact test when there are no ties), two-sided unless stated;
  * all values in per cent / percentage points.
Bootstrap end points can differ by about +-0.1 pp from those printed in the manuscript,
which were produced by analysis scripts with other RNG seeds or resampling loops
(provenance/env2/analysis, provenance/env2/extended_validation*, scripts/extra/stats.py).
Dependencies: numpy, scipy (pandas not required).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
B = 20000
RNG_SEED = 12345
VALUES = {}          # machine-readable copy of every printed value -> aggregate/aggregate_values.json


# ------------------------------------------------------------------------- helpers
def seedvals(dirs, key='final_macro_f1', idx=None, scale=100.0):
    """{seed: value} from seed_*.json in one or more result dirs (later dirs add seeds)."""
    out = {}
    for d in dirs:
        fs = sorted(glob.glob(str(R / d / 'seed_*.json')))
        assert fs, f'no seed files in {d}'
        for f in fs:
            j = json.load(open(f))
            v = j[key] if idx is None else j[key][idx]
            assert j['seed'] not in out, (d, j['seed'])
            out[j['seed']] = v * scale
    return out


def arr(d, seeds=None):
    seeds = sorted(d) if seeds is None else seeds
    return np.array([d[s] for s in seeds]), seeds


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def signed_rank_exact(d, alternative='two-sided'):
    """Exact Wilcoxon signed-rank p by full enumeration of the 2^n sign assignments, zeros dropped,
    mid-ranks for tied |d|. Equals scipy.stats.wilcoxon(method='exact') when there are no ties; with
    ties it stays exact (scipy's 'exact' method ignores ties)."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-9]
    n = len(d)
    if n == 0:
        return 1.0
    r = stats.rankdata(np.round(np.abs(d), 9))
    t_obs = r[d > 0].sum()
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    t = signs @ r
    mu = r.sum() / 2
    if alternative == 'greater':
        return float(np.mean(t >= t_obs - 1e-9))
    if alternative == 'less':
        return float(np.mean(t <= t_obs + 1e-9))
    return float(min(1.0, np.mean(np.abs(t - mu) >= abs(t_obs - mu) - 1e-9)))


def paired(a, b, alternative='two-sided'):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(RNG_SEED)
    bm = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = signed_rank_exact(d, alternative)
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), lo=float(lo), hi=float(hi), p=p,
                npos=int((d > 0).sum()), nneg=int((d < 0).sum()), n=len(d), nzero=int((d == 0).sum()))


def fmt_ms(x, nd=1):
    m, s = ms(x)
    return f'{m:.{nd}f} +- {s:.{nd}f}'


def fmt_p(r, nd=1, p=True):
    s = f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}] (pos/neg {r['npos']}/{r['nneg']}, n={r['n']}"
    if r['nzero']:
        s += f", zero {r['nzero']}"
    return s + (f", exact Wilcoxon p={r['p']:.4f})" if p else ')')


def put(key, **kw):
    VALUES[key] = kw


def H(title):
    print('\n' + '=' * 100 + '\n' + title + '\n' + '=' * 100)


def per_bearing(dirs):
    """{seed: {bearing: acc%}} from per_bearing.json files."""
    out = {}
    for d in dirs:
        j = json.load(open(R / d / 'per_bearing.json'))
        for s, v in j['seeds'].items():
            out[int(s)] = {b: x['acc'] * 100 for b, x in v['per_bearing'].items()}
    return out


def pb_mean(pb, seeds, b):
    return float(np.mean([pb[s][b] for s in seeds]))


# ------------------------------------------------------------------------- Table 1
def table1():
    H('TABLE 1 (tab:xc) -- XJTU-SY condition 2 -> 3, macro-F1 (%)')
    E1 = 'env1/xjtu_cross/'
    E2 = 'env2/xjtu_cross/'
    rows = {
        # learner: (env, {arm: dirs})
        '1D-CNN': ('1', {'H': [E1 + 'exp_mext_e13_1dcnn_xjtu_cross_leakfree', E1 + 'exp_mext_e13_1dcnn_xjtu_cross_newseed_leakfree'],
                         'V': [E2 + 'cnn_V_only'],
                         'HV': [E1 + 'exp_e1b_xjtu_cross_dual_cnn_leakfree', E1 + 'exp_e1b_xjtu_cross_dual_cnn_newseed_leakfree']}),
        'Mamba-3': ('2', {a: [E2 + f'xc_bm3_{a}', E2 + f'q1_bm3_{a}'] for a in ('H', 'V', 'HV')}),
        'Mamba-2': ('2', {a: [E2 + f'r2_bm2_{a}'] for a in ('H', 'V')}),
    }
    print('Panel A (1D-CNN: H and H+V arms from environment-1 result files; V arm trained later with the same script,')
    print('         result files in results/env2/xjtu_cross/cnn_V_only -- see TABLE_MAP.md)')
    for lr, (env, arms) in rows.items():
        D = {a: seedvals(d) for a, d in arms.items()}
        seeds = sorted(D['H'])
        assert all(sorted(v) == seeds for v in D.values()), lr
        X = {a: arr(v, seeds)[0] for a, v in D.items()}
        line = f'  {lr:8s} (env {env}, seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)}): ' + '  '.join(f'{a}={fmt_ms(x)}' for a, x in X.items())
        print(line)
        put(f'T1A.{lr}.n', v=len(seeds))
        for a, x in X.items():
            put(f'T1A.{lr}.{a}', mean=ms(x)[0], sd=ms(x)[1])
        for a, b in [('HV', 'H'), ('HV', 'V'), ('V', 'H')]:
            if a in X and b in X:
                r = paired(X[a], X[b])
                print(f'      {a}-{b}: {fmt_p(r)}')
                put(f'T1A.{lr}.{a}-{b}', **r)
    print('Panel B (environment 1, seeds 0-7)')
    pb = {'Mamba-2': ([E1 + 'exp_e1b_xjtu_cross_single_bm2_leakfree', E1 + 'exp_e1b_xjtu_cross_single_bm2_newseed_leakfree'],
                      [E1 + 'exp_e1_xjtu_cross_dual_bm2_leakfree', E1 + 'exp_e1_xjtu_cross_dual_bm2_newseed_leakfree']),
          'Mamba-3': ([E1 + 'exp_xjtu_cross_nokin_leakfree', E1 + 'exp_xjtu_cross_nokin_newseed_leakfree'],
                      [E1 + 'exp_xjtu_cross_dual_nokin_leakfree', E1 + 'exp_xjtu_cross_dual_nokin_newseed_leakfree']),
          '1D-CNN': (rows['1D-CNN'][1]['H'], rows['1D-CNN'][1]['HV'])}
    for lr, (h, hv) in pb.items():
        a, s1 = arr(seedvals(h)); b, s2 = arr(seedvals(hv)); assert s1 == s2 == list(range(8))
        r = paired(b, a)
        irh, _ = arr(seedvals(h, 'final_per_class_recall', 1)); irhv, _ = arr(seedvals(hv, 'final_per_class_recall', 1))
        print(f'  {lr:8s}: H={fmt_ms(a)}  H+V={fmt_ms(b)}  H+V-H: {fmt_p(r)}   IR recall H={irh.mean():.1f}  H+V={irhv.mean():.1f}')
        put(f'T1B.{lr}.H', mean=ms(a)[0], sd=ms(a)[1]); put(f'T1B.{lr}.HV', mean=ms(b)[0], sd=ms(b)[1])
        put(f'T1B.{lr}.HV-H', **r); put(f'T1B.{lr}.IRrecall_H', v=float(irh.mean()))
    # footnote b: Mamba-2 env-1 H+V vs env-2 V (cross-environment, descriptive)
    m2hv = ms(arr(seedvals(pb['Mamba-2'][1]))[0])[0]; m2v = ms(arr(seedvals(rows['Mamba-2'][1]['V']))[0])[0]
    print(f'  footnote b: Mamba-2 env-1 H+V ({m2hv:.1f}) minus env-2 V ({m2v:.1f}) = {m2hv - m2v:+.1f} pp (cross-environment, not paired-tested)')
    put('T1.fnb.M2_HVenv1_minus_Venv2', v=m2hv - m2v)
    # footnote a: environment bridge for the 1D-CNN (seeds 0-4)
    e2h = arr(seedvals([E2 + 'xc_cnn_H']))[0]; e1h = arr(seedvals(rows['1D-CNN'][1]['H']), [0, 1, 2, 3, 4])[0]
    e2hv = arr(seedvals([E2 + 'xc_cnn_HV']))[0]; e1hv = arr(seedvals(rows['1D-CNN'][1]['HV']), [0, 1, 2, 3, 4])[0]
    print(f'  footnote a: 1D-CNN env2-env1 per seed 0-4: H {np.round(e2h - e1h, 2).tolist()}  H+V {np.round(e2hv - e1hv, 2).tolist()}'
          f' (max |H+V diff| {np.abs(e2hv - e1hv).max():.2f} pp)')
    put('T1.fna.cnn_bridge_maxabs_HV', v=float(np.abs(e2hv - e1hv).max())); put('T1.fna.cnn_bridge_maxabs_H', v=float(np.abs(e2h - e1h).max()))

    H('Section 4.1 -- numbers quoted in the text')
    # IR-recall of V-only models and the Bearing3_4 share
    cv = arr(seedvals(rows['1D-CNN'][1]['V'], 'final_per_class_recall', 1))[0]
    m2v_ir = arr(seedvals(rows['Mamba-2'][1]['V'], 'final_per_class_recall', 1))[0]
    pbj = json.load(open(R / E2 / 'r2_bm2_V/per_bearing.json'))
    n33 = pbj['seeds']['0']['per_bearing']['Bearing3_3']['n']; n34 = pbj['seeds']['0']['per_bearing']['Bearing3_4']['n']
    print(f'  V-only IR recall: 1D-CNN {cv.mean():.2f} (per seed {np.round(cv, 2).tolist()}), Mamba-2 {m2v_ir.mean():.2f};'
          f' share of Bearing3_4 in IR test windows {n34}/{n33 + n34} = {100 * n34 / (n33 + n34):.2f}')
    put('S41.Vonly_IRrecall_cnn', v=float(cv.mean())); put('S41.Vonly_IRrecall_bm2', v=float(m2v_ir.mean()))
    for nm, dirs in [('1D-CNN (env-2 seeds 0-4, xc_cnn_V)', [E2 + 'xc_cnn_V']), ('Mamba-2 (r2_bm2_V)', [E2 + 'r2_bm2_V'])]:
        p = per_bearing(dirs)
        print(f'    {nm}: per-seed acc Bearing3_3 {sorted(set(round(p[s]["Bearing3_3"], 1) for s in p))}, Bearing3_4 {sorted(set(round(p[s]["Bearing3_4"], 1) for s in p))}')
    # Mamba-3 env2 per-bearing and IR recall
    p = per_bearing([E2 + 'xc_bm3_V', E2 + 'q1_bm3_V']); q = per_bearing([E2 + 'xc_bm3_HV', E2 + 'q1_bm3_HV']); s8 = list(range(8))
    irv = arr(seedvals([E2 + 'xc_bm3_V', E2 + 'q1_bm3_V'], 'final_per_class_recall', 1))[0]
    irhv = arr(seedvals([E2 + 'xc_bm3_HV', E2 + 'q1_bm3_HV'], 'final_per_class_recall', 1))[0]
    print(f'  Mamba-3 env2 IR recall: H+V {irhv.mean():.1f}  V {irv.mean():.1f}')
    put('S41.M3_IRrecall_HV', v=float(irhv.mean())); put('S41.M3_IRrecall_V', v=float(irv.mean()))
    for b in ('Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5'):
        print(f'    {b}: V {pb_mean(p, s8, b):.0f}  H+V {pb_mean(q, s8, b):.0f}')
        put(f'S41.M3_pb_V.{b}', v=pb_mean(p, s8, b)); put(f'S41.M3_pb_HV.{b}', v=pb_mean(q, s8, b))
    r = VALUES['T1A.Mamba-3.HV-V']
    print(f"  Mamba-3 H+V-V: SD of paired differences {r['sd']:.1f} pp, standard uncertainty of the mean {r['sd'] / np.sqrt(r['n']):.1f} pp (section 3.4)")
    # late fusion (seeds 0-4)
    F = json.load(open(R / E2 / 'late_fusion/fusion_cross_bm3.json'))
    lf = np.array([x['fusion_sub']['f1'] for x in F]) * 100; seeds = [x['seed'] for x in F]
    ef = arr(seedvals([E2 + 'xc_bm3_HV']), seeds)[0]
    rr = paired(lf, ef)
    print(f'  late fusion Mamba-3 (seeds {seeds}, weight from source hold-out, sub-models): {fmt_ms(lf)};  late - early fusion {fmt_p(rr)}')
    put('S41.latefusion_M3', mean=ms(lf)[0], sd=ms(lf)[1]); put('S41.latefusion_M3_minus_early', **rr)
    lff = np.array([x['fusion_full']['f1'] for x in F]) * 100
    print(f'  late fusion Mamba-3, same weight applied to full-data models: {fmt_ms(lff)}')
    put('S41.latefusion_M3_full', mean=ms(lff)[0], sd=ms(lff)[1])
    Fc = json.load(open(R / E2 / 'late_fusion/fusion_cross_cnn1d.json'))
    lfc = np.array([x['fusion_sub']['f1'] for x in Fc]) * 100; vc = np.array([x['V_sub']['f1'] for x in Fc]) * 100
    lfcf = np.array([x['fusion_full']['f1'] for x in Fc]) * 100
    print(f'  late fusion 1D-CNN, full-data models: {fmt_ms(lfcf, 2)}')
    print(f'  late fusion 1D-CNN: {fmt_ms(lfc, 2)} vs its V sub-model {fmt_ms(vc, 2)} and V-only (Table 1) {fmt_ms(arr(seedvals(rows["1D-CNN"][1]["V"]))[0], 2)}')
    put('S41.latefusion_cnn', mean=ms(lfc)[0], sd=ms(lfc)[1])
    tc = json.load(open(R / E1 / 'exp_xjtu_cross_nokin_leakfree/seed_0.json'))['train_class_counts']
    print(f'  condition-2 training windows OR:IR = {tc[0]}:{tc[1]} = {tc[0] / tc[1]:.1f}:1 (section 3.1)')


# ------------------------------------------------------------------------- Table 2
def table2():
    H('TABLE 2 (tab:loco) -- leave-one-condition-out, macro-F1 (%), environment 2')
    L = 'env2/xjtu_loco/'
    spec = {('Cond. 2', 'Mamba-3'): lambda a: [L + f'bm3_L2_{a}', L + f'r1_bm3_L2_{a}'],
            ('Cond. 2', '1D-CNN'): lambda a: [L + f'cnn_L2_{a}', L + f'r1_cnn_L2_{a}'],
            ('Cond. 3', 'Mamba-3'): lambda a: [L + f'q3_bm3_L3_{a}', L + f'r1_bm3_L3_{a}'],
            ('Cond. 3', '1D-CNN'): lambda a: [L + f'q3_cnn_L3_{a}', L + f'r1_cnn_L3_{a}']}
    for (fold, lr), f in spec.items():
        D = {a: seedvals(f(a)) for a in ('H', 'V', 'HV')}
        seeds = sorted(D['H']); assert all(sorted(v) == seeds for v in D.values())
        X = {a: arr(v, seeds)[0] for a, v in D.items()}
        r = paired(X['HV'], X['V'])
        print(f'  {fold} {lr:8s} n={len(seeds)}: H={fmt_ms(X["H"])}  V={fmt_ms(X["V"])}  H+V={fmt_ms(X["HV"])}  (H+V)-V: {fmt_p(r)}  [neg/pos {r["nneg"]}/{r["npos"]}]')
        k = f'T2.{fold}.{lr}'
        for a in X:
            put(f'{k}.{a}', mean=ms(X[a])[0], sd=ms(X[a])[1])
        put(f'{k}.HV-V', **r)
        for a, b in [('HV', 'H'), ('V', 'H')]:
            put(f'{k}.{a}-{b}', **paired(X[a], X[b]))
        P = {a: per_bearing(f(a)) for a in ('H', 'V', 'HV')}
        tb = list(P['H'][seeds[0]])
        pbm = {a: {b: pb_mean(P[a], seeds, b) for b in tb} for a in P}
        print('      per-bearing accuracy H/V/H+V: ' + '; '.join(f'{b} {pbm["H"][b]:.0f}/{pbm["V"][b]:.0f}/{pbm["HV"][b]:.0f}' for b in tb))
        print('      per-bearing H+V - V (pp):    ' + ', '.join(f'{pbm["HV"][b] - pbm["V"][b]:+.0f}' for b in tb))
        for b in tb:
            put(f'{k}.pb.{b}', H=pbm['H'][b], V=pbm['V'][b], HV=pbm['HV'][b])
    for lr, pre in [('Mamba-3', 'bm3'), ('1D-CNN', 'cnn')]:
        accs = []
        for a in ('H', 'V', 'HV'):
            P = per_bearing([L + f'{pre}_L1_{a}'])
            s = sorted(P)
            accs += [pb_mean(P, s, b) for b in P[s[0]]]
        print(f'  Cond. 1 {lr}: seeds {s}, OR recall per bearing (seed means, all systems) {min(accs):.1f}-{max(accs):.1f}')
        put(f'T2.Cond1.{lr}.ORrecall', lo=min(accs), hi=max(accs))


# ------------------------------------------------------------------------- Table 3 and section 4.3
LAB = ['OR', 'IR', 'IR', 'OR']


def lobo_rec(dirs, seeds):
    """present-class recall (%) [fold, seed] from fold{f}_seed{s}.json"""
    M = np.full((4, len(seeds)), np.nan)
    for d in dirs:
        for f in range(4):
            for i, s in enumerate(seeds):
                p = R / d / f'fold{f}_seed{s}.json'
                if p.exists():
                    j = json.load(open(p))
                    M[f, i] = j['per_class_recall'][0 if LAB[f] == 'OR' else 1] * 100
    assert not np.isnan(M).any(), dirs
    return M


def summ(M):
    orf, irf = M[[0, 3]].mean(0), M[[1, 2]].mean(0)
    return orf, irf, (orf + irf) / 2


def table3():
    H('Section 4.3 -- LOBO main comparison (environment 1, Mamba-3, seeds 0-7)')
    L = 'env1/xjtu_lobo/'
    s8 = list(range(8))
    h = lobo_rec([L + 'single_nokin', L + 'single_nokin_n8ext'], s8)
    d = lobo_rec([L + 'dual_nokin', L + 'dual_nokin_n8ext'], s8)
    (ho, hi, hm), (do, di, dm) = summ(h), summ(d)
    r = paired(dm, hm)
    print(f'  macro-recall H={fmt_ms(hm)}  H+V={fmt_ms(dm)}  H+V-H: {fmt_p(r)}')
    print(f'  OR-fold mean H={ho.mean():.1f} H+V={do.mean():.1f};  IR-fold mean H={hi.mean():.1f} H+V={di.mean():.1f}')
    put('S43.LOBOmain.H', mean=ms(hm)[0], sd=ms(hm)[1]); put('S43.LOBOmain.HV', mean=ms(dm)[0], sd=ms(dm)[1]); put('S43.LOBOmain.HV-H', **r)

    H('TABLE 3 (tab:lobo) -- condition-3 LOBO, present-class recall (%), environment 2')
    L2 = 'env2/xjtu_lobo/'
    M = {a: lobo_rec([L2 + f'lobo_bm3_{a}/seeds0-2', L2 + f'lobo_bm3_{a}/seeds3-7'], s8) for a in ('H', 'V', 'HV')}
    S = {a: summ(m) for a, m in M.items()}
    for a, nm in [('H', 'H'), ('V', 'V'), ('HV', 'H+V (early fusion)')]:
        o, i, m = S[a]
        print(f'  Mamba-3 {nm:24s} OR folds {o.mean():.1f}  IR folds {i.mean():.1f}  Mean {m.mean():.1f}   (n=8)')
        put(f'T3.Mamba-3.{a}', OR=float(o.mean()), IR=float(i.mean()), Mean=float(m.mean()))
    F = json.load(open(R / L2 / 'late_fusion/fusion_lobo_bm3.json'))
    lf = {f: np.mean([x['fusion_full'] for x in F if x['fold'] == f]) * 100 for f in range(4)}
    lo_, li_ = (lf[0] + lf[3]) / 2, (lf[1] + lf[2]) / 2
    print(f'  Mamba-3 late fusion (seeds {sorted(set(x["seed"] for x in F))})  OR folds {lo_:.1f}  IR folds {li_:.1f}  Mean {(lo_ + li_) / 2:.1f}'
          '   (weight chosen on source hold-out, applied to full-data H and V models)')
    put('T3.Mamba-3.latefusion', OR=lo_, IR=li_, Mean=(lo_ + li_) / 2)
    for key in ('fusion_sub', 'H_sub', 'V_sub'):
        g = {f: np.mean([x[key] for x in F if x['fold'] == f]) * 100 for f in range(4)}
        go, gi = (g[0] + g[3]) / 2, (g[1] + g[2]) / 2
        print(f'  Mamba-3 {key:10s} (hold-out sub-models)  OR folds {go:.1f}  IR folds {gi:.1f}  Mean {(go + gi) / 2:.1f}')
        put(f'T3.Mamba-3.{key}', OR=go, IR=gi, Mean=(go + gi) / 2)
    s5 = list(range(5))
    for a, nm in [('H', 'H'), ('V', 'V'), ('HV', 'H+V')]:
        o, i, m = summ(lobo_rec([L2 + f'lobo_cnn_{a}'], s5))
        print(f'  1D-CNN  {nm:24s} OR folds {o.mean():.1f}  IR folds {i.mean():.1f}  Mean {m.mean():.1f}   (n=5)')
        put(f'T3.1D-CNN.{a}', OR=float(o.mean()), IR=float(i.mean()), Mean=float(m.mean()))
    print('  Section 4.3 text, OR-fold recall contrasts (Mamba-3, n=8):')
    for b in ('H', 'V'):
        r = paired(S['HV'][0], S[b][0])
        perb = [float((M['HV'][f] - M[b][f]).mean()) for f in (0, 3)]
        print(f'    H+V - {b}: {fmt_p(r)};  per OR test bearing Bearing3_1 {perb[0]:+.1f}, Bearing3_5 {perb[1]:+.1f}')
        put(f'S43.ORfold.HV-{b}', **r)

    H('Section 4.3 -- pre-specified IR (other-class) coverage sweep at Bearing3_1 (environment 1, 5 seeds)')
    P = R / 'env1/xjtu_ir_coverage_p14'
    rec = {}
    for arm in ('single', 'dual'):
        for f in glob.glob(str(P / arm / 'k*_seed*.json')):
            j = json.load(open(f)); rec[(arm, j['fold_tag'], j['seed'])] = j['per_class_recall'][0] * 100
    tags = sorted({t for _, t, _ in rec})
    rng_ = []
    for t in tags:
        tr = t.split('_', 2)[2].split('-')
        seeds = sorted(s for a, tt, s in rec if a == 'single' and tt == t)
        dd = np.array([rec[('dual', t, s)] - rec[('single', t, s)] for s in seeds])
        has_or = 'Bearing3_5' in tr; n_ir = sum(b in ('Bearing3_3', 'Bearing3_4') for b in tr)
        flag = 'single OR bearing + %d IR' % n_ir if has_or and n_ir else 'no OR training bearing' if not has_or else 'OR only'
        print(f'  {t:42s} [{flag:26s}] dual-H OR recall {dd.mean():+.1f} pp (per seed {np.round(dd, 1).tolist()}, negative {int((dd < 0).sum())}/{len(dd)})')
        if has_or and n_ir:
            rng_.append(dd.mean())
            put(f'S43.IRsweep.{t}', mean=float(dd.mean()), nneg=int((dd < 0).sum()), n=len(dd))
    print(f'  range over the subsets with the single OR training bearing plus IR bearings: {-max(rng_):.1f} to {-min(rng_):.1f} pp below H')
    put('S43.IRsweep.range', lo=-max(rng_), hi=-min(rng_))


# ------------------------------------------------------------------------- Table 4 / Figure 3
def table4():
    H('TABLE 4 (tab:cov) and FIGURE 3 (fig:cov) -- OR coverage sweep, Mamba-3, environment 2, 3 seeds; accuracy (%) on the OR test bearing')
    C = R / 'env2/xjtu_or_coverage'
    cells = [('3_1', 'SAME'), ('3_1', 'OTHER'), ('2_2', 'SAME'), ('2_2', 'OTHER'), ('1_1', 'SAME'), ('1_1', 'OTHER')]
    k1_below_both = worst = total = 0
    fig = {}
    for tb, order in cells:
        bname = f'Bearing{tb}'
        mean = {}
        for k in (1, 2, 3):
            for a in ('H', 'V', 'HV'):
                j = json.load(open(C / f'q2_T{tb}_{order}_k{k}_{a}/per_bearing.json'))
                v = {int(s): x['per_bearing'][bname]['acc'] * 100 for s, x in j['seeds'].items()}
                mean[(k, a)] = float(np.mean(list(v.values())))
                fig[f'{tb}_{order}_k{k}_{a}'] = [round(v[s], 2) for s in sorted(v)]
        db = {k: mean[(k, 'HV')] - max(mean[(k, 'H')], mean[(k, 'V')]) for k in (1, 2, 3)}
        o = 'same' if order == 'SAME' else 'other'
        print(f'  {bname} {o:5s}: k=1 H {mean[(1, "H")]:.0f}  V {mean[(1, "V")]:.0f}  H+V {mean[(1, "HV")]:.0f}  '
              f'D_best(k=1) {db[1]:+.0f}  D_best(k=2) {db[2]:+.0f}  D_best(k=3) {db[3]:+.0f}')
        put(f'T4.{bname}.{o}', H1=mean[(1, 'H')], V1=mean[(1, 'V')], HV1=mean[(1, 'HV')], D1=db[1], D3=db[3])
        k1_below_both += mean[(1, 'HV')] < min(mean[(1, 'H')], mean[(1, 'V')])
        for k in (1, 2, 3):
            total += 1; worst += mean[(k, 'HV')] < min(mean[(k, 'H')], mean[(k, 'V')])
    print(f'  k=1 cells in which H+V is below both single sensors: {k1_below_both}/6;  all cells (k=1,2,3) in which H+V is the worst of three: {worst}/{total}')
    put('T4.k1_below_both', v=int(k1_below_both)); put('T4.worst_cells', v=int(worst), n=total)
    print('  Figure 3 points (per seed 0,1,2):')
    for key, v in fig.items():
        print(f'    {key:22s} {v}')


# ------------------------------------------------------------------------- Figure 2
def figure2():
    H('FIGURE 2 (fig:forest) -- dual minus single sensor, seed-paired, 95% bootstrap interval (pp)')
    V = VALUES
    rows = [('Condition 2->3', '1D-CNN', 'H', V['T1B.1D-CNN.HV-H']), ('Condition 2->3', '1D-CNN', 'V', V['T1A.1D-CNN.HV-V']),
            ('Condition 2->3', 'Mamba-2', 'H', V['T1B.Mamba-2.HV-H']),
            ('Condition 2->3', 'Mamba-3', 'H', V['T1A.Mamba-3.HV-H']), ('Condition 2->3', 'Mamba-3', 'V', V['T1A.Mamba-3.HV-V'])]
    for fold, k in [('Held-out condition 2', 'Cond. 2'), ('Held-out condition 3', 'Cond. 3')]:
        for lr in ('Mamba-3', '1D-CNN'):
            for b in ('H', 'V'):
                rows.append((fold, lr, b, V[f'T2.{k}.{lr}.HV-{b}']))
    for b in ('H', 'V'):
        rows.append(('Held-out bearing (LOBO, OR folds)', 'Mamba-3', b, V[f'S43.ORfold.HV-{b}']))
    for fold, lr, b, r in rows:
        print(f'  {fold:36s} {lr:8s} dual-{b}: {r["mean"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}]')
        put(f'F2.{fold}.{lr}.{b}', **r)
    print(f'  {"Condition 2->3":36s} {"Mamba-2":8s} dual-V: {V["T1.fnb.M2_HVenv1_minus_Venv2"]["v"]:+.2f} (cross-environment, no interval)')


# ------------------------------------------------------------------------- Table 5 and section 4.4
def table5():
    H('TABLE 5 (tab:cwru) -- CWRU accuracy (%) under AWGN, seeds 0-4 (test accuracy at the validation-selected epoch)')
    C1, C2 = 'env1/cwru/', 'env2/cwru/'
    K = 'test_acc_at_best_val'
    s5 = [0, 1, 2, 3, 4]
    src = {'Mamba-3': {'FE': lambda s: [C2 + f'bm3_FE_snr-{s}'], 'DE': lambda s: [C1 + f'exp02_snr-{s}_nokin'],
                       'DEFE': lambda s: [C1 + f'exp_b2_dual_nokin_snrm{s}']},
           '1D-CNN': {a: (lambda s, a=a: [C2 + f'cnn_{a}_snr-{s}']) for a in ('FE', 'DE', 'DEFE')}}
    fe_gap, fe_pairs, fe_dir = [], 0, 0
    for lr, arms in src.items():
        for s in (4, 6, 8):
            X = {a: arr(seedvals(f(s), K), s5)[0] for a, f in arms.items()}
            print(f'  {lr:8s} -{s} dB: FE-only {fmt_ms(X["FE"], 2)}  DE-only {fmt_ms(X["DE"], 2)}  DE+FE {fmt_ms(X["DEFE"], 2)}')
            for a in X:
                put(f'T5.{lr}.-{s}.{a}', mean=ms(X[a])[0], sd=ms(X[a])[1])
            g = X['DE'] - X['FE']; fe_gap.append(g.mean()); fe_pairs += len(g); fe_dir += int((g > 0).sum())
            r = paired(X['DEFE'], X['DE'], 'greater')
            r2 = paired(X['DEFE'], X['DE'])
            print(f'      DE+FE - DE: {fmt_p(r, 2, p=False)}  one-sided p={r["p"]:.4f}  two-sided p={r2["p"]:.4f}')
            put(f'S44.gain5.{lr}.-{s}', **r)
    print(f'  FE-only below DE-only by {min(fe_gap):.1f}-{max(fe_gap):.1f} pp (cell means); seed pairs with DE > FE: {fe_dir}/{fe_pairs}')
    put('S44.FEgap', lo=float(min(fe_gap)), hi=float(max(fe_gap)), ndir=fe_dir, n=fe_pairs)

    H('Section 4.4 -- eight-seed Mamba-3 pool (environment 1, seeds 0-7), one-sided exact Wilcoxon (pre-specified DE+FE > DE)')
    for s in (4, 6, 8):
        de = arr(seedvals([C1 + f'exp02_snr-{s}_nokin', C1 + f'exp_e6_single_nokin_snrm{s}_newseed_leakfree'], K))[0]
        df = arr(seedvals([C1 + f'exp_b2_dual_nokin_snrm{s}', C1 + f'exp_e6_dual_nokin_snrm{s}_newseed_leakfree'], K))[0]
        r = paired(df, de, 'greater')
        print(f'  -{s} dB: DE {fmt_ms(de, 2)}  DE+FE {fmt_ms(df, 2)}  gain {r["mean"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}]  SD of differences {r["sd"]:.2f}  one-sided p={r["p"]:.4f}  (pos/neg {r["npos"]}/{r["nneg"]})')
        put(f'S44.gain8.-{s}', **r); put(f'S44.DE8.-{s}', mean=ms(de)[0])
    ref = json.load(open(R / C1 / 'b2_n8_leakfree_summary.json'))
    print('  cross-check against results/env1/cwru/b2_n8_leakfree_summary.json: ' +
          ', '.join(f'-{s} dB {ref[f"-{s}"]["delta_nokin_n8_pp"]:+.2f} (p={ref[f"-{s}"]["p_nokin_n8_one_sided"]:.4f})' for s in (4, 6, 8)))

    H('Section 4.4 -- pink noise (Mamba-3, environment 1, seeds 0-4)')
    pg = {}
    for s in (4, 6, 8):
        de = arr(seedvals([C1 + f'exp_e5_single_pink_snr-{s}_leakfree'], K))[0]
        df = arr(seedvals([C1 + f'exp_e5_dual_pink_snr-{s}_leakfree'], K))[0]
        r = paired(df, de)
        pg[s] = df - de
        print(f'  -{s} dB pink: DE {fmt_ms(de, 2)}  DE+FE {fmt_ms(df, 2)}  DE+FE-DE {fmt_p(r, 2)}')
        put(f'S44.pink.-{s}', **r); put(f'S44.pinkDE.-{s}', mean=ms(de)[0])
    print(f"  DE-only at nominal -8 dB: pink {VALUES['S44.pinkDE.-8']['mean']:.1f}  AWGN (8 seeds) {VALUES['S44.DE8.-8']['mean']:.1f}")
    # matched single-sensor accuracy: AWGN -4 dB (8 seeds) vs pink -8 dB (5 seeds); independent groups
    ga = (arr(seedvals([C1 + 'exp_b2_dual_nokin_snrm4', C1 + 'exp_e6_dual_nokin_snrm4_newseed_leakfree'], K))[0]
          - arr(seedvals([C1 + 'exp02_snr-4_nokin', C1 + 'exp_e6_single_nokin_snrm4_newseed_leakfree'], K))[0])
    gp = pg[8]
    rng = np.random.default_rng(RNG_SEED)
    bd = ga[rng.integers(0, len(ga), (B, len(ga)))].mean(1) - gp[rng.integers(0, len(gp), (B, len(gp)))].mean(1)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    print(f'  matched accuracy, gain(AWGN -4 dB, n={len(ga)}) - gain(pink -8 dB, n={len(gp)}): {ga.mean() - gp.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] (independent-group bootstrap)')
    put('S44.matched', mean=float(ga.mean() - gp.mean()), lo=float(lo), hi=float(hi))
    ib = json.load(open(R / 'env2/cwru_noise_analysis/inband.json'))
    print(f"  in-band SNR, 2-5 kHz resonance band, nominal -8 dB (stored output of provenance/env2/analysis/inband.py): "
          f"pink {ib['pink@-8']['2-5 kHz (resonance)'][0]:+.1f} dB, AWGN {ib['awgn@-8']['2-5 kHz (resonance)'][0]:+.1f} dB")
    put('S44.inband', pink=ib['pink@-8']['2-5 kHz (resonance)'][0], awgn=ib['awgn@-8']['2-5 kHz (resonance)'][0])
    import csv
    co = {r['condition']: float(r['coherence']) for r in csv.DictReader(open(R / 'env1/cwru_coherence/gain_vs_coherence_points_n8_leakfree.csv')) if r['dataset'] == 'CWRU'}
    print(f"  DE-FE fault-band magnitude-squared coherence (stored output, results/env1/cwru_coherence): 0 dB {co['SNR=0dB']:.3f}, -8 dB {co['SNR=-8dB']:.3f}")
    put('S44.coherence', c0=co['SNR=0dB'], c8=co['SNR=-8dB'])


if __name__ == '__main__':
    print(f'aggregate_main.py -- bootstrap B={B}, numpy.random.default_rng({RNG_SEED}) per contrast; numpy {np.__version__}, scipy {__import__("scipy").__version__}')
    table1()
    table2()
    table3()
    table4()
    figure2()
    table5()
    json.dump(VALUES, open(ROOT / 'aggregate/aggregate_values.json', 'w'), indent=1)
    print('\nwrote aggregate/aggregate_values.json')
