"""Round 6 analysis, as specified in prereg_mst_round6.md (commit 44e5d27, deviation 353684d).
Reads out_r6/ only (round-6 runs); never pairs with earlier rounds. Writes analysis/an_r6_results.md and .json.
Missing configurations are reported as missing, never imputed."""
import json, numpy as np, yaml
from pathlib import Path
_W = Path(__file__).resolve().parents[1]
_REL = _W / 'results' / 'mst_round6_20261106'
RUN = _REL if (_REL / 'A_m3_H_H').exists() else _W / 'out_r6'   # released results, else local run directory
CFG = RUN / 'cfg_r6' if (RUN / 'cfg_r6').exists() else _W / 'cfg_r6'
ENV_CSV = (RUN / 'envelope' / 'envelope_onset.csv') if (RUN / 'envelope').exists() else _W / 'r6analysis' / 'envelope_onset.csv'
from scipy import stats
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../scripts/stats_exact'))
from exact_wilcoxon import wilcoxon as exact_wilcoxon  # zero differences dropped, mid-ranks, full sign enumeration (2026-11-13)
O = RUN
A = Path(__file__).resolve().parent
NB = 20000
OUT, MD = {}, []
def P(s=''): MD.append(s)

def boot(d, seed=12345):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    m = rng.choice(d, (NB, len(d))).mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def wil(d):
    d = np.asarray(d, float)
    if np.all(d == 0): return float('nan')
    return float(exact_wilcoxon(d, method='exact').pvalue)
def summ(d):
    d = np.asarray(d, float)
    return dict(mean=float(d.mean()), ci=boot(d), p=wil(d), pos=int((d > 0).sum()), neg=int((d < 0).sum()), n=len(d), d=[float(x) for x in d])
def fmt(r, scale=100, u='pp'):
    if r is None: return 'missing'
    return f"{r['mean']*scale:+.2f} {u} [{r['ci'][0]*scale:+.2f}, {r['ci'][1]*scale:+.2f}], p={r['p']:.4f}, {r['pos']}+/{r['neg']}- (n={r['n']})"

def xc(name):
    f = O / name / 'r6eval' / 'summary.json'
    if not f.exists(): return None
    s = json.load(open(f)); return {int(k[4:]): v for k, v in s.items() if k.startswith('seed')}
def lobo(name):
    f = O / name / 'r6eval' / 'summary.json'
    if not f.exists(): return None
    s = json.load(open(f)); out = {}
    for k, v in s.items():
        fo, se = k.split('_'); out.setdefault(int(se[4:]), {})[int(fo[4:])] = v
    return out
def cw(name):
    fs = sorted((O / name).glob('seed_*.json'))
    if not fs: return None
    out = {}
    for f in fs:
        j = json.load(open(f)); out[j['seed']] = j['test_acc_at_best_val']
    return out
def cw_eval(name, noise):
    f = O / 'evalcwru' / name / f'test_{noise}_r0.json'
    if not f.exists(): return None
    return {int(k): v['test_other'] for k, v in json.load(open(f))['seeds'].items()}

def metric(rec, view, m):
    v = rec[view]
    if m == 'macro_f1': return v['macro_f1']
    if m == 'macro_recall': return float(np.nanmean(v['per_class_recall']))
    if m == 'beq': return v['bearing_equal_balanced_acc']
def paired(a, b, f):
    if a is None or b is None: return None
    S = sorted(set(a) & set(b)); return summ([f(a[s]) - f(b[s]) for s in S])
def per_bearing(rec_by_seed, view='own'):
    bs = rec_by_seed[min(rec_by_seed)][view]['per_bearing']
    return {b: float(np.mean([rec_by_seed[s][view]['per_bearing'][b]['acc'] for s in rec_by_seed])) for b in bs}

L = {'m3': 'Mamba-3', 'cnn': '1D-CNN', 'm2': 'Mamba-2'}
prim = {}

# ---------------- Block A
P('# Round 6 results (prereg_mst_round6.md)'); P()
P('## Block A: onset rules x learners, condition 2 -> 3'); P()
OUT['A'] = {}
for l in L:
    for rule in ['H', 'V', 'last15']:
        R = {a: xc(f'A_{l}_{rule}_{a}') for a in ['H', 'V', 'HV']}
        if any(v is None for v in R.values()): P(f'- {L[l]} rule {rule}: missing {[a for a,v in R.items() if v is None]}'); continue
        row = {}
        P(f'### {L[l]}, rule {rule} (seeds {sorted(R["H"])})')
        P('| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |'); P('|---|---|---|---|---|---|')
        for a, r in R.items():
            vals = {m: np.array([metric(r[s], 'own', m) for s in r]) for m in ['macro_f1', 'macro_recall', 'beq']}
            vc = np.array([metric(r[s], 'common', 'macro_f1') for s in r]); pb = per_bearing(r)
            P(f"| {a} | {vals['macro_f1'].mean()*100:.1f} ± {vals['macro_f1'].std(ddof=1)*100:.1f} | {vals['macro_recall'].mean()*100:.1f} | {vals['beq'].mean()*100:.1f} | {vc.mean()*100:.1f} | " + ', '.join(f'{b[7:]}:{v:.2f}' for b, v in pb.items()) + ' |')
            row[a] = dict(macro_f1=vals['macro_f1'].tolist(), macro_recall=vals['macro_recall'].tolist(), beq=vals['beq'].tolist(), common_f1=vc.tolist(), per_bearing=pb)
        P(); P('| contrast | metric / view | result |'); P('|---|---|---|')
        for x, y in [('V', 'H'), ('HV', 'H'), ('HV', 'V')]:
            for m in ['macro_f1', 'macro_recall', 'beq']:
                for view in ['own', 'common']:
                    r = paired(R[x], R[y], lambda q, m=m, view=view: metric(q, view, m)); row[f'{x}-{y}|{m}|{view}'] = r
                    P(f'| {x}-{y} | {m} / {view} | {fmt(r)} |')
            pbx, pby = per_bearing(R[x]), per_bearing(R[y])
            d = {b: pbx[b] - pby[b] for b in pbx}; row[f'{x}-{y}|per_bearing'] = d
            cons = all(v >= -0.02 for v in d.values()) or all(v <= 0.02 for v in d.values())
            P(f"| {x}-{y} | per bearing (own) | " + ', '.join(f'{b[7:]}:{v:+.2f}' for b, v in d.items()) + f" — signs {'consistent' if cons else 'differ across bearings'} |")
        P()
        OUT['A'][f'{l}_{rule}'] = row
        if rule == 'H': prim[f'P{["m3","cnn","m2"].index(l)+1}: {L[l]} V-H, bearing-eq bal. acc, H rule'] = row['V-H|beq|own']
        if l == 'm3' and rule == 'last15': prim['P4: Mamba-3 V-H, macro-F1 on common test set, last15 rule'] = row['V-H|macro_f1|common']

# ---------------- Block B
P('## Block B: decomposition of the last-15% effect (common test set, macro-F1)'); P()
OUT['B'] = {}
for l, S0 in [('m3', range(5)), ('cnn', range(8))]:
    G = {'H': {a: xc(f'A_{l}_H_{a}') for a in ['H', 'V', 'HV']}, 'last15': {a: xc(f'A_{l}_last15_{a}') for a in ['H', 'V', 'HV']},
         'seg': {a: xc(f'B_{l}_seg_{a}') for a in ['H', 'V', 'HV']}, 'size': {a: xc(f'B_{l}_size_{a}') for a in ['H', 'V', 'HV']}}
    if any(v is None for g in G.values() for v in g.values()): P(f'- {L[l]}: missing configs'); continue
    S = [s for s in S0 if all(s in v for g in G.values() for v in g.values())]
    def C(var, x, y, view):
        return np.array([metric(G[var][x][s], view, 'macro_f1') - metric(G[var][y][s], view, 'macro_f1') for s in S])
    P(f'### {L[l]} (seeds {S})'); P('| contrast | C_c(H rule) | test composition | training segmentation | bearing weighting | sample size | C_own(last15) |'); P('|---|---|---|---|---|---|---|')
    row = {}
    for x, y in [('V', 'H'), ('HV', 'H'), ('HV', 'V')]:
        cH, cl, cs, cz, clo = C('H', x, y, 'common'), C('last15', x, y, 'common'), C('seg', x, y, 'common'), C('size', x, y, 'common'), C('last15', x, y, 'own')
        parts = dict(test_composition=summ(clo - cl), training_segmentation=summ(cs - cH), bearing_weighting=summ(cz - cs), sample_size=summ(cl - cz))
        row[f'{x}-{y}'] = dict(C_common_Hrule=summ(cH), C_own_last15=summ(clo), C_common_last15=summ(cl), C_common_seg=summ(cs), C_common_size=summ(cz), **parts)
        P(f"| {x}-{y} | {cH.mean()*100:+.1f} | " + ' | '.join(f"{parts[k]['mean']*100:+.1f} [{parts[k]['ci'][0]*100:+.1f}, {parts[k]['ci'][1]*100:+.1f}]" for k in parts) + f" | {clo.mean()*100:+.1f} |")
    P(); OUT['B'][l] = row

# ---------------- Block C (LOBO concat) and E (gates)
P('## Block C: LOBO OR folds, Mamba-3 (own-class recall on the test bearing)'); P()
OUT['C'] = {}
def lobo_rec(R, view='own'):   # seed -> fold -> recall of the test bearing's class
    return {s: {f: list(v[view]['per_bearing'].values())[0]['acc'] for f, v in fs.items()} for s, fs in R.items()}
for rule in ['H', 'last15']:
    R = {a: lobo(f'C_lobo_m3_{rule}_{a}') for a in ['H', 'V', 'HV']}
    if any(v is None for v in R.values()): P(f'- rule {rule}: missing'); continue
    rr = {a: lobo_rec(v) for a, v in R.items()}; S = sorted(set.intersection(*[set(v) for v in rr.values()]))
    P(f'### rule {rule} (seeds {S})'); P('| arm | fold 0 (3_1) | fold 3 (3_5) | OR mean |'); P('|---|---|---|---|')
    row = {}
    for a in rr:
        f0 = float(np.mean([rr[a][s][0] for s in S])); f3 = float(np.mean([rr[a][s][3] for s in S])); row[a] = dict(f0=f0, f3=f3)
        P(f'| {a} | {f0*100:.1f} | {f3*100:.1f} | {(f0+f3)/2*100:.1f} |')
    for x, y in [('HV', 'H'), ('HV', 'V'), ('V', 'H')]:
        r = summ([np.mean([rr[x][s][f] for f in (0, 3)]) - np.mean([rr[y][s][f] for f in (0, 3)]) for s in S]); row[f'{x}-{y}'] = r
        P(f'- {x}-{y} (OR mean): {fmt(r)}')
    P(); OUT['C'][rule] = row

P('## Block E: gated fusion and fixed-gate ablation'); P()
OUT['E'] = {}
for n, base in [('E_m3_gated', 'm3'), ('E_m3_tgated', 'm3'), ('E_cnn_gated', 'cnn'), ('E_cnn_tgated', 'cnn')]:
    G = xc(n); Hc = {a: xc(f'A_{base}_H_{a}') for a in ['H', 'V', 'HV']}
    if G is None: P(f'- {n}: missing'); continue
    row = {}
    f1 = np.array([G[s]['own']['macro_f1'] for s in G]); ff = np.array([G[s]['fixed_gate_own']['macro_f1'] for s in G])
    row['gate_mean'] = [G[s]['gate_mean'] for s in G]; row['frac_changed'] = [G[s]['frac_pred_changed_by_fixing'] for s in G]
    abl = summ(f1 - ff); row['adaptive_minus_fixed'] = abl
    functional = abs(abl['mean']) >= 0.01 and (abl['ci'][0] > 0 or abl['ci'][1] < 0); row['functional'] = bool(functional)
    P(f"### {n}: macro-F1 {f1.mean()*100:.1f} ± {f1.std(ddof=1)*100:.1f}; fixed gate {ff.mean()*100:.1f}; adaptive - fixed {fmt(abl)}; functional gate: {functional}")
    P(f"- mean gate (H, V) per seed: {[[round(x,3) for x in g] for g in row['gate_mean']]}; fraction of predictions changed by fixing: {[round(x,4) for x in row['frac_changed']]}")
    for a in ['H', 'V', 'HV']:
        r = paired(G, Hc[a], lambda q: q['own']['macro_f1']); row[f'gated-{a}'] = r; P(f'- gated - {a} (macro-F1): {fmt(r)}')
    P(); OUT['E'][n] = row
for n in ['E_lobo_m3_gated', 'E_lobo_m3_tgated']:
    R = lobo(n); Cc = {a: lobo(f'C_lobo_m3_H_{a}') for a in ['H', 'V', 'HV']}
    if R is None: P(f'- {n}: missing'); continue
    row = {}
    g = lobo_rec(R); gf = {s: {f: list(v['fixed_gate_own']['per_bearing'].values())[0]['acc'] for f, v in fs.items()} for s, fs in R.items()}
    S = sorted(g); orm = np.array([np.mean([g[s][f] for f in (0, 3)]) for s in S]); orf = np.array([np.mean([gf[s][f] for f in (0, 3)]) for s in S])
    abl = summ(orm - orf); functional = abs(abl['mean']) >= 0.01 and (abl['ci'][0] > 0 or abl['ci'][1] < 0)
    row.update(OR_mean=orm.tolist(), fixed=orf.tolist(), adaptive_minus_fixed=abl, functional=bool(functional))
    P(f'### {n}: OR recall {orm.mean()*100:.1f}; fixed gate {orf.mean()*100:.1f}; adaptive - fixed {fmt(abl)}; functional gate: {functional}')
    if all(v is not None for v in Cc.values()):
        cr = {a: lobo_rec(v) for a, v in Cc.items()}
        for a in cr:
            SS = [s for s in S if s in cr[a]]; r = summ([np.mean([g[s][f] for f in (0, 3)]) - np.mean([cr[a][s][f] for f in (0, 3)]) for s in SS]); row[f'gated-{a}'] = r
            P(f'- gated - {a} (OR mean): {fmt(r)}')
    P(); OUT['E'][n] = row

# ---------------- Block D
P('## Block D: CWRU common-noise controls (Mamba-3, -8 dB unless stated; test accuracy)'); P()
names = ['D0_DE', 'D0_DEFE_rho0', 'D0_DEFE_rho0.9', 'D2_SUM_rho0', 'D2_SUM_rho0.9', 'D2_DIFF_rho0', 'D2_DIFF_rho0.9',
         'D3_DEFE_rho0.9_d4', 'D3_DEFE_rho0.9_d64', 'D4_DE_r1', 'D4_DEFE_rho0_r1', 'D4_DEFE_rho0.9_r1', 'D4_DE_r2', 'D4_DEFE_rho0_r2', 'D4_DEFE_rho0.9_r2']
D = {n: cw(n) for n in names}
OUT['D'] = {'acc': D}
P('| configuration | accuracy (%) per seed | mean |'); P('|---|---|---|')
for k, v in D.items():
    if v is None: P(f'| {k} | missing | |'); continue
    P(f"| {k} | {', '.join(f'{v[s]*100:.2f}' for s in sorted(v))} | {np.mean(list(v.values()))*100:.2f} |")
P()
def dd(a, b):
    if a is None or b is None: return None
    S = sorted(set(a) & set(b)); return summ([a[s] - b[s] for s in S])
res = {}
res['gain_rho0'] = dd(D['D0_DEFE_rho0'], D['D0_DE']); res['gain_rho0.9'] = dd(D['D0_DEFE_rho0.9'], D['D0_DE'])
res['P6_DIFF0.9-DE'] = dd(D['D2_DIFF_rho0.9'], D['D0_DE']); res['DIFF0.9-DIFF0'] = dd(D['D2_DIFF_rho0.9'], D['D2_DIFF_rho0'])
res['SUM0.9-SUM0'] = dd(D['D2_SUM_rho0.9'], D['D2_SUM_rho0']); res['DIFF0.9-DEFE0.9'] = dd(D['D2_DIFF_rho0.9'], D['D0_DEFE_rho0.9'])
res['SUM0-DE'] = dd(D['D2_SUM_rho0'], D['D0_DE']); res['DIFF0-DE'] = dd(D['D2_DIFF_rho0'], D['D0_DE']); res['SUM0.9-DE'] = dd(D['D2_SUM_rho0.9'], D['D0_DE'])
for dl in [4, 64]: res[f'gain_rho0.9_d{dl}'] = dd(D[f'D3_DEFE_rho0.9_d{dl}'], D['D0_DE'])
for r in [1, 2]:
    res[f'gain_rho0_r{r}'] = dd(D[f'D4_DEFE_rho0_r{r}'], D[f'D4_DE_r{r}']); res[f'gain_rho0.9_r{r}'] = dd(D[f'D4_DEFE_rho0.9_r{r}'], D[f'D4_DE_r{r}'])
for key, te in [('D0_DEFE_rho0.9', 'awgn_corr0.0'), ('D0_DEFE_rho0', 'awgn_corr0.9')]:
    e = cw_eval(key, te)
    if e is not None and D[key] is not None: res[f'mismatch_{key}_same_minus_{te}'] = dd(D[key], e)
for snr in [-6, -8]:
    for tr, te in [('0.9', 'awgn_corr0.0'), ('0', 'awgn_corr0.9')]:
        f = O / 'evalcwru' / f'D1_r4_rho{tr}_snr{snr}_test{te[-3:]}' / f'test_{te}_r0.json'
        if f.exists():
            j = json.load(open(f))['seeds']; res[f'mismatch_r4_rho{tr}_snr{snr}_same_minus_{te}'] = summ([j[s]['recomputed_same'] - j[s]['test_other'] for s in sorted(j)])
for k, v in res.items(): P(f'- {k}: {fmt(v)}')
prim['P5: rho0.9-trained, tested rho0.9 minus tested rho0 (-8 dB, D0)'] = res.get('mismatch_D0_DEFE_rho0.9_same_minus_awgn_corr0.0')
prim['P6: DIFF-only minus DE-only at rho0.9, -8 dB'] = res.get('P6_DIFF0.9-DE')
OUT['D']['contrasts'] = res; P()

# ---------------- Block F
P('## Block F: controlled coverage (test-bearing own-class recall, H+V - H, mean over seeds per unit)'); P()
units = yaml.safe_load(open(CFG / 'F_units.yaml'))
rows = []
for u in units:
    T = u['test'][7:]; tag = f"F_{T}_k{u['k']}_s{u['subset']}"
    h, hv = xc(tag + '_H'), xc(tag + '_HV')
    if h is None or hv is None: rows.append(dict(**u, missing=True)); continue
    S = sorted(set(h) & set(hv)); rh = np.mean([h[s]['own']['per_bearing'][u['test']]['acc'] for s in S]); rhv = np.mean([hv[s]['own']['per_bearing'][u['test']]['acc'] for s in S])
    rows.append(dict(**u, H=float(rh), HV=float(rhv), diff=float(rhv - rh), n_conditions=len({b[7] for b in u['or_bearings']}), seeds=S))
P('| test | k | subset | OR training bearings | conditions | H | H+V | H+V - H |'); P('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r.get('missing'): P(f"| {r['test']} | {r['k']} | {r['subset']} | {r['or_bearings']} | | missing | | |"); continue
    P(f"| {r['test'][7:]} | {r['k']} | {r['subset']} | {', '.join(b[7:] for b in r['or_bearings'])} | {r['n_conditions']} | {r['H']:.2f} | {r['HV']:.2f} | {r['diff']:+.2f} |")
ok = [r for r in rows if not r.get('missing')]
for k in [1, 2, 3]:
    d = [r['diff'] for r in ok if r['k'] == k]
    if d: P(f'- k={k}: mean H+V - H {np.mean(d):+.3f} over {len(d)} units; mean H {np.mean([r["H"] for r in ok if r["k"]==k]):.3f}, H+V {np.mean([r["HV"] for r in ok if r["k"]==k]):.3f}')
d1 = np.array([r['diff'] for r in ok if r['k'] == 1]); d3 = np.array([r['diff'] for r in ok if r['k'] == 3])
if len(d1) and len(d3):
    rng = np.random.default_rng(12345)
    bs = [rng.choice(d3, len(d3)).mean() - rng.choice(d1, len(d1)).mean() for _ in range(NB)]
    obs = d3.mean() - d1.mean(); pool = np.concatenate([d1, d3]); cnt = 0; rp = np.random.default_rng(12345)
    for _ in range(NB):
        q = rp.permutation(pool); cnt += abs(q[len(d1):].mean() - q[:len(d1)].mean()) >= abs(obs) - 1e-12
    p7 = dict(mean=float(obs), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=float((cnt + 1) / (NB + 1)), pos=int((d3 > 0).sum()), neg=int((d3 < 0).sum()), n=int(len(d1) + len(d3)), d=[])
    prim['P7: coverage, mean(H+V - H) at k=3 minus at k=1 (bootstrap over units; two-sided permutation p)'] = p7
    P(f'- P7: k=3 minus k=1: {fmt(p7)}  (pos/neg counts refer to k=3 units)')
OUT['F'] = rows; P()

# ---------------- Block G
P('## Block G: numerical replicates (identical configuration and seed)'); P()
Gd = {}
for rep, orig, kind in [('G_m3_H_HV_rep', 'A_m3_H_HV', 'xc'), ('G_m2_H_H_rep', 'A_m2_H_H', 'xc'), ('G_DEFE_rho0.9_rep', 'D0_DEFE_rho0.9', 'cw')]:
    if kind == 'xc':
        a, b = xc(rep), xc(orig)
        if a is None or b is None: P(f'- {rep}: missing'); continue
        d = [a[s]['own']['macro_f1'] - b[s]['own']['macro_f1'] for s in sorted(set(a) & set(b))]
    else:
        a, b = cw(rep), cw(orig)
        if a is None or b is None: P(f'- {rep}: missing'); continue
        d = [a[s] - b[s] for s in sorted(set(a) & set(b))]
    Gd[rep] = d; P(f"- {rep} vs {orig}: |diff| per seed (pp) {[round(abs(x)*100,2) for x in d]}; max {max(abs(x) for x in d)*100:.2f}")
allxc = [abs(x) for k in ['G_m3_H_HV_rep', 'G_m2_H_H_rep'] for x in Gd.get(k, [])]
fx = float(np.percentile(allxc, 95)) if allxc else None
fc = float(np.percentile([abs(x) for x in Gd['G_DEFE_rho0.9_rep']], 95)) if 'G_DEFE_rho0.9_rep' in Gd else None
OUT['G'] = dict(diffs=Gd, floor_xjtu_macro_f1=fx, floor_cwru_acc=fc)
P(f"- numerical-noise floor (95th percentile of |repeat - original|): XJTU macro-F1 {('%.2f pp' % (fx*100)) if fx is not None else 'not yet available'}; CWRU accuracy {('%.2f pp' % (fc*100)) if fc is not None else 'not yet available'}"); P()

# ---------------- Primary family, Holm
P('## Primary contrasts (Holm over P1-P7, alpha = 0.05)'); P()
ps = sorted([(v['p'], k) for k, v in prim.items() if v is not None and not np.isnan(v['p'])]); m = len(ps); adj = {}; run = 0
for i, (p, k) in enumerate(ps): run = max(run, min(1.0, (m - i) * p)); adj[k] = run
P(f'(Holm over the {m} contrasts available; the family is P1-P7.)'); P()
P('| contrast | result | Holm-adjusted p | significant |'); P('|---|---|---|---|')
for k in sorted(prim):
    v = prim[k]
    if v is None: P(f'| {k} | missing | | |'); continue
    a = adj.get(k, float('nan')); P(f'| {k} | {fmt(v)} | {a:.4f} | {a < 0.05} |')
P(); P('Note: with n = 5 seeds the smallest two-sided exact Wilcoxon p is 0.0625, so P5 and P6 (n = 5) cannot reach alpha = 0.05 by construction; their direction and seed counts are reported.')
OUT['primary'] = {k: (dict(v, holm=adj.get(k)) if v is not None else None) for k, v in prim.items()}
json.dump(OUT, open(A / 'an_r6_results.json', 'w'), indent=1, default=float)
open(A / 'an_r6_results.md', 'w').write('\n'.join(MD) + '\n')
print('written', len(MD), 'lines')
