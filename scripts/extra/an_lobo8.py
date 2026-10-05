"""Cond3 LOBO BM3 single-channel controls, 8 seeds: seeds 0-2 from out/ (original control run), 3-7 from out_s/ (same YAML, seeds only)."""
import sys, os, json, numpy as np
sys.path.insert(0, 'scripts/extra'); from stats import paired, ms
BEAR = ['Bearing3_1','Bearing3_3','Bearing3_4','Bearing3_5']; LAB = ['OR','IR','IR','OR']
def load(a):
    rec = np.full((4, 8), np.nan)
    for s in range(8):
        d = 'out' if s < 3 else 'out_s'
        for f in range(4):
            p = f'{d}/lobo_bm3_{a}/fold{f}_seed{s}.json'
            if os.path.exists(p):
                rec[f, s] = json.load(open(p))['per_class_recall'][0 if LAB[f]=='OR' else 1]
    return rec
D = {a: load(a) for a in ['H','V','HV']}
seeds = [s for s in range(8) if all(not np.isnan(D[a][:, s]).any() for a in D)]
L = [f'# Cond3 LOBO BM3 controls, present-class recall, seeds {seeds} (n={len(seeds)})', '',
     '| arm | ' + ' | '.join(f'{b} ({l})' for b,l in zip(BEAR,LAB)) + ' | OR-fold mean | IR-fold mean | 4-fold mean |', '|---|' + '---|'*7]
for a, r in D.items():
    r = r[:, seeds]; orm = r[[0,3]].mean(0); irm = r[[1,2]].mean(0); m4 = r.mean(0)
    L.append(f'| {a} | ' + ' | '.join('%.3f ± %.3f' % ms(r[f]) for f in range(4)) + ' | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |' % (*ms(orm), *ms(irm), *ms(m4)))
L += ['', '| contrast | unit | mean diff | 95% CI | exact Wilcoxon p | n+/n- |', '|---|---|---|---|---|---|']
for x, y in [('V','H'),('HV','H'),('HV','V')]:
    X, Y = D[x][:, seeds], D[y][:, seeds]
    units = [(f'{BEAR[f]} ({LAB[f]})', X[f]-Y[f]) for f in range(4)] + [('OR-fold mean', X[[0,3]].mean(0)-Y[[0,3]].mean(0)), ('IR-fold mean', X[[1,2]].mean(0)-Y[[1,2]].mean(0)), ('4-fold mean', X.mean(0)-Y.mean(0))]
    for nm, d in units:
        r = paired(d, np.zeros_like(d))
        L.append(f"| {x}-{y} | {nm} | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4f} | {r['n_pos']}/{r['n_neg']} |")
L += ['', 'Per-seed present-class recall:'] + [f'- {a} {BEAR[f]}: ' + ', '.join('%.3f' % v for v in D[a][f, seeds]) for a in D for f in range(4)]
open('summary_lobo_bm3_8seed.md', 'w').write('\n'.join(L)); print('\n'.join(L[:24]))
