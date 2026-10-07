"""Round 4 (CWRU noise-key correction): analysis + LaTeX rows, from result files only.
Statistics as fixed in prereg_mst_round4_noisefix.md."""
import json, glob, os, numpy as np
from scipy import stats
_D = os.path.dirname(os.path.abspath(__file__))
O = (os.path.join(_D, 'out_r4') if os.path.isdir(os.path.join(_D, 'out_r4')) else _D) + '/'   # run dir or release folder
SEEDS = [0, 1, 2, 3, 4]
def acc(n):
    d = {}
    for f in glob.glob(O + n + '/seed_*.json'):
        j = json.load(open(f)); d[j['seed']] = j['test_acc_at_best_val'] * 100
    return d
def done(n): return os.path.exists(O + 'done_' + n)
def paired(a, b, side):
    d = np.array([a[s] - b[s] for s in SEEDS]); rng = np.random.default_rng(12345)
    ci = np.percentile([rng.choice(d, len(d)).mean() for _ in range(20000)], [2.5, 97.5])
    p = stats.wilcoxon(d, alternative=side, method='exact').pvalue if np.any(d != 0) else float('nan')
    return dict(mean=float(d.mean()), lo=float(ci[0]), hi=float(ci[1]), p=float(p), pos=int((d > 0).sum()), neg=int((d < 0).sum()), d=d.round(2).tolist())
R = {'arms': {}, 'contrasts': {}}
out = []
for snr in [-8, -6, -4]:
    names = {'DE': f'r4_DE_snr{snr}', 'FE': f'r4_FE_snr{snr}', 'DEFE0': f'r4_DEFE_rho0_snr{snr}'}
    if snr != -4: names.update({'DEFE0.5': f'r4_DEFE_rho0.5_snr{snr}', 'DEFE0.9': f'r4_DEFE_rho0.9_snr{snr}'})
    A = {k: acc(v) for k, v in names.items() if done(v)}
    for k, v in A.items():
        x = np.array([v[s] for s in SEEDS]); R['arms'][f'{k}@{snr}'] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), seeds=x.round(2).tolist())
        out.append(f'{snr} dB {k:8s} {x.mean():6.2f} ± {x.std(ddof=1):4.2f}  {x.round(2).tolist()}')
    if 'DE' in A:
        for k in [k for k in A if k.startswith('DEFE')] + (['FE'] if 'FE' in A else []):
            c = paired(A[k], A['DE'], 'greater' if k.startswith('DEFE') else 'two-sided'); R['contrasts'][f'{k}-DE@{snr}'] = c
            out.append(f'   {k}-DE  {c["mean"]:+.2f} [{c["lo"]:+.2f},{c["hi"]:+.2f}] p={c["p"]:.4f} {c["pos"]}/{c["neg"]}')
    for r in ['0.5', '0.9']:
        if f'DEFE{r}' in A and 'DEFE0' in A:
            c = paired(A[f'DEFE{r}'], A['DEFE0'], 'two-sided'); R['contrasts'][f'dG{r}@{snr}'] = c
            out.append(f'   dG(rho={r})  {c["mean"]:+.2f} [{c["lo"]:+.2f},{c["hi"]:+.2f}] p={c["p"]:.4f} {c["pos"]}/{c["neg"]}')
json.dump(R, open(os.path.join(_D, 'an_r4_results.json'), 'w'), indent=1)
open(os.path.join(_D, 'an_r4_results.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
