"""Holm adjustment of the seven primary contrasts with P7's p replaced by the test-bearing-stratified exact
permutation p (p7_detail.json). Sensitivity analysis requested in the second internal review; the reported
primary family (an_r6_results.json) is unchanged. Output: holm_strat.json."""
import json
from pathlib import Path
A = Path(__file__).parent
P = json.load(open(A / 'an_r6_results.json'))['primary']; p7 = json.load(open(A / 'p7_detail.json'))
ps = {k.split(':')[0]: v['p'] for k, v in P.items()}
ps_s = dict(ps); ps_s['P7'] = p7['stratified_exact_permutation_p_two_sided']
def holm(d):
    ks = sorted(d, key=d.get); m = len(ks); out = {}; run = 0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * d[k])); out[k] = run
    return out
H0 = {k.split(':')[0]: v['holm'] for k, v in P.items()}
H1 = holm(ps_s)
assert all(abs(holm(ps)[k] - H0[k]) < 1e-12 for k in H0), 'reproduces reported Holm'
res = dict(raw_p_stratified=ps_s, holm_reported=H0, holm_with_stratified_P7=H1)
json.dump(res, open(A / 'holm_strat.json', 'w'), indent=1)
for k in sorted(H1): print(k, round(ps_s[k], 5), round(H0[k], 4), '->', round(H1[k], 4))
