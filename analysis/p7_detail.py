"""P7 details for the MST-140500 rewrite (post hoc, requested by the 2026-11-13 internal review).
(1) the pre-specified bootstrap over units (unchanged); (2) the unstratified permutation p used in the Holm
family (documented); (3) a test-bearing-stratified analysis: exact permutation of the k labels within each
test bearing (C(6,3)^3 = 8000 assignments) and a bootstrap that resamples subsets within each (test bearing, k)
cell; (4) per-test-bearing k=3 minus k=1; (5) unique OR training windows per unit (sampling with replacement
where a bearing has fewer windows than its share). Inputs: an_r6_results.json, cfg_r6/F_units.yaml,
audit/onset_audit.csv. Output: p7_detail.json, p7_detail.md."""
import json, itertools, csv, numpy as np, yaml
from pathlib import Path
_W = Path(__file__).resolve().parents[1]
_REL = _W / 'results' / 'mst_round6_20261106'
RUN = _REL if (_REL / 'A_m3_H_H').exists() else _W / 'out_r6'   # released results, else local run directory
CFG = RUN / 'cfg_r6' if (RUN / 'cfg_r6').exists() else _W / 'cfg_r6'
ENV_CSV = (RUN / 'envelope' / 'envelope_onset.csv') if (RUN / 'envelope').exists() else _W / 'r6analysis' / 'envelope_onset.csv'
A = Path(__file__).resolve().parent; W = A.parent
J = json.load(open(A / 'an_r6_results.json'))
F = [r for r in J['F'] if not r.get('missing')]
TB = ['Bearing1_1', 'Bearing2_2', 'Bearing3_1']
avail = {r['bearing']: int(r['windows_H']) for r in csv.DictReader(open(W / 'audit' / 'onset_audit.csv'))}
d = {(t, k): np.array([r['diff'] for r in F if r['test'] == t and r['k'] == k]) for t in TB for k in [1, 2, 3]}
obs = np.mean([d[(t, 3)].mean() - d[(t, 1)].mean() for t in TB])  # equal weight per test bearing (= pooled mean here)
pooled = np.concatenate([d[(t, 3)] for t in TB]).mean() - np.concatenate([d[(t, 1)] for t in TB]).mean()
# stratified exact permutation
per = []
for t in TB:
    v = np.concatenate([d[(t, 1)], d[(t, 3)]]); opts = []
    for c in itertools.combinations(range(6), 3):
        m = np.zeros(6, bool); m[list(c)] = True; opts.append(v[m].mean() - v[~m].mean())
    per.append(np.array(opts))
null = np.array([a + b + c for a in per[0] for b in per[1] for c in per[2]]) / 3
p_strat = float(np.mean(np.abs(null) >= abs(obs) - 1e-12)); p_strat_one = float(np.mean(null >= obs - 1e-12))
# stratified bootstrap
rng = np.random.default_rng(12345); B = 20000; bs = np.empty(B)
for i in range(B):
    bs[i] = np.mean([rng.choice(d[(t, 3)], 3).mean() - rng.choice(d[(t, 1)], 3).mean() for t in TB])
ci_strat = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
pre = J['primary'][[k for k in J['primary'] if k.startswith('P7')][0]]
units = yaml.safe_load(open(CFG / 'F_units.yaml'))
uw = []
for u in units:
    share = 1200 // u['k']; uniq = sum(min(share, avail[b]) for b in u['or_bearings'])
    uw.append(dict(test=u['test'], k=u['k'], subset=u['subset'], or_bearings=u['or_bearings'], unique_or_windows=uniq, replaced=[b for b in u['or_bearings'] if avail[b] < share]))
R = dict(observed=float(obs), pooled_check=float(pooled), prespecified_bootstrap_ci=pre['ci'], permutation_unstratified_p=pre['p'], holm=pre['holm'],
         stratified_exact_permutation_p_two_sided=p_strat, stratified_exact_permutation_p_one_sided=p_strat_one, n_assignments=int(len(null)),
         stratified_bootstrap_ci=ci_strat, per_test_bearing={t: dict(k1=float(d[(t, 1)].mean()), k3=float(d[(t, 3)].mean()), diff=float(d[(t, 3)].mean() - d[(t, 1)].mean())) for t in TB},
         unit_windows=uw)
json.dump(R, open(A / 'p7_detail.json', 'w'), indent=1)
L = ['# P7 details', '', f'observed k3-k1 = {obs*100:+.2f} pp (pooled {pooled*100:+.2f})',
     f"pre-specified bootstrap over units: [{pre['ci'][0]*100:+.1f}, {pre['ci'][1]*100:+.1f}]; unstratified permutation p = {pre['p']:.4f}; Holm {pre['holm']:.3f}",
     f'stratified exact permutation (k labels within test bearing, {len(null)} assignments): two-sided p = {p_strat:.4f}, one-sided p = {p_strat_one:.4f}',
     f'stratified bootstrap (subsets within test bearing x k): [{ci_strat[0]*100:+.1f}, {ci_strat[1]*100:+.1f}]', '']
for t, v in R['per_test_bearing'].items(): L.append(f"{t}: k1 {v['k1']*100:+.1f}, k3 {v['k3']*100:+.1f}, k3-k1 {v['diff']*100:+.1f}")
L.append(''); L.append('units with sampling with replacement:')
for x in uw:
    if x['replaced']: L.append(f"  {x['test']} k={x['k']} s{x['subset']} {x['or_bearings']}: unique OR windows {x['unique_or_windows']} of 1200 (short: {x['replaced']})")
L.append(f"unique OR windows by k: " + '; '.join(f"k={k}: {min(x['unique_or_windows'] for x in uw if x['k']==k)}-{max(x['unique_or_windows'] for x in uw if x['k']==k)}" for k in [1,2,3]))
open(A / 'p7_detail.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
