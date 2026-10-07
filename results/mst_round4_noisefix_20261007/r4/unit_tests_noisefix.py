"""Unit tests for the split-isolated CWRU noise key (noise_key='v2').
Run from the repository root: python results/mst_round4_noisefix_20261007/r4/unit_tests_noisefix.py"""
import sys, importlib.util, numpy as np, json, yaml
sys.path.insert(0, '.')
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
new = load('bearmamba3/data_cwru_leakfree.py', 'dcl_new'); old = load('results/mst_round4_noisefix_20261007/r4/dcl_v1_reference.py', 'dcl_old')
R = {}
rng = np.random.default_rng(7); L = 2048
def fake(n, fid0):  # 2-channel windows with different power per channel
    return [(rng.standard_normal((2, L)).astype(np.float32) * np.array([[1.0], [0.4]], np.float32), i % 4, 1797.0, fid0 + i // 50, 1024 * (i % 50)) for i in range(n)]
tr, te = fake(200, 100), fake(80, 100)
# 1. v1 bit-identical to the superseded implementation
ok = True
for nt in ['awgn', 'awgn_corr0.9']:
    for seed in [0, 3]:
        a = old.CWRULeakfreeDataset([s[:3] for s in te], noise_snr_db=-8.0, noise_type=nt, seed=seed)
        b = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, noise_type=nt, seed=seed)
        ok &= all(np.array_equal(a[i][0].numpy(), b[i][0].numpy()) for i in range(len(te)))
R['1_v1_bit_identical_to_old'] = bool(ok)
def raw_noise(ds, i):
    w = ds.samples[i][0]; ds2 = ds; ds2.normalize = False
    x = ds2[i][0].numpy(); ds2.normalize = True
    n = x - w; return n / n.std(axis=1, keepdims=True)
# 2. cross-split independence (old vs new): same local index, train vs test
def xcorr(mod_kw, nt):
    a = new.CWRULeakfreeDataset(tr, noise_snr_db=-8.0, noise_type=nt, seed=0, split='train', **mod_kw)
    b = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, noise_type=nt, seed=0, split='test', **mod_kw)
    return float(np.mean([abs(np.corrcoef(raw_noise(a, i)[0], raw_noise(b, i)[0])[0, 1]) for i in range(len(te))]))
R['2a_v1_cross_split_|corr|_awgn'] = round(xcorr(dict(noise_key='v1'), 'awgn'), 4)
R['2b_v2_cross_split_|corr|_awgn'] = round(xcorr(dict(noise_key='v2', channels=['DE', 'FE']), 'awgn'), 4)
R['2c_v2_cross_split_|corr|_rho0.9'] = round(xcorr(dict(noise_key='v2', channels=['DE', 'FE']), 'awgn_corr0.9'), 4)
# also identical (fid,start) in train and test -> still independent (split code)
same = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=0, split='train', noise_key='v2', channels=['DE', 'FE'])
tst = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=0, split='test', noise_key='v2', channels=['DE', 'FE'])
R['2d_v2_same_window_id_train_vs_test_|corr|'] = round(float(np.mean([abs(np.corrcoef(raw_noise(same, i)[0], raw_noise(tst, i)[0])[0, 1]) for i in range(len(te))])), 4)
# 3. seed invariance of a given test window's noise
def noise_by_id(ds):
    return {(s[3], s[4]): raw_noise(ds, i) for i, s in enumerate(ds.samples)}
a = noise_by_id(new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=0, split='test', noise_key='v2', channels=['DE', 'FE']))
b = noise_by_id(new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=4, split='test', noise_key='v2', channels=['DE', 'FE']))
R['3_v2_seed_invariant_test_noise'] = bool(all(np.allclose(a[k], b[k], atol=1e-5) for k in a))
o0 = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=0); o4 = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, seed=4)
a1 = {(s[3], s[4]): raw_noise(o0, i) for i, s in enumerate(o0.samples)}; b1 = {(s[3], s[4]): raw_noise(o4, i) for i, s in enumerate(o4.samples)}
R['3b_v1_seed_changes_test_noise_mean|corr|'] = round(float(np.mean([abs(np.corrcoef(a1[k][0], b1[k][0])[0, 1]) for k in a1])), 4)
# 4. physical channel identity and shared draws across arms / rho (unnormalised additive noise, standardised)
te1 = [(s[0][:1], *s[1:]) for s in te]; teF = [(s[0][1:], *s[1:]) for s in te]
DE = new.CWRULeakfreeDataset(te1, noise_snr_db=-8.0, seed=0, split='test', noise_key='v2', channels=['DE'])
FE = new.CWRULeakfreeDataset(teF, noise_snr_db=-8.0, seed=0, split='test', noise_key='v2', channels=['FE'])
D0 = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, noise_type='awgn', seed=0, split='test', noise_key='v2', channels=['DE', 'FE'])
D9 = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, noise_type='awgn_corr0.9', seed=0, split='test', noise_key='v2', channels=['DE', 'FE'])
D00 = new.CWRULeakfreeDataset(te, noise_snr_db=-8.0, noise_type='awgn_corr0.0', seed=0, split='test', noise_key='v2', channels=['DE', 'FE'])
R['4a_DEonly_equals_dual_DE_row'] = bool(all(np.allclose(raw_noise(DE, i)[0], raw_noise(D0, i)[0], atol=1e-5) for i in range(len(te))))
R['4b_FEonly_equals_dual_FE_row_rho0'] = bool(all(np.allclose(raw_noise(FE, i)[0], raw_noise(D0, i)[1], atol=1e-5) for i in range(len(te))))
R['4c_DE_row_shared_across_rho'] = bool(all(np.allclose(raw_noise(D9, i)[0], raw_noise(D0, i)[0], atol=1e-5) for i in range(len(te))))
R['4d_awgn_equals_awgn_corr0.0'] = bool(all(np.array_equal(D0[i][0].numpy(), D00[i][0].numpy()) for i in range(len(te))))
# 5. empirical rho and SNR
r9 = [np.corrcoef(*raw_noise(D9, i))[0, 1] for i in range(len(te))]; r0 = [np.corrcoef(*raw_noise(D0, i))[0, 1] for i in range(len(te))]
R['5a_empirical_rho_0.9'] = round(float(np.mean(r9)), 3); R['5b_empirical_rho_0'] = round(float(np.mean(r0)), 3)
D9.normalize = False
snr = [10 * np.log10((D9.samples[i][0] ** 2).mean(1) / ((D9[i][0].numpy() - D9.samples[i][0]) ** 2).mean(1)) for i in range(len(te))]
R['5c_mean_SNR_per_channel_dB'] = [round(float(x), 3) for x in np.mean(snr, 0)]
# 6. noise fixed per window across epochs (deterministic __getitem__)
R['6_deterministic_getitem'] = bool(np.array_equal(D0[5][0].numpy(), D0[5][0].numpy()))
print(json.dumps(R, indent=1)); json.dump(R, open('/dev/null', 'w'), indent=1)
assert R['1_v1_bit_identical_to_old'] and R['2b_v2_cross_split_|corr|_awgn'] < 0.05 and R['2c_v2_cross_split_|corr|_rho0.9'] < 0.05 and R['2d_v2_same_window_id_train_vs_test_|corr|'] < 0.05
assert R['3_v2_seed_invariant_test_noise'] and R['4a_DEonly_equals_dual_DE_row'] and R['4b_FEonly_equals_dual_FE_row_rho0'] and R['4c_DE_row_shared_across_rho'] and R['4d_awgn_equals_awgn_corr0.0']
assert abs(R['5a_empirical_rho_0.9'] - 0.9) < 0.01 and abs(R['5b_empirical_rho_0']) < 0.01 and all(abs(x + 8) < 0.05 for x in R['5c_mean_SNR_per_channel_dB'])
print('ALL PASS')
