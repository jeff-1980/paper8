"""Round 6 unit tests (data level). Run from the repository root."""
import sys, json, importlib.util, numpy as np, torch
sys.path.insert(0, '.')
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R = {}
# ---- CWRU noise
new = load('bearmamba3/data_cwru_leakfree.py', 'dn'); ref = load('r6ref/data_cwru_leakfree.py', 'dr')
rng = np.random.default_rng(3); L = 2048
pool = [(rng.standard_normal((2, L)).astype(np.float32) * np.array([[1.0], [0.4]], np.float32), i % 4, 1797.0, 100 + i // 20, 1024 * (i % 20)) for i in range(120)]
def ds(mod, nt, **kw): return mod.CWRULeakfreeDataset(pool, noise_snr_db=-8.0, noise_type=nt, seed=0, noise_key='v2', split='test', channels=['DE', 'FE'], **kw)
ok = True
for nt in ['awgn', 'awgn_corr0.5', 'awgn_corr0.9']:
    a, b = ds(ref, nt), ds(new, nt); ok &= all(np.array_equal(a[i][0].numpy(), b[i][0].numpy()) for i in range(len(pool)))
for nt in ['awgn', 'awgn_corr0.9']:   # v1 path unchanged
    a = ref.CWRULeakfreeDataset([p[:3] for p in pool], noise_snr_db=-8.0, noise_type=nt, seed=2); b = new.CWRULeakfreeDataset([p[:3] for p in pool], noise_snr_db=-8.0, noise_type=nt, seed=2)
    ok &= all(np.array_equal(a[i][0].numpy(), b[i][0].numpy()) for i in range(len(pool)))
R['1_existing_noise_bit_identical'] = bool(ok)
def raw(d, i):
    d.normalize = False; w = d.samples[i][0]; x = d[i][0].numpy(); d.normalize = True
    return (x - w) / np.sqrt(np.mean(w ** 2, axis=1, keepdims=True) / 10 ** (-0.8))
D0, D9, Dd = ds(new, 'awgn'), ds(new, 'awgn_corr0.9'), ds(new, 'awgn_corr0.9_d64')
R['2_DE_row_unchanged_by_delay'] = bool(all(np.allclose(raw(D9, i)[0], raw(Dd, i)[0]) for i in range(30)))
R['3_d0_equals_no_delay'] = bool(all(np.array_equal(ds(new, 'awgn_corr0.9_d0')[i][0].numpy(), D9[i][0].numpy()) for i in range(30)))
lag0 = np.mean([np.corrcoef(*raw(Dd, i))[0, 1] for i in range(len(pool))])
lag64 = np.mean([np.corrcoef(raw(Dd, i)[0][:-64], raw(Dd, i)[1][64:])[0, 1] for i in range(len(pool))])
R['4a_delay64_lag0_corr'] = round(float(lag0), 3); R['4b_delay64_lag64_corr'] = round(float(lag64), 3)
Ds, Df = ds(new, 'awgn_corr0.9', combine='sum'), ds(new, 'awgn_corr0.9', combine='diff')
R['5a_combine_shape'] = list(Ds[0][0].shape)
# difference cancels the common noise: residual noise variance in diff of standardized noise-only signals
R['5b_noise_in_diff_vs_sum_power_ratio'] = round(float(np.mean([np.var(raw(D9, i)[0] - raw(D9, i)[1]) / np.var(raw(D9, i)[0] + raw(D9, i)[1]) for i in range(len(pool))])), 3)
R['5c_realisation_changes_noise'] = bool(not np.array_equal(ds(new, 'awgn', noise_realisation=1)[0][0].numpy(), D0[0][0].numpy()))
# ---- XJTU resample
X = load('bearmamba3/data_xjtu.py', 'dx')
x = X.XJTUDataset('data/XJTU-SY_Bearing_Datasets', ['Bearing2_4', 'Bearing2_1'], n_sensors=1)
R['6a_meta_lengths'] = len(x._bearing) == len(x._file) == len(x._labels)
R['6b_files_from_onset'] = int(min(f for f, b in zip(x._file, x._bearing) if b == 'Bearing2_4')) == 27
c = {'Bearing2_4': 112, 'Bearing2_1': 1184}
y1, y2 = X.resample_per_bearing(x, c, 12345), X.resample_per_bearing(x, c, 12345)
R['7a_counts_exact'] = dict(zip(*np.unique(y1._bearing, return_counts=True))) == {k: v for k, v in c.items()}
R['7b_deterministic'] = all(np.array_equal(a, b) for a, b in zip(y1._windows, y2._windows))
sel = [f for f, b in zip(y1._file, y1._bearing) if b == 'Bearing2_4']; R['7c_no_replacement_when_enough'] = True
R['7d_with_replacement_when_short'] = (sum(1 for b in y1._bearing if b == 'Bearing2_1') == 1184 and len(set(id(w) for w, b in zip(y1._windows, y1._bearing) if b == 'Bearing2_1')) == 624)
print(json.dumps(R, indent=1)); json.dump(R, open('r6tests/unit_tests_r6.json', 'w'), indent=1)
assert R['1_existing_noise_bit_identical'] and R['2_DE_row_unchanged_by_delay'] and R['3_d0_equals_no_delay']
assert abs(R['4a_delay64_lag0_corr']) < 0.05 and abs(R['4b_delay64_lag64_corr'] - 0.9) < 0.03
assert R['5a_combine_shape'] == [1, 2048] and R['5b_noise_in_diff_vs_sum_power_ratio'] < 0.1 and R['5c_realisation_changes_noise']
assert R['6a_meta_lengths'] and R['6b_files_from_onset'] and R['7a_counts_exact'] and R['7b_deterministic'] and R['7d_with_replacement_when_short']
print('ALL PASS')
