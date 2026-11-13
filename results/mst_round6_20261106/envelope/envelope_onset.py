"""R1.3: independent fault indicator for the XJTU-SY degraded phase (prereg_mst_round6.md section 9).
Read-only analysis of the raw CSVs; no training. Outputs envelope_onset.csv and envelope_traces.json."""
import glob, os, json, csv, numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
R = 'data/XJTU-SY_Bearing_Datasets'; FS = 25600.0
COND = {'1': ('35Hz12kN', 35.0), '2': ('37.5Hz11kN', 37.5), '3': ('40Hz10kN', 40.0)}
B = {'Bearing1_1': 'OR', 'Bearing1_2': 'OR', 'Bearing1_3': 'OR', 'Bearing2_1': 'IR', 'Bearing2_2': 'OR', 'Bearing2_4': 'OR',
     'Bearing2_5': 'OR', 'Bearing3_1': 'OR', 'Bearing3_3': 'IR', 'Bearing3_4': 'IR', 'Bearing3_5': 'OR'}
NB, d, D = 8, 7.92, 34.55
SOS = butter(4, [1000, 10000], btype='bandpass', fs=FS, output='sos')
AUD = {r['bearing']: r for r in csv.DictReader(open('audit/onset_audit.csv'))}
def indicator(x, fc):
    e = np.abs(hilbert(sosfiltfilt(SOS, x - x.mean()))) ** 2; e -= e.mean()
    S = np.abs(np.fft.rfft(e * np.hanning(len(e)))); f = np.fft.rfftfreq(len(e), 1 / FS)
    bg = np.median(S[(f >= 10) & (f <= 500)])
    peak = sum(S[(f >= h * fc * 0.98) & (f <= h * fc * 1.02)].max() for h in (1, 2, 3))
    return float(peak / bg)
def onset(v, w=5):
    n = len(v); nb = max(1, min(20, n // 5)); mu, sd = np.mean(v[:nb]), np.std(v[:nb], ddof=1) if nb > 1 else 0.0
    r = np.convolve(v, np.ones(w) / w, 'valid'); ex = np.where(r > mu + 3 * sd)[0]
    return (int(ex[0]) if len(ex) else None), float(mu), float(sd)
rows, traces = [], {}
for b, cls in B.items():
    cond, fr = COND[b[7]]
    fc = (NB / 2) * fr * (1 - d / D) if cls == 'OR' else (NB / 2) * fr * (1 + d / D)
    fs_ = sorted(glob.glob(f'{R}/{cond}/{b}/*.csv'), key=lambda p: int(os.path.basename(p)[:-4]))
    ind = {0: [], 1: []}
    for p in fs_:
        x = np.loadtxt(p, delimiter=',', skiprows=1)
        for c in (0, 1): ind[c].append(indicator(x[:, c], fc))
    n = len(fs_); a = AUD[b]
    rules = {'H': int(a['onset_H']), 'V': int(a['onset_V']), 'last15': int(n * 0.85)}
    traces[b] = {'fc_hz': fc, 'H': ind[0], 'V': ind[1]}
    for ch, c in (('H', 0), ('V', 1)):
        o, mu, sd = onset(np.array(ind[c]))
        row = dict(bearing=b, cls=cls, n_files=n, fault_freq_hz=round(fc, 2), channel=ch, indicator_onset=o, baseline_mean=round(mu, 3), baseline_sd=round(sd, 3))
        for rn, ro in rules.items():
            row[f'onset_{rn}'] = ro
            row[f'offset_{rn}_minus_indicator'] = None if o is None else ro - o
            row[f'frac_{rn}_segment_before_indicator'] = None if o is None else round(max(0, o - ro) / (n - ro), 3)
        rows.append(row)
    print(b, cls, n, 'indicator onset H/V:', [r['indicator_onset'] for r in rows[-2:]], 'rules', rules, flush=True)
with open('r6analysis/envelope_onset.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(traces, open('r6analysis/envelope_traces.json', 'w'))
