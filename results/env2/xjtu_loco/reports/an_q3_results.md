## BM3 L3 (train Cond1+Cond2, test Cond3) [('Bearing3_1', 'OR'), ('Bearing3_3', 'IR'), ('Bearing3_4', 'IR'), ('Bearing3_5', 'OR')]
| arm | macro-F1 | Bearing3_1 (OR) | Bearing3_3 (IR) | Bearing3_4 (IR) | Bearing3_5 (OR) |
|---|---|---|---|---|---|
| H | 59.7 ± 9.8 (per seed 53.4 71.0 54.6) | 1.00 | 0.47 | 0.13 | 1.00 |
| V | 79.8 ± 2.3 (per seed 80.9 81.4 77.2) | 1.00 | 0.01 | 1.00 | 0.69 |
| HV | 88.4 ± 8.9 (per seed 78.4 95.6 91.1) | 1.00 | 0.86 | 0.67 | 1.00 |
V-H: +0.00 -0.46 +0.87 -0.31   macroF1 per-seed diff: +27.5 +10.4 +22.6
HV-H: +0.00 +0.39 +0.54 -0.00   macroF1 per-seed diff: +24.9 +24.6 +36.5
HV-V: +0.00 +0.85 -0.33 +0.31   macroF1 per-seed diff: -2.5 +14.2 +13.9

## CNN L3 (train Cond1+Cond2, test Cond3) [('Bearing3_1', 'OR'), ('Bearing3_3', 'IR'), ('Bearing3_4', 'IR'), ('Bearing3_5', 'OR')]
| arm | macro-F1 | Bearing3_1 (OR) | Bearing3_3 (IR) | Bearing3_4 (IR) | Bearing3_5 (OR) |
|---|---|---|---|---|---|
| H | 40.1 ± 0.0 (per seed 40.0 40.0 40.1) | 1.00 | 0.00 | 0.00 | 1.00 |
| V | 88.6 ± 0.2 (per seed 88.6 88.5 88.8) | 1.00 | 0.00 | 0.99 | 1.00 |
| HV | 41.2 ± 1.2 (per seed 40.2 40.9 42.5) | 1.00 | 0.02 | 0.01 | 1.00 |
V-H: +0.00 -0.00 +0.99 +0.00   macroF1 per-seed diff: +48.6 +48.4 +48.7
HV-H: +0.00 +0.02 +0.01 +0.00   macroF1 per-seed diff: +0.2 +0.9 +2.4
HV-V: +0.00 +0.02 -0.98 +0.00   macroF1 per-seed diff: -48.4 -47.6 -46.3
