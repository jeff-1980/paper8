## E1 Mamba-2 condition 2->3, environment 2, seeds 0-7
seeds [0, 1, 2, 3, 4, 5, 6, 7] H 44.59±2.42 V 88.51±0.46 H+V 77.59±10.34
  H+V-V -10.92 [-18.35, -5.24] p=0.0078 +0/-8
  H+V-H +33.00 [+25.11, +39.67] p=0.0078 +8/-0
  V-H +43.92 [+41.85, +45.33] p=0.0078 +8/-0
  per-bearing H+V: {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(76.5), 'Bearing3_4': np.float64(41.5), 'Bearing3_5': np.float64(100.0)}
  per-bearing V  : {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(0.0), 'Bearing3_4': np.float64(100.0), 'Bearing3_5': np.float64(99.0)}
## E3 e3_cnn_gated (cnn), condition 2->3, seeds 0-4
seeds [0, 1, 2, 3, 4] gated 52.37±19.29 H 40.15±0.08 V 88.82±0.00 concat 45.14±4.56
  gated-concat +7.23 [-5.19, +26.58] p=1.0000 +3/-2
  gated-V -36.45 [-45.64, -19.06] p=0.0625 +0/-5
  gated-H +12.22 [+3.04, +29.61] p=0.0625 +5/-0
  per-bearing gated: {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(11.1), 'Bearing3_4': np.float64(18.5), 'Bearing3_5': np.float64(100.0)}
  gate SD over test windows / token SD: {'0': (0.0075, None), '1': (0.0101, None), '2': (0.0315, None), '3': (0.0028, None), '4': (0.0075, None), 'adaptive_rule_met': False}
## E3 e3_bm3_gated (bm3), condition 2->3, seeds 0-4
seeds [0, 1, 2, 3, 4] gated 89.93±7.16 H 61.49±9.26 V 83.80±5.48 concat 94.64±3.88
  gated-concat -4.72 [-10.06, +0.73] p=0.3125 +1/-4
  gated-V +6.13 [+1.12, +10.92] p=0.1875 +4/-1
  gated-H +28.44 [+22.13, +33.79] p=0.0625 +5/-0
  per-bearing gated: {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(90.8), 'Bearing3_4': np.float64(70.1), 'Bearing3_5': np.float64(100.0)}
  gate SD over test windows / token SD: {'0': (0.0016, None), '1': (0.0038, None), '2': (0.004, None), '3': (0.0023, None), '4': (0.0067, None), 'adaptive_rule_met': False}
## E3 e3t_cnn_tgated (cnn), condition 2->3, seeds 0-4
seeds [0, 1, 2, 3, 4] gated 47.96±6.74 H 40.15±0.08 V 88.82±0.00 concat 45.14±4.56
  gated-concat +2.82 [-1.39, +7.04] p=0.4375 +3/-2
  gated-V -40.86 [-45.66, -35.23] p=0.0625 +0/-5
  gated-H +7.81 [+3.01, +13.45] p=0.0625 +5/-0
  per-bearing gated: {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(0.3), 'Bearing3_4': np.float64(11.2), 'Bearing3_5': np.float64(100.0)}
  gate SD over test windows / token SD: {'0': (0.0017, 0.034), '1': (0.0016, 0.034), '2': (0.0018, 0.043), '3': (0.0025, 0.038), '4': (0.001, 0.053), 'adaptive_rule_met': False}
## E3 e3t_bm3_tgated (bm3), condition 2->3, seeds 0-4
seeds [0, 1, 2, 3, 4] gated 93.64±4.28 H 61.49±9.26 V 83.80±5.48 concat 94.64±3.88
  gated-concat -1.00 [-3.57, +1.94] p=0.8125 +2/-3
  gated-V +9.85 [+5.89, +13.80] p=0.0625 +5/-0
  gated-H +32.15 [+25.58, +37.26] p=0.0625 +5/-0
  per-bearing gated: {'Bearing3_1': np.float64(100.0), 'Bearing3_3': np.float64(93.2), 'Bearing3_4': np.float64(80.6), 'Bearing3_5': np.float64(100.0)}
  gate SD over test windows / token SD: {'0': (0.0036, 0.047), '1': (0.0026, 0.065), '2': (0.0019, 0.063), '3': (0.0034, 0.073), '4': (0.0036, 0.077), 'adaptive_rule_met': False}
## E2 onset rule V: condition 2->3 Mamba-3, seeds 0-2 (H-rule comparator: xc_bm3_*, seeds 0-2)
  rule V {'H': '61.30±12.09', 'V': '87.47±2.52', 'HV': '94.84±3.61'} | H rule {'H': '64.49±11.34', 'V': '82.63±1.51', 'HV': '93.05±4.50'}
   V-H: rule V +26.17 (seeds [40.8 12.6 25.1]) | H rule +18.14 -> stable
   H+V-V: rule V +7.36 (seeds [ 1.8 12.6  7.7]) | H rule +10.42 -> stable
   H+V-H: rule V +33.53 (seeds [42.6 25.2 32.8]) | H rule +28.55 -> stable
## E2 onset rule V: LOBO OR folds (3_1, 3_5), Mamba-3, seeds 0-2
   H rule V OR recall 74.94±3.94 per fold {0: np.float64(97.5), 3: np.float64(52.4)} | H rule 72.60±3.74
   V rule V OR recall 80.31±7.52 per fold {0: np.float64(74.1), 3: np.float64(86.5)} | H rule 77.59±6.21
   HV rule V OR recall 31.37±6.58 per fold {0: np.float64(40.2), 3: np.float64(22.5)} | H rule 30.39±8.26
   H+V-H: rule V -43.57 (seeds [-45.2 -39.7 -45.9]) | H rule -42.22 -> stable
   H+V-V: rule V -48.94 (seeds [-54.8 -49.5 -42.5]) | H rule -47.20 -> stable
## E2 onset rule last15: condition 2->3 Mamba-3, seeds 0-2 (H-rule comparator: xc_bm3_*, seeds 0-2)
  rule last15 {'H': '50.28±4.52', 'V': '38.16±0.62', 'HV': '44.52±4.08'} | H rule {'H': '64.49±11.34', 'V': '82.63±1.51', 'HV': '93.05±4.50'}
   V-H: rule last15 -12.12 (seeds [-11.4 -17.6  -7.4]) | H rule +18.14 -> NOT stable
   H+V-V: rule last15 +6.36 (seeds [ 5.5  3.4 10.2]) | H rule +10.42 -> stable
   H+V-H: rule last15 -5.76 (seeds [ -5.8 -14.2   2.8]) | H rule +28.55 -> NOT stable
## E2 onset rule last15: LOBO OR folds (3_1, 3_5), Mamba-3, seeds 0-2
   H rule last15 OR recall 10.17±5.78 per fold {0: np.float64(11.1), 3: np.float64(9.3)} | H rule 72.60±3.74
   V rule last15 OR recall 51.15±0.03 per fold {0: np.float64(2.3), 3: np.float64(100.0)} | H rule 77.59±6.21
   HV rule last15 OR recall 4.41±2.89 per fold {0: np.float64(0.9), 3: np.float64(7.9)} | H rule 30.39±8.26
   H+V-H: rule last15 -5.76 (seeds [  1.1  -5.2 -13.2]) | H rule -42.22 -> stable
   H+V-V: rule last15 -46.74 (seeds [-46.4 -44.  -49.8]) | H rule -47.20 -> stable
## E3 e3_lobo_bm3_gated, LOBO OR folds, Mamba-3, seeds 0-2
  gated 67.06±17.80 {0: np.float64(93.5), 3: np.float64(40.6)} | H 72.60±3.74 V 77.59±6.21 concat 30.39±8.26
   gated-H: -5.54 seeds [ -3.6 -24.3  11.3]
   gated-V: -10.53 seeds [ -9.4 -34.2  12. ]
   gated-HV: +36.67 seeds [40.2 10.8 59. ]
  gate stats: {'fold3_seed1': 0.0008, 'fold3_seed0': 0.002, 'fold0_seed1': 0.0032, 'fold3_seed2': 0.0007, 'fold0_seed0': 0.0016, 'fold0_seed2': 0.0028, 'adaptive_rule_met': False}
## E3 e3t_lobo_bm3_tgated, LOBO OR folds, Mamba-3, seeds 0-2
  gated 61.47±19.40 {0: np.float64(89.1), 3: np.float64(33.8)} | H 72.60±3.74 V 77.59±6.21 concat 30.39±8.26
   gated-H: -11.14 seeds [-14.3 -28.    8.8]
   gated-V: -16.13 seeds [-20.  -37.9   9.5]
   gated-HV: +31.08 seeds [29.6  7.1 56.5]
  gate stats: {'fold3_seed1': 0.0013, 'fold3_seed0': 0.0015, 'fold0_seed1': 0.0019, 'fold3_seed2': 0.0032, 'fold0_seed0': 0.0031, 'fold0_seed2': 0.0023, 'adaptive_rule_met': False}
## E4 CWRU Mamba-3, environment 2, seeds 0-4 (accuracy at validation-selected epoch)
  -4 dB  FE 92.55±0.91  DE 97.15±1.16  DE+FE 97.67±0.46 | DE+FE-DE +0.52 [-0.24, +1.27] p=0.4375 +3/-2 one-sided p=0.2188 | FE-DE -4.61 [-4.88, -4.24] p=0.0625 +0/-5
  -6 dB  FE 84.42±0.96  DE 92.42±1.55  DE+FE 94.30±0.95 | DE+FE-DE +1.88 [+0.15, +3.73] p=0.1875 +4/-1 one-sided p=0.0938 | FE-DE -8.00 [-9.33, -6.30] p=0.0625 +0/-5
  -8 dB  FE 76.09±2.20  DE 84.76±0.72  DE+FE 88.03±0.79 | DE+FE-DE +3.27 [+2.39, +4.18] p=0.0625 +5/-0 one-sided p=0.0312 | FE-DE -8.67 [-10.09, -7.24] p=0.0625 +0/-5
## E5 correlated noise, Mamba-3 DE+FE gain over DE, seeds 0-4
  -6 dB rho=0.0: DE+FE 94.30±0.95  gain +1.88 [+0.15, +3.73] p=0.1875 +4/-1
  -6 dB rho=0.5: DE+FE 95.91±1.16  gain +3.48 [+1.76, +5.42] p=0.0625 +5/-0
  -6 dB rho=0.9: DE+FE 99.03±0.68  gain +6.61 [+5.21, +8.18] p=0.0625 +5/-0
   ordering g(0)>g(0.5)>g(0.9): False; g(0.9)<g(0) in 0/5 seeds; gain(0.9)-gain(0) +4.73 [+4.30, +5.24] p=0.0625 +5/-0
  -8 dB rho=0.0: DE+FE 88.03±0.79  gain +3.27 [+2.39, +4.21] p=0.0625 +5/-0
  -8 dB rho=0.5: DE+FE 90.58±0.25  gain +5.82 [+5.36, +6.24] p=0.0625 +5/-0
  -8 dB rho=0.9: DE+FE 97.85±0.89  gain +13.09 [+11.88, +14.12] p=0.0625 +5/-0
   ordering g(0)>g(0.5)>g(0.9): False; g(0.9)<g(0) in 0/5 seeds; gain(0.9)-gain(0) +9.82 [+8.67, +11.00] p=0.0625 +5/-0