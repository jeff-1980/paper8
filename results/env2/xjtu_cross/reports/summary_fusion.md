# Validation-selected late fusion and channel-attention baseline

Late fusion: logits = w*z_H + (1-w)*z_V, w in {0,0.05,...,1} selected per seed (per fold in LOBO) on a source-domain hold-out (last 20% of windows of every source bearing, time-contiguous; excluded from training of the 'sub-models'); criterion = class-balanced NLL. Test data never used. Because the main-arm H/V models were trained on 100% of the source windows (no validation split exists in the trainers), fusion is reported (i) with the 80%-trained sub-models and (ii) with the selected w applied to the full-data models.

## Cross-condition (Cond2->Cond3), backbone cnn1d, seeds [0, 1, 2, 3, 4]

| model | macro-F1 mean±SD | IR recall mean±SD | per-seed macro-F1 |
|---|---|---|---|
| H (full) | 0.4015 ± 0.0008 | 0.0010 ± 0.0007 | 0.402, 0.402, 0.402, 0.400, 0.401 |
| V (full) | 0.8882 ± 0.0000 | 0.7248 ± 0.0000 | 0.888, 0.888, 0.888, 0.888, 0.888 |
| HV early fusion (full) | 0.4514 ± 0.0456 | 0.0508 ± 0.0466 | 0.432, 0.496, 0.422, 0.403, 0.504 |
| ATT channel-attention early fusion (full) | 0.5010 ± 0.0507 | 0.1038 ± 0.0559 | 0.439, 0.528, 0.461, 0.564, 0.514 |
| H_sub (80% source) | 0.4016 ± 0.0012 | 0.0011 ± 0.0011 | 0.403, 0.403, 0.401, 0.401, 0.400 |
| V_sub (80% source) | 0.8861 ± 0.0027 | 0.7201 ± 0.0059 | 0.888, 0.883, 0.888, 0.883, 0.888 |
| Late fusion, val-selected w (sub-models) | 0.8859 ± 0.0025 | 0.7196 ± 0.0056 | 0.887, 0.883, 0.888, 0.883, 0.888 |
| Late fusion, w applied to full H,V models | 0.8882 ± 0.0000 | 0.7248 ± 0.0000 | 0.888, 0.888, 0.888, 0.888, 0.888 |
| [not selected] equal-weight average (sub-models) | 0.4874 ± 0.0462 | 0.0885 ± 0.0509 | 0.556, 0.475, 0.510, 0.455, 0.442 |

Selected w (weight on H logits) per seed: [0.1, 0.0, 0.0, 0.0, 0.0]

Seed-paired differences in macro-F1 (bootstrap 95% CI, 20000 resamples; exact Wilcoxon):

| contrast | mean diff | 95% CI | Wilcoxon p | n+/n- |
|---|---|---|---|---|
| Late fusion, val-selected w (sub-models) - HV early fusion (full) | +0.4345 | [+0.3995, +0.4695] | 0.0625 | 5/0 |
| Late fusion, val-selected w (sub-models) - H (full) | +0.4844 | [+0.4825, +0.4863] | 0.0625 | 5/0 |
| Late fusion, val-selected w (sub-models) - V (full) | -0.0023 | [-0.0042, -0.0004] | 0.1250 | 0/4 |
| Late fusion, val-selected w (sub-models) - V_sub (80% source) | -0.0002 | [-0.0006, +0.0000] | 1.0000 | 0/1 |
| Late fusion, val-selected w (sub-models) - H_sub (80% source) | +0.4843 | [+0.4819, +0.4866] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V models - HV early fusion (full) | +0.4368 | [+0.4018, +0.4718] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V models - V (full) | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 0/0 |
| ATT channel-attention early fusion (full) - HV early fusion (full) | +0.0497 | [+0.0132, +0.1054] | 0.0625 | 5/0 |
| ATT channel-attention early fusion (full) - H (full) | +0.0995 | [+0.0588, +0.1381] | 0.0625 | 5/0 |
| ATT channel-attention early fusion (full) - V (full) | -0.3871 | [-0.4272, -0.3488] | 0.0625 | 0/5 |

Per-test-bearing accuracy of the fusion (sub-models; Bearing3_1 OR, 3_3 IR, 3_4 IR, 3_5 OR):

| seed | w | 3_1 | 3_3 | 3_4 | 3_5 |
|---|---|---|---|---|---|
| 0 | 0.1 | 1.000 | 0.000 | 0.997 | 1.000 |
| 1 | 0.0 | 1.000 | 0.000 | 0.985 | 1.000 |
| 2 | 0.0 | 1.000 | 0.000 | 1.000 | 1.000 |
| 3 | 0.0 | 1.000 | 0.000 | 0.984 | 1.000 |
| 4 | 0.0 | 1.000 | 0.000 | 0.998 | 1.000 |

Per-test-bearing accuracy, fusion with w applied to full models:

| seed | 3_1 | 3_3 | 3_4 | 3_5 |
|---|---|---|---|---|
| 0 | 1.000 | 0.000 | 1.000 | 1.000 |
| 1 | 1.000 | 0.000 | 1.000 | 1.000 |
| 2 | 1.000 | 0.000 | 1.000 | 1.000 |
| 3 | 1.000 | 0.000 | 1.000 | 1.000 |
| 4 | 1.000 | 0.000 | 1.000 | 1.000 |

## LOBO (Cond3), backbone cnn1d, seeds [0, 1, 2, 3, 4]; present-class recall

| model | Bearing3_1 (OR) | Bearing3_3 (IR) | Bearing3_4 (IR) | Bearing3_5 (OR) | OR-fold mean | IR-fold mean | 4-fold mean |
|---|---|---|---|---|---|---|---|
| H | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.007 | 0.789 ± 0.128 | 0.894 ± 0.064 | 0.998 ± 0.004 | 0.946 ± 0.031 |
| V | 0.651 ± 0.111 | 0.168 ± 0.176 | 0.656 ± 0.116 | 0.890 ± 0.046 | 0.771 ± 0.073 | 0.412 ± 0.127 | 0.591 ± 0.035 |
| HV | 0.624 ± 0.262 | 0.986 ± 0.019 | 0.193 ± 0.168 | 0.007 ± 0.007 | 0.316 ± 0.133 | 0.590 ± 0.088 | 0.453 ± 0.061 |
| ATT | 0.423 ± 0.139 | 0.985 ± 0.021 | 0.313 ± 0.227 | 0.043 ± 0.068 | 0.233 ± 0.081 | 0.649 ± 0.119 | 0.441 ± 0.090 |
| Late fusion val-w (sub-models) | 0.905 ± 0.064 | 0.132 ± 0.133 | 1.000 ± 0.000 | 0.483 ± 0.146 | 0.694 ± 0.061 | 0.566 ± 0.067 | 0.630 ± 0.038 |
| Late fusion, w applied to full H,V | 0.841 ± 0.106 | 0.168 ± 0.176 | 0.998 ± 0.005 | 0.789 ± 0.128 | 0.815 ± 0.023 | 0.583 ± 0.089 | 0.699 ± 0.039 |
| H_sub | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0.483 ± 0.146 | 0.741 ± 0.073 | 1.000 ± 0.000 | 0.871 ± 0.037 |
| V_sub | 0.751 ± 0.119 | 0.132 ± 0.133 | 0.688 ± 0.078 | 0.242 ± 0.128 | 0.497 ± 0.106 | 0.410 ± 0.105 | 0.453 ± 0.036 |

Selected w per fold (rows) x seed: Bearing3_1: [0.1, 0.05, 0.25, 0.0, 0.35]; Bearing3_3: [0.0, 0.0, 0.0, 0.0, 0.0]; Bearing3_4: [0.75, 0.65, 0.7, 0.6, 0.75]; Bearing3_5: [1.0, 1.0, 1.0, 1.0, 1.0]

Seed-paired differences (bootstrap 95% CI, exact Wilcoxon):

| contrast | unit | mean diff | 95% CI | Wilcoxon p | n+/n- |
|---|---|---|---|---|---|
| Late fusion val-w (sub-models) - H | Bearing3_1 | -0.0947 | [-0.1453, -0.0468] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H | Bearing3_3 | -0.8679 | [-0.9683, -0.7588] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H | Bearing3_4 | +0.0030 | [-0.0005, +0.0095] | 1.0000 | 1/1 |
| Late fusion val-w (sub-models) - H | Bearing3_5 | -0.3062 | [-0.3867, -0.2286] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H | OR-fold mean | -0.2005 | [-0.2576, -0.1523] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H | IR-fold mean | -0.4325 | [-0.4812, -0.3778] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H | 4-fold mean | -0.3165 | [-0.3627, -0.2703] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - V | Bearing3_1 | +0.2544 | [+0.1404, +0.3695] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - V | Bearing3_3 | -0.0358 | [-0.1454, +0.0783] | 0.6250 | 2/3 |
| Late fusion val-w (sub-models) - V | Bearing3_4 | +0.3434 | [+0.2522, +0.4345] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - V | Bearing3_5 | -0.4077 | [-0.4761, -0.2931] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - V | OR-fold mean | -0.0766 | [-0.1481, -0.0155] | 0.1250 | 1/4 |
| Late fusion val-w (sub-models) - V | IR-fold mean | +0.1538 | [+0.0912, +0.2163] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - V | 4-fold mean | +0.0386 | [+0.0006, +0.0772] | 0.1875 | 4/1 |
| Late fusion val-w (sub-models) - HV | Bearing3_1 | +0.2811 | [+0.0826, +0.4817] | 0.1250 | 4/1 |
| Late fusion val-w (sub-models) - HV | Bearing3_3 | -0.8538 | [-0.9642, -0.7408] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - HV | Bearing3_4 | +0.8065 | [+0.6748, +0.9381] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - HV | Bearing3_5 | +0.4753 | [+0.3860, +0.6043] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - HV | OR-fold mean | +0.3782 | [+0.2401, +0.5162] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - HV | IR-fold mean | -0.0236 | [-0.0982, +0.0540] | 0.6250 | 2/3 |
| Late fusion val-w (sub-models) - HV | 4-fold mean | +0.1773 | [+0.1148, +0.2409] | 0.0625 | 5/0 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_1 | -0.0947 | [-0.1453, -0.0468] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_3 | -0.8679 | [-0.9683, -0.7588] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_4 | +0.0000 | [-0.0006, +0.0008] | 1.0000 | 1/2 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 0/0 |
| Late fusion val-w (sub-models) - H_sub | OR-fold mean | -0.0474 | [-0.0727, -0.0234] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H_sub | IR-fold mean | -0.4340 | [-0.4843, -0.3795] | 0.0625 | 0/5 |
| Late fusion val-w (sub-models) - H_sub | 4-fold mean | -0.2407 | [-0.2770, -0.2043] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - H | Bearing3_1 | -0.1592 | [-0.2389, -0.0701] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - H | Bearing3_3 | -0.8321 | [-0.9583, -0.6817] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - H | Bearing3_4 | +0.0011 | [+0.0000, +0.0033] | 1.0000 | 1/0 |
| Late fusion, w applied to full H,V - H | Bearing3_5 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 0/0 |
| Late fusion, w applied to full H,V - H | OR-fold mean | -0.0796 | [-0.1194, -0.0351] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - H | IR-fold mean | -0.4155 | [-0.4781, -0.3408] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - H | 4-fold mean | -0.2476 | [-0.2900, -0.2058] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - HV | Bearing3_1 | +0.2166 | [+0.0633, +0.3699] | 0.1250 | 4/1 |
| Late fusion, w applied to full H,V - HV | Bearing3_3 | -0.8179 | [-0.9446, -0.6729] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V - HV | Bearing3_4 | +0.8046 | [+0.6750, +0.9342] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V - HV | Bearing3_5 | +0.7815 | [+0.6742, +0.8772] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V - HV | OR-fold mean | +0.4990 | [+0.3993, +0.6069] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V - HV | IR-fold mean | -0.0067 | [-0.0366, +0.0233] | 0.8125 | 3/2 |
| Late fusion, w applied to full H,V - HV | 4-fold mean | +0.2462 | [+0.1978, +0.2946] | 0.0625 | 5/0 |
| ATT - H | Bearing3_1 | -0.5771 | [-0.7003, -0.4932] | 0.0625 | 0/5 |
| ATT - H | Bearing3_3 | -0.0154 | [-0.0312, +0.0000] | 0.5000 | 0/2 |
| ATT - H | Bearing3_4 | -0.6837 | [-0.8619, -0.4930] | 0.0625 | 0/5 |
| ATT - H | Bearing3_5 | -0.7462 | [-0.8312, -0.6620] | 0.0625 | 0/5 |
| ATT - H | OR-fold mean | -0.6617 | [-0.6959, -0.6239] | 0.0625 | 0/5 |
| ATT - H | IR-fold mean | -0.3496 | [-0.4464, -0.2467] | 0.0625 | 0/5 |
| ATT - H | 4-fold mean | -0.5056 | [-0.5721, -0.4352] | 0.0625 | 0/5 |
| ATT - HV | Bearing3_1 | -0.2013 | [-0.4986, +0.0886] | 0.4375 | 2/3 |
| ATT - HV | Bearing3_3 | -0.0012 | [-0.0175, +0.0096] | 0.8750 | 3/1 |
| ATT - HV | Bearing3_4 | +0.1198 | [-0.0267, +0.2663] | 0.3125 | 3/2 |
| ATT - HV | Bearing3_5 | +0.0352 | [-0.0036, +0.0984] | 0.3750 | 3/1 |
| ATT - HV | OR-fold mean | -0.0830 | [-0.2362, +0.0512] | 0.4375 | 2/3 |
| ATT - HV | IR-fold mean | +0.0593 | [-0.0111, +0.1296] | 0.3125 | 3/2 |
| ATT - HV | 4-fold mean | -0.0119 | [-0.1027, +0.0633] | 1.0000 | 3/2 |

## Cross-condition (Cond2->Cond3), backbone bm3, seeds [0, 1, 2, 3, 4]

| model | macro-F1 mean±SD | IR recall mean±SD | per-seed macro-F1 |
|---|---|---|---|
| H (full) | 0.6149 ± 0.0926 | 0.2503 ± 0.1373 | 0.547, 0.769, 0.618, 0.600, 0.540 |
| V (full) | 0.8380 ± 0.0548 | 0.7990 ± 0.1059 | 0.817, 0.844, 0.818, 0.928, 0.783 |
| HV early fusion (full) | 0.9464 ± 0.0388 | 0.8646 ± 0.0950 | 0.903, 0.982, 0.906, 0.965, 0.976 |
| H_sub (80% source) | 0.5760 ± 0.0796 | 0.1966 ± 0.1103 | 0.539, 0.715, 0.517, 0.566, 0.542 |
| V_sub (80% source) | 0.8511 ± 0.0440 | 0.8169 ± 0.1142 | 0.824, 0.858, 0.843, 0.922, 0.808 |
| Late fusion, val-selected w (sub-models) | 0.8652 ± 0.0513 | 0.8150 ± 0.1111 | 0.828, 0.863, 0.843, 0.954, 0.837 |
| Late fusion, w applied to full H,V models | 0.8492 ± 0.0587 | 0.7931 ± 0.0969 | 0.822, 0.852, 0.818, 0.949, 0.805 |
| [not selected] equal-weight average (sub-models) | 0.8941 ± 0.0616 | 0.7494 ± 0.1373 | 0.884, 0.907, 0.794, 0.953, 0.933 |

Selected w (weight on H logits) per seed: [0.05, 0.1, 0.0, 0.25, 0.15]

Seed-paired differences in macro-F1 (bootstrap 95% CI, 20000 resamples; exact Wilcoxon):

| contrast | mean diff | 95% CI | Wilcoxon p | n+/n- |
|---|---|---|---|---|
| Late fusion, val-selected w (sub-models) - HV early fusion (full) | -0.0813 | [-0.1196, -0.0429] | 0.0625 | 0/5 |
| Late fusion, val-selected w (sub-models) - H (full) | +0.2503 | [+0.1688, +0.3170] | 0.0625 | 5/0 |
| Late fusion, val-selected w (sub-models) - V (full) | +0.0272 | [+0.0166, +0.0418] | 0.0625 | 5/0 |
| Late fusion, val-selected w (sub-models) - V_sub (80% source) | +0.0141 | [+0.0027, +0.0259] | 0.1250 | 4/0 |
| Late fusion, val-selected w (sub-models) - H_sub (80% source) | +0.2892 | [+0.2120, +0.3508] | 0.0625 | 5/0 |
| Late fusion, w applied to full H,V models - HV early fusion (full) | -0.0973 | [-0.1451, -0.0515] | 0.0625 | 0/5 |
| Late fusion, w applied to full H,V models - V (full) | +0.0112 | [+0.0035, +0.0189] | 0.1250 | 4/0 |

Per-test-bearing accuracy of the fusion (sub-models; Bearing3_1 OR, 3_3 IR, 3_4 IR, 3_5 OR):

| seed | w | 3_1 | 3_3 | 3_4 | 3_5 |
|---|---|---|---|---|---|
| 0 | 0.05 | 1.000 | 0.133 | 1.000 | 0.744 |
| 1 | 0.1 | 1.000 | 0.006 | 1.000 | 0.917 |
| 2 | 0.0 | 1.000 | 0.013 | 1.000 | 0.847 |
| 3 | 0.25 | 0.991 | 0.925 | 1.000 | 0.889 |
| 4 | 0.15 | 0.921 | 0.562 | 1.000 | 0.715 |

Per-test-bearing accuracy, fusion with w applied to full models:

| seed | 3_1 | 3_3 | 3_4 | 3_5 |
|---|---|---|---|---|
| 0 | 1.000 | 0.050 | 1.000 | 0.756 |
| 1 | 1.000 | 0.048 | 1.000 | 0.863 |
| 2 | 1.000 | 0.013 | 1.000 | 0.758 |
| 3 | 0.993 | 0.848 | 1.000 | 0.899 |
| 4 | 0.958 | 0.283 | 1.000 | 0.655 |

## LOBO (Cond3), backbone bm3, seeds [0, 1, 2]; present-class recall

| model | Bearing3_1 (OR) | Bearing3_3 (IR) | Bearing3_4 (IR) | Bearing3_5 (OR) | OR-fold mean | IR-fold mean | 4-fold mean |
|---|---|---|---|---|---|---|---|
| H | 0.987 ± 0.022 | 0.994 ± 0.011 | 0.951 ± 0.042 | 0.465 ± 0.053 | 0.726 ± 0.037 | 0.972 ± 0.026 | 0.849 ± 0.032 |
| V | 0.730 ± 0.138 | 0.999 ± 0.002 | 0.544 ± 0.127 | 0.822 ± 0.028 | 0.776 ± 0.062 | 0.771 ± 0.064 | 0.774 ± 0.028 |
| HV | 0.387 ± 0.031 | 0.994 ± 0.002 | 0.948 ± 0.015 | 0.221 ± 0.190 | 0.304 ± 0.083 | 0.971 ± 0.008 | 0.637 ± 0.037 |
| Late fusion val-w (sub-models) | 0.989 ± 0.019 | 1.000 ± 0.000 | 0.942 ± 0.063 | 0.262 ± 0.091 | 0.626 ± 0.054 | 0.971 ± 0.032 | 0.798 ± 0.041 |
| Late fusion, w applied to full H,V | 0.990 ± 0.017 | 1.000 ± 0.000 | 0.920 ± 0.095 | 0.551 ± 0.084 | 0.771 ± 0.033 | 0.960 ± 0.048 | 0.865 ± 0.009 |
| H_sub | 0.986 ± 0.023 | 0.994 ± 0.010 | 0.962 ± 0.030 | 0.297 ± 0.049 | 0.642 ± 0.032 | 0.978 ± 0.019 | 0.810 ± 0.023 |
| V_sub | 0.768 ± 0.142 | 0.997 ± 0.005 | 0.575 ± 0.122 | 0.164 ± 0.039 | 0.466 ± 0.075 | 0.786 ± 0.062 | 0.626 ± 0.037 |

Selected w per fold (rows) x seed: Bearing3_1: [0.8, 0.8, 0.75]; Bearing3_3: [0.25, 0.85, 0.8]; Bearing3_4: [0.35, 1.0, 1.0]; Bearing3_5: [0.6, 0.95, 1.0]

Seed-paired differences (bootstrap 95% CI, exact Wilcoxon):

| contrast | unit | mean diff | 95% CI | Wilcoxon p | n+/n- |
|---|---|---|---|---|---|
| Late fusion val-w (sub-models) - H | Bearing3_1 | +0.0020 | [+0.0000, +0.0060] | 1.0000 | 1/0 |
| Late fusion val-w (sub-models) - H | Bearing3_3 | +0.0063 | [+0.0000, +0.0188] | 1.0000 | 1/0 |
| Late fusion val-w (sub-models) - H | Bearing3_4 | -0.0090 | [-0.0340, +0.0040] | 1.0000 | 2/1 |
| Late fusion val-w (sub-models) - H | Bearing3_5 | -0.2027 | [-0.2352, -0.1405] | 0.2500 | 0/3 |
| Late fusion val-w (sub-models) - H | OR-fold mean | -0.1004 | [-0.1176, -0.0702] | 0.2500 | 0/3 |
| Late fusion val-w (sub-models) - H | IR-fold mean | -0.0014 | [-0.0076, +0.0020] | 1.0000 | 2/1 |
| Late fusion val-w (sub-models) - H | 4-fold mean | -0.0509 | [-0.0605, -0.0341] | 0.2500 | 0/3 |
| Late fusion val-w (sub-models) - V | Bearing3_1 | +0.2591 | [+0.1126, +0.3398] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - V | Bearing3_3 | +0.0014 | [+0.0000, +0.0042] | 1.0000 | 1/0 |
| Late fusion val-w (sub-models) - V | Bearing3_4 | +0.3979 | [+0.1820, +0.5380] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - V | Bearing3_5 | -0.5596 | [-0.6821, -0.4509] | 0.2500 | 0/3 |
| Late fusion val-w (sub-models) - V | OR-fold mean | -0.1502 | [-0.2165, -0.0630] | 0.2500 | 0/3 |
| Late fusion val-w (sub-models) - V | IR-fold mean | +0.1997 | [+0.0910, +0.2690] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - V | 4-fold mean | +0.0247 | [-0.0401, +0.0880] | 0.7500 | 2/1 |
| Late fusion val-w (sub-models) - HV | Bearing3_1 | +0.6023 | [+0.5804, +0.6438] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - HV | Bearing3_3 | +0.0062 | [+0.0042, +0.0083] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - HV | Bearing3_4 | -0.0055 | [-0.0902, +0.0570] | 1.0000 | 2/1 |
| Late fusion val-w (sub-models) - HV | Bearing3_5 | +0.0412 | [-0.1781, +0.2312] | 0.7500 | 2/1 |
| Late fusion val-w (sub-models) - HV | OR-fold mean | +0.3218 | [+0.2329, +0.4070] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - HV | IR-fold mean | +0.0004 | [-0.0420, +0.0326] | 1.0000 | 2/1 |
| Late fusion val-w (sub-models) - HV | 4-fold mean | +0.1611 | [+0.1328, +0.2087] | 0.2500 | 3/0 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_1 | +0.0028 | [+0.0000, +0.0084] | 1.0000 | 1/0 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_3 | +0.0056 | [+0.0000, +0.0167] | 1.0000 | 1/0 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_4 | -0.0200 | [-0.0601, +0.0000] | 1.0000 | 0/1 |
| Late fusion val-w (sub-models) - H_sub | Bearing3_5 | -0.0349 | [-0.0934, +0.0000] | 0.5000 | 0/2 |
| Late fusion val-w (sub-models) - H_sub | OR-fold mean | -0.0161 | [-0.0425, +0.0000] | 0.5000 | 0/2 |
| Late fusion val-w (sub-models) - H_sub | IR-fold mean | -0.0072 | [-0.0217, +0.0000] | 1.0000 | 0/1 |
| Late fusion val-w (sub-models) - H_sub | 4-fold mean | -0.0117 | [-0.0321, +0.0000] | 0.5000 | 0/2 |
| Late fusion, w applied to full H,V - H | Bearing3_1 | +0.0026 | [+0.0000, +0.0079] | 1.0000 | 1/0 |
| Late fusion, w applied to full H,V - H | Bearing3_3 | +0.0063 | [+0.0000, +0.0188] | 1.0000 | 1/0 |
| Late fusion, w applied to full H,V - H | Bearing3_4 | -0.0311 | [-0.0934, +0.0000] | 1.0000 | 0/1 |
| Late fusion, w applied to full H,V - H | Bearing3_5 | +0.0865 | [+0.0000, +0.2433] | 0.5000 | 2/0 |
| Late fusion, w applied to full H,V - H | OR-fold mean | +0.0446 | [+0.0000, +0.1256] | 0.5000 | 2/0 |
| Late fusion, w applied to full H,V - H | IR-fold mean | -0.0124 | [-0.0373, +0.0000] | 1.0000 | 0/1 |
| Late fusion, w applied to full H,V - H | 4-fold mean | +0.0161 | [+0.0000, +0.0442] | 0.5000 | 2/0 |
| Late fusion, w applied to full H,V - HV | Bearing3_1 | +0.6030 | [+0.5823, +0.6438] | 0.2500 | 3/0 |
| Late fusion, w applied to full H,V - HV | Bearing3_3 | +0.0062 | [+0.0042, +0.0083] | 0.2500 | 3/0 |
| Late fusion, w applied to full H,V - HV | Bearing3_4 | -0.0277 | [-0.1495, +0.0538] | 1.0000 | 2/1 |
| Late fusion, w applied to full H,V - HV | Bearing3_5 | +0.3304 | [+0.0733, +0.5464] | 0.2500 | 3/0 |
| Late fusion, w applied to full H,V - HV | OR-fold mean | +0.4667 | [+0.3586, +0.5644] | 0.2500 | 3/0 |
| Late fusion, w applied to full H,V - HV | IR-fold mean | -0.0107 | [-0.0716, +0.0311] | 1.0000 | 2/1 |
| Late fusion, w applied to full H,V - HV | 4-fold mean | +0.2280 | [+0.1948, +0.2464] | 0.2500 | 3/0 |