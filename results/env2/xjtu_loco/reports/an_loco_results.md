## BM3 L2: test bearings ['Bearing2_1', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5'] labels ['IR', 'OR', 'OR', 'OR']; seeds used 3
| arm | macro-F1 (mean±SD over seeds) | Bearing2_1 (IR) | Bearing2_2 (OR) | Bearing2_4 (OR) | Bearing2_5 (OR) |
|---|---|---|---|---|---|
| H | 45.2 ± 1.0 | 0.94±0.02 | 0.93±0.03 | 1.00±0.00 | 0.02±0.01 |
| V | 94.5 ± 2.0 | 0.85±0.04 | 0.99±0.01 | 1.00±0.00 | 1.00±0.00 |
| HV | 74.8 ± 7.7 | 0.86±0.03 | 0.98±0.02 | 1.00±0.00 | 0.74±0.13 |

Contrasts, per test bearing (mean over seeds of acc difference) and number of bearings with positive difference:
| contrast | Bearing2_1 | Bearing2_2 | Bearing2_4 | Bearing2_5 | #bearings + / - | mean over bearings [bootstrap over bearings] |
|---|---|---|---|---|---|---|
| V-H | -0.10 | +0.06 | +0.00 | +0.98 | 2/1 | +0.234 [-0.059, +0.733] |
| HV-H | -0.09 | +0.05 | +0.00 | +0.72 | 2/1 | +0.170 [-0.054, +0.541] |
| HV-V | +0.01 | -0.01 | +0.00 | -0.26 | 0/1 | -0.064 [-0.193, +0.006] |

## BM3 L1: test bearings ['Bearing1_1', 'Bearing1_2', 'Bearing1_3'] labels ['OR', 'OR', 'OR']; seeds used 3
| arm | macro-F1 (mean±SD over seeds) | Bearing1_1 (OR) | Bearing1_2 (OR) | Bearing1_3 (OR) |
|---|---|---|---|---|
| H | 49.9 ± 0.1 | 0.99±0.01 | 1.00±0.00 | 1.00±0.00 |
| V | 50.0 ± 0.0 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| HV | 49.8 ± 0.2 | 1.00±0.01 | 0.98±0.02 | 1.00±0.00 |

Contrasts, per test bearing (mean over seeds of acc difference) and number of bearings with positive difference:
| contrast | Bearing1_1 | Bearing1_2 | Bearing1_3 | #bearings + / - | mean over bearings [bootstrap over bearings] |
|---|---|---|---|---|---|
| V-H | +0.01 | -0.00 | +0.00 | 0/0 | +0.002 [-0.002, +0.008] |
| HV-H | +0.01 | -0.02 | +0.00 | 0/0 | -0.004 [-0.017, +0.006] |
| HV-V | -0.00 | -0.02 | +0.00 | 0/0 | -0.006 [-0.016, +0.000] |

## CNN L2: test bearings ['Bearing2_1', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5'] labels ['IR', 'OR', 'OR', 'OR']; seeds used 3
| arm | macro-F1 (mean±SD over seeds) | Bearing2_1 (IR) | Bearing2_2 (OR) | Bearing2_4 (OR) | Bearing2_5 (OR) |
|---|---|---|---|---|---|
| H | 45.4 ± 0.8 | 0.86±0.06 | 0.99±0.00 | 1.00±0.00 | 0.02±0.02 |
| V | 92.1 ± 1.0 | 1.00±0.00 | 0.89±0.02 | 1.00±0.00 | 1.00±0.00 |
| HV | 70.1 ± 8.5 | 1.00±0.00 | 0.99±0.00 | 1.00±0.00 | 0.57±0.17 |

Contrasts, per test bearing (mean over seeds of acc difference) and number of bearings with positive difference:
| contrast | Bearing2_1 | Bearing2_2 | Bearing2_4 | Bearing2_5 | #bearings + / - | mean over bearings [bootstrap over bearings] |
|---|---|---|---|---|---|---|
| V-H | +0.14 | -0.10 | +0.00 | +0.98 | 2/1 | +0.252 [-0.051, +0.732] |
| HV-H | +0.14 | -0.00 | +0.00 | +0.55 | 2/0 | +0.170 [-0.001, +0.412] |
| HV-V | -0.00 | +0.10 | +0.00 | -0.43 | 1/1 | -0.082 [-0.320, +0.075] |

## CNN L1: test bearings ['Bearing1_1', 'Bearing1_2', 'Bearing1_3'] labels ['OR', 'OR', 'OR']; seeds used 3
| arm | macro-F1 (mean±SD over seeds) | Bearing1_1 (OR) | Bearing1_2 (OR) | Bearing1_3 (OR) |
|---|---|---|---|---|
| H | 50.0 ± 0.0 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| V | 50.0 ± 0.0 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| HV | 50.0 ± 0.0 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

Contrasts, per test bearing (mean over seeds of acc difference) and number of bearings with positive difference:
| contrast | Bearing1_1 | Bearing1_2 | Bearing1_3 | #bearings + / - | mean over bearings [bootstrap over bearings] |
|---|---|---|---|---|---|
| V-H | +0.00 | +0.00 | -0.00 | 0/0 | -0.000 [-0.001, +0.000] |
| HV-H | +0.00 | +0.00 | +0.00 | 0/0 | +0.000 [+0.000, +0.000] |
| HV-V | +0.00 | +0.00 | +0.00 | 0/0 | +0.000 [+0.000, +0.001] |

## OR coverage sweep: test bearing Bearing3_1 (OR); IR training bearings fixed (2_1,3_3,3_4); OR training bearings k
| k OR train bearings | H acc (per seed) | HV acc (per seed) | HV-H mean |
|---|---|---|---|
| 1 | 0.96, 1.00 | 0.26, 0.19 | -0.755 |
| 2 | 1.00, 1.00 | 0.92, 0.98 | -0.047 |
| 4 | 1.00, 1.00 | 1.00, 1.00 | +0.000 |
| 7 | 1.00, 1.00 | 1.00, 1.00 | +0.000 |