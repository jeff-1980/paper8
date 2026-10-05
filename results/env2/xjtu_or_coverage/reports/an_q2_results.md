Accuracy on the test bearing (own-class recall; all windows of a bearing share one label), per seed 0,1,2. IR training bearings fixed: 2_1, 3_3, 3_4.

| test | ordering | k | OR training bearings added | H acc | HV acc | HV-H mean |
|---|---|---|---|---|---|---|
| Bearing3_1 (Cond3) | SAME | 1 | 3_5 | 0.95, 1.00, 1.00 | 0.26, 0.19, 0.22 | -0.764 |
| Bearing3_1 (Cond3) | SAME | 2 | 3_5, 2_2 | 1.00, 1.00, 1.00 | 0.92, 0.98, 1.00 | -0.033 |
| Bearing3_1 (Cond3) | SAME | 3 | 3_5, 2_2, 2_4 | - | 1.00, 1.00, 1.00 | - |
| Bearing3_1 (Cond3) | OTHER | 1 | 2_2 | 1.00, 1.00, 1.00 | 0.18, 0.51, 0.59 | -0.574 |
| Bearing3_1 (Cond3) | OTHER | 2 | 2_2, 2_4 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 | -0.001 |
| Bearing3_1 (Cond3) | OTHER | 3 | 2_2, 2_4, 2_5 | - | 1.00, 1.00, 1.00 | - |
| Bearing2_2 (Cond2) | SAME | 1 | 2_4 | 0.67, 0.60, 0.70 | 0.10, 0.23, 0.04 | -0.531 |
| Bearing2_2 (Cond2) | SAME | 2 | 2_4, 2_5 | 0.86, 0.95, 0.94 | 0.78, 0.86, 0.80 | -0.102 |
| Bearing2_2 (Cond2) | SAME | 3 | 2_4, 2_5, 3_5 | - | 0.83, 0.87, 0.85 | - |
| Bearing2_2 (Cond2) | OTHER | 1 | 3_1 | 0.32, 0.40, 0.42 | 0.00, 0.11, 0.00 | -0.342 |
| Bearing2_2 (Cond2) | OTHER | 2 | 3_1, 3_5 | 0.71, 0.88, 0.85 | 0.63, 0.78, 0.75 | -0.093 |
| Bearing2_2 (Cond2) | OTHER | 3 | 3_1, 3_5, 1_1 | - | 0.84, 0.93, 0.86 | - |
| Bearing1_1 (Cond1) | SAME | 1 | 1_2 | 0.95, 0.96, 0.99 | 0.52, 0.75, 0.67 | -0.321 |
| Bearing1_1 (Cond1) | SAME | 2 | 1_2, 1_3 | 1.00, 1.00, 0.99 | 0.61, 0.71, 0.86 | -0.270 |
| Bearing1_1 (Cond1) | SAME | 3 | 1_2, 1_3, 3_1 | - | 0.98, 0.99, 0.92 | - |
| Bearing1_1 (Cond1) | OTHER | 1 | 3_1 | 0.65, 0.68, 0.87 | 0.74, 0.54, 0.61 | -0.106 |
| Bearing1_1 (Cond1) | OTHER | 2 | 3_1, 3_5 | 0.44, 0.49, 0.55 | 0.42, 0.43, 0.43 | -0.070 |
| Bearing1_1 (Cond1) | OTHER | 3 | 3_1, 3_5, 2_2 | - | 1.00, 0.99, 1.00 | - |