# Round 6 results (prereg_mst_round6.md)

## Block A: onset rules x learners, condition 2 -> 3

### Mamba-3, rule H (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 59.6 ± 11.9 | 61.6 | 63.4 | 54.4 | 3_1:1.00, 3_3:0.35, 3_4:0.19, 3_5:1.00 |
| V | 83.2 ± 4.6 | 83.5 | 73.9 | 88.6 | 3_1:0.98, 3_3:0.23, 3_4:1.00, 3_5:0.75 |
| HV | 91.4 ± 5.8 | 89.5 | 91.6 | 89.4 | 3_1:1.00, 3_3:0.93, 3_4:0.74, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +23.54 pp [+15.62, +31.18], p=0.0078, 8+/0- (n=8) |
| V-H | macro_f1 / common | +34.20 pp [+25.20, +42.74], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / own | +21.86 pp [+15.73, +27.69], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / common | +26.95 pp [+20.66, +32.74], p=0.0078, 8+/0- (n=8) |
| V-H | beq / own | +10.46 pp [+2.91, +18.44], p=0.0781, 6+/2- (n=8) |
| V-H | beq / common | +16.74 pp [+9.04, +24.62], p=0.0078, 8+/0- (n=8) |
| V-H | per bearing (own) | 3_1:-0.02, 3_3:-0.12, 3_4:+0.81, 3_5:-0.25 — signs differ across bearings |
| HV-H | macro_f1 / own | +31.74 pp [+23.26, +39.02], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_f1 / common | +35.03 pp [+25.98, +42.76], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / own | +27.89 pp [+21.20, +33.39], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / common | +27.85 pp [+21.16, +33.33], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / own | +28.17 pp [+21.98, +33.54], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / common | +28.16 pp [+21.97, +33.52], p=0.0078, 8+/0- (n=8) |
| HV-H | per bearing (own) | 3_1:-0.00, 3_3:+0.58, 3_4:+0.55, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | +8.19 pp [+3.97, +12.79], p=0.0156, 7+/1- (n=8) |
| HV-V | macro_f1 / common | +0.83 pp [-4.00, +5.66], p=0.5469, 5+/3- (n=8) |
| HV-V | macro_recall / own | +6.04 pp [+1.26, +11.07], p=0.0781, 6+/2- (n=8) |
| HV-V | macro_recall / common | +0.90 pp [-3.82, +5.63], p=0.5469, 5+/3- (n=8) |
| HV-V | beq / own | +17.71 pp [+11.55, +23.25], p=0.0078, 8+/0- (n=8) |
| HV-V | beq / common | +11.42 pp [+5.69, +16.62], p=0.0391, 6+/2- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.02, 3_3:+0.70, 3_4:-0.26, 3_5:+0.25 — signs differ across bearings |

### Mamba-3, rule V (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 59.5 ± 10.5 | 62.3 | 64.6 | 56.1 | 3_1:1.00, 3_3:0.40, 3_4:0.18, 3_5:1.00 |
| V | 88.0 ± 4.9 | 87.8 | 81.8 | 92.3 | 3_1:0.99, 3_3:0.44, 3_4:1.00, 3_5:0.85 |
| HV | 93.8 ± 4.9 | 92.7 | 93.5 | 93.0 | 3_1:1.00, 3_3:0.91, 3_4:0.84, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +28.42 pp [+20.36, +36.68], p=0.0078, 8+/0- (n=8) |
| V-H | macro_f1 / common | +36.14 pp [+26.89, +45.18], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / own | +25.48 pp [+18.95, +32.13], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / common | +29.68 pp [+22.72, +36.47], p=0.0078, 8+/0- (n=8) |
| V-H | beq / own | +17.19 pp [+8.68, +26.05], p=0.0078, 8+/0- (n=8) |
| V-H | beq / common | +21.15 pp [+11.75, +30.64], p=0.0078, 8+/0- (n=8) |
| V-H | per bearing (own) | 3_1:-0.01, 3_3:+0.04, 3_4:+0.82, 3_5:-0.15 — signs differ across bearings |
| HV-H | macro_f1 / own | +34.25 pp [+26.30, +41.14], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_f1 / common | +36.92 pp [+28.62, +44.13], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / own | +30.40 pp [+24.19, +35.25], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / common | +30.44 pp [+24.33, +35.25], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / own | +28.92 pp [+22.98, +33.93], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / common | +28.79 pp [+23.07, +33.84], p=0.0078, 8+/0- (n=8) |
| HV-H | per bearing (own) | 3_1:-0.00, 3_3:+0.51, 3_4:+0.65, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | +5.83 pp [+2.18, +9.49], p=0.0391, 6+/2- (n=8) |
| HV-V | macro_f1 / common | +0.77 pp [-3.14, +4.93], p=0.9453, 4+/4- (n=8) |
| HV-V | macro_recall / own | +4.92 pp [+0.86, +9.11], p=0.0781, 6+/2- (n=8) |
| HV-V | macro_recall / common | +0.76 pp [-3.13, +4.94], p=0.9453, 4+/4- (n=8) |
| HV-V | beq / own | +11.73 pp [+5.71, +17.90], p=0.0156, 7+/1- (n=8) |
| HV-V | beq / common | +7.64 pp [+1.45, +14.23], p=0.1484, 5+/3- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.01, 3_3:+0.47, 3_4:-0.16, 3_5:+0.15 — signs differ across bearings |

### Mamba-3, rule last15 (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 47.3 ± 12.1 | 49.6 | 62.9 | 65.4 | 3_1:0.75, 3_3:0.63, 3_4:0.14, 3_5:1.00 |
| V | 37.9 ± 0.4 | 54.1 | 75.9 | 58.2 | 3_1:0.04, 3_3:1.00, 3_4:1.00, 3_5:1.00 |
| HV | 47.8 ± 5.2 | 59.1 | 77.6 | 77.8 | 3_1:0.16, 3_3:0.99, 3_4:0.98, 3_5:0.97 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | -9.35 pp [-17.30, -1.29], p=0.1094, 1+/7- (n=8) |
| V-H | macro_f1 / common | -7.18 pp [-13.13, -1.35], p=0.0391, 2+/6- (n=8) |
| V-H | macro_recall / own | +4.46 pp [-3.28, +13.16], p=0.3828, 6+/2- (n=8) |
| V-H | macro_recall / common | -4.65 pp [-9.13, -0.36], p=0.1484, 2+/6- (n=8) |
| V-H | beq / own | +13.10 pp [+7.24, +19.22], p=0.0156, 7+/1- (n=8) |
| V-H | beq / common | +2.18 pp [-1.33, +5.64], p=0.3828, 6+/2- (n=8) |
| V-H | per bearing (own) | 3_1:-0.71, 3_3:+0.37, 3_4:+0.86, 3_5:+0.00 — signs differ across bearings |
| HV-H | macro_f1 / own | +0.55 pp [-9.52, +11.74], p=1.0000, 4+/4- (n=8) |
| HV-H | macro_f1 / common | +12.42 pp [+4.39, +21.24], p=0.0234, 7+/1- (n=8) |
| HV-H | macro_recall / own | +9.47 pp [+0.53, +19.95], p=0.1094, 6+/2- (n=8) |
| HV-H | macro_recall / common | +10.37 pp [+3.96, +17.39], p=0.0234, 7+/1- (n=8) |
| HV-H | beq / own | +14.72 pp [+8.12, +21.99], p=0.0156, 7+/1- (n=8) |
| HV-H | beq / common | +10.13 pp [+4.89, +15.62], p=0.0156, 7+/1- (n=8) |
| HV-H | per bearing (own) | 3_1:-0.58, 3_3:+0.36, 3_4:+0.84, 3_5:-0.03 — signs differ across bearings |
| HV-V | macro_f1 / own | +9.90 pp [+6.86, +13.28], p=0.0078, 8+/0- (n=8) |
| HV-V | macro_f1 / common | +19.61 pp [+15.06, +23.96], p=0.0078, 8+/0- (n=8) |
| HV-V | macro_recall / own | +5.01 pp [+3.36, +7.01], p=0.0078, 8+/0- (n=8) |
| HV-V | macro_recall / common | +15.01 pp [+11.18, +18.85], p=0.0078, 8+/0- (n=8) |
| HV-V | beq / own | +1.62 pp [-0.59, +3.34], p=0.1484, 7+/1- (n=8) |
| HV-V | beq / common | +7.95 pp [+4.65, +10.77], p=0.0156, 7+/1- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.12, 3_3:-0.01, 3_4:-0.02, 3_5:-0.03 — signs differ across bearings |

### 1D-CNN, rule H (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 40.2 ± 0.2 | 50.1 | 50.2 | 34.1 | 3_1:1.00, 3_3:0.01, 3_4:0.00, 3_5:1.00 |
| V | 88.8 ± 0.0 | 86.2 | 75.0 | 86.3 | 3_1:1.00, 3_3:0.00, 3_4:1.00, 3_5:1.00 |
| HV | 45.5 ± 4.3 | 52.7 | 51.9 | 39.6 | 3_1:1.00, 3_3:0.00, 3_4:0.07, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +48.59 pp [+48.49, +48.69], p=0.0078, 8+/0- (n=8) |
| V-H | macro_f1 / common | +52.14 pp [+52.03, +52.23], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / own | +36.15 pp [+36.10, +36.20], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / common | +36.15 pp [+36.10, +36.20], p=0.0078, 8+/0- (n=8) |
| V-H | beq / own | +24.84 pp [+24.75, +24.92], p=0.0078, 8+/0- (n=8) |
| V-H | beq / common | +24.84 pp [+24.75, +24.92], p=0.0078, 8+/0- (n=8) |
| V-H | per bearing (own) | 3_1:+0.00, 3_3:-0.01, 3_4:+1.00, 3_5:+0.00 — signs consistent |
| HV-H | macro_f1 / own | +5.29 pp [+2.53, +8.05], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_f1 / common | +5.45 pp [+2.61, +8.30], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / own | +2.64 pp [+1.24, +4.05], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / common | +2.64 pp [+1.24, +4.05], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / own | +1.76 pp [+0.81, +2.71], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / common | +1.76 pp [+0.81, +2.71], p=0.0078, 8+/0- (n=8) |
| HV-H | per bearing (own) | 3_1:+0.00, 3_3:-0.00, 3_4:+0.07, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | -43.31 pp [-46.12, -40.49], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_f1 / common | -46.69 pp [-49.59, -43.78], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_recall / own | -33.51 pp [-34.94, -32.09], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_recall / common | -33.51 pp [-34.94, -32.09], p=0.0078, 0+/8- (n=8) |
| HV-V | beq / own | -23.09 pp [-24.08, -22.09], p=0.0078, 0+/8- (n=8) |
| HV-V | beq / common | -23.09 pp [-24.08, -22.09], p=0.0078, 0+/8- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.00, 3_3:+0.00, 3_4:-0.93, 3_5:+0.00 — signs consistent |

### 1D-CNN, rule V (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 41.0 ± 1.2 | 51.3 | 52.2 | 35.4 | 3_1:1.00, 3_3:0.09, 3_4:0.00, 3_5:1.00 |
| V | 87.3 ± 0.0 | 85.4 | 75.0 | 86.2 | 3_1:1.00, 3_3:0.00, 3_4:1.00, 3_5:1.00 |
| HV | 48.5 ± 5.9 | 55.3 | 53.9 | 44.1 | 3_1:1.00, 3_3:0.01, 3_4:0.15, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +46.36 pp [+45.63, +47.18], p=0.0078, 8+/0- (n=8) |
| V-H | macro_f1 / common | +50.88 pp [+50.37, +51.42], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / own | +34.12 pp [+33.77, +34.50], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / common | +35.58 pp [+35.34, +35.82], p=0.0078, 8+/0- (n=8) |
| V-H | beq / own | +22.81 pp [+22.21, +23.47], p=0.0078, 8+/0- (n=8) |
| V-H | beq / common | +23.80 pp [+23.37, +24.25], p=0.0078, 8+/0- (n=8) |
| V-H | per bearing (own) | 3_1:+0.00, 3_3:-0.09, 3_4:+1.00, 3_5:+0.00 — signs differ across bearings |
| HV-H | macro_f1 / own | +7.57 pp [+3.98, +11.25], p=0.0156, 7+/1- (n=8) |
| HV-H | macro_f1 / common | +8.78 pp [+4.98, +12.82], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / own | +4.02 pp [+2.04, +6.10], p=0.0156, 7+/1- (n=8) |
| HV-H | macro_recall / common | +4.48 pp [+2.43, +6.66], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / own | +1.71 pp [+0.25, +3.05], p=0.0781, 6+/2- (n=8) |
| HV-H | beq / common | +2.52 pp [+1.07, +3.94], p=0.0234, 7+/1- (n=8) |
| HV-H | per bearing (own) | 3_1:+0.00, 3_3:-0.08, 3_4:+0.15, 3_5:+0.00 — signs differ across bearings |
| HV-V | macro_f1 / own | -38.79 pp [-42.51, -34.87], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_f1 / common | -42.11 pp [-45.98, -38.02], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_recall / own | -30.10 pp [-32.15, -27.90], p=0.0078, 0+/8- (n=8) |
| HV-V | macro_recall / common | -31.09 pp [-33.16, -28.87], p=0.0078, 0+/8- (n=8) |
| HV-V | beq / own | -21.10 pp [-22.55, -19.59], p=0.0078, 0+/8- (n=8) |
| HV-V | beq / common | -21.28 pp [-22.71, -19.78], p=0.0078, 0+/8- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.00, 3_3:+0.01, 3_4:-0.85, 3_5:+0.00 — signs consistent |

### 1D-CNN, rule last15 (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 48.9 ± 3.1 | 55.1 | 65.4 | 45.7 | 3_1:0.96, 3_3:0.64, 3_4:0.02, 3_5:1.00 |
| V | 88.3 ± 2.4 | 87.8 | 75.4 | 86.7 | 3_1:0.93, 3_3:0.08, 3_4:1.00, 3_5:1.00 |
| HV | 92.5 ± 3.2 | 91.6 | 91.4 | 82.1 | 3_1:1.00, 3_3:0.82, 3_4:0.84, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +39.33 pp [+36.76, +41.62], p=0.0078, 8+/0- (n=8) |
| V-H | macro_f1 / common | +40.99 pp [+37.45, +44.44], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / own | +32.64 pp [+30.68, +34.31], p=0.0078, 8+/0- (n=8) |
| V-H | macro_recall / common | +30.71 pp [+28.71, +32.58], p=0.0078, 8+/0- (n=8) |
| V-H | beq / own | +10.05 pp [+7.30, +12.45], p=0.0078, 8+/0- (n=8) |
| V-H | beq / common | +15.45 pp [+11.95, +18.68], p=0.0078, 8+/0- (n=8) |
| V-H | per bearing (own) | 3_1:-0.03, 3_3:-0.55, 3_4:+0.98, 3_5:+0.00 — signs differ across bearings |
| HV-H | macro_f1 / own | +43.59 pp [+41.01, +46.27], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_f1 / common | +36.43 pp [+32.47, +40.23], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / own | +36.46 pp [+33.57, +39.76], p=0.0078, 8+/0- (n=8) |
| HV-H | macro_recall / common | +26.60 pp [+23.11, +29.72], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / own | +25.98 pp [+23.63, +28.55], p=0.0078, 8+/0- (n=8) |
| HV-H | beq / common | +24.03 pp [+20.60, +27.47], p=0.0078, 8+/0- (n=8) |
| HV-H | per bearing (own) | 3_1:+0.04, 3_3:+0.18, 3_4:+0.82, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | +4.26 pp [+0.99, +7.43], p=0.0547, 7+/1- (n=8) |
| HV-V | macro_f1 / common | -4.57 pp [-9.23, -0.73], p=0.1094, 2+/6- (n=8) |
| HV-V | macro_recall / own | +3.83 pp [+0.62, +6.91], p=0.0547, 7+/1- (n=8) |
| HV-V | macro_recall / common | -4.11 pp [-8.20, -0.65], p=0.1094, 2+/6- (n=8) |
| HV-V | beq / own | +15.93 pp [+13.01, +18.68], p=0.0078, 8+/0- (n=8) |
| HV-V | beq / common | +8.58 pp [+4.34, +11.96], p=0.0156, 7+/1- (n=8) |
| HV-V | per bearing (own) | 3_1:+0.07, 3_3:+0.73, 3_4:-0.16, 3_5:+0.00 — signs differ across bearings |

### Mamba-2, rule H (seeds [0, 1, 2, 3, 4])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 44.0 ± 1.0 | 51.9 | 53.5 | 38.0 | 3_1:1.00, 3_3:0.14, 3_4:0.00, 3_5:1.00 |
| V | 88.6 ± 0.2 | 86.1 | 74.8 | 86.3 | 3_1:1.00, 3_3:0.00, 3_4:1.00, 3_5:0.99 |
| HV | 75.1 ± 12.6 | 73.5 | 77.6 | 71.0 | 3_1:1.00, 3_3:0.74, 3_4:0.37, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +44.62 pp [+43.74, +45.49], p=0.0625, 5+/0- (n=5) |
| V-H | macro_f1 / common | +48.25 pp [+47.43, +49.07], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / own | +34.20 pp [+33.75, +34.65], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / common | +34.33 pp [+33.94, +34.73], p=0.0625, 5+/0- (n=5) |
| V-H | beq / own | +21.38 pp [+20.60, +22.16], p=0.0625, 5+/0- (n=5) |
| V-H | beq / common | +21.54 pp [+20.82, +22.26], p=0.0625, 5+/0- (n=5) |
| V-H | per bearing (own) | 3_1:+0.00, 3_3:-0.14, 3_4:+1.00, 3_5:-0.01 — signs differ across bearings |
| HV-H | macro_f1 / own | +31.07 pp [+20.27, +40.63], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_f1 / common | +32.99 pp [+21.40, +43.35], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / own | +21.65 pp [+13.37, +29.46], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / common | +21.65 pp [+13.37, +29.46], p=0.0625, 5+/0- (n=5) |
| HV-H | beq / own | +24.18 pp [+16.74, +31.12], p=0.0625, 5+/0- (n=5) |
| HV-H | beq / common | +24.18 pp [+16.74, +31.12], p=0.0625, 5+/0- (n=5) |
| HV-H | per bearing (own) | 3_1:+0.00, 3_3:+0.60, 3_4:+0.37, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | -13.54 pp [-23.62, -4.86], p=0.0625, 0+/5- (n=5) |
| HV-V | macro_f1 / common | -15.26 pp [-26.23, -5.71], p=0.0625, 0+/5- (n=5) |
| HV-V | macro_recall / own | -12.56 pp [-20.45, -5.19], p=0.0625, 0+/5- (n=5) |
| HV-V | macro_recall / common | -12.69 pp [-20.67, -5.28], p=0.0625, 0+/5- (n=5) |
| HV-V | beq / own | +2.80 pp [-4.07, +8.98], p=0.6250, 3+/2- (n=5) |
| HV-V | beq / common | +2.64 pp [-4.43, +8.94], p=0.6250, 3+/2- (n=5) |
| HV-V | per bearing (own) | 3_1:+0.00, 3_3:+0.74, 3_4:-0.63, 3_5:+0.01 — signs differ across bearings |

### Mamba-2, rule V (seeds [0, 1, 2, 3, 4])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 45.2 ± 2.0 | 53.4 | 55.7 | 41.0 | 3_1:1.00, 3_3:0.22, 3_4:0.00, 3_5:1.00 |
| V | 87.2 ± 0.2 | 85.3 | 74.9 | 86.3 | 3_1:1.00, 3_3:0.00, 3_4:1.00, 3_5:0.99 |
| HV | 77.9 ± 13.1 | 77.1 | 80.6 | 75.6 | 3_1:1.00, 3_3:0.78, 3_4:0.44, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +41.99 pp [+40.38, +43.42], p=0.0625, 5+/0- (n=5) |
| V-H | macro_f1 / common | +45.23 pp [+43.46, +46.72], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / own | +31.89 pp [+31.02, +32.63], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / common | +32.84 pp [+31.93, +33.59], p=0.0625, 5+/0- (n=5) |
| V-H | beq / own | +19.19 pp [+17.82, +20.36], p=0.0625, 5+/0- (n=5) |
| V-H | beq / common | +18.91 pp [+17.41, +20.19], p=0.0625, 5+/0- (n=5) |
| V-H | per bearing (own) | 3_1:+0.00, 3_3:-0.22, 3_4:+1.00, 3_5:-0.01 — signs differ across bearings |
| HV-H | macro_f1 / own | +32.67 pp [+21.22, +42.88], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_f1 / common | +34.56 pp [+22.18, +45.80], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / own | +23.67 pp [+14.34, +32.16], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / common | +24.00 pp [+14.21, +32.82], p=0.0625, 5+/0- (n=5) |
| HV-H | beq / own | +24.89 pp [+16.86, +32.21], p=0.0625, 5+/0- (n=5) |
| HV-H | beq / common | +25.86 pp [+17.30, +33.41], p=0.0625, 5+/0- (n=5) |
| HV-H | per bearing (own) | 3_1:+0.00, 3_3:+0.56, 3_4:+0.44, 3_5:+0.00 — signs consistent |
| HV-V | macro_f1 / own | -9.32 pp [-20.20, -0.37], p=0.1875, 1+/4- (n=5) |
| HV-V | macro_f1 / common | -10.67 pp [-22.51, -0.75], p=0.1250, 1+/4- (n=5) |
| HV-V | macro_recall / own | -8.22 pp [-17.14, -0.40], p=0.1875, 1+/4- (n=5) |
| HV-V | macro_recall / common | -8.84 pp [-18.03, -0.69], p=0.1250, 1+/4- (n=5) |
| HV-V | beq / own | +5.70 pp [-1.42, +11.86], p=0.1875, 4+/1- (n=5) |
| HV-V | beq / common | +6.96 pp [-0.28, +13.26], p=0.1875, 4+/1- (n=5) |
| HV-V | per bearing (own) | 3_1:+0.00, 3_3:+0.78, 3_4:-0.56, 3_5:+0.01 — signs differ across bearings |

### Mamba-2, rule last15 (seeds [0, 1, 2, 3, 4])
| arm | macro-F1 own | macro-recall own | bearing-eq bal. acc own | macro-F1 common | per-bearing acc (own) |
|---|---|---|---|---|---|
| H | 28.3 ± 3.8 | 29.1 | 48.8 | 53.6 | 3_1:0.45, 3_3:0.49, 3_4:0.01, 3_5:1.00 |
| V | 58.3 ± 14.0 | 63.5 | 66.6 | 73.6 | 3_1:0.38, 3_3:0.28, 3_4:1.00, 3_5:1.00 |
| HV | 45.5 ± 6.0 | 53.3 | 74.0 | 70.6 | 3_1:0.19, 3_3:0.96, 3_4:0.81, 3_5:1.00 |

| contrast | metric / view | result |
|---|---|---|
| V-H | macro_f1 / own | +29.94 pp [+19.17, +43.52], p=0.0625, 5+/0- (n=5) |
| V-H | macro_f1 / common | +20.02 pp [+13.46, +27.50], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / own | +34.38 pp [+26.14, +44.95], p=0.0625, 5+/0- (n=5) |
| V-H | macro_recall / common | +13.50 pp [+7.39, +19.61], p=0.0625, 5+/0- (n=5) |
| V-H | beq / own | +17.79 pp [+13.37, +24.59], p=0.0625, 5+/0- (n=5) |
| V-H | beq / common | +2.48 pp [-0.92, +7.18], p=0.4375, 4+/1- (n=5) |
| V-H | per bearing (own) | 3_1:-0.07, 3_3:-0.21, 3_4:+0.99, 3_5:+0.00 — signs differ across bearings |
| HV-H | macro_f1 / own | +17.21 pp [+13.30, +21.38], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_f1 / common | +16.98 pp [+9.98, +23.65], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / own | +24.19 pp [+18.40, +28.63], p=0.0625, 5+/0- (n=5) |
| HV-H | macro_recall / common | +10.70 pp [+3.38, +16.99], p=0.1250, 4+/1- (n=5) |
| HV-H | beq / own | +25.13 pp [+21.83, +28.89], p=0.0625, 5+/0- (n=5) |
| HV-H | beq / common | +11.51 pp [+7.53, +15.26], p=0.0625, 5+/0- (n=5) |
| HV-H | per bearing (own) | 3_1:-0.26, 3_3:+0.47, 3_4:+0.80, 3_5:+0.00 — signs differ across bearings |
| HV-V | macro_f1 / own | -12.72 pp [-23.17, -2.88], p=0.1250, 1+/4- (n=5) |
| HV-V | macro_f1 / common | -3.04 pp [-10.12, +5.17], p=0.6250, 2+/3- (n=5) |
| HV-V | macro_recall / own | -10.19 pp [-17.90, -1.75], p=0.1875, 1+/4- (n=5) |
| HV-V | macro_recall / common | -2.80 pp [-9.46, +5.01], p=0.8125, 2+/3- (n=5) |
| HV-V | beq / own | +7.34 pp [+3.24, +11.28], p=0.0625, 5+/0- (n=5) |
| HV-V | beq / common | +9.03 pp [+4.75, +13.31], p=0.0625, 5+/0- (n=5) |
| HV-V | per bearing (own) | 3_1:-0.19, 3_3:+0.68, 3_4:-0.19, 3_5:+0.00 — signs differ across bearings |

## Block B: decomposition of the last-15% effect (common test set, macro-F1)

### Mamba-3 (seeds [0, 1, 2, 3, 4])
| contrast | C_c(H rule) | test composition | training segmentation | bearing weighting | sample size | C_own(last15) |
|---|---|---|---|---|---|---|
| V-H | +31.5 | -7.6 [-9.7, -5.5] | -34.8 [-44.1, -25.6] | -1.5 [-2.8, +0.2] | -2.9 [-4.5, -1.2] | -15.2 |
| HV-H | +36.1 | -16.5 [-20.2, -13.0] | -24.6 [-34.9, -13.5] | -0.3 [-1.1, +0.6] | -3.0 [-4.9, -1.0] | -8.3 |
| HV-V | +4.6 | -8.9 [-10.8, -6.5] | +10.2 [+1.5, +17.3] | +1.2 [+0.3, +2.0] | -0.1 [-1.1, +0.9] | +7.0 |

### 1D-CNN (seeds [0, 1, 2, 3, 4, 5, 6, 7])
| contrast | C_c(H rule) | test composition | training segmentation | bearing weighting | sample size | C_own(last15) |
|---|---|---|---|---|---|---|
| V-H | +52.1 | -1.7 [-3.8, +0.7] | -11.1 [-13.8, -8.5] | -0.1 [-1.6, +1.2] | -0.0 [-1.6, +1.9] | +39.3 |
| HV-H | +5.5 | +7.2 [+4.8, +9.6] | +26.5 [+22.3, +30.2] | +0.2 [-2.5, +2.8] | +4.3 [+0.9, +7.8] | +43.6 |
| HV-V | -46.7 | +8.8 [+7.2, +10.7] | +37.5 [+32.3, +42.3] | +0.3 [-1.6, +2.4] | +4.3 [+1.0, +7.8] | +4.3 |

## Block C: LOBO OR folds, Mamba-3 (own-class recall on the test bearing)

### rule H (seeds [0, 1, 2, 3, 4])
| arm | fold 0 (3_1) | fold 3 (3_5) | OR mean |
|---|---|---|---|
| H | 99.2 | 48.9 | 74.1 |
| V | 72.4 | 76.0 | 74.2 |
| HV | 43.0 | 17.0 | 30.0 |
- HV-H (OR mean): -44.04 pp [-54.46, -34.83], p=0.0625, 0+/5- (n=5)
- HV-V (OR mean): -44.20 pp [-48.51, -37.57], p=0.0625, 0+/5- (n=5)
- V-H (OR mean): +0.16 pp [-7.74, +6.27], p=0.8125, 3+/2- (n=5)

### rule last15 (seeds [0, 1, 2, 3, 4])
| arm | fold 0 (3_1) | fold 3 (3_5) | OR mean |
|---|---|---|---|
| H | 12.8 | 6.3 | 9.6 |
| V | 1.9 | 99.9 | 50.9 |
| HV | 0.8 | 5.1 | 3.0 |
- HV-H (OR mean): -6.59 pp [-11.02, -2.32], p=0.1250, 1+/4- (n=5)
- HV-V (OR mean): -47.97 pp [-49.90, -45.59], p=0.0625, 0+/5- (n=5)
- V-H (OR mean): +41.38 pp [+37.93, +45.28], p=0.0625, 5+/0- (n=5)

## Block E: gated fusion and fixed-gate ablation

### E_m3_gated: macro-F1 90.8 ± 6.5; fixed gate 90.5; adaptive - fixed +0.28 pp [-0.10, +0.84], p=0.9375, 3+/4- (n=8); functional gate: False
- mean gate (H, V) per seed: [[0.645, 0.355], [0.551, 0.449], [0.532, 0.468], [0.55, 0.45], [0.736, 0.264], [0.496, 0.504], [0.533, 0.467], [0.627, 0.373]]; fraction of predictions changed by fixing: [0.0004, 0.0004, 0.0017, 0.0006, 0.0025, 0.008, 0.0004, 0.0151]
- gated - H (macro-F1): +31.12 pp [+24.85, +37.72], p=0.0078, 8+/0- (n=8)
- gated - V (macro-F1): +7.57 pp [+2.72, +11.91], p=0.0391, 7+/1- (n=8)
- gated - HV (macro-F1): -0.62 pp [-6.45, +5.19], p=1.0000, 4+/4- (n=8)

### E_m3_tgated: macro-F1 93.9 ± 4.2; fixed gate 90.5; adaptive - fixed +3.40 pp [+0.23, +7.48], p=0.1094, 6+/2- (n=8); functional gate: True
- mean gate (H, V) per seed: [[0.506, 0.494], [0.502, 0.498], [0.548, 0.452], [0.503, 0.497], [0.651, 0.349], [0.517, 0.483], [0.465, 0.535], [0.593, 0.407]]; fraction of predictions changed by fixing: [0.0198, 0.0248, 0.0105, 0.0109, 0.0215, 0.0194, 0.0554, 0.1033]
- gated - H (macro-F1): +34.29 pp [+28.35, +40.24], p=0.0078, 8+/0- (n=8)
- gated - V (macro-F1): +10.75 pp [+6.60, +14.58], p=0.0078, 8+/0- (n=8)
- gated - HV (macro-F1): +2.55 pp [-1.39, +7.46], p=0.5469, 5+/3- (n=8)

### E_cnn_gated: macro-F1 61.9 ± 20.7; fixed gate 40.6; adaptive - fixed +21.28 pp [+7.93, +35.03], p=0.0078, 8+/0- (n=8); functional gate: True
- mean gate (H, V) per seed: [[0.379, 0.621], [0.561, 0.439], [0.468, 0.532], [0.518, 0.482], [0.443, 0.557], [0.518, 0.482], [0.496, 0.504], [0.411, 0.589]]; fraction of predictions changed by fixing: [0.0118, 0.013, 0.2325, 0.0042, 0.0051, 0.2195, 0.0911, 0.1957]
- gated - H (macro-F1): +21.63 pp [+8.64, +35.02], p=0.0078, 8+/0- (n=8)
- gated - V (macro-F1): -26.97 pp [-39.99, -13.56], p=0.0078, 0+/8- (n=8)
- gated - HV (macro-F1): +16.34 pp [+2.48, +30.86], p=0.1484, 6+/2- (n=8)

### E_cnn_tgated: macro-F1 50.7 ± 7.7; fixed gate 50.3; adaptive - fixed +0.47 pp [-0.61, +1.55], p=0.6406, 5+/3- (n=8); functional gate: False
- mean gate (H, V) per seed: [[0.436, 0.564], [0.456, 0.544], [0.455, 0.545], [0.493, 0.507], [0.431, 0.569], [0.474, 0.526], [0.491, 0.509], [0.496, 0.504]]; fraction of predictions changed by fixing: [0.0013, 0.0011, 0.0084, 0.0011, 0.0139, 0.0032, 0.0059, 0.0112]
- gated - H (macro-F1): +10.50 pp [+5.87, +15.61], p=0.0078, 8+/0- (n=8)
- gated - V (macro-F1): -38.09 pp [-42.77, -32.96], p=0.0078, 0+/8- (n=8)
- gated - HV (macro-F1): +5.21 pp [+1.34, +8.99], p=0.0547, 6+/2- (n=8)

### E_lobo_m3_gated: OR recall 59.9; fixed gate 59.5; adaptive - fixed +0.45 pp [-0.12, +1.13], p=0.4375, 3+/2- (n=5); functional gate: False
- gated - H (OR mean): -14.12 pp [-26.95, +1.26], p=0.1875, 1+/4- (n=5)
- gated - V (OR mean): -14.28 pp [-26.87, -0.47], p=0.1875, 1+/4- (n=5)
- gated - HV (OR mean): +29.91 pp [+13.89, +45.94], p=0.0625, 5+/0- (n=5)

### E_lobo_m3_tgated: OR recall 56.3; fixed gate 61.7; adaptive - fixed -5.46 pp [-12.74, +1.82], p=0.6250, 2+/3- (n=5); functional gate: False
- gated - H (OR mean): -17.77 pp [-29.29, -3.17], p=0.1250, 1+/4- (n=5)
- gated - V (OR mean): -17.93 pp [-30.88, -2.42], p=0.1250, 1+/4- (n=5)
- gated - HV (OR mean): +26.27 pp [+11.89, +41.79], p=0.0625, 5+/0- (n=5)

## Block D: CWRU common-noise controls (Mamba-3, -8 dB unless stated; test accuracy)

| configuration | accuracy (%) per seed | mean |
|---|---|---|
| D0_DE | 81.97, 82.88, 86.36, 84.70, 84.85 | 84.15 |
| D0_DEFE_rho0 | 88.03, 89.55, 88.18, 86.97, 88.18 | 88.18 |
| D0_DEFE_rho0.9 | 98.64, 96.67, 97.58, 96.36, 97.73 | 97.39 |
| D2_SUM_rho0 | 78.64, 77.12, 79.24, 80.00, 78.48 | 78.70 |
| D2_SUM_rho0.9 | 64.70, 62.73, 66.21, 66.82, 65.45 | 65.18 |
| D2_DIFF_rho0 | 74.09, 72.27, 74.24, 74.39, 72.42 | 73.48 |
| D2_DIFF_rho0.9 | 99.55, 99.55, 99.24, 99.55, 99.24 | 99.42 |
| D3_DEFE_rho0.9_d4 | 95.91, 94.55, 93.48, 93.94, 95.00 | 94.58 |
| D3_DEFE_rho0.9_d64 | 89.24, 88.79, 89.09, 89.09, 88.79 | 89.00 |
| D4_DE_r1 | 84.55, 83.94, 86.97, 86.82, 85.30 | 85.52 |
| D4_DEFE_rho0_r1 | 88.64, 90.91, 90.45, 87.73, 88.48 | 89.24 |
| D4_DEFE_rho0.9_r1 | 98.64, 97.42, 97.58, 97.12, 97.27 | 97.61 |
| D4_DE_r2 | 83.03, 86.36, 88.48, 87.42, 84.55 | 85.97 |
| D4_DEFE_rho0_r2 | 86.97, 88.33, 88.33, 88.48, 88.33 | 88.09 |
| D4_DEFE_rho0.9_r2 | 98.18, 97.88, 98.18, 95.45, 96.52 | 97.24 |

- gain_rho0: +4.03 pp [+2.30, +5.76], p=0.0625, 5+/0- (n=5)
- gain_rho0.9: +13.24 pp [+11.73, +15.09], p=0.0625, 5+/0- (n=5)
- P6_DIFF0.9-DE: +15.27 pp [+13.88, +16.76], p=0.0625, 5+/0- (n=5)
- DIFF0.9-DIFF0: +25.94 pp [+25.15, +26.76], p=0.0625, 5+/0- (n=5)
- SUM0.9-SUM0: -13.52 pp [-14.03, -13.06], p=0.0625, 0+/5- (n=5)
- DIFF0.9-DEFE0.9: +2.03 pp [+1.30, +2.79], p=0.0625, 5+/0- (n=5)
- SUM0-DE: -5.45 pp [-6.55, -4.21], p=0.0625, 0+/5- (n=5)
- DIFF0-DE: -10.67 pp [-11.94, -9.21], p=0.0625, 0+/5- (n=5)
- SUM0.9-DE: -18.97 pp [-20.00, -17.94], p=0.0625, 0+/5- (n=5)
- gain_rho0.9_d4: +10.42 pp [+8.45, +12.55], p=0.0625, 5+/0- (n=5)
- gain_rho0.9_d64: +4.85 pp [+3.55, +6.33], p=0.0625, 5+/0- (n=5)
- gain_rho0_r1: +3.73 pp [+2.06, +5.58], p=0.0625, 5+/0- (n=5)
- gain_rho0.9_r1: +12.09 pp [+10.76, +13.42], p=0.0625, 5+/0- (n=5)
- gain_rho0_r2: +2.12 pp [+0.76, +3.48], p=0.1250, 4+/1- (n=5)
- gain_rho0.9_r2: +11.27 pp [+9.36, +13.42], p=0.0625, 5+/0- (n=5)
- mismatch_D0_DEFE_rho0.9_same_minus_awgn_corr0.0: +61.09 pp [+57.24, +63.52], p=0.0625, 5+/0- (n=5)
- mismatch_D0_DEFE_rho0_same_minus_awgn_corr0.9: +47.42 pp [+39.09, +53.94], p=0.0625, 5+/0- (n=5)
- mismatch_r4_rho0.9_snr-6_same_minus_awgn_corr0.0: +66.45 pp [+64.06, +70.85], p=0.0625, 5+/0- (n=5)
- mismatch_r4_rho0_snr-6_same_minus_awgn_corr0.9: +43.97 pp [+34.82, +53.73], p=0.0625, 5+/0- (n=5)
- mismatch_r4_rho0.9_snr-8_same_minus_awgn_corr0.0: +60.24 pp [+54.67, +63.42], p=0.0625, 5+/0- (n=5)
- mismatch_r4_rho0_snr-8_same_minus_awgn_corr0.9: +46.97 pp [+41.36, +51.76], p=0.0625, 5+/0- (n=5)

## Block F: controlled coverage (test-bearing own-class recall, H+V - H, mean over seeds per unit)

| test | k | subset | OR training bearings | conditions | H | H+V | H+V - H |
|---|---|---|---|---|---|---|---|
| 1_1 | 1 | 0 | 1_2 | 1 | 0.98 | 0.59 | -0.39 |
| 1_1 | 1 | 1 | 2_5 | 1 | 0.00 | 0.05 | +0.05 |
| 1_1 | 1 | 2 | 1_3 | 1 | 1.00 | 0.41 | -0.59 |
| 1_1 | 2 | 0 | 1_3, 2_4 | 2 | 0.99 | 0.72 | -0.27 |
| 1_1 | 2 | 1 | 1_3, 2_5 | 2 | 0.80 | 0.74 | -0.06 |
| 1_1 | 2 | 2 | 1_2, 2_2 | 2 | 1.00 | 0.96 | -0.04 |
| 1_1 | 3 | 0 | 2_2, 2_4, 3_5 | 2 | 1.00 | 0.91 | -0.08 |
| 1_1 | 3 | 1 | 2_2, 2_4, 3_1 | 2 | 1.00 | 1.00 | -0.00 |
| 1_1 | 3 | 2 | 1_3, 3_1, 3_5 | 2 | 0.99 | 0.51 | -0.48 |
| 2_2 | 1 | 0 | 3_1 | 1 | 0.53 | 0.04 | -0.49 |
| 2_2 | 1 | 1 | 2_5 | 1 | 0.57 | 0.51 | -0.06 |
| 2_2 | 1 | 2 | 1_1 | 1 | 0.34 | 0.00 | -0.34 |
| 2_2 | 2 | 0 | 2_5, 3_5 | 2 | 0.71 | 0.82 | +0.11 |
| 2_2 | 2 | 1 | 1_1, 2_5 | 2 | 0.97 | 0.76 | -0.22 |
| 2_2 | 2 | 2 | 2_4, 2_5 | 1 | 0.93 | 0.84 | -0.10 |
| 2_2 | 3 | 0 | 1_3, 2_4, 3_1 | 3 | 0.88 | 0.70 | -0.18 |
| 2_2 | 3 | 1 | 2_4, 3_1, 3_5 | 2 | 0.83 | 0.78 | -0.05 |
| 2_2 | 3 | 2 | 1_2, 2_4, 3_1 | 3 | 0.72 | 0.84 | +0.11 |
| 3_1 | 1 | 0 | 3_5 | 1 | 0.98 | 0.22 | -0.76 |
| 3_1 | 1 | 1 | 2_2 | 1 | 1.00 | 0.29 | -0.71 |
| 3_1 | 1 | 2 | 2_4 | 1 | 1.00 | 1.00 | -0.00 |
| 3_1 | 2 | 0 | 1_1, 2_4 | 2 | 1.00 | 1.00 | -0.00 |
| 3_1 | 2 | 1 | 1_1, 3_5 | 2 | 1.00 | 1.00 | +0.00 |
| 3_1 | 2 | 2 | 1_2, 1_3 | 1 | 1.00 | 0.67 | -0.33 |
| 3_1 | 3 | 0 | 1_1, 1_2, 2_5 | 2 | 1.00 | 1.00 | +0.00 |
| 3_1 | 3 | 1 | 1_3, 2_4, 2_5 | 2 | 1.00 | 1.00 | +0.00 |
| 3_1 | 3 | 2 | 1_2, 1_3, 2_5 | 2 | 1.00 | 1.00 | +0.00 |
- k=1: mean H+V - H -0.366 over 9 units; mean H 0.711, H+V 0.345
- k=2: mean H+V - H -0.101 over 9 units; mean H 0.935, H+V 0.834
- k=3: mean H+V - H -0.076 over 9 units; mean H 0.936, H+V 0.860
- P7: k=3 minus k=1: +29.01 pp [+7.22, +49.56], p=0.0262, 1+/5- (n=18)  (pos/neg counts refer to k=3 units)

## Block G: numerical replicates (identical configuration and seed)

- G_m3_H_HV_rep vs A_m3_H_HV: |diff| per seed (pp) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; max 0.00
- G_m2_H_H_rep vs A_m2_H_H: |diff| per seed (pp) [0.0, 0.0, 0.0, 0.0, 0.0]; max 0.00
- G_DEFE_rho0.9_rep vs D0_DEFE_rho0.9: |diff| per seed (pp) [0.0, 0.0, 0.0, 0.0, 0.0]; max 0.00
- numerical-noise floor (95th percentile of |repeat - original|): XJTU macro-F1 0.00 pp; CWRU accuracy 0.00 pp

## Primary contrasts (Holm over P1-P7, alpha = 0.05)

(Holm over the 7 contrasts available; the family is P1-P7.)

| contrast | result | Holm-adjusted p | significant |
|---|---|---|---|
| P1: Mamba-3 V-H, bearing-eq bal. acc, H rule | +10.46 pp [+2.91, +18.44], p=0.0781, 6+/2- (n=8) | 0.2500 | False |
| P2: 1D-CNN V-H, bearing-eq bal. acc, H rule | +24.84 pp [+24.75, +24.92], p=0.0078, 8+/0- (n=8) | 0.0547 | False |
| P3: Mamba-2 V-H, bearing-eq bal. acc, H rule | +21.38 pp [+20.60, +22.16], p=0.0625, 5+/0- (n=5) | 0.2500 | False |
| P4: Mamba-3 V-H, macro-F1 on common test set, last15 rule | -7.18 pp [-13.13, -1.35], p=0.0391, 2+/6- (n=8) | 0.1953 | False |
| P5: rho0.9-trained, tested rho0.9 minus tested rho0 (-8 dB, D0) | +61.09 pp [+57.24, +63.52], p=0.0625, 5+/0- (n=5) | 0.2500 | False |
| P6: DIFF-only minus DE-only at rho0.9, -8 dB | +15.27 pp [+13.88, +16.76], p=0.0625, 5+/0- (n=5) | 0.2500 | False |
| P7: coverage, mean(H+V - H) at k=3 minus at k=1 (bootstrap over units; two-sided permutation p) | +29.01 pp [+7.22, +49.56], p=0.0262, 1+/5- (n=18) | 0.1572 | False |

Note: with n = 5 seeds the smallest two-sided exact Wilcoxon p is 0.0625, so P5 and P6 (n = 5) cannot reach alpha = 0.05 by construction; their direction and seed counts are reported.
