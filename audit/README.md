# Degraded-phase (onset) audit, XJTU-SY

`onset_audit.py` is a read-only analysis of the raw XJTU-SY CSV files (no training). For each of
the eleven bearings used it recomputes the degraded-phase start with the rule of manuscript
section 3.1 — first file of the first five-file block in which the mean kurtosis exceeds 5 or the
mean standard deviation exceeds twice its baseline (mean over the first 20 files), whichever comes
first; fallback to the last 15 % of files — separately on the horizontal (H) and the vertical (V)
channel, and compares the H-defined window set (the one used for training and testing) with the
V-defined, joint (min of H and V) and fixed last-15 % alternatives.

```bash
DATA_ROOT=/path/to/data python audit/onset_audit.py    # reads $DATA_ROOT/XJTU-SY_Bearing_Datasets
```

Output `onset_audit.csv` (included; also `onset_audit_traces.json`, not included):
`onset_*` = onset file index (0-based) under each rule, `trigger_*` = criterion that fired,
`fallback_*` = whether the last-15 % fallback was used (never), `windows_*` = number of
2048-sample windows from the onset to failure (16 per 32768-sample file), `jaccard_*` = overlap of
the file sets. Joint onset equals the H onset except Bearing1_3 (100 vs 105) and Bearing3_3 (338 vs 341).
No model was retrained with an alternative onset rule (supplementary section S6.8).

The only edit relative to the version used for the manuscript is the data path, which is now read
from `DATA_ROOT` instead of being hard-coded.
