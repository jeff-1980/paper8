# Superseded entry points (not used for any reported number)

An earlier version of this study used two protocols that are corrected for every result in the
manuscript (supplementary section S7 gives before/after tables):

1. CWRU windows were extracted with overlap from each whole recording and split at window level with
   `random_split`, so training and test windows could share raw samples.
2. XJTU-SY (and Paderborn) checkpoints were selected by the best epoch on the test fold.

The original entry points are kept here only for the record. They are not imported by any script in
this release.

| file | superseded protocol | replaced by |
|---|---|---|
| `exp01_cwru_baseline/train.py` (+ `config.yaml`, `config_kin.yaml`) | overlapping CWRU windows + `random_split` (via `bearmamba3.data_cwru.CWRUDataset`) | `experiments/exp01_cwru_baseline/train_leakfree.py` (time-axis 70/15/15 split with 2048-sample guard gaps, validation-selected epoch) |
| `exp_xjtu/train.py` | XJTU-SY best epoch selected on the test fold | `experiments/exp_bm3_cwru_leakfree/train_xjtu_leakfree.py`, `experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py`, `scripts/xjtu/`, `scripts/lobo/` (fixed final epoch 50) |

Two pieces of the old code remain in the main tree because the leak-free code imports them:

* `bearmamba3/data_cwru.py` — `data_cwru_leakfree.py` imports its recording manifest, label tables
  and `load_signal()`, and the leak-free trainer copies the file into every source snapshot. Its
  `CWRUDataset` class (the superseded overlapping-window dataset) is not used by any entry point.
* `experiments/exp01_cwru_baseline/train.py` in the main tree is **not** the original: it is a
  helper module containing only `build_model()` and `save_kin_snapshot()`, copied verbatim from the
  original, because `train_leakfree.py` imports them with `from train import ...`. Run as a script
  it exits with a message. `train_leakfree.py` itself is byte-identical to the version recorded in the
  result source snapshots.
