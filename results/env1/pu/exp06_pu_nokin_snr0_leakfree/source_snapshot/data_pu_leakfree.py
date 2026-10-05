"""
experiments/exp_bm3_cwru_leakfree/data_pu_leakfree.py

P12 (2026-09-18): file-level train/val split within PU's COND_TRAIN, so
epoch selection no longer touches COND_TEST (the "test-set-used-for-
selection" leak documented in BATCH5_cost_table.md and
experiments/exp06_pu/train.py:56-62).

Every exp06_pu config uses stride == win_len (non-overlapping windows),
and PUDataset windows each .mat file independently (never spanning file
boundaries) — so a file-level split is inherently window-leak-free with
no guard gap needed, unlike CWRU's continuous-per-file signal.

Reuses bearmamba3.data_pu's file-parsing / windowing helpers read-only
(no modification to that module — it still backs the old leaky M4/B1
results, which remain valid provenance for their own claims).
"""
import glob
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from bearmamba3.data_pu import (
    _parse_fname, _load_vib, _sliding_windows,
    LABEL_MAP, LABEL_NAMES, COND_RPM, BEARING_KWARGS, FS, COND_TRAIN, COND_TEST,
)

VAL_FRAC_DEFAULT = 0.2
SPLIT_SEED_DEFAULT = 42   # P12: split-seed decoupled from training seeds


class PUFileSplitBuilder:
    """
    File-level stratified split within `conditions` (default COND_TRAIN).
    Splits independently per (condition, bearing) group so every group
    contributes to both train and val (P12(e): each bearing maps to a
    single fixed damage class, so per-group coverage guarantees the
    14-group aggregate val set covers all 3 classes).
    """

    def __init__(self, data_dir, conditions=COND_TRAIN,
                 val_frac=VAL_FRAC_DEFAULT, split_seed=SPLIT_SEED_DEFAULT):
        self.data_dir = data_dir
        self.conditions = conditions
        self.val_frac = val_frac
        self.split_seed = split_seed

    def build(self) -> dict:
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.mat")))
        groups = {}  # (cond, bearing) -> [file paths]
        for path in all_files:
            cond, bearing = _parse_fname(path)
            if bearing not in LABEL_MAP:
                continue
            if cond not in self.conditions:
                continue
            groups.setdefault((cond, bearing), []).append(path)

        train_files, val_files = [], []
        group_record = {}
        for (cond, bearing), files in sorted(groups.items()):
            files = sorted(files)
            rng = np.random.default_rng(
                (hash((cond, bearing, self.split_seed)) ^ self.split_seed) & 0xFFFFFFFF
            )
            idx = rng.permutation(len(files))
            n_val = max(1, int(round(len(files) * self.val_frac)))
            val_idx = set(idx[:n_val].tolist())
            g_train = [f for i, f in enumerate(files) if i not in val_idx]
            g_val = [f for i, f in enumerate(files) if i in val_idx]
            train_files.extend(g_train)
            val_files.extend(g_val)
            group_record[f"{cond}_{bearing}"] = {
                "label": LABEL_NAMES[LABEL_MAP[bearing]],
                "n_total": len(files),
                "n_train": len(g_train),
                "n_val": len(g_val),
                "train_files": [os.path.basename(f) for f in g_train],
                "val_files": [os.path.basename(f) for f in g_val],
            }

        # P12(e): class-distribution accounting at the 14-group aggregate level
        def class_dist(file_list):
            dist = Counter()
            for f in file_list:
                _, bearing = _parse_fname(f)
                dist[LABEL_NAMES[LABEL_MAP[bearing]]] += 1
            return dict(dist)

        train_class_files = class_dist(train_files)
        val_class_files = class_dist(val_files)
        missing_in_val = [c for c in LABEL_NAMES.values() if c not in val_class_files]
        missing_in_train = [c for c in LABEL_NAMES.values() if c not in train_class_files]
        if missing_in_val or missing_in_train:
            raise RuntimeError(
                f"P12(e) coverage check FAILED: missing_in_val={missing_in_val} "
                f"missing_in_train={missing_in_train}"
            )

        split_record = {
            "val_frac": self.val_frac,
            "split_seed": self.split_seed,
            "conditions": sorted(self.conditions),
            "groups": group_record,
            "train_n_files": len(train_files),
            "val_n_files": len(val_files),
            "train_class_file_counts": train_class_files,
            "val_class_file_counts": val_class_files,
            "coverage_check": "PASS (all 3 classes present in both train and val)",
        }
        return {"train": train_files, "val": val_files, "record": split_record}


def _windows_per_file(win_len: int, stride: int, n_samples_hint: int = 256000) -> int:
    return (n_samples_hint - win_len) // stride + 1


def annotate_window_counts(split_record: dict, win_len: int, stride: int,
                            train_files: list, val_files: list) -> dict:
    """Fill in actual window counts (P12(b): split record must include window
    counts, not just file counts) by loading each file once."""
    def total_windows(file_list):
        total = 0
        counts_by_bearing = Counter()
        for path in file_list:
            vib = _load_vib(path)
            n = len(_sliding_windows(vib, win_len, stride))
            total += n
            _, bearing = _parse_fname(path)
            counts_by_bearing[LABEL_NAMES[LABEL_MAP[bearing]]] += n
        return total, dict(counts_by_bearing)

    train_windows, train_class_windows = total_windows(train_files)
    val_windows, val_class_windows = total_windows(val_files)
    split_record["train_n_windows"] = train_windows
    split_record["val_n_windows"] = val_windows
    split_record["train_class_window_counts"] = train_class_windows
    split_record["val_class_window_counts"] = val_class_windows
    return split_record


class PUDatasetFromFiles(Dataset):
    """Same __getitem__ behavior as bearmamba3.data_pu.PUDataset (noise/
    normalize logic copied verbatim), but built from an explicit file list
    instead of a `conditions` filter — used for the leak-free train/val
    split (train/val) and reused as-is for the untouched test condition."""

    def __init__(self, file_list, win_len=4096, stride=None,
                 normalize=True, noise_snr_db=None):
        stride = stride or win_len
        self.win_len = win_len
        self.normalize = normalize
        self.noise_snr_db = noise_snr_db

        segs, labels, rpms = [], [], []
        for path in file_list:
            cond, bearing = _parse_fname(path)
            label = LABEL_MAP[bearing]
            rpm = float(COND_RPM.get(cond, 1500))
            vib = _load_vib(path)
            wins = _sliding_windows(vib, win_len, stride)
            n = len(wins)
            segs.append(wins)
            labels.append(np.full(n, label, dtype=np.int64))
            rpms.append(np.full(n, rpm, dtype=np.float32))

        self._data = np.concatenate(segs, axis=0)
        self._labels = np.concatenate(labels, axis=0)
        self._rpms = np.concatenate(rpms, axis=0)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        w = self._data[idx].copy()

        if self.noise_snr_db is not None:
            rng = np.random.default_rng(hash((idx, self.noise_snr_db)) & 0xFFFFFFFF)
            sig_pwr = np.mean(w ** 2, keepdims=True).clip(min=1e-12)
            noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
            w = w + rng.standard_normal(w.shape).astype(np.float32) * noise_std

        if self.normalize:
            mu = w.mean()
            std = w.std() + 1e-8
            w = (w - mu) / std

        x = torch.from_numpy(w[None, :])
        lbl = torch.tensor(self._labels[idx], dtype=torch.long)
        rpm = torch.tensor(self._rpms[idx], dtype=torch.float32)
        return x, lbl, rpm
