"""
bearmamba3/data_cwru_leakfree.py — leak-free CWRU windowing

STEP 2 of 故障诊断Wiki/_tasks/bm3-cwru-leakage-fix.md.

Problem this fixes: bearmamba3/data_cwru.py::CWRUDataset windows each file in
full (default 50% overlap) and pools ALL windows before any split is made;
experiments/exp01_cwru_baseline/train.py then does
torch.utils.data.random_split() over that pooled window list. Adjacent
overlapping windows can land on opposite sides of the split, so train and
val share raw samples ("textbook pattern B" leakage). There was also no
held-out test set — the val set used for checkpoint selection was the same
set whose accuracy got reported as the final number.

Fix (identical strategy to the already-verified
~/论文1/src/data/cwru_loader_leakfree.py): for each source file, split along
the TIME AXIS into train/val/test segments first, drop a guard gap at each
of the two segment boundaries, and only THEN slide windows — independently —
within each segment. Because the three segments are disjoint, non-adjacent
(guard-gapped) sub-arrays of the same 1-D signal, no window built from one
segment can share a raw sample index with a window built from another
segment. This holds by construction, for any window length/stride/overlap.

This is a NEW module. bearmamba3/data_cwru.py (MANIFEST, LABEL10, label maps,
load_signal, bearing geometry) is imported read-only and left unmodified —
it is still the code path for any pre-existing / comparison run.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from bearmamba3.data_cwru import (
    DE_BEARING,
    FE_BEARING,
    FS,
    LABEL10,
    LABEL10_NAMES,
    LABEL_NAMES,
    MANIFEST,
    load_signal,
)

# Guard gap: default = one full window length at the default win_len=2048
# used by every affected config (exp01-05, B2). Matches the paperA precedent
# of "guard == one window length" rather than an arbitrary fixed constant.
GUARD_SAMPLES_DEFAULT = 2048

# Record-level split fractions (of raw samples per file, before windowing).
# Test gets the remainder after train/val/guards are removed — not tuned,
# only reused directly from the paperA leak-free precedent (70/15/15).
TRAIN_FRAC_DEFAULT = 0.70
VAL_FRAC_DEFAULT = 0.15


def _compute_segment_bounds(
    L: int, guard: int, train_frac: float, val_frac: float
) -> dict[str, tuple[int, int]]:
    """Split [0, L) into (train, val, test) index ranges with a guard gap
    dropped at each of the two internal boundaries. Order along the time
    axis is fixed: train | guard | val | guard | test (test absorbs the
    remainder so rounding error never shrinks the guard)."""
    n_train = int(L * train_frac)
    n_val = int(L * val_frac)

    train_start, train_end = 0, n_train
    val_start = train_end + guard
    val_end = val_start + n_val
    test_start = val_end + guard
    test_end = L

    if train_end <= train_start:
        raise ValueError(f"L={L}: empty train segment")
    if val_end <= val_start:
        raise ValueError(f"L={L}: empty val segment (guard={guard} too large?)")
    if test_end <= test_start:
        raise ValueError(f"L={L}: empty test segment (guard={guard} too large?)")

    return {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }


def _window_segment(signal: np.ndarray, start: int, end: int, win_len: int, stride: int):
    """signal: (C, L) full-file array. Returns list of (C, win_len) windows
    sliced from the [start, end) sub-range, plus their in-segment start offsets."""
    seg = signal[:, start:end]
    seg_len = seg.shape[1]
    starts = list(range(0, seg_len - win_len + 1, stride))
    windows = [seg[:, s : s + win_len].copy() for s in starts]
    return windows, starts


class CWRULeakfreeSplitBuilder:
    """Builds disjoint train/val/test window pools for CWRU, split along the
    time axis (with guard gaps) before windowing, per file.

    Every file contributes windows to all three splits (matching the
    original protocol's property that every split sees every load/condition),
    but no window can straddle a split boundary or share raw samples across
    splits.
    """

    def __init__(
        self,
        data_dir: str | Path,
        ids: Optional[list[int]] = None,
        win_len: int = 2048,
        stride: int = 1024,
        channels: tuple[str, ...] = ("DE",),
        label_mode: str = "4class",
        guard_samples: int = GUARD_SAMPLES_DEFAULT,
        train_frac: float = TRAIN_FRAC_DEFAULT,
        val_frac: float = VAL_FRAC_DEFAULT,
    ):
        self.data_dir = Path(data_dir)
        self.ids = list(ids or MANIFEST.keys())
        self.win_len = win_len
        self.stride = stride
        self.channels = list(channels)
        self.label_mode = label_mode
        self.guard_samples = guard_samples
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.split_index: list[dict] = []

    def build(self) -> dict[str, list[tuple[np.ndarray, int, float]]]:
        pools: dict[str, list[tuple[np.ndarray, int, float]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.split_index = []
        for fid in self.ids:
            path = self.data_dir / f"{fid}.mat"
            if not path.exists():
                raise FileNotFoundError(
                    f"data/cwru_12k_de/{fid}.mat not found — run tools/cwru_inventory.py first"
                )
            signal, rpm = load_signal(path, self.channels)
            label = LABEL10[fid] if self.label_mode == "10class" else MANIFEST[fid][4]
            L = signal.shape[1]
            bounds = _compute_segment_bounds(
                L, self.guard_samples, self.train_frac, self.val_frac
            )
            for split_name, (s, e) in bounds.items():
                windows, starts = _window_segment(signal, s, e, self.win_len, self.stride)
                for w, st in zip(windows, starts):
                    # (window, label, rpm, file_id, absolute start sample in the file)
                    pools[split_name].append((w, label, rpm, int(fid), int(s + st)))
                self.split_index.append(
                    {
                        "file_id": fid,
                        "split": split_name,
                        "file_len": L,
                        "seg_start": s,
                        "seg_end": e,
                        "n_windows": len(windows),
                    }
                )
        return pools

    def verify_disjoint(self, pools: dict[str, list]) -> dict:
        """grep-level evidence (STEP 2.3-2): for every file, assert the
        [seg_start, seg_end) ranges of train/val/test do not overlap, and
        that consecutive ranges are separated by >= guard_samples."""
        by_file: dict[int, dict[str, tuple[int, int]]] = {}
        for row in self.split_index:
            by_file.setdefault(row["file_id"], {})[row["split"]] = (
                row["seg_start"],
                row["seg_end"],
            )
        violations = []
        for fid, segs in by_file.items():
            order = ["train", "val", "test"]
            for a, b in zip(order, order[1:]):
                a_end = segs[a][1]
                b_start = segs[b][0]
                gap = b_start - a_end
                if gap < self.guard_samples:
                    violations.append(
                        {"file_id": fid, "boundary": f"{a}->{b}", "gap": gap}
                    )
        return {
            "n_files_checked": len(by_file),
            "guard_samples_required": self.guard_samples,
            "violations": violations,
            "pass": len(violations) == 0,
        }


class CWRULeakfreeDataset(Dataset):
    """Thin Dataset wrapper around one pre-built split's window pool.

    __getitem__ noise-injection / normalization logic is copied verbatim
    from bearmamba3.data_cwru.CWRUDataset.__getitem__ (P3: no hyperparameter
    or protocol changes beyond the split itself).
    """

    def __init__(
        self,
        samples: list[tuple[np.ndarray, int, float]],
        normalize: bool = True,
        noise_snr_db: Optional[float] = None,
        noise_type: str = "awgn",
        seed: int = 0,
        noise_key: str = "v1",
        split: Optional[str] = None,
        channels: Optional[list] = None,
        noise_realisation: int = 0,
    ):
        # noise_key="v1": original behaviour, noise seeded by (local index, SNR) -- reused across
        #   splits and re-paired with windows whenever the seed changes the order (superseded).
        # noise_key="v2": noise seeded by (split, file id, absolute window start, realisation), drawn
        #   for both physical channels (row 0 = DE, row 1 = FE) and then subset to `channels`, so that
        #   (i) train / val / test noise streams are independent, (ii) a window's noise does not depend
        #   on the training seed or on the sample order, (iii) DE-only, FE-only and dual arms and all rho
        #   share the same underlying draws, rho being applied as z_FE' = rho z_DE + sqrt(1-rho^2) z_FE.
        assert noise_key in ("v1", "v2"), noise_key
        if noise_key == "v2":
            assert split in ("train", "val", "test"), split
            assert channels is not None and all(c in ("DE", "FE") for c in channels), channels
        self.noise_key = noise_key
        self.split = split
        self.channels = list(channels) if channels is not None else None
        self.noise_realisation = int(noise_realisation)
        self.samples = list(samples)
        self.normalize = normalize
        self.noise_snr_db = noise_snr_db
        self.noise_type = noise_type
        rng = np.random.default_rng(seed)
        rng.shuffle(self.samples)  # order-only shuffle WITHIN this split

    def __len__(self):
        return len(self.samples)

    _SPLIT_CODE = {"train": 1, "val": 2, "test": 3}

    def _noise_v2(self, w, fid, start):
        rng = np.random.default_rng([self._SPLIT_CODE[self.split], int(fid), int(start), self.noise_realisation])
        z = rng.standard_normal((2, w.shape[1])).astype(np.float32)   # row 0: DE, row 1: FE
        if self.noise_type == "awgn":
            rho = 0.0
        elif self.noise_type.startswith("awgn_corr"):
            rho = float(self.noise_type[len("awgn_corr"):])
        else:
            raise ValueError(f"noise_key=v2 supports awgn / awgn_corr only, got {self.noise_type!r}")
        if rho != 0.0:
            z[1] = rho * z[0] + np.float32(np.sqrt(1.0 - rho ** 2)) * z[1]
        z = z[[0 if c == "DE" else 1 for c in self.channels]]
        sig_pwr = np.mean(w ** 2, axis=1, keepdims=True).clip(min=1e-12)
        noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
        return w + z * noise_std

    def __getitem__(self, idx):
        w, label, rpm = self.samples[idx][:3]
        if self.noise_snr_db is not None and self.noise_key == "v2":
            fid, start = self.samples[idx][3], self.samples[idx][4]
            w = self._noise_v2(w, fid, start)
        elif self.noise_snr_db is not None:
            rng = np.random.default_rng(hash((idx, self.noise_snr_db)) & 0xFFFFFFFF)
            if self.noise_type == "awgn":
                sig_pwr = np.mean(w ** 2, axis=1, keepdims=True).clip(min=1e-12)
                noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
                w = w + (rng.standard_normal(w.shape).astype(np.float32) * noise_std)
            elif self.noise_type == "pink":
                from bearmamba3.noise import add_noise_at_snr
                w = w.copy()
                for c in range(w.shape[0]):
                    w[c] = add_noise_at_snr(w[c], self.noise_snr_db, rng, noise_type="pink")
            elif self.noise_type.startswith("awgn_corr"):
                rho = float(self.noise_type[len("awgn_corr"):])
                sig_pwr = np.mean(w ** 2, axis=1, keepdims=True).clip(min=1e-12)
                noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
                z = rng.standard_normal(w.shape).astype(np.float32)   # same draw as 'awgn'
                if z.shape[0] == 2:
                    z[1] = rho * z[0] + np.sqrt(1.0 - rho ** 2) * z[1]   # unit variance, corr(z0, z1) = rho
                w = w + z * noise_std
            else:
                raise ValueError(f"unknown noise_type={self.noise_type!r}")
        if self.normalize:
            mu, std = w.mean(axis=1, keepdims=True), w.std(axis=1, keepdims=True) + 1e-8
            w = (w - mu) / std
        return (
            torch.from_numpy(w),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(rpm, dtype=torch.float32),
        )
