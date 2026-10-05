"""Helper module for experiments/exp01_cwru_baseline/train_leakfree.py.

train_leakfree.py imports `build_model` and `save_kin_snapshot` from a module named
`train` in its own directory. The original train.py (overlapping windows split at
window level with random_split; superseded) is kept only under historical/.
This file contains those two functions copied verbatim from it, with its imports,
and refuses to run as an entry point.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.kinematic_loss import instantaneous_freqs, compute_fault_freqs
from bearmamba3.model import BearMamba3


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    backbone = cfg.get("backbone", "mamba3")
    n_classes = cfg.get("n_classes", 4)
    if cfg.get("label_mode", "4class") == "10class":
        n_classes = 10
    if backbone == "mamba2":
        from baselines.mamba2 import BearMamba2
        return BearMamba2(
            d_model=cfg["d_model"],
            d_state=cfg["d_state"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
            dtype=torch.bfloat16,
        ).to(device)
    if backbone == "cnn1d":
        from baselines.cnn1d import BearCNN1D
        return BearCNN1D(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
        ).to(device)
    if backbone == "cnn1d_nobn":
        from baselines.onedcnn_nobn import BearCNN1D_NoBN
        return BearCNN1D_NoBN(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
        ).to(device)
    if backbone == "cnn1d_ln":
        from baselines.cnn1d_ln import BearCNN1D_LN
        return BearCNN1D_LN(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
        ).to(device)
    if backbone == "cnn1d_attnfusion":
        from baselines.cnn1d_attnfusion import BearCNN1DAttnFusion
        return BearCNN1DAttnFusion(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
        ).to(device)
    if backbone == "transformer1d":
        from baselines.transformer1d import BearTransformer1D
        return BearTransformer1D(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=len(cfg["channels"]),
            n_classes=n_classes,
            conv_stride=cfg["conv_stride"],
            nhead=cfg.get("nhead", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
        ).to(device)
    # default: mamba3 SISO
    return BearMamba3(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        n_sensors=len(cfg["channels"]),
        n_classes=n_classes,
        conv_stride=cfg["conv_stride"],
        is_mimo=False,
        use_batchnorm=cfg.get("use_batchnorm", False),
        dtype=torch.bfloat16,
    ).to(device)


def save_kin_snapshot(model, fixed_batch, fs_eff, epoch, tag, results_dir, device):
    """Save time-averaged instantaneous freq per state + fault-freq targets to .npz.

    Saves f_bar (B, S) = mean over time of |f_state|, not the full (B, S, L) tensor.
    This is ~500x smaller (0.25 MB vs 120 MB) and sufficient for Step-5 histograms:
      - per-class distribution of state frequencies vs fault-freq lines.
    """
    x, labels, rpm = [t.to(device) for t in fixed_batch]
    model.eval()
    with torch.no_grad():
        _, kin = model(x, return_kin=True)
    # (B, S, L) → (B, S) time-average; Step-5 histogram is over B×S values per class
    f_bar   = instantaneous_freqs(kin, fs_eff).mean(dim=-1).cpu().numpy()  # (B, S)
    f_fault = compute_fault_freqs(rpm, device=device).cpu().numpy()         # (B, J)
    path = Path(results_dir) / f"{tag}_kin_ep{epoch:03d}.npz"
    np.savez_compressed(
        path,
        f_bar=f_bar.astype(np.float32),
        f_fault=f_fault.astype(np.float32),
        labels=labels.cpu().numpy().astype(np.int32),
        epoch=np.int32(epoch),
        fs_eff=np.float32(fs_eff),
    )


if __name__ == "__main__":
    sys.exit("experiments/exp01_cwru_baseline/train.py is not an entry point in this release; "
             "use train_leakfree.py (see README). The superseded original is in historical/.")
