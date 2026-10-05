#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/train_pu_leakfree.py

P12 (2026-09-18) leak-free PU cross-condition driver. Adapted from
experiments/exp06_pu/train.py. Core fix: COND_TEST is no longer used for
per-epoch checkpoint selection. Instead:
  - train/val are a file-level split WITHIN COND_TRAIN (data_pu_leakfree.py,
    P12(a)(b)(e)), so selection uses only source-domain data.
  - COND_TEST is evaluated exactly once after training, with TWO checkpoints:
    the best-val-epoch weights and the final-epoch weights (P12(g) — both
    are reported, no cherry-picking between them).
  - Per-epoch source-domain val_acc is recorded in full so P12(f)'s epoch-
    selection-stability diagnostics can be computed post hoc.

Usage:
  source venv/bin/activate
  python experiments/exp_bm3_cwru_leakfree/train_pu_leakfree.py \
      --config <config.yaml> [--smoke] [--seeds 0 1 2 3 4]
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_pu import BEARING_KWARGS, COND_TEST, COND_TRAIN, FS, PUDataset
from bearmamba3.kinematic_loss import compute_fault_freqs, instantaneous_freqs, kinematic_loss
from bearmamba3.model import BearMamba3
from experiments.exp_bm3_cwru_leakfree.data_pu_leakfree import (
    PUFileSplitBuilder, PUDatasetFromFiles, annotate_window_counts,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "results_dir"):
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot_source(results_dir: Path, cfg_path: Path):
    snap_dir = results_dir / "source_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path(__file__).parent / "data_pu_leakfree.py",
        cfg_path,
    ]
    manifest = []
    for f in files:
        dst = snap_dir / f.name
        dst.write_bytes(f.read_bytes())
        manifest.append({
            "file": f.name, "md5": md5_of(f), "mtime": f.stat().st_mtime,
        })
    with open(snap_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return [m["file"] for m in manifest]


def get_or_build_split(cfg: dict, results_root: Path, smoke: bool) -> dict:
    """Split is shared across all training seeds of THIS config (P12: split-
    seed decoupled from training seeds) and cached to disk so re-running
    (idempotent resume) doesn't recompute/re-window every file."""
    split_path = results_root / "split_record.json"
    cache_path = results_root / "split_files.json"
    if split_path.exists() and cache_path.exists() and not smoke:
        record = json.load(open(split_path))
        files = json.load(open(cache_path))
        return {"train": files["train"], "val": files["val"], "record": record}

    builder = PUFileSplitBuilder(cfg["data_dir"], conditions=COND_TRAIN,
                                  val_frac=cfg.get("val_frac", 0.2),
                                  split_seed=cfg.get("split_seed", 42))
    split = builder.build()
    split["record"] = annotate_window_counts(
        split["record"], cfg["win_len"], cfg.get("stride", cfg["win_len"]),
        split["train"], split["val"],
    )
    if not smoke:
        results_root.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split["record"], f, indent=2)
        with open(cache_path, "w") as f:
            json.dump({"train": split["train"], "val": split["val"]}, f, indent=2)
    return split


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    backbone = cfg.get("backbone", "mamba3")
    if backbone == "mamba2":
        from baselines.mamba2 import BearMamba2
        return BearMamba2(
            d_model=cfg["d_model"], d_state=cfg["d_state"], n_layers=cfg["n_layers"],
            n_sensors=1, n_classes=3, conv_stride=cfg["conv_stride"], dtype=torch.bfloat16,
        ).to(device)
    return BearMamba3(
        d_model=cfg["d_model"], d_state=cfg["d_state"], n_layers=cfg["n_layers"],
        n_sensors=1, n_classes=3, conv_stride=cfg["conv_stride"], is_mimo=False,
        dtype=torch.bfloat16,
    ).to(device)


def eval_acc(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_seed_leakfree(cfg: dict, seed: int, split: dict, device: torch.device,
                             results_dir: Path, smoke: bool) -> dict:
    set_seed(seed)
    win_len = cfg["win_len"]
    stride = cfg.get("stride", win_len)
    noise_snr = cfg.get("noise_snr_db", None)

    train_files = split["train"][:8] if smoke else split["train"]
    val_files = split["val"][:4] if smoke else split["val"]

    train_ds = PUDatasetFromFiles(train_files, win_len, stride, noise_snr_db=noise_snr)
    val_ds = PUDatasetFromFiles(val_files, win_len, stride, noise_snr_db=noise_snr)
    test_ds = PUDataset(cfg["data_dir"], conditions=COND_TEST, win_len=win_len,
                         stride=stride, noise_snr_db=noise_snr)

    bs = cfg["batch_size"]
    nw = 0 if smoke else cfg.get("num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_epochs = 2 if smoke else cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    backbone = cfg.get("backbone", "mamba3")
    lambda_kin = cfg.get("lambda_kin", 0.0)
    kin_variant = cfg.get("kin_variant", "cover")
    fs_hz = cfg.get("fs_hz", float(FS))
    fs_eff = fs_hz / cfg["conv_stride"]
    do_kin = (lambda_kin > 0) and (backbone not in ("mamba2", "cnn1d", "transformer1d"))

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_ce = running_kin = n_steps = 0
        for step, (x, labels, rpm) in enumerate(train_loader):
            if smoke and step >= 2:
                break
            x, labels, rpm = x.to(device), labels.to(device), rpm.to(device)
            if do_kin:
                out, kin = model(x, return_kin=True)
                l_ce = nn.functional.cross_entropy(out, labels)
                l_kin = kinematic_loss(kin, rpm, fs_eff, variant=kin_variant, bearing_kwargs=BEARING_KWARGS)
                loss = l_ce + lambda_kin * l_kin
            else:
                out = model(x)
                l_ce = nn.functional.cross_entropy(out, labels)
                l_kin = torch.tensor(0.0)
                loss = l_ce
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()
            running_ce += l_ce.item()
            running_kin += l_kin.item()
            n_steps += 1
        scheduler.step()

        val_acc = eval_acc(model, val_loader, device)  # P12(a): source-domain val, NOT test
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        log = {"epoch": epoch, "l_ce": running_ce / max(n_steps, 1),
               "l_kin": running_kin / max(n_steps, 1), "val_acc": val_acc}
        history.append(log)

        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(f"  [P12 source-val] seed={seed} epoch={epoch:3d}/{n_epochs}"
                  f"  l_ce={log['l_ce']:.4f}  l_kin={log['l_kin']:.4f}"
                  f"  val_acc(source)={val_acc:.4f}  [{elapsed:.0f}s]")

    last_state = copy.deepcopy(model.state_dict())

    # P12(g): evaluate test exactly once with best-val weights AND once with
    # last-epoch weights; both reported, no cherry-picking.
    model.load_state_dict(best_state)
    test_acc_best = eval_acc(model, test_loader, device)
    model.load_state_dict(last_state)
    test_acc_last = eval_acc(model, test_loader, device)

    elapsed = time.time() - t0
    print(f"  [P12 fixed-post] seed={seed} best_epoch={best_epoch}/{n_epochs}"
          f"  test_acc(best-val)={test_acc_best:.4f}  test_acc(last-epoch)={test_acc_last:.4f}"
          f"  (first & only test-loader touch, x2 checkpoints)")

    # P12(f) diagnostics: raw ingredients only, family-level judgment at analysis time
    val_accs = [h["val_acc"] for h in history]
    last5_mean = float(np.mean(val_accs[-5:])) if len(val_accs) >= 5 else float(np.mean(val_accs))
    window = val_accs[max(0, best_epoch - 3):best_epoch + 2] if best_epoch > 0 else val_accs[:1]
    local_jitter = float(np.std(window)) if len(window) > 1 else 0.0

    if not smoke:
        ckpt_dir = results_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": best_epoch, "val_acc": best_val_acc, "model_state": best_state},
                   ckpt_dir / f"best_epoch{best_epoch:03d}_seed{seed}.pt")
        torch.save({"epoch": n_epochs, "val_acc": val_accs[-1], "model_state": last_state},
                   ckpt_dir / f"last_epoch{n_epochs:03d}_seed{seed}.pt")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc_source": best_val_acc,
        "last_val_acc_source": val_accs[-1],
        "last5_mean_val_acc_source": last5_mean,
        "best_minus_last5_gap": best_val_acc - last5_mean,
        "local_jitter_std_around_best": local_jitter,
        "best_epoch_in_first10": best_epoch <= 10,
        "test_acc_best_val": test_acc_best,
        "test_acc_last_epoch": test_acc_last,
        "elapsed_s": elapsed,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg = load_config(cfg_path)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).expanduser())
    seeds = args.seeds or cfg["seeds"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "mamba3")
    print(f"[P12 leak-free PU] Device: {device}  backbone={backbone}"
          f"  lambda_kin={cfg.get('lambda_kin',0)}  snr={cfg.get('noise_snr_db')}dB"
          f"  seeds={seeds}  epochs={2 if args.smoke else cfg['epochs']}"
          f"  val_frac={cfg.get('val_frac', 0.2)}  split_seed={cfg.get('split_seed', 42)}")

    results_dir = Path(cfg["results_dir"])
    if not args.smoke:
        results_dir.mkdir(parents=True, exist_ok=True)
        snapshot_source(results_dir, cfg_path)

    split = get_or_build_split(cfg, results_dir, args.smoke)
    print(f"  split: train={len(split['train'])} files, val={len(split['val'])} files"
          f"  val_class_file_counts={split['record'].get('val_class_file_counts')}")

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}")
        seed_path = results_dir / f"seed_{seed}.json"
        if seed_path.exists() and not args.smoke:
            print(f"  [SKIP] {seed_path.name} exists")
            all_results.append(json.load(open(seed_path)))
            continue
        result = train_one_seed_leakfree(cfg, seed, split, device, results_dir, args.smoke)
        all_results.append(result)
        if not args.smoke:
            with open(seed_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  -> saved {seed_path}")

    if not args.smoke:
        best_accs = [r["test_acc_best_val"] for r in all_results]
        last_accs = [r["test_acc_last_epoch"] for r in all_results]
        best_epochs = [r["best_epoch"] for r in all_results]
        n_first10 = sum(1 for r in all_results if r["best_epoch_in_first10"])
        max_jitter = max(r["local_jitter_std_around_best"] for r in all_results)
        summary = {
            "config": cfg_path.name,
            "protocol": "leakfree_pu_v1 (P12: source-domain-val selection, test evaluated once)",
            "lambda_kin": cfg.get("lambda_kin", 0.0),
            "noise_snr_db": cfg.get("noise_snr_db"),
            "seeds": seeds,
            "test_acc_best_val_per_seed": best_accs,
            "test_acc_last_epoch_per_seed": last_accs,
            "mean_test_acc_best_val": float(np.mean(best_accs)),
            "std_test_acc_best_val": float(np.std(best_accs, ddof=1)) if len(best_accs) > 1 else 0.0,
            "mean_test_acc_last_epoch": float(np.mean(last_accs)),
            "std_test_acc_last_epoch": float(np.std(last_accs, ddof=1)) if len(last_accs) > 1 else 0.0,
            "best_epochs_per_seed": best_epochs,
            "n_runs_best_epoch_in_first10": n_first10,
            "max_local_jitter_std_around_best": max_jitter,
            "stability_flag_P12f": "UNSTABLE" if (n_first10 > len(seeds) / 2 or max_jitter > 0.02) else "OK",
        }
        summary_path = results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"RESULTS: test_acc(best-val) = {summary['mean_test_acc_best_val']*100:.2f}"
              f"±{summary['std_test_acc_best_val']*100:.2f}%")
        print(f"         test_acc(last-epoch) = {summary['mean_test_acc_last_epoch']*100:.2f}"
              f"±{summary['std_test_acc_last_epoch']*100:.2f}%")
        print(f"         P12(f) stability_flag = {summary['stability_flag_P12f']}")
        print(f"Summary saved to {summary_path}")
    else:
        print("\n✅ Smoke test passed (no summary.json written)")


if __name__ == "__main__":
    main()
