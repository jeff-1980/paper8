"""
experiments/exp01_cwru_baseline/train_leakfree.py

Leak-free counterpart of train.py — STEP 2/4/5 of
故障诊断Wiki/_tasks/bm3-cwru-leakage-fix.md.

Same model / optimizer / training loop / hyperparameters as train.py
(P3: no tuning — only the data split changes). Differences from train.py:
  1. Uses bearmamba3.data_cwru_leakfree.CWRULeakfreeSplitBuilder instead of
     CWRUDataset + torch.utils.data.random_split. Train/val/test windows are
     built from disjoint, guard-gapped time-axis segments of each file.
  2. Three-way split: train (fit) / val (early-stopping & checkpoint
     selection only) / test (final report only, never touched during
     training or selection).
  3. Reports test_acc_at_best_val (test accuracy of the checkpoint that had
     the best val accuracy) as the number that would go in the paper —
     best_val_acc is kept in the output for reference/selection audit only.
  4. Writes split_index.json (once per experiment, not per seed — the split
     itself does not depend on seed) so the partition is inspectable.
  5. Writes disjointness_check.json (STEP 2.3 evidence).

Usage:
  source venv/bin/activate
  python experiments/exp01_cwru_baseline/train_leakfree.py --config config.yaml [--smoke]
"""
import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Infrastructure piece #2 (source snapshot) — files whose content determines
# this script's training/eval behaviour. Copied byte-for-byte into every
# run's results_dir with md5s, so a later edit to the live files (e.g. for a
# different experiment) can never make a past run's exact code unrecoverable.
# Lesson: KG-cSSM's driver.py was edited between two batch runs and the code
# version behind 16 earlier runs was lost with no way to reconstruct it.
SNAPSHOT_FILES = [
    Path(__file__),
    PROJECT_ROOT / "bearmamba3" / "data_cwru_leakfree.py",
    PROJECT_ROOT / "bearmamba3" / "data_cwru.py",
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot_source(results_dir: Path, cfg_path: Path) -> list[dict]:
    snap_dir = Path(results_dir) / "source_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in SNAPSHOT_FILES + [Path(cfg_path)]:
        dst = snap_dir / src.name
        shutil.copy2(src, dst)
        manifest.append({
            "original_path": str(src),
            "snapshot_path": str(dst),
            "md5": _md5(src),
            "mtime_utc": src.stat().st_mtime,
        })
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

from bearmamba3.data_cwru import LABEL10_NAMES
from bearmamba3.data_cwru_leakfree import CWRULeakfreeDataset, CWRULeakfreeSplitBuilder
from bearmamba3.kinematic_loss import kinematic_loss, instantaneous_freqs, compute_fault_freqs

# reuse build_model / save_kin_snapshot unchanged from train.py (same dir)
from train import build_model, save_kin_snapshot  # noqa: E402


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


def build_split(cfg: dict):
    """Build the (seed-independent) train/val/test window pools once."""
    label_mode = cfg.get("label_mode", "4class")
    builder = CWRULeakfreeSplitBuilder(
        data_dir=cfg["data_dir"],
        win_len=cfg["win_len"],
        stride=cfg["stride"],
        channels=cfg["channels"],
        label_mode=label_mode,
        guard_samples=cfg.get("guard_samples", 2048),
        train_frac=cfg.get("train_frac", 0.70),
        val_frac=cfg.get("val_frac", 0.15),
    )
    pools = builder.build()
    disjoint = builder.verify_disjoint(pools)
    if not disjoint["pass"]:
        raise RuntimeError(f"Leak-free split verification FAILED: {disjoint}")
    return pools, builder.split_index, disjoint


def build_loaders(cfg: dict, seed: int, smoke: bool, pools: dict):
    noise_snr = cfg.get("noise_snr_db", None)
    noise_type = cfg.get("noise_type", "awgn")

    train_ds = CWRULeakfreeDataset(pools["train"], normalize=True, seed=seed,
                                    noise_snr_db=noise_snr, noise_type=noise_type)
    val_ds = CWRULeakfreeDataset(pools["val"], normalize=True, seed=seed,
                                  noise_snr_db=noise_snr, noise_type=noise_type)
    test_ds = CWRULeakfreeDataset(pools["test"], normalize=True, seed=seed,
                                   noise_snr_db=noise_snr, noise_type=noise_type)

    bs = cfg["batch_size"]
    nw = 0 if smoke else cfg.get("num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=nw, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        num_workers=nw, pin_memory=True
    )
    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


def eval_epoch(model, loader, device):
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


def eval_and_dump_test(model, loader, device, dump_path: Path) -> float:
    """STEP 5: per-sample output must be saved alongside the summary
    accuracy (predicted label, softmax probs, penultimate-layer feature) —
    a scalar accuracy alone is not sufficient evidence for STEP 6 tables/figs.

    Captures the classifier's INPUT (the (B, d_model) feature right after
    model.norm(...).mean(dim=1), before the linear head) via a forward hook,
    rather than touching bearmamba3/model.py, since that file is shared by
    every other experiment in the project.
    """
    model.eval()
    feats_buf = []

    def _hook(module, inputs, output):
        feats_buf.append(inputs[0].detach().float().cpu())

    handle = model.classifier.register_forward_hook(_hook)

    all_pred, all_true, all_prob, all_feat, all_rpm = [], [], [], [], []
    correct = total = 0
    with torch.no_grad():
        for x, labels, rpm in loader:
            x, labels = x.to(device), labels.to(device)
            feats_buf.clear()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), dim=-1)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_pred.append(pred.cpu().numpy())
            all_true.append(labels.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            all_feat.append(feats_buf[0].numpy())
            all_rpm.append(rpm.numpy())
    handle.remove()

    np.savez_compressed(
        dump_path,
        pred_label=np.concatenate(all_pred).astype(np.int32),
        true_label=np.concatenate(all_true).astype(np.int32),
        softmax_prob=np.concatenate(all_prob).astype(np.float32),
        penultimate_feat=np.concatenate(all_feat).astype(np.float32),
        rpm=np.concatenate(all_rpm).astype(np.float32),
    )
    return correct / total if total > 0 else 0.0


def train_one_seed(cfg: dict, seed: int, device: torch.device, pools: dict,
                    smoke: bool = False) -> dict:
    set_seed(seed)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_loaders(
        cfg, seed, smoke, pools
    )
    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    n_epochs = 2 if smoke else cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    lambda_kin    = cfg.get("lambda_kin", 0.0)
    kin_variant   = cfg.get("kin_variant", "cover")
    bearing_kwargs= cfg.get("bearing_kwargs", {})
    fs_hz         = cfg.get("fs_hz", 12_000.0)
    fs_eff        = fs_hz / cfg["conv_stride"]
    backbone      = cfg.get("backbone", "mamba3")
    do_kin = (lambda_kin > 0) and (backbone != "mamba2")

    snap_batch = None
    snap_tag = None
    if do_kin and not smoke:
        snap_batch = next(iter(val_loader))
        exp_name = cfg.get("name", "exp")
        lk_str = f"lk{lambda_kin:.0e}".replace("e-0", "e-").replace("e+0", "e")
        snap_tag = f"{exp_name}_seed{seed}_{lk_str}"
    snap_epochs = {1} | {e for e in range(10, n_epochs + 1, 10)}

    results_dir = cfg["results_dir"]
    # Infrastructure piece #1 (per-epoch checkpoints) — KG-cSSM only kept
    # best.pt in memory; when a later re-evaluation wanted a different
    # checkpoint-selection rule, there were no per-epoch weights to fall
    # back on and the run had to be redone from scratch. Disk cost is
    # negligible here (~1.14MB/checkpoint measured, ~11.7GB for the full
    # 42-config x 5-seed x 50-epoch rerun), so every epoch is saved, not
    # just a reduced set of selection-rule candidates.
    ckpt_dir = Path(results_dir) / "checkpoints" / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = 0.0
    best_val_epoch = 0
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
                l_ce  = nn.functional.cross_entropy(out, labels)
                l_kin = kinematic_loss(kin, rpm, fs_eff, variant=kin_variant,
                                      bearing_kwargs=bearing_kwargs)
                loss  = l_ce + lambda_kin * l_kin
            else:
                out   = model(x)
                l_ce  = nn.functional.cross_entropy(out, labels)
                l_kin = torch.tensor(0.0)
                loss  = l_ce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()

            running_ce  += l_ce.item()
            running_kin += l_kin.item()
            n_steps += 1

        scheduler.step()
        val_acc = eval_epoch(model, val_loader, device)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            # STEP 2: test set is NEVER touched here — only val drives selection.
            best_state = copy.deepcopy(model.state_dict())

        log = {
            "epoch": epoch,
            "l_ce": running_ce / max(n_steps, 1),
            "l_kin": running_kin / max(n_steps, 1),
            "val_acc": val_acc,
        }
        history.append(log)

        # Infrastructure piece #1: persist this epoch's weights + optimizer
        # state unconditionally (not gated on being the current best), so any
        # checkpoint-selection rule can be re-derived later without retraining.
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "l_ce": log["l_ce"],
            "l_kin": log["l_kin"],
        }, ckpt_dir / f"epoch{epoch:03d}.pt")

        if snap_batch is not None and epoch in snap_epochs:
            save_kin_snapshot(model, snap_batch, fs_eff, epoch, snap_tag,
                              results_dir, device)

        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(
                f"  seed={seed} epoch={epoch:3d}/{n_epochs}"
                f"  l_ce={log['l_ce']:.4f}  l_kin={log['l_kin']:.4f}"
                f"  val_acc={val_acc:.4f}  [{elapsed:.0f}s]"
            )

    # STEP 2: test set touched exactly once, after training/selection is over,
    # using the checkpoint selected purely by val accuracy.
    if best_state is not None:
        model.load_state_dict(best_state)
    # STEP 5: per-sample predictions/probs/features dumped alongside the
    # scalar test accuracy — required evidence, not merely diagnostic.
    dump_path = Path(results_dir) / f"seed_{seed}_test_predictions.npz"
    test_acc = eval_and_dump_test(model, test_loader, device, dump_path)

    elapsed = time.time() - t0
    return {
        "seed": seed,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_acc_at_best_val": test_acc,
        "final_val_acc": history[-1]["val_acc"],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "elapsed_s": elapsed,
        "history": history,
        "checkpoint_dir": str(ckpt_dir),
        "n_checkpoints_saved": len(list(ckpt_dir.glob("epoch*.pt"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml",
                    help="YAML config relative to this script's dir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--backbone", choices=["mamba3", "mamba2"], default=None,
                    help="Override backbone in config (default: use config value)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg = load_config(cfg_path)
    if args.backbone is not None:
        cfg["backbone"] = args.backbone
    seeds = args.seeds or cfg["seeds"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "mamba3")
    snr = cfg.get("noise_snr_db", None)
    noise_type = cfg.get("noise_type", "awgn")
    print(f"[leakfree] Device: {device}  backbone={backbone}  "
          f"lambda_kin={cfg.get('lambda_kin',0)}  snr={snr}dB  noise_type={noise_type}  "
          f"seeds={seeds}  epochs={2 if args.smoke else cfg['epochs']}")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Infrastructure piece #2: snapshot the exact source + config for this
    # run before anything else happens, so provenance never depends on the
    # live .py files staying unchanged after this run completes.
    snapshot_manifest = snapshot_source(results_dir, cfg_path)
    print(f"[leakfree] source snapshot: {len(snapshot_manifest)} files -> "
          f"{results_dir / 'source_snapshot'}")

    # STEP 2: build the (seed-independent) leak-free split ONCE, verify
    # disjointness, and persist evidence before any training happens.
    pools, split_index, disjoint = build_split(cfg)
    with open(results_dir / "split_index.json", "w") as f:
        json.dump(split_index, f, indent=2)
    with open(results_dir / "disjointness_check.json", "w") as f:
        json.dump(disjoint, f, indent=2)
    print(f"[leakfree] split sizes (raw windows, pre-shuffle): "
          f"train={len(pools['train'])} val={len(pools['val'])} test={len(pools['test'])}")
    print(f"[leakfree] disjointness check: {'PASS' if disjoint['pass'] else 'FAIL'} "
          f"({disjoint['n_files_checked']} files, guard={disjoint['guard_samples_required']})")
    if not disjoint["pass"]:
        print(f"[leakfree] VIOLATIONS: {disjoint['violations']}")
        sys.exit(1)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        result = train_one_seed(cfg, seed, device, pools, smoke=args.smoke)
        all_results.append(result)

        seed_path = results_dir / f"seed_{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → saved {seed_path}")

    if not args.smoke:
        # test_acc_at_best_val is the number that goes in the paper; best_val_acc
        # is retained for selection-audit purposes only (never reported as the
        # headline accuracy — that would reopen the same leakage this fixes).
        test_accs = [r["test_acc_at_best_val"] for r in all_results]
        val_accs  = [r["best_val_acc"] for r in all_results]
        mean_test = float(np.mean(test_accs))
        std_test  = float(np.std(test_accs, ddof=1))
        summary = {
            "config": cfg_path.name,
            "backbone": backbone,
            "lambda_kin": cfg.get("lambda_kin", 0.0),
            "noise_snr_db": snr,
            "noise_type": noise_type,
            "label_mode": cfg.get("label_mode", "4class"),
            "seeds": seeds,
            "test_accs_at_best_val": test_accs,
            "mean_test_acc": mean_test,
            "std_test_acc": std_test,
            "best_val_accs": val_accs,
            "protocol": "leakfree_v2 (train/val/test time-axis split, guard-gapped, per-file; "
                        "per-epoch checkpoints + source snapshot, see checkpoints/ and source_snapshot/)",
            "disjointness_pass": disjoint["pass"],
            "disjointness_check_file": str(results_dir / "disjointness_check.json"),
            "source_snapshot_dir": str(results_dir / "source_snapshot"),
            "source_snapshot_files": [m["original_path"] for m in snapshot_manifest],
        }
        summary_path = results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"RESULTS (test, leak-free): mean±std = {mean_test*100:.2f}±{std_test*100:.2f}%")
        print(f"Per-seed test: {[f'{a*100:.2f}' for a in test_accs]}")
        print(f"(for reference) per-seed best_val: {[f'{a*100:.2f}' for a in val_accs]}")
        print(f"Summary saved to {summary_path}")
    else:
        print("\n✅ Smoke test passed")


if __name__ == "__main__":
    main()
