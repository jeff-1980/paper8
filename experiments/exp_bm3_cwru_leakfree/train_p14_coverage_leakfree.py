"""
experiments/exp_bm3_cwru_leakfree/train_p14_coverage_leakfree.py

P14 (prereg_bm3_leakfree.md, 2026-09-20/28): XJTU-SY Cond3 training-bearing-
coverage sweep. Verbatim copy of experiments/exp_e3_lobo_leakfree/
train_lobo_leakfree.py (same train_one_run_leakfree(), same P1/P2/P3
discipline: fixed_epoch selection, test-fold-invisible training loop,
final-epoch checkpoint saved before the one-and-only test evaluation) —
the ONLY change is run_lobo_leakfree()'s fold generator, replaced by
run_coverage_sweep() below, which enumerates C(3,k) combinations of the
non-test bearing pool for k in {1,2,3} instead of the standard 4-fold LOBO
rotation (make_lobo_folds()).

Fixed test bearing: Bearing3_1 (OR, fold 0 in the original LOBO table) —
chosen per P14's write-up because the LOBO negative-transfer signal this
sweep is meant to explain concentrates almost entirely in the OR folds
(tab:xjtu_lobo_strat: fold0 -45.6pp, fold3 -23.2pp CE-only; IR folds
-0.05/-4.9pp, negligible), and Bearing3_1's near-ceiling, low-variance
single-sensor baseline (99.53+/-0.73%) isolates the coverage effect more
cleanly than Bearing3_5's already-noisy baseline (51.35+/-8.39%).
Training pool (the other 3 usable Cond3 bearings; Bearing3_2 is excluded
project-wide per D26, mixed-fault label): Bearing3_3(IR), Bearing3_4(IR),
Bearing3_5(OR).

Usage:
  source venv/bin/activate
  python experiments/exp_bm3_cwru_leakfree/train_p14_coverage_leakfree.py \
      --config experiments/exp_bm3_cwru_leakfree/configs_p14/config_p14_single.yaml --smoke
"""
import argparse
import itertools
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

from bearmamba3.data_xjtu import XJTUDataset, BEARING_FAILURE
from bearmamba3.kinematic_loss import kinematic_loss
from bearmamba3.model import BearMamba3

# P14 fixed test bearing + training pool (see module docstring for rationale).
TEST_BEARING = "Bearing3_1"
TRAIN_POOL = ["Bearing3_3", "Bearing3_4", "Bearing3_5"]


def make_coverage_combos():
    """All C(3,k) combos of TRAIN_POOL for k in {1,2,3}, paired with the
    fixed TEST_BEARING. Returns list of (k, combo_idx, train_bearings,
    test_bearings) tuples, deterministic itertools.combinations order."""
    combos = []
    for k in (1, 2, 3):
        for combo_idx, subset in enumerate(itertools.combinations(TRAIN_POOL, k)):
            combos.append((k, combo_idx, list(subset), [TEST_BEARING]))
    return combos


# ── Utilities (verbatim from train_lobo_leakfree.py) ───────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "results_dir"):
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    return BearMamba3(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        n_sensors=cfg.get("n_sensors", 1),
        n_classes=cfg.get("n_classes", 2),
        conv_stride=cfg["conv_stride"],
        is_mimo=False,
        use_batchnorm=cfg.get("use_batchnorm", False),
        dtype=torch.bfloat16,
    ).to(device)


def make_train_loader(dataset: XJTUDataset, batch_size: int, num_workers: int, smoke: bool):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if smoke else num_workers,
        pin_memory=True, drop_last=True,
    )


def eval_recall_f1(model, loader, device, n_classes: int = 2):
    """Return per-class recall, macro recall, macro F1."""
    model.eval()
    tp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(1).cpu().numpy()
            labs = labels.cpu().numpy()
            for c in range(n_classes):
                tp[c] += ((preds == c) & (labs == c)).sum()
                fn[c] += ((preds != c) & (labs == c)).sum()
                fp[c] += ((preds == c) & (labs != c)).sum()
    recall = tp / np.maximum(tp + fn, 1).astype(float)
    precision = tp / np.maximum(tp + fp, 1).astype(float)
    f1_per = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    macro_recall = recall.mean()
    macro_f1 = f1_per.mean()
    return recall, macro_recall, macro_f1


# ── Per-combo / per-seed training (P1 + P2 + P3, verbatim) ─────────────────

def train_one_run_leakfree(
    cfg: dict,
    seed: int,
    train_ds: XJTUDataset,
    test_ds: XJTUDataset,
    device: torch.device,
    smoke: bool,
    fold_tag: str,
) -> dict:
    set_seed(seed)

    selection_mode = cfg.get("selection_mode", "fixed_epoch")
    assert selection_mode == "fixed_epoch", (
        f"P1 violation: selection_mode={selection_mode!r} not supported — "
        f"this script only implements the zero-selection fixed_epoch control (prereg P1)."
    )

    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)
    lambda_kin = cfg.get("lambda_kin", 0.0)
    kin_variant = cfg.get("kin_variant", "cover")
    bkw = cfg.get("bearing_kwargs", {})
    fs_eff = float(cfg["fs_eff"])
    n_epochs = 2 if smoke else cfg["epochs"]
    results_dir = cfg["results_dir"]
    do_kin = lambda_kin > 0

    train_loader = make_train_loader(train_ds, bs, nw, smoke)
    # P2: test_loader is constructed here but MUST NOT be iterated until after the
    # training loop below has fully completed.
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        num_workers=0 if smoke else nw, pin_memory=True,
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    t0 = time.time()

    # ── P2: TEST-BEARING-INVISIBLE TRAINING LOOP ───────────────────────────
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
                l_kin = kinematic_loss(kin, rpm, fs_eff,
                                        variant=kin_variant, bearing_kwargs=bkw)
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

        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(
                f"  [P2 test-bearing-invisible] {fold_tag} seed={seed} ep={epoch:3d}/{n_epochs}"
                f"  l_ce={running_ce/max(n_steps,1):.4f}"
                f"  l_kin={running_kin/max(n_steps,1):.4f}"
                f"  [{elapsed:.0f}s]"
            )
    # ── end of test-bearing-invisible training loop ────────────────────────

    final_epoch = n_epochs
    recall, macro_recall, macro_f1 = eval_recall_f1(model, test_loader, device)
    print(
        f"  [P1 fixed_epoch] {fold_tag} seed={seed} final_epoch={final_epoch}"
        f"  recall=[{recall[0]:.3f},{recall[1]:.3f}]  macro_recall={macro_recall:.4f}"
        f"  macro_f1={macro_f1:.4f}  (first & only test-loader touch)"
    )

    checkpoint_path = None
    if cfg.get("save_checkpoint", False):
        ckpt_dir = Path(results_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"{fold_tag}_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": final_epoch,
            "fold_tag": fold_tag,
            "seed": seed,
            "config_name": cfg.get("name"),
        }, checkpoint_path)

    return {
        "fold_tag": fold_tag,
        "seed": seed,
        "selection_mode": "fixed_epoch",
        "final_epoch": final_epoch,
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_recall": recall.tolist(),
        "elapsed_s": time.time() - t0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


# ── P14 coverage-sweep mode ──────────────────────────────────────────────────

def run_coverage_sweep(cfg: dict, device: torch.device, smoke: bool,
                        max_combos: int = None, max_seeds: int = None):
    data_root = cfg["data_root"]
    seeds = cfg["seeds"] if max_seeds is None else cfg["seeds"][:max_seeds]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = make_coverage_combos()  # 7 combos (k=1:3, k=2:3, k=3:1)
    if max_combos is not None:
        combos = combos[:max_combos]

    test_label = BEARING_FAILURE[TEST_BEARING]  # 'OR'
    n_sensors = cfg.get("n_sensors", 1)

    all_rows = []  # one row per (combo, seed)

    for k, combo_idx, train_bearings, test_bearings in combos:
        combo_tag = f"k{k}_c{combo_idx}_{'-'.join(train_bearings)}"

        test_ds = XJTUDataset(data_root, test_bearings, n_sensors=n_sensors)
        if smoke:
            test_ds._windows = test_ds._windows[:64]
            test_ds._labels = test_ds._labels[:64]
            test_ds._rpms = test_ds._rpms[:64]

        for seed in seeds:
            out_path = results_dir / f"{combo_tag}_seed{seed}.json"
            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                with open(out_path) as f:
                    res = json.load(f)
                all_rows.append({"k": k, "combo_idx": combo_idx,
                                  "train_bearings": train_bearings, **res})
                continue

            print(f"\n{'='*60}")
            print(f"[P14] k={k} combo={combo_idx} train={train_bearings} "
                  f"test={TEST_BEARING}({test_label}) seed={seed}")

            train_ds = XJTUDataset(data_root, train_bearings, n_sensors=n_sensors)
            if smoke:
                train_ds._windows = train_ds._windows[:128]
                train_ds._labels = train_ds._labels[:128]
                train_ds._rpms = train_ds._rpms[:128]

            res = train_one_run_leakfree(cfg, seed, train_ds, test_ds, device, smoke, combo_tag)

            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  -> saved {out_path}")

            all_rows.append({"k": k, "combo_idx": combo_idx,
                              "train_bearings": train_bearings, **res})

    if smoke:
        print("\n[P14] Smoke test passed (no summary.json written; "
              "per-run JSONs written under the --results-dir override only).")
        return

    # ── aggregate: per-k mean/std of the TEST_BEARING's true recall
    # (per_class_recall[idx], idx=0 for OR since TEST_BEARING is OR) ────────
    idx = 0 if test_label == "OR" else 1
    by_k = {1: [], 2: [], 3: []}
    for row in all_rows:
        by_k[row["k"]].append(row["per_class_recall"][idx])

    summary = {
        "mode": "p14_coverage_sweep",
        "selection_mode": "fixed_epoch",
        "config": cfg.get("name"),
        "lambda_kin": cfg.get("lambda_kin", 0.0),
        "n_sensors": n_sensors,
        "test_bearing": TEST_BEARING,
        "test_label": test_label,
        "train_pool": TRAIN_POOL,
        "seeds": seeds,
        "per_k_recall": {
            str(k): {
                "values": vals,
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
            for k, vals in by_k.items()
        },
        "all_rows": all_rows,
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"[P14] COVERAGE SWEEP RESULTS ({TEST_BEARING}, {test_label} recall):")
    for k in (1, 2, 3):
        s = summary["per_k_recall"][str(k)]
        print(f"  k={k}: {s['mean']*100:.2f}+/-{s['std']*100:.2f}%  (n={len(s['values'])})")
    print(f"Summary saved to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--max-combos", type=int, default=None,
                     help="restrict number of coverage combos run (unit-check use only)")
    ap.add_argument("--results-dir", default=None,
                     help="override cfg['results_dir'] (isolated-scratch smoke testing)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg = load_config(cfg_path)
    if args.results_dir:
        cfg["results_dir"] = str(Path(args.results_dir).expanduser())
    if args.seeds:
        cfg["seeds"] = args.seeds

    assert "fs_eff" in cfg, "config.yaml must contain fs_eff (expected 12800 for XJTU)"
    assert abs(cfg["fs_eff"] - 12800.0) < 1, \
        f"fs_eff={cfg['fs_eff']} — expected 12800 (fs=25600 / conv_stride=2)"
    assert cfg.get("selection_mode", "fixed_epoch") == "fixed_epoch", \
        "P1 violation: this script only supports selection_mode=fixed_epoch"
    assert cfg.get("lambda_kin", 0.0) == 0.0, \
        "P14 spec: BM3 CE-only only (lambda_kin excluded as a confound, per prereg)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P14] Device: {device}  selection_mode=fixed_epoch"
          f"  n_sensors={cfg.get('n_sensors',1)}  test_bearing={TEST_BEARING}"
          f"  fs_eff={cfg['fs_eff']}  seeds={cfg['seeds']}"
          f"  epochs={2 if args.smoke else cfg['epochs']}"
          f"  save_checkpoint={cfg.get('save_checkpoint', False)}")

    run_coverage_sweep(cfg, device, args.smoke, max_combos=args.max_combos)


if __name__ == "__main__":
    main()
