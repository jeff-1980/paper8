#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/train_sksvm_leakfree.py

Leak-free counterpart of experiments/exp07_baselines/train_sksvm.py.

Replaces train_sksvm.py:76-83 (CWRUDataset + random permutation train/val
split — the same leaky window-overlap + random-split pattern this whole
task fixes) with bearmamba3.data_cwru_leakfree.CWRULeakfreeSplitBuilder:
per-file time-axis train(70%)/val(15%)/test(15%) split with a guard gap,
THEN windowing independently within each segment. Feature extraction
(extract_features, unchanged, copied verbatim from train_sksvm.py) runs on
windows drawn from each split's own pool — no window can straddle a split
boundary or share raw samples across splits.

Fit uses the train pool only (parity with the neural-network leak-free
scripts, which also only gradient-fit on "train"). Reported accuracy is on
the TEST pool (never touched before this point) — the val pool is built
and disjointness-checked for structural parity but unused for fitting,
since the SVM here has fixed hyperparameters (no epoch/selection loop to
protect with a validation set).

Note on "seeds": under the ORIGINAL leaky protocol, each seed reshuffled
which windows landed in train vs val, so 5 seeds gave 5 genuinely different
splits. Under leak-free splitting, the train/val/test partition is
file-segment-deterministic and does NOT depend on seed — only
SVC(random_state=seed) varies, which has near-zero effect on a
deterministic RBF-SVC fit with no probability estimation. Per-seed
variance may therefore collapse to ~0 here; this is reported honestly
rather than concealed, not a bug to paper over.

Provenance artifacts (replaces the checkpoint/epoch concept, which does
not apply to a single SVM fit):
  - model.pkl               : fitted sklearn Pipeline (scaler + SVC)
  - train_features_sha256   : sha256 of the (X_tr, y_tr) arrays, so the
                               exact training feature matrix behind a
                               reported number can be verified later
  - source_snapshot/         : this script + data_cwru_leakfree.py + data_cwru.py, md5+mtime
  - split_index.json / disjointness_check.json : same format as train_leakfree.py

Usage:
  source venv/bin/activate
  python experiments/exp_bm3_cwru_leakfree/train_sksvm_leakfree.py --snr_db -4 --seed 0 [--smoke]
  python experiments/exp_bm3_cwru_leakfree/train_sksvm_leakfree.py --run_all
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_cwru_leakfree import CWRULeakfreeSplitBuilder, CWRULeakfreeDataset

SNR_GRID = [None, 0.0, -2.0, -4.0, -6.0, -8.0]
RESULTS_ROOT = PROJECT_ROOT / "results" / "bm3_cwru_leakfree_20260911-leakfree"

SNAPSHOT_FILES = [
    Path(__file__),
    PROJECT_ROOT / "bearmamba3" / "data_cwru_leakfree.py",
    PROJECT_ROOT / "bearmamba3" / "data_cwru.py",
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def snapshot_source(out_dir: Path) -> list[dict]:
    snap_dir = out_dir / "source_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in SNAPSHOT_FILES:
        dst = snap_dir / src.name
        shutil.copy2(src, dst)
        manifest.append({"original_path": str(src), "md5": _md5(src), "mtime_utc": src.stat().st_mtime})
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ── feature extraction — copied verbatim from train_sksvm.py ────────────────

def extract_features(x: np.ndarray) -> np.ndarray:
    """x: (win_len,) — single-channel signal → feature vector (12,)"""
    rms        = np.sqrt(np.mean(x**2))
    kurt       = float(stats.kurtosis(x, fisher=True))
    skew       = float(stats.skew(x))
    crest      = np.max(np.abs(x)) / (rms + 1e-8)
    pp         = np.ptp(x)

    N     = len(x)
    mag   = np.abs(np.fft.rfft(x)) / N
    freqs = np.arange(len(mag))

    top5_energy = np.sort(mag**2)[-5:].sum() / (np.sum(mag**2) + 1e-8)
    spec_cent = np.sum(freqs * mag) / (np.sum(mag) + 1e-8)

    mu   = spec_cent
    sig2 = np.sum(mag * (freqs - mu)**2) / (np.sum(mag) + 1e-8)
    sig4 = np.sum(mag * (freqs - mu)**4) / (np.sum(mag) + 1e-8)
    spec_kurt = sig4 / (sig2**2 + 1e-8) - 3.0

    env = np.abs(hilbert(x))
    env_kurt = float(stats.kurtosis(env, fisher=True))

    q1 = mag[:N//8].sum()
    q4 = mag[-N//8:].sum()
    band_ratio = q1 / (q4 + 1e-8)

    return np.array([rms, kurt, skew, crest, pp,
                     top5_energy, spec_cent, spec_kurt,
                     env_kurt, band_ratio,
                     np.log1p(rms), np.log1p(np.abs(spec_cent))], dtype=np.float32)


def build_split(snr_db, win_len=2048, stride=1024, guard=2048):
    builder = CWRULeakfreeSplitBuilder(
        data_dir=str(Path("<DATA_ROOT>/cwru_12k_de").expanduser()),
        win_len=win_len, stride=stride, channels=["DE"], label_mode="4class",
        guard_samples=guard,
    )
    pools = builder.build()
    disjoint = builder.verify_disjoint(pools)
    if not disjoint["pass"]:
        raise RuntimeError(f"disjointness check failed: {disjoint}")
    return pools, builder.split_index, disjoint


def extract_split_features(samples, snr_db, seed, smoke=False):
    """samples: list of (window, label, rpm). Applies the same noise-injection
    convention as CWRULeakfreeDataset.__getitem__ (AWGN at snr_db) before
    feature extraction, so features reflect the noisy signal (parity with
    the original train_sksvm.py, which extracted features AFTER
    CWRUDataset's noise injection)."""
    ds = CWRULeakfreeDataset(samples, normalize=True, seed=seed, noise_snr_db=snr_db)
    n = len(ds) if not smoke else min(200, len(ds))
    X, y = [], []
    for i in range(n):
        x_t, lbl, _ = ds[i]
        X.append(extract_features(x_t[0].numpy()))
        y.append(int(lbl))
    return np.stack(X), np.array(y)


def run_one(snr_db, seed, pools, split_index, disjoint, smoke=False):
    X_tr, y_tr = extract_split_features(pools["train"], snr_db, seed, smoke)
    X_te, y_te = extract_split_features(pools["test"], snr_db, seed, smoke)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       decision_function_shape="ovr", random_state=seed)),
    ])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    train_feat_sha256 = hashlib.sha256(X_tr.tobytes()).hexdigest()
    return acc, model, train_feat_sha256


def snr_tag(snr_db):
    return "clean" if snr_db is None else f"snr{int(snr_db):+d}".replace("+", "")


def run_snr_point(snr_db, seeds, smoke=False):
    tag = snr_tag(snr_db)
    out_dir = RESULTS_ROOT / f"exp07_sksvm_{tag}_leakfree"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = snapshot_source(out_dir)
    print(f"[leakfree-sksvm] source snapshot: {len(manifest)} files -> {out_dir/'source_snapshot'}")

    pools, split_index, disjoint = build_split(snr_db)
    with open(out_dir / "split_index.json", "w") as f:
        json.dump(split_index, f, indent=2)
    with open(out_dir / "disjointness_check.json", "w") as f:
        json.dump(disjoint, f, indent=2)
    print(f"[leakfree-sksvm] SNR={tag}  split: train={len(pools['train'])} "
          f"val={len(pools['val'])} test={len(pools['test'])}  "
          f"disjoint={'PASS' if disjoint['pass'] else 'FAIL'}")
    if not disjoint["pass"]:
        sys.exit(1)

    accs = []
    for s in seeds:
        print(f"SK-SVM leak-free | {tag} | seed={s}", flush=True)
        acc, model, sha = run_one(snr_db, s, pools, split_index, disjoint, smoke=smoke)
        accs.append(acc)
        print(f"  test_acc={acc*100:.2f}%  train_feat_sha256={sha[:16]}...")

        model_dir = out_dir / "models"
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / f"seed{s}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        seed_result = {
            "acc": acc, "snr_db": snr_db, "seed": s,
            "train_feat_sha256": sha,
            "model_pkl": str(model_dir / f"seed{s}_model.pkl"),
            "protocol": "leakfree_v1 (train/test time-axis split, guard-gapped, per-file; "
                        "fit on train pool only, reported acc on held-out test pool)",
        }
        with open(out_dir / f"seed{s}.json", "w") as f:
            json.dump(seed_result, f, indent=2)

    if not smoke:
        mean = float(np.mean(accs)); std = float(np.std(accs, ddof=1))
        print(f"  {tag}: mean={mean*100:.2f}±{std*100:.2f}%")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({
                "mean_acc": mean, "std_acc": std, "per_seed": accs,
                "snr_db": snr_db, "backbone": "sksvm", "n_seeds": len(seeds),
                "protocol": "leakfree_v1",
                "disjointness_pass": disjoint["pass"],
            }, f, indent=2)
        print(f"  Saved -> {out_dir}/metrics.json")
    else:
        print("\n  Smoke test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db",  type=float, default=None)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--seeds",   type=int,   nargs="+", default=None)
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument("--smoke",   action="store_true")
    args = ap.parse_args()

    if args.run_all:
        for snr in SNR_GRID:
            run_snr_point(snr, [0, 1, 2, 3, 4], smoke=args.smoke)
    else:
        seeds = args.seeds or [args.seed]
        run_snr_point(args.snr_db, seeds, smoke=args.smoke)


if __name__ == "__main__":
    main()
