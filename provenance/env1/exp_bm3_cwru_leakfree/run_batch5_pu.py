#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/run_batch5_pu.py

P12 PU leak-free driver: runs the 7 configs in run_manifest_batch5_pu.json
(35 runs total) through train_pu_leakfree.py. Idempotent: skips any entry
whose results_dir/summary.json already exists (train_pu_leakfree.py itself
also skips per-seed if seed_{s}.json exists, for within-config resume).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
HERE = Path(__file__).parent
TRAIN = HERE / "train_pu_leakfree.py"


def main():
    manifest = json.load(open(HERE / "run_manifest_batch5_pu.json"))
    print(f"[run_batch5_pu] {len(manifest)} configs to process (35 runs total)")
    t_start = time.time()
    for i, row in enumerate(manifest, 1):
        results_dir = Path(row["results_dir"]).expanduser()
        summary = results_dir / "summary.json"
        if summary.exists():
            print(f"[{i}/{len(manifest)}] [SKIP] {results_dir.name} — summary.json exists")
            continue
        cfg_path = ROOT / row["leakfree_config"]
        print(f"\n[{i}/{len(manifest)}] [RUN ] ({row['family']}) {cfg_path.name}  -> {results_dir}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(TRAIN), "--config", str(cfg_path)],
            cwd=str(ROOT),
        )
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"[{i}/{len(manifest)}] [ERR ] exit={result.returncode} after {dt:.0f}s — STOPPING")
            sys.exit(result.returncode)
        print(f"[{i}/{len(manifest)}] [DONE] {dt:.0f}s  (elapsed total {time.time()-t_start:.0f}s)")

    print(f"\n[run_batch5_pu] ALL DONE in {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
