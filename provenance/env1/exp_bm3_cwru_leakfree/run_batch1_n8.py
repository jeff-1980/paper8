#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/run_batch1_n8.py

BATCH 1 driver: runs the 9 configs in run_manifest_batch1_n8.json (seeds
[5,6,7] each = 27 runs total) through train_leakfree.py. Idempotent: skips
any entry whose results_dir/summary.json already exists.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
HERE = Path(__file__).parent
TRAIN = ROOT / "experiments/exp01_cwru_baseline/train_leakfree.py"


def main():
    manifest = json.load(open(HERE / "run_manifest_batch1_n8.json"))
    print(f"[run_batch1_n8] {len(manifest)} configs to process (27 runs total)")
    t_start = time.time()
    for i, row in enumerate(manifest, 1):
        results_dir = Path(row["results_dir"]).expanduser()
        summary = results_dir / "summary.json"
        if summary.exists():
            print(f"[{i}/{len(manifest)}] [SKIP] {results_dir.name} — summary.json exists")
            continue
        cfg_path = ROOT / row["leakfree_config"]
        print(f"\n[{i}/{len(manifest)}] [RUN ] {cfg_path.name}  -> {results_dir}")
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

    print(f"\n[run_batch1_n8] ALL DONE in {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
