#!/usr/bin/env python3
"""
experiments/exp_bm3_cwru_leakfree/run_batch5_lobo_n8.py

P11 LOBO n=8 extension driver. Idempotent at the per-fold-per-seed level
(train_lobo_leakfree.py itself skips any fold{f}_seed{s}.json that already
exists), so this wrapper just invokes the 4 configs in sequence and lets
the inner script's own skip logic handle resume.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
HERE = Path(__file__).parent
TRAIN = ROOT / "experiments/exp_e3_lobo_leakfree/train_lobo_leakfree.py"


def main():
    manifest = json.load(open(HERE / "run_manifest_batch5_lobo_n8.json"))
    print(f"[run_batch5_lobo_n8] {len(manifest)} configs to process "
          f"({sum(len(r['seeds']) * 4 for r in manifest)} fold-runs total)")
    t_start = time.time()
    for i, row in enumerate(manifest, 1):
        cfg_path = ROOT / row["n8_config"]
        summary = Path(row["n8_results_dir"]).expanduser() / "summary.json"
        if summary.exists():
            print(f"[{i}/{len(manifest)}] [SKIP] {row['family']} — summary.json exists")
            continue
        print(f"\n[{i}/{len(manifest)}] [RUN ] ({row['family']}) {cfg_path.name}")
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

    print(f"\n[run_batch5_lobo_n8] ALL DONE in {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
