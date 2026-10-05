"""Train H-only and V-only models on the source domain MINUS a fresh time-contiguous hold-out
(last HOLD fraction of the windows of every source bearing), same recipe/hyper-parameters as the
main arms, then dump hold-out and test logits for validation-selected late fusion.
usage: fusion_holdout.py <cfg.yaml> <mode: cross|lobo> [--seeds ...] [--hold 0.2]
cfg is a normal single-channel config (backbone, seeds, results_dir, ...); sensor_channel is overridden."""
import sys, json, argparse, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("mode")
ap.add_argument("--seeds", nargs="+", type=int); ap.add_argument("--hold", type=float, default=0.2)
a = ap.parse_args()
dev = torch.device("cuda")
trx = load_module("trx", "scripts/xjtu/train_xjtu_leakfree_v2.py")
trl = load_module("trl", "scripts/lobo/train_lobo_leakfree_v2.py")
from bearmamba3.data_xjtu import make_cross_condition_split, make_lobo_folds
cfg = trx.load_config(Path(a.cfg)); seeds = a.seeds or cfg["seeds"]
rd = Path(cfg["results_dir"]); rd.mkdir(parents=True, exist_ok=True)
cfg["n_sensors"] = 1
if a.mode == "cross":
    src, tst = make_cross_condition_split(cfg.get("train_condition", "37.5Hz11kN"), cfg.get("test_condition", "40Hz10kN"))
    tasks = [("cross", src, tst)]
else:
    folds = make_lobo_folds(trl.COND3_LOBO_BEARINGS)
    tasks = [(f"fold{i}_{te[0]}", tr_, te) for i, (tr_, te) in enumerate(folds)]
cache = {}
def ds_for(b, ch):
    if (b, ch) not in cache:
        cache[(b, ch)] = XJTUDataset(cfg["data_root"], [b], n_sensors=1, channel=ch)
    return cache[(b, ch)]
for seed in seeds:
    for tag, src, tst in tasks:
        for ch in (0, 1):
            out = rd / f"{tag}_seed{seed}_ch{ch}.npz"
            if out.exists(): continue
            parts = [ds_for(b, ch) for b in src]
            trn = [sub(d, 0, int(len(d._labels) * (1 - a.hold))) for d in parts]
            hol = [sub(d, int(len(d._labels) * (1 - a.hold)), len(d._labels)) for d in parts]
            trn_ds, _ = merge(trn); hol_ds, hol_b = merge(hol)
            test_ds, tb = merge([ds_for(b, ch) for b in tst])
            c2 = dict(cfg, sensor_channel=ch, results_dir=str(rd / f"ckpt_ch{ch}"))
            ftag = f"{tag}_seed{seed}"
            t0 = time.time()
            if a.mode == "cross":
                r = trx.train_one_run(c2, seed, trn_ds, hol_ds, dev, False, f"cross_seed{seed}", use_class_weights=True)
                ck = Path(c2["results_dir"]) / "checkpoints" / f"cross_seed{seed}_seed{seed}_final.pt"
            else:
                c2["save_checkpoint"] = True
                r = trl.train_one_run_leakfree(c2, seed, trn_ds, hol_ds, dev, False, tag)
                ck = Path(c2["results_dir"]) / "checkpoints" / f"{tag}_seed{seed}.pt"
            m = trx.build_model(c2, dev); _d = torch.load(ck, map_location=dev); m.load_state_dict(_d.get("model_state", _d.get("model_state_dict")))
            np.savez_compressed(out, hold_logits=get_logits(m, hol_ds, dev), hold_y=np.array(hol_ds._labels), hold_b=hol_b,
                                test_logits=get_logits(m, test_ds, dev), test_y=np.array(test_ds._labels), test_b=tb,
                                train_n=len(trn_ds), wall_s=time.time() - t0)
            ck.unlink()  # remove checkpoint of the sub-model (logits are what we need)
            print("done", ftag, "ch", ch, f"{time.time()-t0:.0f}s", flush=True)
