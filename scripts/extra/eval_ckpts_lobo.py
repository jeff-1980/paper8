"""Logits of the saved full-data LOBO checkpoints (single-channel arms) on each test fold. usage: eval_ckpts_lobo.py cfg..."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
tr = load_module("trl", "scripts/lobo/train_lobo_leakfree_v2.py")
from bearmamba3.data_xjtu import make_lobo_folds
dev = torch.device("cuda")
for cp in sys.argv[1:]:
    cfg = tr.load_config(Path(cp)); rd = Path(cfg["results_dir"])
    for i, (trb, te) in enumerate(make_lobo_folds(tr.COND3_LOBO_BEARINGS)):
        ds = XJTUDataset(cfg["data_root"], te, n_sensors=cfg["n_sensors"], channel=cfg.get("sensor_channel", 0)); y = np.array(ds._labels)
        for seed in cfg["seeds"]:
            ck = rd / "checkpoints" / f"fold{i}_{te[0]}_seed{seed}.pt"
            if not ck.exists(): continue
            m = tr.build_model(cfg, dev); _d = torch.load(ck, map_location=dev); m.load_state_dict(_d.get("model_state", _d.get("model_state_dict")))
            lg = get_logits(m, ds, dev); rec = float((lg.argmax(1) == y).mean())
            np.savez_compressed(rd / f"test_logits_fold{i}_seed{seed}.npz", logits=lg, y=y)
            rr = json.load(open(rd / f"fold{i}_seed{seed}.json"))["per_class_recall"][y[0]]
            print(cfg["name"], i, seed, round(rec, 4), round(rr, 4), flush=True)
