"""Evaluate saved final checkpoints of a cross-condition arm per test bearing; save logits.
usage: eval_ckpts_cross.py <cfg.yaml> [<cfg2.yaml> ...]"""
import sys, json, yaml, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
tr = load_module("trx", "scripts/xjtu/train_xjtu_leakfree_v3.py")
from bearmamba3.data_xjtu import make_cross_condition_split
dev = torch.device("cuda")
for cp in sys.argv[1:]:
    cfg = tr.load_config(Path(cp)); rd = Path(cfg["results_dir"])
    _, tb = make_cross_condition_split(cfg.get("train_condition", "37.5Hz11kN"), cfg.get("test_condition", "40Hz10kN"))
    if cfg.get("test_bearings"): tb = list(cfg["test_bearings"])
    dss = [XJTUDataset(cfg["data_root"], [b], n_sensors=cfg["n_sensors"], channel=cfg.get("sensor_channel", 0)) for b in tb]
    test, bidx = merge(dss); y = np.array(test._labels)
    res = {}
    for seed in cfg["seeds"]:
        ck = rd / "checkpoints" / f"cross_seed{seed}_seed{seed}_final.pt"
        if not ck.exists(): continue
        m = tr.build_model(cfg, dev); m.load_state_dict(torch.load(ck, map_location=dev)["model_state"])
        lg = get_logits(m, test, dev); pred = lg.argmax(1)
        # sanity: identical to recorded json
        rec = json.load(open(rd / f"seed_{seed}.json"))
        rc = [float((pred[y == c] == c).mean()) for c in (0, 1)]
        per_b = {b: {"label": BEARING_FAILURE[b], "n": int((bidx == i).sum()),
                     "acc": float((pred[bidx == i] == y[bidx == i]).mean())} for i, b in enumerate(tb)}
        res[seed] = {"per_bearing": per_b, "recall_recomputed": rc, "recall_recorded": rec["final_per_class_recall"]}
        np.savez_compressed(rd / f"test_logits_seed{seed}.npz", logits=lg, y=y, bidx=bidx)
        print(cfg["name"], seed, {b: round(v["acc"], 3) for b, v in per_b.items()}, rc, rec["final_per_class_recall"], flush=True)
    json.dump({"test_bearings": tb, "seeds": res}, open(rd / "per_bearing.json", "w"), indent=1)
