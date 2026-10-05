"""Validation-selected late fusion. Weight w (on H logits; 1-w on V) chosen on the source-domain HOLD-OUT
(last 20% of every source bearing, excluded from training of the sub-models) by minimising class-balanced NLL of the
fused logits; grid 0..1 step 0.05, ties -> closest to 0.5. Test data never used for w."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stats import *
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def bal_nll(z, y): l = -lsm(z)[np.arange(len(y)), y]; return np.mean([l[y == c].mean() for c in np.unique(y)])
def pick_w(zh, zv, y):
    sc = np.array([bal_nll(w * zh + (1 - w) * zv, y) for w in GRID]); best = sc.min()
    cand = GRID[sc <= best + 1e-12]; return float(cand[np.argmin(np.abs(cand - 0.5))]), sc
def cls_metrics(pred, y):
    rec = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in (0, 1)])
    f1 = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1.append(2 * p * r / max(p + r, 1e-8))
    return rec, float(np.mean(f1))
def cross(prefix_fus, prefix_full, arm_full_HV, seeds, tb_names=('Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5')):
    rows = []
    for s in seeds:
        h = np.load(f"out/{prefix_fus}/cross_seed{s}_ch0.npz"); v = np.load(f"out/{prefix_fus}/cross_seed{s}_ch1.npz")
        assert (h["hold_y"] == v["hold_y"]).all() and (h["test_y"] == v["test_y"]).all()
        w, sc = pick_w(h["hold_logits"], v["hold_logits"], h["hold_y"]); y = h["test_y"]; b = h["test_b"]
        r = dict(seed=s, w=w)
        for nm, z in [("fusion_sub", w * h["test_logits"] + (1 - w) * v["test_logits"]), ("H_sub", h["test_logits"]), ("V_sub", v["test_logits"]),
                      ("avg05_sub", .5 * h["test_logits"] + .5 * v["test_logits"])]:
            rec, f1 = cls_metrics(z.argmax(1), y); r[nm] = dict(f1=f1, orr=rec[0], irr=rec[1], per_bearing=[float((z.argmax(1)[b == i] == y[b == i]).mean()) for i in range(4)])
        # same w applied to FULL-data H and V models (their own hold-out is in-sample, so w comes from sub-models)
        fh = np.load(f"out/{prefix_full}_H/test_logits_seed{s}.npz"); fv = np.load(f"out/{prefix_full}_V/test_logits_seed{s}.npz")
        z = w * fh["logits"] + (1 - w) * fv["logits"]; rec, f1 = cls_metrics(z.argmax(1), y)
        r["fusion_full"] = dict(f1=f1, orr=rec[0], irr=rec[1], per_bearing=[float((z.argmax(1)[b == i] == y[b == i]).mean()) for i in range(4)])
        rows.append(r)
    return rows
def lobo(prefix_fus, prefix_full, seeds, lab=('OR', 'IR', 'IR', 'OR'), bears=('Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5')):
    rows = []
    for f in range(4):
        for s in seeds:
            tag = f"fold{f}_{bears[f]}"; h = np.load(f"out/{prefix_fus}/{tag}_seed{s}_ch0.npz"); v = np.load(f"out/{prefix_fus}/{tag}_seed{s}_ch1.npz")
            w, _ = pick_w(h["hold_logits"], v["hold_logits"], h["hold_y"]); y = h["test_y"]; c = int(y[0]); r = dict(fold=f, seed=s, w=w)
            for nm, z in [("fusion_sub", w * h["test_logits"] + (1 - w) * v["test_logits"]), ("H_sub", h["test_logits"]), ("V_sub", v["test_logits"])]:
                r[nm] = float((z.argmax(1) == y).mean())
            fh = np.load(f"out/{prefix_full}_H/test_logits_fold{f}_seed{s}.npz")["logits"]; fv = np.load(f"out/{prefix_full}_V/test_logits_fold{f}_seed{s}.npz")["logits"]
            r["fusion_full"] = float(((w * fh + (1 - w) * fv).argmax(1) == y).mean()); rows.append(r)
    return rows
