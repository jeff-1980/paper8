"""Round 6 evaluation of saved XJTU-SY checkpoints (cross-condition or LOBO).
usage: XJTU_ONSET_RULE=<rule used for training> python scripts/extra/eval_r6_xjtu.py <cfg.yaml> [...]
For every seed (and LOBO fold) writes, to <results_dir>/r6eval/:
  * per-window logits with bearing and file index (own-rule test set);
  * per-bearing accuracy on the own-rule test set and on the COMMON test set (files at or after
    max(onset_H, onset_V, floor(0.85 n)) of each test bearing, the windows that all three degraded-phase
    rules label as faulty), and macro-F1 / bearing-equal balanced accuracy on both;
  * for gated backbones, the same metrics with the gate fixed to its mean over the training windows.
The first number of each seed is checked against the trainer's recorded result."""
import sys, os, json, csv, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from bearmamba3.data_xjtu import make_cross_condition_split, make_lobo_folds
RULE = os.environ.get('XJTU_ONSET_RULE', 'H')
dev = torch.device("cuda")
AUD = {r['bearing']: r for r in csv.DictReader(open(ROOT / 'audit' / 'onset_audit.csv'))}
def common_start(b):
    r = AUD[b]; return max(int(r['onset_H']), int(r['onset_V']), int(int(r['n_files']) * 0.85))

def metrics(pred, y, bear, mask=None):
    if mask is not None: pred, y, bear = pred[mask], y[mask], bear[mask]
    per_b = {b: {"label": BEARING_FAILURE[b], "n": int((bear == b).sum()), "acc": float((pred[bear == b] == y[bear == b]).mean())}
             for b in sorted(set(bear))}
    f1 = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f1.append(0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn)))
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else float('nan') for c in (0, 1)]
    beq = [np.mean([v["acc"] for v in per_b.values() if v["label"] == L]) if any(v["label"] == L for v in per_b.values()) else float('nan') for L in ("OR", "IR")]
    return dict(per_bearing=per_b, macro_f1=float(np.mean(f1)), per_class_recall=rec, bearing_equal_balanced_acc=float(np.nanmean(beq)))

def run_eval(m, test, gated, train=None):
    out = {}
    lg = get_logits(m, test, dev); out['logits'] = lg
    if gated:
        emb = m.conv_embed; gs = []
        loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=False, num_workers=0)
        m.eval()
        with torch.no_grad():
            for x, _, _ in loader:
                m(x.to(dev)); gs.append(emb.last_gates.float().cpu().numpy())
        gmean = np.concatenate(gs).mean(0); emb.fixed_gate = torch.tensor(gmean)
        out['fixed_logits'] = get_logits(m, test, dev); out['gate_mean'] = gmean.tolist(); emb.fixed_gate = None
    return out

for cp in sys.argv[1:]:
    import yaml
    cfg = yaml.safe_load(open(cp)); rd = Path(cfg["results_dir"]); od = rd / "r6eval"; od.mkdir(exist_ok=True)
    mode = cfg.get('mode', 'cross'); gated = 'gated' in cfg.get('backbone', 'mamba3')
    if mode == 'cross':
        tr = load_module("trx", "scripts/xjtu/train_xjtu_leakfree_v3.py")
        trb, teb = make_cross_condition_split(cfg.get("train_condition", "37.5Hz11kN"), cfg.get("test_condition", "40Hz10kN"))
        if cfg.get("train_bearings"): trb, teb = list(cfg["train_bearings"]), list(cfg["test_bearings"])
        jobs = [(None, trb, teb)]
    else:
        tr = load_module("trl", "scripts/lobo/train_lobo_leakfree_v3.py")
        jobs = [(i, a, b) for i, (a, b) in enumerate(make_lobo_folds(tr.COND3_LOBO_BEARINGS)) if cfg.get('only_folds') is None or i in cfg['only_folds']]
    summary = {}
    for fold, trb, teb in jobs:
        dss = [XJTUDataset(cfg["data_root"], [b], n_sensors=cfg["n_sensors"], channel=cfg.get("sensor_channel", 0)) for b in teb]
        test, _ = merge(dss)
        y = np.array(test._labels); bear = np.array(sum([d._bearing for d in dss], [])); fil = np.array(sum([d._file for d in dss], []))
        cmask = np.array([f >= common_start(b) for b, f in zip(bear, fil)])
        train = None
        if gated:
            train, _ = merge([XJTUDataset(cfg["data_root"], [b], n_sensors=cfg["n_sensors"]) for b in trb])
            if cfg.get('train_counts'):
                from bearmamba3.data_xjtu import resample_per_bearing
                train = resample_per_bearing(train, cfg['train_counts'], seed=cfg.get('resample_seed', 12345))
        for seed in cfg["seeds"]:
            if mode == 'cross':
                ck = rd / "checkpoints" / f"cross_seed{seed}_seed{seed}_final.pt"; tag = f"seed{seed}"
            else:
                ck = rd / "checkpoints" / f"fold{fold}_{teb[0]}_seed{seed}.pt"; tag = f"fold{fold}_seed{seed}"
            if not ck.exists(): print('missing', ck); continue
            m = tr.build_model(cfg, dev); d = torch.load(ck, map_location=dev); m.load_state_dict(d.get("model_state", d.get("model_state_dict")))
            r = run_eval(m, test, gated, train); pred = r['logits'].argmax(1)
            res = dict(rule=RULE, own=metrics(pred, y, bear), common=metrics(pred, y, bear, cmask), n_common=int(cmask.sum()), n_own=int(len(y)))
            if mode == 'cross':
                rec = json.load(open(rd / f"seed_{seed}.json")); res['recorded_macro_f1'] = rec['final_macro_f1']
            else:
                rec = json.load(open(rd / f"fold{fold}_seed{seed}.json")); res['recorded_recall'] = rec['per_class_recall']
            if gated:
                fp = r['fixed_logits'].argmax(1); res['gate_mean'] = r['gate_mean']
                res['fixed_gate_own'] = metrics(fp, y, bear); res['fixed_gate_common'] = metrics(fp, y, bear, cmask)
                res['frac_pred_changed_by_fixing'] = float((fp != pred).mean())
            np.savez_compressed(od / f"logits_{tag}.npz", logits=r['logits'], y=y, bearing=bear, file_idx=fil, common=cmask,
                                **({'fixed_logits': r['fixed_logits']} if gated else {}))
            summary[tag] = res
            print(cfg['name'], tag, RULE, 'own F1 %.4f (rec %s)' % (res['own']['macro_f1'], res.get('recorded_macro_f1', res.get('recorded_recall'))),
                  'common F1 %.4f' % res['common']['macro_f1'], ('fixed-gate F1 %.4f' % res['fixed_gate_own']['macro_f1']) if gated else '', flush=True)
    json.dump(summary, open(od / "summary.json", "w"), indent=1)
