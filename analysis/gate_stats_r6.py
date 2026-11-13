"""Round-3 adaptivity criterion, unchanged (prereg_mst_round6.md section 6): SD over test windows of the
per-window gate weight on channel 1 (vertical); 'adaptive' requires SD > 0.01 in every seed. For the token
gate the window value is the mean over tokens, and the mean token-level SD within windows is also reported.
Applied to the round-6 gated checkpoints. usage: python analysis/gate_stats_r6.py <cfg names>"""
import sys, json, numpy as np, torch, importlib.util
from pathlib import Path
NAMES = sys.argv[1:]
spec = importlib.util.spec_from_file_location('tr', 'scripts/xjtu/train_xjtu_leakfree_v3.py'); tr = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(tr)
from bearmamba3.data_xjtu import XJTUDataset, make_cross_condition_split, make_lobo_folds
import yaml
dev = torch.device('cuda')
TOK = []
def gates(m, ds):
    g = []; X = torch.stack([torch.as_tensor(w) for w in ds._windows]).float()
    X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-8); m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(X), 256):
            m(X[i:i + 256].to(dev)); g.append(m.conv_embed.last_gates[:, 1].float().cpu())
            if getattr(m.conv_embed, 'last_token_gates', None) is not None: TOK.append(m.conv_embed.last_token_gates[..., 1].float().std(-1).mean().item())
    return torch.cat(g).numpy()
for name in NAMES:
    cfg = yaml.safe_load(open(f'cfg_r6/{name}.yaml')); rd = Path(cfg['results_dir']); res = {}
    if cfg['mode'] == 'cross':
        _, tb = make_cross_condition_split(cfg['train_condition'], cfg['test_condition'])
        ds = XJTUDataset(cfg['data_root'], tb, n_sensors=2)
        for s in cfg['seeds']:
            m = tr.build_model(cfg, dev); m.load_state_dict(torch.load(rd / 'checkpoints' / f'cross_seed{s}_seed{s}_final.pt', map_location=dev)['model_state'])
            TOK.clear(); g = gates(m, ds)
            res[f'seed{s}'] = dict(mean=float(g.mean()), sd=float(g.std()), token_sd_mean=float(np.mean(TOK)) if TOK else None)
    else:
        folds = make_lobo_folds(tr.COND3_LOBO_BEARINGS)
        for fo in cfg['only_folds']:
            teb = folds[fo][1]; ds = XJTUDataset(cfg['data_root'], teb, n_sensors=2)
            for s in cfg['seeds']:
                d = torch.load(rd / 'checkpoints' / f'fold{fo}_{teb[0]}_seed{s}.pt', map_location=dev)
                m = tr.build_model(cfg, dev); m.load_state_dict(d.get('model_state', d.get('model_state_dict')))
                TOK.clear(); g = gates(m, ds)
                res[f'fold{fo}_seed{s}'] = dict(mean=float(g.mean()), sd=float(g.std()), token_sd_mean=float(np.mean(TOK)) if TOK else None)
    res['adaptive_rule_met'] = all(v['sd'] > 0.01 for v in res.values() if isinstance(v, dict))
    json.dump(res, open(rd / 'r6eval' / 'gate_stats.json', 'w'), indent=1)
    print(name, 'max SD %.4f' % max(v['sd'] for v in res.values() if isinstance(v, dict)), 'rule met', res['adaptive_rule_met'],
          'token SD', None if not any(v.get('token_sd_mean') for v in res.values() if isinstance(v, dict)) else round(np.mean([v['token_sd_mean'] for v in res.values() if isinstance(v, dict)]), 3))
