"""Round 6: evaluate the validation-selected CWRU checkpoints of a configuration under a different test noise.
usage: python scripts/extra/eval_r6_cwru.py <trained cfg.yaml> <test noise_type> [<test noise_realisation>]
Uses the same split, the same window identities and the v2 noise key; only the test noise_type (rho) and/or the
realisation number differ from training. Writes <results_dir>/r6eval/test_<noise>_r<k>.json."""
import sys, json, yaml, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments' / 'exp01_cwru_baseline'))
import train_leakfree as T
cfg = yaml.safe_load(open(sys.argv[1])); nt = sys.argv[2]; real = int(sys.argv[3]) if len(sys.argv) > 3 else 0
assert cfg.get('noise_key') == 'v2'
rd = Path(cfg['results_dir']); dev = torch.device('cuda')
pools, _, _ = T.build_split(cfg)
out = {}
for f in sorted(rd.glob('seed_*.json')):
    j = json.load(open(f)); s = j['seed']; e = j['best_val_epoch']
    ck = rd / 'checkpoints' / f'seed{s}' / f'epoch{e:03d}.pt'
    m = T.build_model(dict(cfg, channels=["SUMDIFF"]) if cfg.get("combine") else cfg, dev)
    m.load_state_dict(torch.load(ck, map_location=dev)['model_state']); m.eval()
    def acc(noise, r):
        ds = T.CWRULeakfreeDataset(pools['test'], normalize=True, seed=s, noise_snr_db=cfg['noise_snr_db'], noise_type=noise,
                                   split='test', noise_key='v2', channels=cfg['channels'], noise_realisation=r, combine=cfg.get('combine'))
        return T.eval_epoch(m, torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), dev)
    same = acc(cfg.get('noise_type', 'awgn'), cfg.get('noise_realisation', 0)); other = acc(nt, real)
    out[s] = dict(recorded=j['test_acc_at_best_val'], recomputed_same=same, test_other=other, epoch=e)
    print(cfg['name'], s, 'rec %.4f same %.4f -> %s r%d: %.4f' % (j['test_acc_at_best_val'], same, nt, real, other), flush=True)
import os
od = Path(os.environ.get('R6_EVAL_DIR', str(rd / 'r6eval'))) / Path(sys.argv[1]).stem; od.mkdir(parents=True, exist_ok=True)
json.dump(dict(trained_results_dir=str(rd), test_noise=nt, test_realisation=real, seeds=out), open(od / f'test_{nt}_r{real}.json', 'w'), indent=1)
