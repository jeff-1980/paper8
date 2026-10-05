"""Write the split manifests in splits/.

CWRU  : per recording, the train / val / test segment boundaries, the guard gaps and the
        window index ranges, produced by the leak-free data module
        (bearmamba3.data_cwru_leakfree.CWRULeakfreeSplitBuilder) with the parameters of the
        CWRU configs (win_len 2048, stride 1024, guard 2048, 70/15/15). Needs the CWRU .mat
        files in $DATA_ROOT/cwru_12k_de (default ./data). The result is checked against the
        split_index.json written at training time (results/env1/cwru/exp02_snr-4_nokin/).
XJTU-SY: per design, the train and test bearing lists, and per bearing the degraded-phase
        onset file index and window count taken from audit/onset_audit.csv (H-channel rule,
        16 non-overlapping 2048-sample windows per 32768-sample file).

usage (from the repository root):  DATA_ROOT=/path/to/data python splits/make_split_manifests.py
"""
import csv, glob, json, os, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'splits'
DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'data'))


def cwru():
    from bearmamba3.data_cwru import MANIFEST
    from bearmamba3.data_cwru_leakfree import CWRULeakfreeSplitBuilder
    cfg = yaml.safe_load(open(ROOT / 'configs/env1/cwru/exp02_snr-4_nokin.yaml'))
    b = CWRULeakfreeSplitBuilder(DATA_ROOT / 'cwru_12k_de', win_len=cfg['win_len'], stride=cfg['stride'],
                                 channels=('DE',), label_mode='4class',
                                 guard_samples=cfg.get('guard_samples', 2048),
                                 train_frac=cfg.get('train_frac', 0.70), val_frac=cfg.get('val_frac', 0.15))
    pools = b.build()
    chk = b.verify_disjoint(pools)
    ref = json.load(open(ROOT / 'results/env1/cwru/exp02_snr-4_nokin/split_index.json'))
    assert ref == b.split_index, 'split differs from the split_index.json written at training time'
    by = {}
    for r in b.split_index:
        by.setdefault(r['file_id'], {})[r['split']] = r
    rows = []
    for fid in [r['file_id'] for r in b.split_index if r['split'] == 'train']:   # pool order
        ft, size, load, rpm, lab = MANIFEST[fid]
        d = by[fid]
        row = dict(file_id=fid, fault=ft, fault_size_in=size, load_hp=load, nominal_rpm=rpm, label=lab,
                   file_len=d['train']['file_len'])
        for s in ('train', 'val', 'test'):
            n = d[s]['n_windows']
            row[f'{s}_seg_start'] = d[s]['seg_start']
            row[f'{s}_seg_end'] = d[s]['seg_end']
            row[f'{s}_n_windows'] = n
            row[f'{s}_last_window_end'] = d[s]['seg_start'] + (n - 1) * cfg['stride'] + cfg['win_len']
        row['guard1'] = f"[{d['train']['seg_end']},{d['val']['seg_start']})"
        row['guard2'] = f"[{d['val']['seg_end']},{d['test']['seg_start']})"
        row['guard1_len'] = d['val']['seg_start'] - d['train']['seg_end']
        row['guard2_len'] = d['test']['seg_start'] - d['val']['seg_end']
        rows.append(row)
    cum = {'train': 0, 'val': 0, 'test': 0}       # index ranges of each recording's windows in the pooled split
    for row in rows:
        for s in cum:
            n = row[f'{s}_n_windows']
            row[f'{s}_pool_index'] = f'[{cum[s]},{cum[s] + n})'
            cum[s] += n
    with open(OUT / 'cwru_split_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"CWRU: {len(rows)} recordings; windows train/val/test = {cum['train']}/{cum['val']}/{cum['test']}; "
          f"disjoint={chk['pass']} (violations {len(chk['violations'])}); identical to training-time split_index.json")


def xjtu():
    from bearmamba3.data_xjtu import make_cross_condition_split, make_lobo_folds, valid_bearings
    onset = {r['bearing']: r for r in csv.DictReader(open(ROOT / 'audit/onset_audit.csv'))}
    def info(bs):
        return [dict(bearing=b, cls=onset[b]['cls'], onset_file_H=int(onset[b]['onset_H']),
                     n_files=int(onset[b]['n_files']), windows=int(onset[b]['windows_H'])) for b in bs]
    designs = []
    tr, te = make_cross_condition_split()
    designs.append(dict(design='cross_condition_2to3', env='1 and 2', train=tr, test=te))
    for i, (tr, te) in enumerate(make_lobo_folds(valid_bearings('40Hz10kN'))):
        designs.append(dict(design=f'lobo_cond3_fold{i}', env='1 and 2', train=tr, test=te))
    seen = set()
    for p in sorted(glob.glob(str(ROOT / 'configs/env2/xjtu_loco/*.yaml'))) + sorted(glob.glob(str(ROOT / 'configs/env2/xjtu_or_coverage/*.yaml'))):
        c = yaml.safe_load(open(p))
        if not c.get('train_bearings'):
            continue
        key = (tuple(c['train_bearings']), tuple(c['test_bearings']))
        if key in seen:
            continue
        seen.add(key)
        designs.append(dict(design=Path(p).stem.rsplit('_', 1)[0], env='2',
                            train=list(c['train_bearings']), test=list(c['test_bearings'])))
    for f in sorted(glob.glob(str(ROOT / 'results/env1/xjtu_ir_coverage_p14/dual/k*_seed0.json'))):
        tag = json.load(open(f))['fold_tag']
        designs.append(dict(design=f'ir_coverage_p14_{tag}', env='1', train=tag.split('_', 2)[2].split('-'),
                            test=['Bearing3_1']))
    out = []
    for d in designs:
        d['train_info'], d['test_info'] = info(d['train']), info(d['test'])
        d['n_train_windows'] = sum(x['windows'] for x in d['train_info'])
        d['n_test_windows'] = sum(x['windows'] for x in d['test_info'])
        out.append(d)
    json.dump(out, open(OUT / 'xjtu_designs.json', 'w'), indent=1)
    with open(OUT / 'xjtu_designs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['design', 'env', 'role', 'bearing', 'class', 'onset_file_H', 'n_files', 'windows'])
        for d in out:
            for role in ('train', 'test'):
                for x in d[f'{role}_info']:
                    w.writerow([d['design'], d['env'], role, x['bearing'], x['cls'], x['onset_file_H'], x['n_files'], x['windows']])
    print(f'XJTU-SY: {len(out)} designs written')


if __name__ == '__main__':
    xjtu()
    if (DATA_ROOT / 'cwru_12k_de').exists():
        cwru()
    else:
        print(f'CWRU skipped: {DATA_ROOT / "cwru_12k_de"} not found')
