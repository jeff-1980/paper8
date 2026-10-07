"""Keep only the validation-selected checkpoint per seed (epoch = best_val_epoch in seed_<s>.json)."""
import sys, json, glob, os
d = sys.argv[1]
for f in sorted(glob.glob(f'{d}/seed_*.json')):
    j = json.load(open(f)); s = j['seed']; e = j['best_val_epoch']
    keep = f'{d}/checkpoints/seed{s}/epoch{e:03d}.pt'
    assert os.path.exists(keep), keep
    for c in glob.glob(f'{d}/checkpoints/seed{s}/epoch*.pt'):
        if c != keep: os.remove(c)
print('kept best-val checkpoints in', d)
