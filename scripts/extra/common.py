"""Shared helpers: per-bearing datasets, logits extraction, loading trainer modules."""
import copy, importlib.util, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bearmamba3.data_xjtu import XJTUDataset, BEARING_FAILURE

def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

def merge(datasets):
    """Concatenate XJTUDataset objects (window lists) into one; returns (ds, bearing_index array)."""
    ds = copy.copy(datasets[0])
    ds._windows = sum([list(d._windows) for d in datasets], [])
    ds._labels = sum([list(d._labels) for d in datasets], [])
    ds._rpms = sum([list(d._rpms) for d in datasets], [])
    bidx = np.concatenate([np.full(len(d._labels), i) for i, d in enumerate(datasets)])
    return ds, bidx

def sub(ds, lo, hi):
    d = copy.copy(ds)
    d._windows = ds._windows[lo:hi]; d._labels = ds._labels[lo:hi]; d._rpms = ds._rpms[lo:hi]
    return d

@torch.no_grad()
def get_logits(model, ds, device, bs=256):
    model.eval(); out = []
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    for x, y, _ in loader:
        o = model(x.to(device)); o = o[0] if isinstance(o, tuple) else o
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)
