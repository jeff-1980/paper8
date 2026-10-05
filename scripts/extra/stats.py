import numpy as np, json
from scipy import stats
def paired(a, b, n_boot=20000, seed=12345):
    """a-b, seed-paired. Returns mean diff, bootstrap 95% CI over paired diffs (percentile), exact two-sided Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
    if np.all(d == 0): p = 1.0
    else: p = float(stats.wilcoxon(d, method="exact", alternative="two-sided").pvalue)
    return dict(mean_diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), wilcoxon_p=p, n=len(d), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))
def ms(x): x = np.asarray(x, float); return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
