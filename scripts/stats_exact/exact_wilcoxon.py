"""Exact Wilcoxon signed-rank test that handles zero differences and ties correctly.
Differences are rounded to 9 decimals before ranking (removes floating-point noise that would otherwise break
theoretical ties), zero differences are dropped (Wilcoxon's rule), tied |d| receive mid-ranks, and the null
distribution of the positive-rank sum is obtained by full enumeration of the 2^n sign assignments.
Two-sided p = P(|T+ - E| >= |t - E|), E = sum(ranks)/2. Returns an object with .statistic and .pvalue."""
import itertools, numpy as np
from types import SimpleNamespace
from scipy.stats import rankdata
def wilcoxon(x, y=None, alternative='two-sided', decimals=9, **kw):
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    d = np.round(d, decimals); d = d[d != 0]
    n = len(d)
    if n == 0: return SimpleNamespace(statistic=np.nan, pvalue=1.0, n=0)
    r = rankdata(np.abs(d)); t = r[d > 0].sum(); E = r.sum() / 2
    S = np.array([np.dot(s, r) for s in itertools.product((0, 1), repeat=n)])
    eps = 1e-9
    if alternative == 'two-sided': p = np.mean(np.abs(S - E) >= abs(t - E) - eps)
    elif alternative == 'greater': p = np.mean(S >= t - eps)
    else: p = np.mean(S <= t + eps)
    return SimpleNamespace(statistic=float(t), pvalue=float(p), n=n)
