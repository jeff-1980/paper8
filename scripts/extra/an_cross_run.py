import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stats import *
from an_cross import load
def report(prefix, arms, seeds, out):
    D = {a: load(f"out/{prefix}_{a}", seeds) for a in arms}
    PB = {a: json.load(open(f"out/{prefix}_{a}/per_bearing.json")) for a in arms if os.path.exists(f"out/{prefix}_{a}/per_bearing.json")}
    L = [f"### {prefix}: seeds {seeds}\n", "| arm | macro-F1 mean±SD | macro-recall | OR recall | IR recall | per-seed macro-F1 | mean wall/run (s) |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        d = D[a]; f = lambda k: "%.4f ± %.4f" % ms(d[k])
        L.append(f"| {a} | {f('f1')} | {f('rec')} | {f('orr')} | {f('irr')} | {', '.join('%.4f'%x for x in d['f1'])} | {np.mean(d['wall']):.0f} |")
    L += ["", "Paired differences (seed-paired; bootstrap 95% CI, 20000 resamples; exact two-sided Wilcoxon):", "",
          "| contrast | metric | mean diff | 95% CI | Wilcoxon p | n+ / n- |", "|---|---|---|---|---|---|"]
    for x, y in [("V", "H"), ("HV", "H"), ("HV", "V"), ("ATT", "H"), ("ATT", "V"), ("ATT", "HV")]:
        if x in D and y in D:
            for k, nm in [("f1", "macro-F1"), ("irr", "IR recall")]:
                r = paired(D[x][k], D[y][k]); L.append(f"| {x}-{y} | {nm} | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4f} | {r['n_pos']}/{r['n_neg']} |")
    if PB:
        bs = PB[arms[0]]["test_bearings"]
        L += ["", "Per-test-bearing window accuracy (recall of the true class), per seed; bearings: " + ", ".join(f"{b}({PB[arms[0]]['seeds']['0']['per_bearing'][b]['label']}, n={PB[arms[0]]['seeds']['0']['per_bearing'][b]['n']})" for b in bs), "",
              "| arm | seed | " + " | ".join(bs) + " | bearings >=0.5 acc |", "|---|---|" + "---|" * (len(bs) + 1)]
        for a in arms:
            for s in seeds:
                pb = PB[a]["seeds"][str(s)]["per_bearing"]
                L.append(f"| {a} | {s} | " + " | ".join("%.3f" % pb[b]["acc"] for b in bs) + f" | {sum(pb[b]['acc']>=0.5 for b in bs)}/{len(bs)} |")
        L += ["", "Mean per-bearing accuracy over seeds (mean ± SD) and # (seed) runs with acc>=0.9 / <0.5:", "", "| arm | " + " | ".join(bs) + " |", "|---|" + "---|" * len(bs)]
        for a in arms:
            row = []
            for b in bs:
                v = [PB[a]["seeds"][str(s)]["per_bearing"][b]["acc"] for s in seeds]
                row.append("%.3f ± %.3f (>=.9: %d, <.5: %d)" % (*ms(v), sum(x >= .9 for x in v), sum(x < .5 for x in v)))
            L.append(f"| {a} | " + " | ".join(row) + " |")
    open(out, "w").write("\n".join(L)); print("\n".join(L))
if __name__ == "__main__":
    report("xc_bm3", ["H", "V", "HV"], [0, 1, 2, 3, 4], "out/summary_xc_bm3.md")
