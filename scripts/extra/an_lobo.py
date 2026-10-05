import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from stats import *
BEAR = ['Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']; LAB = ['OR', 'IR', 'IR', 'OR']
def load(rd, seeds):
    """returns rec[fold][seed] = present-class recall from per_class_recall; wall[fold][seed]"""
    rec = np.full((4, len(seeds)), np.nan); wall = np.full((4, len(seeds)), np.nan)
    for f in range(4):
        for j, s in enumerate(seeds):
            p = f"{rd}/fold{f}_seed{s}.json"
            if os.path.exists(p):
                r = json.load(open(p)); rec[f, j] = r["per_class_recall"][0 if LAB[f] == 'OR' else 1]; wall[f, j] = r["elapsed_s"]
    return rec, wall
def report(prefix, arms, seeds, out, title):
    D = {a: load(f"out/{prefix}_{a}", seeds) for a in arms}
    L = [f"### {title}: seeds {seeds} (present-class recall from per_class_recall)\n",
         "| arm | " + " | ".join(f"{b} ({l})" for b, l in zip(BEAR, LAB)) + " | OR-fold mean | IR-fold mean | mean of 4 folds | mean wall/run (s) |", "|---|" + "---|" * 8]
    for a in arms:
        r, w = D[a]
        cells = ["%.3f ± %.3f (n=%d)" % (*ms(r[f][~np.isnan(r[f])]), (~np.isnan(r[f])).sum()) if (~np.isnan(r[f])).any() else "not run" for f in range(4)]
        orm = np.nanmean(r[[0, 3]], axis=0); irm = np.nanmean(r[[1, 2]], axis=0); m4 = np.nanmean(r, axis=0)
        L.append(f"| {a} | " + " | ".join(cells) + f" | {ms(orm)[0]:.3f} ± {ms(orm)[1]:.3f} | {ms(irm)[0]:.3f} ± {ms(irm)[1]:.3f} | {ms(m4)[0]:.3f} ± {ms(m4)[1]:.3f} | {np.nanmean(w):.0f} |")
    L += ["", "Per-seed present-class recall (rows: fold; columns: seed):", ""]
    for a in arms:
        r, _ = D[a]; L.append(f"- {a}: " + "; ".join(f"{BEAR[f]}: " + ", ".join("%.3f" % x for x in r[f]) for f in range(4)))
    L += ["", "Seed-paired differences (mean diff, bootstrap 95% CI [20000 resamples], exact Wilcoxon p; n = number of seeds — with n=3 the smallest attainable two-sided exact p is 0.25 and the bootstrap CI is coarse):", "",
          "| contrast | unit | mean diff | 95% CI | Wilcoxon p | n+/n- |", "|---|---|---|---|---|---|"]
    for x, y in [("V", "H"), ("HV", "H"), ("HV", "V"), ("ATT", "H"), ("ATT", "V"), ("ATT", "HV")]:
        if x not in D or y not in D: continue
        units = [(f"{BEAR[f]} ({LAB[f]})", D[x][0][f] - D[y][0][f]) for f in range(4)]
        units += [("OR-fold mean", np.mean(D[x][0][[0, 3]], 0) - np.mean(D[y][0][[0, 3]], 0)), ("IR-fold mean", np.mean(D[x][0][[1, 2]], 0) - np.mean(D[y][0][[1, 2]], 0)),
                  ("4-fold mean", np.mean(D[x][0], 0) - np.mean(D[y][0], 0))]
        for nm, d in units:
            if np.isnan(d).any(): L.append(f"| {x}-{y} | {nm} | not run |  |  |  |"); continue
            r = paired(d, np.zeros_like(d)); L.append(f"| {x}-{y} | {nm} | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4f} | {r['n_pos']}/{r['n_neg']} |")
    open(out, "w").write("\n".join(L)); print("\n".join(L))
if __name__ == "__main__":
    report("lobo_bm3", ["H", "V", "HV"], [0, 1, 2], "out/summary_lobo_bm3.md", "LOBO BM3")
