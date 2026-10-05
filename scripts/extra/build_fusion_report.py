import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *; import an_fusion as F; import an_cross as AC; import an_lobo as AL
def pt(x, y, nm):
    r = paired(x, y); return f"| {nm} | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4f} | {r['n_pos']}/{r['n_neg']} |"
def cross_section(bb, fus, full, seeds, att):
    rows = F.cross(fus, full, None, seeds); L = [f"## Cross-condition (Cond2->Cond3), backbone {bb}, seeds {seeds}\n"]
    arms = {a: AC.load(f"out/{full}_{a}", seeds) for a in (["H", "V", "HV"] + (["ATT"] if att else []))}
    g = lambda k, m: np.array([r[k][m] for r in rows])
    T = {"H (full)": (arms["H"]["f1"], arms["H"]["irr"]), "V (full)": (arms["V"]["f1"], arms["V"]["irr"]), "HV early fusion (full)": (arms["HV"]["f1"], arms["HV"]["irr"])}
    if att: T["ATT channel-attention early fusion (full)"] = (arms["ATT"]["f1"], arms["ATT"]["irr"])
    T.update({"H_sub (80% source)": (g("H_sub", "f1"), g("H_sub", "irr")), "V_sub (80% source)": (g("V_sub", "f1"), g("V_sub", "irr")),
              "Late fusion, val-selected w (sub-models)": (g("fusion_sub", "f1"), g("fusion_sub", "irr")),
              "Late fusion, w applied to full H,V models": (g("fusion_full", "f1"), g("fusion_full", "irr")),
              "[not selected] equal-weight average (sub-models)": (g("avg05_sub", "f1"), g("avg05_sub", "irr"))})
    L += ["| model | macro-F1 mean±SD | IR recall mean±SD | per-seed macro-F1 |", "|---|---|---|---|"]
    for k, (f, i) in T.items(): L.append(f"| {k} | {ms(f)[0]:.4f} ± {ms(f)[1]:.4f} | {ms(i)[0]:.4f} ± {ms(i)[1]:.4f} | {', '.join('%.3f' % x for x in f)} |")
    L.append(f"\nSelected w (weight on H logits) per seed: {[r['w'] for r in rows]}\n")
    L += ["Seed-paired differences in macro-F1 (bootstrap 95% CI, 20000 resamples; exact Wilcoxon):\n", "| contrast | mean diff | 95% CI | Wilcoxon p | n+/n- |", "|---|---|---|---|---|"]
    for a, b in [("Late fusion, val-selected w (sub-models)", "HV early fusion (full)"), ("Late fusion, val-selected w (sub-models)", "H (full)"), ("Late fusion, val-selected w (sub-models)", "V (full)"),
                 ("Late fusion, val-selected w (sub-models)", "V_sub (80% source)"), ("Late fusion, val-selected w (sub-models)", "H_sub (80% source)"),
                 ("Late fusion, w applied to full H,V models", "HV early fusion (full)"), ("Late fusion, w applied to full H,V models", "V (full)")] + \
                ([("ATT channel-attention early fusion (full)", "HV early fusion (full)"), ("ATT channel-attention early fusion (full)", "H (full)"), ("ATT channel-attention early fusion (full)", "V (full)")] if att else []):
        L.append(pt(T[a][0], T[b][0], f"{a} - {b}"))
    L += ["", "Per-test-bearing accuracy of the fusion (sub-models; Bearing3_1 OR, 3_3 IR, 3_4 IR, 3_5 OR):", "", "| seed | w | 3_1 | 3_3 | 3_4 | 3_5 |", "|---|---|---|---|---|---|"]
    for r in rows: L.append(f"| {r['seed']} | {r['w']} | " + " | ".join("%.3f" % x for x in r["fusion_sub"]["per_bearing"]) + " |")
    L += ["", "Per-test-bearing accuracy, fusion with w applied to full models:", "", "| seed | 3_1 | 3_3 | 3_4 | 3_5 |", "|---|---|---|---|---|"]
    for r in rows: L.append(f"| {r['seed']} | " + " | ".join("%.3f" % x for x in r["fusion_full"]["per_bearing"]) + " |")
    json.dump(rows, open(f"out/fusion_cross_{bb}.json", "w"), default=float, indent=1)
    return L
def lobo_section(bb, fus, full, seeds, att):
    rows = F.lobo(fus, full, seeds); L = [f"## LOBO (Cond3), backbone {bb}, seeds {seeds}; present-class recall\n"]
    D = {a: AL.load(f"out/{full}_{a}", seeds)[0] for a in (["H", "V", "HV"] + (["ATT"] if att else []))}
    def arr(k): return np.array([[next(r[k] for r in rows if r['fold'] == f and r['seed'] == s) for s in seeds] for f in range(4)])
    D["Late fusion val-w (sub-models)"] = arr("fusion_sub"); D["Late fusion, w applied to full H,V"] = arr("fusion_full"); D["H_sub"] = arr("H_sub"); D["V_sub"] = arr("V_sub")
    L += ["| model | " + " | ".join(f"{b} ({l})" for b, l in zip(AL.BEAR, AL.LAB)) + " | OR-fold mean | IR-fold mean | 4-fold mean |", "|---|" + "---|" * 7]
    for k, r in D.items():
        c = ["%.3f ± %.3f" % ms(r[f]) for f in range(4)]; o = r[[0, 3]].mean(0); i = r[[1, 2]].mean(0); m = r.mean(0)
        L.append(f"| {k} | " + " | ".join(c) + f" | {ms(o)[0]:.3f} ± {ms(o)[1]:.3f} | {ms(i)[0]:.3f} ± {ms(i)[1]:.3f} | {ms(m)[0]:.3f} ± {ms(m)[1]:.3f} |")
    L.append("\nSelected w per fold (rows) x seed: " + "; ".join(f"{AL.BEAR[f]}: {[next(r['w'] for r in rows if r['fold']==f and r['seed']==s) for s in seeds]}" for f in range(4)) + "\n")
    L += ["Seed-paired differences (bootstrap 95% CI, exact Wilcoxon):\n", "| contrast | unit | mean diff | 95% CI | Wilcoxon p | n+/n- |", "|---|---|---|---|---|---|"]
    for a, b in [("Late fusion val-w (sub-models)", "H"), ("Late fusion val-w (sub-models)", "V"), ("Late fusion val-w (sub-models)", "HV"), ("Late fusion val-w (sub-models)", "H_sub"),
                 ("Late fusion, w applied to full H,V", "H"), ("Late fusion, w applied to full H,V", "HV")] + ([("ATT", "H"), ("ATT", "HV")] if att else []):
        units = [(f"{AL.BEAR[f]}", D[a][f] - D[b][f]) for f in range(4)] + [("OR-fold mean", D[a][[0, 3]].mean(0) - D[b][[0, 3]].mean(0)), ("IR-fold mean", D[a][[1, 2]].mean(0) - D[b][[1, 2]].mean(0)), ("4-fold mean", D[a].mean(0) - D[b].mean(0))]
        for u, d in units:
            r = paired(d, np.zeros_like(d)); L.append(f"| {a} - {b} | {u} | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4f} | {r['n_pos']}/{r['n_neg']} |")
    json.dump(rows, open(f"out/fusion_lobo_{bb}.json", "w"), default=float, indent=1)
    return L
if __name__ == "__main__":
    which = sys.argv[1:]; L = ["# Validation-selected late fusion and channel-attention baseline\n",
       "Late fusion: logits = w*z_H + (1-w)*z_V, w in {0,0.05,...,1} selected per seed (per fold in LOBO) on a source-domain hold-out (last 20% of windows of every source bearing, time-contiguous; excluded from training of the 'sub-models'); criterion = class-balanced NLL. Test data never used. Because the main-arm H/V models were trained on 100% of the source windows (no validation split exists in the trainers), fusion is reported (i) with the 80%-trained sub-models and (ii) with the selected w applied to the full-data models.\n"]
    if "cross_cnn" in which: L += cross_section("cnn1d", "fus_xc_cnn", "xc_cnn", [0, 1, 2, 3, 4], True)
    if "lobo_cnn" in which: L += [""] + lobo_section("cnn1d", "fus_lobo_cnn", "lobo_cnn", [0, 1, 2, 3, 4], True)
    if "cross_bm3" in which: L += [""] + cross_section("bm3", "fus_xc_bm3", "xc_bm3", [0, 1, 2, 3, 4], False)
    if "lobo_bm3" in which: L += [""] + lobo_section("bm3", "fus_lobo_bm3", "lobo_bm3", [0, 1, 2], False)
    open("out/summary_fusion.md", "w").write("\n".join(L)); print("\n".join(L))
