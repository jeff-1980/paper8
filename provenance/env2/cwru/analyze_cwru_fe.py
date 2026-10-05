import json, numpy as np, os
from scipy.stats import wilcoxon
R="<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree"
CLS=["normal","inner","ball","outer"]
SNRS=[-4,-6,-8]; SEEDS=range(5)
def load(d):
    return [json.load(open(f"{d}/seed_{s}.json")) for s in SEEDS]
def acc(rs): return np.array([r["test_acc_at_best_val"] for r in rs])
def val(rs): return np.array([r["best_val_acc"] for r in rs])
def recalls(d):
    out=[]
    for s in SEEDS:
        z=np.load(f"{d}/seed_{s}_test_predictions.npz"); t,p=z["true_label"],z["pred_label"]
        out.append([np.mean(p[t==c]==c) for c in range(4)])
    return np.array(out)
def orig(bb,kind,snr):
    if bb=="bm3":
        n={"DE":f"exp02_snr{snr}_nokin","DEFE":f"exp_b2_dual_nokin_snrm{abs(snr)}"}[kind]
    else:
        assert kind=="DE"; n=f"exp07_cnn1d_snr{snr}_leakfree"
    return f"{R}/{n}"
def boot(d,n=20000,seed=12345):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(d),(n,len(d)))
    m=d[idx].mean(1); return np.percentile(m,[2.5,97.5])
def wil(d):
    if np.allclose(d,0): return 1.0
    return wilcoxon(d,alternative="two-sided",method="exact").pvalue
def ms(a): return f"{a.mean()*100:.2f} ± {a.std(ddof=1)*100:.2f}"
def pd_row(name,d):
    lo,hi=boot(d); return f"| {name} | {d.mean()*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | {wil(d):.4f} | {int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())} |"
L=[]; RAW={}
# environment checks
L.append("## 1. Environment check (this environment vs original result files, test_acc_at_best_val, seeds 0-4)\n")
L.append("| backbone | arm | this env (mean ± SD) | original (mean ± SD) | mean diff (pts) | max abs per-seed diff (pts) |\n|---|---|---|---|---|---|")
envdiff={}
for bb,snr in [("bm3",-8),("cnn",-8),("cnn",-6),("cnn",-4)]:
    a=acc(load(f"results/{bb}_DE_snr{snr}")); o=acc(load(orig(bb,"DE",snr)))
    envdiff[(bb,snr)]=(a.mean()-o.mean())*100
    L.append(f"| {bb} | DE-only {snr} dB | {ms(a)} | {ms(o)} | {(a.mean()-o.mean())*100:+.2f} | {np.abs(a-o).max()*100:.2f} |")
L.append("\nBM3: |diff| < 1 pt at -8 dB, so original DE-only and DE+FE files are used as comparators (BM3 DE/DE+FE were NOT rerun). CNN: DE-only rerun at all three SNRs and DE+FE rerun here (no leak-free original exists for CNN DE+FE; exp_mext_e14 used the pre-leak-free random split).\n")
for bb in ["bm3","cnn"]:
    tag={"bm3":"BM3 (Mamba-3, CE only, lambda_kin=0)","cnn":"1D-CNN (baselines/cnn1d)"}[bb]
    L.append(f"## {'2' if bb=='bm3' else '3'}. {tag}\n")
    L.append("Accuracy = test_acc_at_best_val (checkpoint chosen by validation accuracy), mean ± SD over seeds 0-4, %.\n")
    L.append("| SNR | FE-only | DE-only | DE+FE | source of DE / DE+FE |\n|---|---|---|---|---|")
    comps=[]; recs=[]
    for snr in SNRS:
        fe=load(f"results/{bb}_FE_snr{snr}")
        if bb=="bm3":
            de=load(orig("bm3","DE",snr)); df=load(orig("bm3","DEFE",snr)); src="original files"
            deD=orig("bm3","DE",snr); dfD=orig("bm3","DEFE",snr)
        else:
            de=load(f"results/cnn_DE_snr{snr}"); df=load(f"results/cnn_DEFE_snr{snr}"); src="rerun here"
            deD=f"results/cnn_DE_snr{snr}"; dfD=f"results/cnn_DEFE_snr{snr}"
        A={"FE":acc(fe),"DE":acc(de),"DEFE":acc(df)}; V={"FE":val(fe),"DE":val(de),"DEFE":val(df)}
        RAW[f"{bb}_{snr}"]={k:v.tolist() for k,v in A.items()}
        L.append(f"| {snr} dB | {ms(A['FE'])} | {ms(A['DE'])} | {ms(A['DEFE'])} | {src} |")
        best_val="FE" if V["FE"].mean()>V["DE"].mean() else "DE"
        best_test="FE" if A["FE"].mean()>A["DE"].mean() else "DE"
        comps.append((snr,A,best_val,best_test,fe,de,df,deD,dfD))
    L.append("\nPaired differences (percentage points, seed-paired, n=5). Columns: mean diff, bootstrap 95% CI (20000 resamples over seed-paired differences), exact two-sided Wilcoxon p (**n=5 floor is 2/32 = 0.0625, so p<0.05 is unattainable**), #seeds positive/negative/zero.\n")
    L.append("| SNR | comparison | mean diff (pts) | bootstrap 95% CI | Wilcoxon p (exact) | +/-/0 |\n|---|---|---|---|---|---|")
    for snr,A,bv,bt,*_ in comps:
        L.append(pd_row(f"{snr} dB | DE+FE − DE",A["DEFE"]-A["DE"]).replace("| "," | ",0))
        L.append(pd_row(f"{snr} dB | DE+FE − better single (by mean val-acc: {bv})",A["DEFE"]-A[bv]))
        if bt!=bv: L.append(pd_row(f"{snr} dB | DE+FE − better single (by mean test acc: {bt}; sensitivity)",A["DEFE"]-A[bt]))
        L.append(pd_row(f"{snr} dB | FE − DE",A["FE"]-A["DE"]))
    L.append("\nPer-class recall (%), mean over seeds 0-4 (test set, checkpoint chosen by val). Classes: "+", ".join(CLS)+".\n")
    L.append("| SNR | arm | "+" | ".join(CLS)+" |\n|---|---|---|---|---|---|")
    for snr,A,bv,bt,fe,de,df,deD,dfD in comps:
        for nm,d in [("FE",f"results/{bb}_FE_snr{snr}"),("DE",deD),("DE+FE",dfD)]:
            try: r=recalls(d).mean(0)*100; L.append(f"| {snr} dB | {nm} | "+" | ".join(f"{x:.1f}" for x in r)+" |")
            except Exception as e: L.append(f"| {snr} dB | {nm} | n/a ({type(e).__name__}) |||||")
    L.append("")
    L.append("Per-seed test accuracy (%):\n\n| SNR | arm | s0 | s1 | s2 | s3 | s4 |\n|---|---|---|---|---|---|---|")
    for snr,A,*_ in comps:
        for k in ["FE","DE","DEFE"]: L.append(f"| {snr} dB | {k} | "+" | ".join(f"{x*100:.2f}" for x in A[k])+" |")
    L.append("")
open("summary_cwru_fe.md","w").write("\n".join(L)); json.dump(RAW,open("cwru_fe_accuracies.json","w"),indent=1)
print("\n".join(L))
