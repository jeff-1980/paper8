"""Round-3 analysis (pre-registered contrasts, prereg_mst_round3.md). Reads result files only."""
import json, glob, os, numpy as np
from scipy import stats
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R=os.path.join(ROOT,'results','env2')+'/'; O=os.path.dirname(os.path.abspath(__file__))+'/'
rng=np.random.default_rng(12345)
out=[]
def P(*a): s=' '.join(str(x) for x in a); print(s); out.append(s)
def xc(d):
    o={}
    for f in glob.glob(d+'/seed_*.json'): j=json.load(open(f)); o[j['seed']]=j['final_macro_f1']*100
    return o
def pb(d):
    f=d+'/per_bearing.json'
    if not os.path.exists(f): return {}
    j=json.load(open(f)); return {int(s):{b:v['acc']*100 for b,v in x['per_bearing'].items()} for s,x in j['seeds'].items()}
def paired(a,b,seeds):
    d=np.array([a[s]-b[s] for s in seeds]); n=len(d)
    ci=np.percentile([rng.choice(d,n).mean() for _ in range(20000)],[2.5,97.5])
    p=stats.wilcoxon(d,method='exact').pvalue if np.any(d!=0) and n>=5 else float('nan')
    return d.mean(),ci,p,(d>0).sum(),(d<0).sum(),d
def fmt(r): m,ci,p,np_,nn,_=r; return f'{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.4f} +{np_}/-{nn}'
def ms(o,S): v=np.array([o[s] for s in S]); return f'{v.mean():.2f}±{v.std(ddof=1):.2f}'
def lobo(d, folds=(0,3)):
    o={}
    for f in glob.glob(d+'/**/fold*_seed*.json',recursive=True):
        j=json.load(open(f)); fo=int(os.path.basename(f)[4]); 
        if fo in folds: o.setdefault(j['seed'],{})[fo]=j['per_class_recall'][0]*100   # OR folds -> OR recall
    return {s:np.mean([v[k] for k in folds]) for s,v in o.items() if all(k in v for k in folds)}, o
# ---------------- E1
if os.path.exists(O+'done_e1_bm2_HV'):
    P('## E1 Mamba-2 condition 2->3, environment 2, seeds 0-7')
    H,V,D=xc(R+'xjtu_cross/r2_bm2_H'),xc(R+'xjtu_cross/r2_bm2_V'),xc(O+'e1_bm2_HV'); S=sorted(set(H)&set(V)&set(D))
    P('seeds',S,'H',ms(H,S),'V',ms(V,S),'H+V',ms(D,S))
    for nm,a,b in [('H+V-V',D,V),('H+V-H',D,H),('V-H',V,H)]: P(' ',nm,fmt(paired(a,b,S)))
    pd_,pv=pb(O+'e1_bm2_HV'),pb(R+'xjtu_cross/r2_bm2_V')
    if pd_: 
        bs=list(pd_[S[0]].keys()); P('  per-bearing H+V:',{b:round(np.mean([pd_[s][b] for s in S]),1) for b in bs}); 
        if pv: P('  per-bearing V  :',{b:round(np.mean([pv[s][b] for s in S]),1) for b in bs})
# ---------------- E3 cross
for L,base,n in [('cnn','xc_cnn','e3_cnn_gated'),('bm3','xc_bm3','e3_bm3_gated'),('cnn','xc_cnn','e3t_cnn_tgated'),('bm3','xc_bm3','e3t_bm3_tgated')]:
    if not os.path.exists(O+'done_'+n): continue
    P(f'## E3 {n} ({L}), condition 2->3, seeds 0-4')
    G=xc(O+n); H,V,D=xc(R+f'xjtu_cross/{base}_H'),xc(R+f'xjtu_cross/{base}_V'),xc(R+f'xjtu_cross/{base}_HV'); S=[s for s in range(5) if s in G and s in H and s in V and s in D]
    P('seeds',S,'gated',ms(G,S),'H',ms(H,S),'V',ms(V,S),'concat',ms(D,S))
    for nm,b in [('gated-concat',D),('gated-V',V),('gated-H',H)]: P(' ',nm,fmt(paired(G,b,S)))
    g=pb(O+n)
    if g: P('  per-bearing gated:',{b:round(np.mean([g[s][b] for s in S]),1) for b in g[S[0]]})
    gf=O+n+'/gate_stats.json'
    if os.path.exists(gf): P('  gate SD over test windows / token SD:',{k:((round(v['sd'],4),round(v['token_sd_mean'],3) if v.get('token_sd_mean') else None) if isinstance(v,dict) else v) for k,v in json.load(open(gf)).items()})
# ---------------- E2
for rule in ['V','last15']:
    ok=all(os.path.exists(O+f'done_e2_{rule}_xc_bm3_{a}') for a in ['H','V','HV'])
    if ok:
        P(f'## E2 onset rule {rule}: condition 2->3 Mamba-3, seeds 0-2 (H-rule comparator: xc_bm3_*, seeds 0-2)')
        A={a:xc(O+f'e2_{rule}_xc_bm3_{a}') for a in ['H','V','HV']}; B={a:xc(R+f'xjtu_cross/xc_bm3_{a}') for a in ['H','V','HV']}; S=[0,1,2]
        P('  rule',rule,{a:ms(A[a],S) for a in A},'| H rule',{a:ms(B[a],S) for a in B})
        for nm,x,y in [('V-H','V','H'),('H+V-V','HV','V'),('H+V-H','HV','H')]:
            r=paired(A[x],A[y],S); r0=paired(B[x],B[y],S)
            stable=np.sign(r[0])==np.sign(r0[0]) and (np.sum(np.sign(r[5])==np.sign(r0[0]))>=2)
            P(f'   {nm}: rule {rule} {r[0]:+.2f} (seeds {np.round(r[5],1)}) | H rule {r0[0]:+.2f} -> {"stable" if stable else "NOT stable"}')
    ok=all(os.path.exists(O+f'done_e2_{rule}_lobo_bm3_{a}') for a in ['H','V','HV'])
    if ok:
        P(f'## E2 onset rule {rule}: LOBO OR folds (3_1, 3_5), Mamba-3, seeds 0-2')
        A={a:lobo(O+f'e2_{rule}_lobo_bm3_{a}') for a in ['H','V','HV']}; B={a:lobo(R+f'xjtu_lobo/lobo_bm3_{a}') for a in ['H','V','HV']}; S=[0,1,2]
        for a in A: P('  ',a,'rule',rule,'OR recall',ms(A[a][0],S),'per fold',{f:round(np.mean([A[a][1][s][f] for s in S]),1) for f in (0,3)},'| H rule',ms(B[a][0],S))
        for nm,x,y in [('H+V-H','HV','H'),('H+V-V','HV','V')]:
            r=paired(A[x][0],A[y][0],S); r0=paired(B[x][0],B[y][0],S)
            stable=np.sign(r[0])==np.sign(r0[0]) and (np.sum(np.sign(r[5])==np.sign(r0[0]))>=2)
            P(f'   {nm}: rule {rule} {r[0]:+.2f} (seeds {np.round(r[5],1)}) | H rule {r0[0]:+.2f} -> {"stable" if stable else "NOT stable"}')
# ---------------- E3 LOBO
for n in ['e3_lobo_bm3_gated','e3t_lobo_bm3_tgated']:
  if os.path.exists(O+'done_'+n):
    P(f'## E3 {n}, LOBO OR folds, Mamba-3, seeds 0-2')
    G=lobo(O+n); B={a:lobo(R+f'xjtu_lobo/lobo_bm3_{a}') for a in ['H','V','HV']}; S=[0,1,2]
    B={a:lobo(R+f'xjtu_lobo/lobo_bm3_{a}') for a in ['H','V','HV']}
    P('  gated',ms(G[0],S),{f:round(np.mean([G[1][s][f] for s in S]),1) for f in (0,3)},'| H',ms(B['H'][0],S),'V',ms(B['V'][0],S),'concat',ms(B['HV'][0],S))
    for a in ['H','V','HV']: r=paired(G[0],B[a][0],S); P(f'   gated-{a}: {r[0]:+.2f} seeds {np.round(r[5],1)}')
    gf=O+n+'/gate_stats.json'
    if os.path.exists(gf): P('  gate stats:',{k:(round(v['sd'],4) if isinstance(v,dict) else v) for k,v in json.load(open(gf)).items()})
# ---------------- E4 / E5
def cw(d):
    o={}
    for f in glob.glob(d+'/seed_*.json'): j=json.load(open(f)); o[j['seed']]=j['test_acc_at_best_val']*100
    return o
for snr in [-4,-6,-8]:
    if all(os.path.exists(O+f'done_e4_bm3_{a}_snr{snr}') for a in ['DE','DEFE']):
        if snr==-4: P('## E4 CWRU Mamba-3, environment 2, seeds 0-4 (accuracy at validation-selected epoch)')
        DE,DF,FE=cw(O+f'e4_bm3_DE_snr{snr}'),cw(O+f'e4_bm3_DEFE_snr{snr}'),cw(R+f'cwru/bm3_FE_snr{snr}'); S=sorted(set(DE)&set(DF)&set(FE))
        r=paired(DF,DE,S); p1=stats.wilcoxon(r[5],alternative='greater',method='exact').pvalue
        P(f'  {snr} dB  FE {ms(FE,S)}  DE {ms(DE,S)}  DE+FE {ms(DF,S)} | DE+FE-DE {fmt(r)} one-sided p={p1:.4f} | FE-DE {fmt(paired(FE,DE,S))}')
for snr in [-6,-8]:
    names=[f'e5_bm3_DEFE_snr{snr}_rho{r}' for r in (0.5,0.9)]
    if all(os.path.exists(O+'done_'+n) for n in names) and os.path.exists(O+f'done_e4_bm3_DE_snr{snr}'):
        if snr==-6: P('## E5 correlated noise, Mamba-3 DE+FE gain over DE, seeds 0-4')
        DE=cw(O+f'e4_bm3_DE_snr{snr}'); G={0.0:cw(O+f'e4_bm3_DEFE_snr{snr}'),0.5:cw(O+names[0]),0.9:cw(O+names[1])}; S=sorted(set(DE).intersection(*[set(v) for v in G.values()]))
        g={rho:np.array([G[rho][s]-DE[s] for s in S]) for rho in G}
        for rho in G: P(f'  {snr} dB rho={rho}: DE+FE {ms(G[rho],S)}  gain {fmt(paired(G[rho],DE,S))}')
        mono=g[0.0].mean()>g[0.5].mean()>g[0.9].mean(); cnt=int((g[0.9]<g[0.0]).sum())
        P(f'   ordering g(0)>g(0.5)>g(0.9): {mono}; g(0.9)<g(0) in {cnt}/{len(S)} seeds; gain(0.9)-gain(0) {fmt(paired({s:g[0.9][i] for i,s in enumerate(S)},{s:g[0.0][i] for i,s in enumerate(S)},S))}')
open('an_r3_results.md','w').write('\n'.join(out))
