import json,numpy as np,sys
from scipy import stats
rng=np.random.default_rng(7)
def load(fold,pre,a,old):
    mf={};pb={}
    for d in old+[f'out_q/r1_{pre}_{fold}_{a}']:
        try: j=json.load(open(f'{d}/per_bearing.json'))
        except FileNotFoundError: return None
        for s,v in j['seeds'].items(): pb[int(s)]=v['per_bearing']
        import glob
        for f in glob.glob(f'{d}/seed_*.json'):
            r=json.load(open(f)); mf[int(r['seed'])]=r['final_macro_f1']*100
    ss=sorted(mf); tb=list(pb[ss[0]].keys())
    return ss,np.array([mf[s] for s in ss]),np.array([[pb[s][b]['acc'] for b in tb] for s in ss]),tb
IV='../iv/independent_validation/results/'
old={('L2','bm3'):lambda a:[IV+f'bm3_L2_{a}'],('L3','bm3'):lambda a:[f'out_q/q3_bm3_L3_{a}'],
     ('L2','cnn'):lambda a:[IV+f'cnn_L2_{a}'],('L3','cnn'):lambda a:[f'out_q/q3_cnn_L3_{a}']}
out=[]
def P(x=''): out.append(x); print(x)
for fold in ['L2','L3']:
  for pre in ['bm3','cnn']:
    D={a:load(fold,pre,a,old[(fold,pre)](a)) for a in ['H','V','HV']}
    if any(v is None for v in D.values()): P(f'## {pre} {fold}: incomplete'); continue
    n=len(D['H'][0]); 
    if not all(len(v[0])==n for v in D.values()): P(f'## {pre} {fold}: unequal seeds'); continue
    tb=D['H'][3]
    P(f'## {pre.upper()} {fold}: {n} seeds {D["H"][0]}'); P('| arm | macro-F1 | '+' | '.join(tb)+' |'); P('|---|---|'+'---|'*len(tb))
    for a,v in D.items(): P(f'| {a} | {v[1].mean():.1f} ± {v[1].std(ddof=1):.1f} | '+' | '.join(f'{x:.2f}' for x in v[2].mean(0))+' |')
    P('| contrast | mean pp | 95% CI | exact Wilcoxon p | n+/n- |'); P('|---|---|---|---|---|')
    for x,y in [('V','H'),('HV','H'),('HV','V')]:
        d=D[x][1]-D[y][1]; ci=np.percentile([rng.choice(d,n).mean() for _ in range(20000)],[2.5,97.5])
        p=stats.wilcoxon(d,method='exact').pvalue if np.any(d!=0) else float('nan')
        P(f'| {x}-{y} | {d.mean():+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {p:.4f} | {(d>0).sum()}/{(d<0).sum()} |')
    P()
open('an_r1_results.md','w').write('\n'.join(out))
