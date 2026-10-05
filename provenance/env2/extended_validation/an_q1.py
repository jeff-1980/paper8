import json,numpy as np
from scipy import stats
rng=np.random.default_rng(5)
def load(a):
    mf={};pb={}
    for d in [f'out/xc_bm3_{a}',f'out_q/q1_bm3_{a}']:
        j=json.load(open(f'{d}/per_bearing.json'))
        for s,v in j['seeds'].items(): pb[int(s)]=v['per_bearing']
        import glob
        for f in glob.glob(f'{d}/seed_*.json'):
            r=json.load(open(f)); mf[int(r['seed'])]=r['final_macro_f1']*100
    ss=sorted(mf); assert ss==list(range(8)),ss
    return np.array([mf[s] for s in ss]),pb
D={a:load(a) for a in ['H','V','HV']}
tb=['Bearing3_1','Bearing3_3','Bearing3_4','Bearing3_5']
out=[]
def P(x=''): out.append(x); print(x)
P('| arm | macro-F1 mean±SD (8 seeds) | '+' | '.join(tb)+' |'); P('|---|---|'+'---|'*4)
for a,(m,pb) in D.items():
    acc=np.array([[pb[s][b]['acc'] for b in tb] for s in range(8)])
    P(f'| {a} | {m.mean():.2f} ± {m.std(ddof=1):.2f} | '+' | '.join(f'{x:.2f}' for x in acc.mean(0))+' |')
def ci(d): return np.percentile([rng.choice(d,8).mean() for _ in range(20000)],[2.5,97.5])
P(); P('| contrast | mean diff (pp) | 95% CI | Wilcoxon exact p | n+/n- |'); P('|---|---|---|---|---|')
for x,y in [('V','H'),('HV','H'),('HV','V')]:
    d=D[x][0]-D[y][0]; lo,hi=ci(d)
    P(f'| {x}-{y} | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {stats.wilcoxon(d,method="exact").pvalue:.4f} | {(d>0).sum()}/{(d<0).sum()} |')
P(); P('Per-seed macro-F1:')
for a in D: P(a+' '+' '.join(f'{v:.1f}' for v in D[a][0]))
open('an_q1_results.md','w').write('\n'.join(out))
