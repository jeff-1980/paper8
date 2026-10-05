import json,glob,numpy as np
from scipy import stats
rng=np.random.default_rng(11)
def get(a):
    pbd={}; mf={}
    d=f'out_q/r2_bm2_{a}'
    j=json.load(open(f'{d}/per_bearing.json'))
    for s,v in j['seeds'].items(): pbd[int(s)]=v['per_bearing']
    for f in glob.glob(f'{d}/seed_*.json'):
        r=json.load(open(f)); mf[int(r['seed'])]=r['final_macro_f1']*100
    ss=sorted(mf); return ss,np.array([mf[s] for s in ss]),pbd
H=get('H'); V=get('V'); print(H[0],V[0])
# original 8-seed BM2 H (main) and dual from result dirs
R='<REPO_ROOT>/results/xjtu_leakfree_20260917/'
import os
print(os.path.exists(R))
d=V[1]-H[1]
ci=np.percentile([rng.choice(d,8).mean() for _ in range(20000)],[2.5,97.5])
print('BM2 H %.2f±%.2f  V %.2f±%.2f'%(H[1].mean(),H[1].std(ddof=1),V[1].mean(),V[1].std(ddof=1)))
print('V-H %+.2f [%+.2f,%+.2f] p=%.4f %d/%d'%(d.mean(),ci[0],ci[1],stats.wilcoxon(d,method='exact').pvalue,(d>0).sum(),(d<0).sum()))
tb=list(V[2][0].keys()); print(tb)
for n,x in [('H',H),('V',V)]: print(n,np.round([[x[2][s][b]['acc'] for b in tb] for s in range(8)],2).mean(0))
print('per-seed H',H[1].round(1),'V',V[1].round(1))

def orig(dirs):
    o={}
    for dd in dirs:
        for f in glob.glob(R+dd+'/seed_*.json'):
            j=json.load(open(f)); o[j['seed']]=j['final_macro_f1']*100
    return np.array([o[s] for s in range(8)])
OH=orig(['exp_e1b_xjtu_cross_single_bm2_leakfree','exp_e1b_xjtu_cross_single_bm2_newseed_leakfree'])
OD=orig(['exp_e1_xjtu_cross_dual_bm2_leakfree','exp_e1_xjtu_cross_dual_bm2_newseed_leakfree'])
print('orig H %.2f±%.2f  orig dual %.2f±%.2f'%(OH.mean(),OH.std(ddof=1),OD.mean(),OD.std(ddof=1)))
print('H repro minus orig per seed',(H[1]-OH).round(1),'mean abs',np.abs(H[1]-OH).mean().round(2))
print('V - orig dual (cross-environment, NOT paired-tested): %.2f; V>dual in %d/8 seeds'%((V[1]-OD).mean(),((V[1]-OD)>0).sum()))
