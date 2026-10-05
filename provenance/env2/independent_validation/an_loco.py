import json,glob,os,numpy as np
rng=np.random.default_rng(3)
R='out_loco/'
def load(name):
    p=f'{R}{name}/per_bearing.json'
    if not os.path.exists(p): return None
    j=json.load(open(p)); tb=j['test_bearings']
    seeds=sorted(j['seeds'],key=int)
    M=np.array([[j['seeds'][s]['per_bearing'][b]['acc'] for b in tb] for s in seeds])  # seeds x bearings
    lab=[j['seeds'][seeds[0]]['per_bearing'][b]['label'] for b in tb]
    mf=[json.load(open(f'{R}{name}/seed_{s}.json'))['final_macro_f1'] for s in seeds]
    return dict(tb=tb,lab=lab,M=M,seeds=seeds,mf=np.array(mf))
def boot(d,B=20000):
    d=np.asarray(d); return np.percentile([rng.choice(d,len(d)).mean() for _ in range(B)],[2.5,97.5])
out=[]
def P(s=''): out.append(s); print(s)
for pre in ['bm3','cnn']:
  for f in ['L2','L1']:
    D={a:load(f'{pre}_{f}_{a}') for a in ['H','V','HV']}
    if any(v is None for v in D.values()): P(f'## {pre} {f}: incomplete (missing {[a for a,v in D.items() if v is None]})'); continue
    n=min(len(v['seeds']) for v in D.values())
    tb=D['H']['tb']; lab=D['H']['lab']
    P(f'## {pre.upper()} {f}: test bearings {tb} labels {lab}; seeds used {n}')
    P('| arm | macro-F1 (mean±SD over seeds) | '+' | '.join(f'{b} ({l})' for b,l in zip(tb,lab))+' |'); P('|---|---|'+'---|'*len(tb))
    for a,v in D.items():
        m=v['M'][:n].mean(0); sd=v['M'][:n].std(0,ddof=1) if n>1 else np.zeros(len(tb))
        P(f"| {a} | {v['mf'][:n].mean()*100:.1f} ± {v['mf'][:n].std(ddof=1)*100 if n>1 else 0:.1f} | "+' | '.join(f'{x:.2f}±{y:.2f}' for x,y in zip(m,sd))+' |')
    P(); P('Contrasts, per test bearing (mean over seeds of acc difference) and number of bearings with positive difference:')
    P('| contrast | '+' | '.join(tb)+' | #bearings + / - | mean over bearings [bootstrap over bearings] |'); P('|---|'+'---|'*(len(tb)+2))
    for x,y in [('V','H'),('HV','H'),('HV','V')]:
        d=(D[x]['M'][:n]-D[y]['M'][:n]).mean(0)
        ci=boot(d) if len(d)>1 else (np.nan,np.nan)
        P(f'| {x}-{y} | '+' | '.join(f'{v:+.2f}' for v in d)+f' | {(d>0.02).sum()}/{(d<-0.02).sum()} | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
    P()
# coverage
rows=[]
for k in [1,2,4,7]:
    H=load(f'bm3_COV{k}_H'); HV=load(f'bm3_COV{k}_HV')
    if H is None or HV is None: continue
    n=min(len(H['seeds']),len(HV['seeds']))
    rows.append((k,H['M'][:n,0],HV['M'][:n,0]))
if rows:
    P('## OR coverage sweep: test bearing Bearing3_1 (OR); IR training bearings fixed (2_1,3_3,3_4); OR training bearings k')
    P('| k OR train bearings | H acc (per seed) | HV acc (per seed) | HV-H mean |'); P('|---|---|---|---|')
    for k,h,hv in rows: P(f'| {k} | '+', '.join(f'{v:.2f}' for v in h)+' | '+', '.join(f'{v:.2f}' for v in hv)+f' | {(hv-h).mean():+.3f} |')
open('an_loco_results.md','w').write('\n'.join(out))
