import json,numpy as np
def load(name):
    j=json.load(open(f'out_q/{name}/per_bearing.json')); ss=sorted(j['seeds'],key=int); tb=j['test_bearings']
    M=np.array([[j['seeds'][s]['per_bearing'][b]['acc'] for b in tb] for s in ss])
    mf=np.array([json.load(open(f'out_q/{name}/seed_{s}.json'))['final_macro_f1'] for s in ss])*100
    return tb,[j['seeds'][ss[0]]['per_bearing'][b]['label'] for b in tb],M,mf
out=[]
def P(x=''): out.append(x); print(x)
for pre in ['bm3','cnn']:
    D={a:load(f'q3_{pre}_L3_{a}') for a in ['H','V','HV']}
    tb,lab=D['H'][:2]; P(f'## {pre.upper()} L3 (train Cond1+Cond2, test Cond3) {list(zip(tb,lab))}')
    P('| arm | macro-F1 | '+' | '.join(f'{b} ({l})' for b,l in zip(tb,lab))+' |'); P('|---|---|'+'---|'*len(tb))
    for a,(_,_,M,mf) in D.items(): P(f'| {a} | {mf.mean():.1f} ± {mf.std(ddof=1):.1f} (per seed {" ".join(f"{x:.1f}" for x in mf)}) | '+' | '.join(f'{x:.2f}' for x in M.mean(0))+' |')
    for x,y in [('V','H'),('HV','H'),('HV','V')]:
        d=(D[x][2]-D[y][2]).mean(0); P(f'{x}-{y}: '+' '.join(f'{v:+.2f}' for v in d)+f'   macroF1 per-seed diff: '+' '.join(f'{v:+.1f}' for v in D[x][3]-D[y][3]))
    P()
open('an_q3_results.md','w').write('\n'.join(out))
