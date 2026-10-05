import json,os,glob,numpy as np
rows=[]
def acc(name):
    p=f'out_q/{name}/per_bearing.json'
    if not os.path.exists(p) or not os.path.exists(f'out_q/{name}/done.flag'): return None
    j=json.load(open(p)); ss=sorted(j['seeds'],key=int); b=j['test_bearings'][0]
    return np.array([j['seeds'][s]['per_bearing'][b]['acc'] for s in ss])
out=[]
def P(x=''): out.append(x); print(x)
order={ '3_1':{'SAME':['3_5','2_2','2_4'],'OTHER':['2_2','2_4','2_5']},
        '2_2':{'SAME':['2_4','2_5','3_5'],'OTHER':['3_1','3_5','1_1']},
        '1_1':{'SAME':['1_2','1_3','3_1'],'OTHER':['3_1','3_5','2_2']}}
cond=lambda b:b[0]
P('Accuracy on the test bearing (own-class recall; all windows of a bearing share one label), per seed 0,1,2. IR training bearings fixed: 2_1, 3_3, 3_4.'); P()
P('| test | ordering | k | OR training bearings added | H acc | HV acc | HV-H mean |'); P('|---|---|---|---|---|---|---|')
for t,o in order.items():
    for ordn,lst in o.items():
        for k in [1,2,3]:
            h=acc(f'q2_T{t}_{ordn}_k{k}_H'); hv=acc(f'q2_T{t}_{ordn}_k{k}_HV')
            fmt=lambda a:'-' if a is None else ', '.join(f'{x:.2f}' for x in a)
            d='-' if (h is None or hv is None) else f'{(hv-h).mean():+.3f}'
            P(f'| Bearing{t} (Cond{t[0]}) | {ordn} | {k} | {", ".join(lst[:k])} | {fmt(h)} | {fmt(hv)} | {d} |')
open('an_q2_results.md','w').write('\n'.join(out))
