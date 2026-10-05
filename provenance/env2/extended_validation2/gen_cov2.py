import json,numpy as np
Q='<ENV2_WORKDIR>/loco2/out_q/'
def acc(name):
    j=json.load(open(Q+name+'/per_bearing.json')); ss=sorted(j['seeds'],key=int); b=j['test_bearings'][0]
    return [j['seeds'][s]['per_bearing'][b]['acc'] for s in ss]
order={'3_1':{'SAME':['3_5','2_2','2_4'],'OTHER':['2_2','2_4','2_5']},
       '2_2':{'SAME':['2_4','2_5','3_5'],'OTHER':['3_1','3_5','1_1']},
       '1_1':{'SAME':['1_2','1_3','3_1'],'OTHER':['3_1','3_5','2_2']}}
BS='\\\\'
us=lambda x:x.replace('_','\\_')
lines=[]
keys=[(t,o) for t in order for o in order[t]]
for idx,(t,ordn) in enumerate(keys):
    lst=order[t][ordn]
    for k in [1,2,3]:
        hv=acc(f'q2_T{t}_{ordn}_k{k}_HV'); h=acc(f'q2_T{t}_{ordn}_k{k}_H'); v=acc(f'q2_T{t}_{ordn}_k{k}_V')
        fm=lambda a:'---' if a is None else ',\\ '.join(f'{x:.2f}' for x in a)
        dm=f'{np.mean(hv)-np.mean(h):+.2f}'; dv=f'{np.mean(hv)-np.mean(v):+.2f}'
        lab='Bearing'+us(t) if (ordn=='SAME' and k==1) else ''
        added='+'.join(us(x) for x in lst[:k])
        lines.append(f'{lab} & {ordn.lower()} & {k} & {added} & ${fm(h)}$ & ${fm(v)}$ & ${fm(hv)}$ & ${dm}$ & ${dv}$ {BS}')
    if idx<len(keys)-1: lines.append('\\midrule')
open('cov_rows2.tex','w').write('\n'.join(lines))
print('\n'.join(lines[:5]))
