import json,glob,os,itertools,numpy as np
from scipy import stats
R='<REPO_ROOT>/results/'
X=R+'xjtu_leakfree_20260917/'
rng=np.random.default_rng(12345)
def seedvals(dirs,key='final_macro_f1',sub=None):
    out={}
    for d in dirs:
        for f in glob.glob(d+'/seed_*.json'):
            j=json.load(open(f)); out[j['seed']]=j[key] if sub is None else j[key][sub]
    return np.array([out[s] for s in sorted(out)]),sorted(out)
def wil(d,alt='two-sided'):
    d=np.asarray(d)
    if np.all(d==0): return 1.0
    return stats.wilcoxon(d,alternative=alt,method='exact').pvalue
def boot_ci(d,B=20000):
    d=np.asarray(d); m=[rng.choice(d,len(d)).mean() for _ in range(B)]
    return np.percentile(m,[2.5,97.5])
def paired(name,a,b,scale=100):
    d=(b-a)*scale
    ci=boot_ci(d)
    return dict(name=name,n=len(d),mean_diff=d.mean(),sd_diff=d.std(ddof=1),ci=tuple(ci),p=wil(d),conc=int((d>0).sum()),dpos=d.tolist())
res={}
bb={'BM3':(['exp_xjtu_cross_nokin_leakfree','exp_xjtu_cross_nokin_newseed_leakfree'],['exp_xjtu_cross_dual_nokin_leakfree','exp_xjtu_cross_dual_nokin_newseed_leakfree']),
    'BM2':(['exp_e1b_xjtu_cross_single_bm2_leakfree','exp_e1b_xjtu_cross_single_bm2_newseed_leakfree'],['exp_e1_xjtu_cross_dual_bm2_leakfree','exp_e1_xjtu_cross_dual_bm2_newseed_leakfree']),
    'CNN':(['exp_mext_e13_1dcnn_xjtu_cross_leakfree','exp_mext_e13_1dcnn_xjtu_cross_newseed_leakfree'],['exp_e1b_xjtu_cross_dual_cnn_leakfree','exp_e1b_xjtu_cross_dual_cnn_newseed_leakfree'])}
cross={}
for k,(s,d) in bb.items():
    a,sa=seedvals([X+x for x in s]);b,sb=seedvals([X+x for x in d]);assert sa==sb==list(range(8)),(k,sa,sb)
    cross[k]=(a,b)
    r=paired('cross F1 dual-single '+k,a,b);r['single']=(a.mean()*100,a.std(ddof=1)*100);r['dual']=(b.mean()*100,b.std(ddof=1)*100)
    ira,_=seedvals([X+x for x in s],'final_per_class_recall',1);irb,_=seedvals([X+x for x in d],'final_per_class_recall',1)
    r['IR']=(ira.mean()*100,irb.mean()*100,paired('ir',ira,irb)['p'])
    res['cross_'+k]=r
# LOBO
L=R+'e3_lobo_leakfree_20260709-205255/'
def lobo(arm):
    M=np.zeros((4,8))
    for fold in range(4):
        for s in range(8):
            fs=glob.glob(f'{L}{arm}/fold{fold}_*seed{s}.json')+glob.glob(f'{L}{arm}_n8ext/fold{fold}_*seed{s}.json')
            fs=[f for f in fs if os.path.basename(f).startswith(f'fold{fold}_') and f.endswith(f'seed{s}.json') and not os.path.basename(f).startswith(f'fold{fold}_Bearing') or os.path.basename(f)==f'fold{fold}_seed{s}.json']
            fs=list(set(fs)); assert len(fs)==1,(arm,fold,s,fs)
            M[fold,s]=sum(json.load(open(fs[0]))['per_class_recall'])
    return M
S=lobo('single_nokin');D=lobo('dual_nokin')
# OR folds 0,3 ; IR folds 1,2
def head(M): return (M[[0,3]].mean(0)+M[[1,2]].mean(0))/2
res['lobo_head']=paired('LOBO headline dual-single',head(S),head(D))
res['lobo_head']['single']=(head(S).mean()*100,head(S).std(ddof=1)*100);res['lobo_head']['dual']=(head(D).mean()*100,head(D).std(ddof=1)*100)
folds=['3_1 OR','3_3 IR','3_4 IR','3_5 OR']
res['lobo_folds']={f:paired('fold '+f,S[i],D[i]) for i,f in enumerate(folds)}
# by-fold summary
for i,f in enumerate(folds): res['lobo_folds'][f]['single']=(S[i].mean()*100,S[i].std(ddof=1)*100);res['lobo_folds'][f]['dual']=(D[i].mean()*100,D[i].std(ddof=1)*100)
# leave-one-fold-out sensitivity of headline sign
res['lobo_sens']={f:None for f in folds}
# P14
P=R+'e3_p14_coverage_leakfree_20260928/'
p14={}
for arm in ['single','dual']:
    for f in glob.glob(P+arm+'/k*_seed*.json'):
        j=json.load(open(f)); tag=j['fold_tag'];p14[(arm,tag,j['seed'])]=sum(j['per_class_recall'])
tags=sorted({t for (_,t,_) in p14})
p14res={}
for t in tags:
    a=np.array([p14[('single',t,s)] for s in range(5)]);b=np.array([p14[('dual',t,s)] for s in range(5)])
    p14res[t]=dict(single=(a.mean()*100,a.std(ddof=1)*100),dual=(b.mean()*100,b.std(ddof=1)*100),d=((b-a)*100).tolist(),p=wil((b-a)*100),conc=int((b<a).sum()))
res['p14']=p14res
# k=2 pooled: paired over 10; seed-block: per-seed mean over two combos ->5 pairs
c2=[t for t in tags if t.startswith('k2') and 'Bearing3_5' in t]
d10=np.concatenate([np.array(p14res[t]['d']) for t in c2])
res['p14_k2_pooled']=dict(n=len(d10),mean=d10.mean(),p=wil(d10),ci=tuple(boot_ci(d10)))
seedblock=np.mean([p14res[t]['d'] for t in c2],axis=0)
res['p14_k2_seedblock']=dict(d=seedblock.tolist(),p=wil(seedblock),mean=seedblock.mean())
res['p14_k3']=p14res['k3_c0_Bearing3_3-Bearing3_4-Bearing3_5']
json.dump(res,open('analysis_out.json','w'),indent=1,default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o))
def fmt(r): return f"{r['name']}: n={r['n']} mean_diff={r['mean_diff']:.2f} sd_d={r['sd_diff']:.2f} CI=[{r['ci'][0]:.2f},{r['ci'][1]:.2f}] p={r['p']:.4f} conc={r['conc']}"
for k in ['cross_BM3','cross_BM2','cross_CNN']: print(fmt(res[k]),'single %.2f±%.2f dual %.2f±%.2f'%(*res[k]['single'],*res[k]['dual']),'IR',res[k]['IR'])
print(fmt(res['lobo_head']),res['lobo_head']['single'],res['lobo_head']['dual'])
for f,r in res['lobo_folds'].items(): print(fmt(r),'s %.2f±%.2f d %.2f±%.2f'%(*r['single'],*r['dual']))
for t,r in p14res.items(): print(t,'S %.1f±%.1f D %.1f±%.1f p=%.4f conc=%d'%(*r['single'],*r['dual'],r['p'],r['conc']),np.round(r['d'],1))
print(res['p14_k2_pooled'],res['p14_k2_seedblock'])
