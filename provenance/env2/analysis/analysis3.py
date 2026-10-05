import numpy as np,json
from scipy import stats
rng=np.random.default_rng(7)
z=np.load('cwru_arrays.npz')
def g(d,s,pref=''): return z[f'{pref}D{s}']-z[f'{pref}S{s}']
gp8=z['PD-8']-z['PS-8']
for nm,ga in [('AWGN -2 (n5)',g(0,-2)),('AWGN -4 (n8)',g(0,-4)),('AWGN -4 (seeds0-4)',g(0,-4)[:5])]:
    B=[rng.choice(ga,len(ga)).mean()-rng.choice(gp8,5).mean() for _ in range(20000)]
    print(nm,'gain %.2f vs pink-8 gain %.2f diff %.2f CI'%(ga.mean(),gp8.mean(),ga.mean()-gp8.mean()),np.percentile(B,[2.5,97.5]).round(2),'MW p=%.3f'%stats.mannwhitneyu(ga,gp8).pvalue)
# power of exact Wilcoxon signed-rank; paired diffs ~ N(delta, sd_d)
def power(n,delta,sd,alt,alpha=0.05,N=20000):
    # critical via exact null: enumerate distribution of W+ 
    from itertools import product
    ranks=np.arange(1,n+1);sums=np.array([ (np.array(b)*ranks).sum() for b in product([0,1],repeat=n)])
    tot=n*(n+1)/2
    # two-sided rejection: min(W+,W-)<=c ; find largest c with P<=alpha
    p_le=lambda c:(sums<=c).mean()
    if alt=='one':
        c=max([c for c in range(0,int(tot)+1) if p_le(c)<=alpha],default=-1)
    else:
        c=max([c for c in range(0,int(tot)+1) if 2*p_le(c)<=alpha],default=-1)
    if c<0: return 0.0
    d=rng.normal(delta,sd,(N,n));r=stats.rankdata(np.abs(d),axis=1)
    Wp=(r*(d>0)).sum(1);Wm=tot-Wp
    return float(((Wp<=c)|(Wm<=c)).mean()) if alt=='two' else float((Wm<=c).mean())
rows=[]
for sd in [1,2,3]:
    for n in [5,8]:
        rows.append((sd,n,power(n,2,sd,'two'),power(n,2,sd,'one')))
print(rows)
# z-approx as stated in paper
from scipy.stats import norm
for n in [5,8]: print('z-approx n',n, norm.cdf(2/(1/np.sqrt(n))-1.96))
json.dump(rows,open('power.json','w'))
