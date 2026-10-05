import json,glob,numpy as np
from scipy import stats
exec(open('analysis.py').read().split("res={}")[0])   # helpers
C='<REPO_ROOT>/results/bm3_cwru_leakfree_20260911-leakfree/'
def acc(dirs):
    out={}
    for d in dirs:
        for f in glob.glob(C+d+'/seed_*.json'):
            j=json.load(open(f)); out[j['seed']]=j['test_acc_at_best_val']*100
    return np.array([out[s] for s in sorted(out)])
def sd(x): return x.std(ddof=1)
snr={-4:'-4',-6:'-6',-8:'-8',-2:'-2',0:'0'}
S={};Dd={};Pk_s={};Pk_d={}
for s in [-4,-6,-8]:
    S[s]=acc([f'exp02_snr{s}_nokin',f'exp_e6_single_nokin_snrm{-s}_newseed_leakfree'])
    Dd[s]=acc([f'exp_b2_dual_nokin_snrm{-s}',f'exp_e6_dual_nokin_snrm{-s}_newseed_leakfree'])
    Pk_s[s]=acc([f'exp_e5_single_pink_snr{s}_leakfree']);Pk_d[s]=acc([f'exp_e5_dual_pink_snr{s}_leakfree'])
for s in [-2,0]:
    S[s]=acc([f'exp02_snr{s}_nokin']);Dd[s]=acc([f'exp_b2_dual_nokin_snrm{-s}' if s else 'exp_b2_dual_nokin'])
print('AWGN single/dual n=8 (mean,sd,n)')
for s in S: print(s,len(S[s]),round(S[s].mean(),2),round(sd(S[s]),2),'|',len(Dd[s]),round(Dd[s].mean(),2),round(sd(Dd[s]),2),'gain',round(Dd[s].mean()-S[s].mean(),2))
print('n=5 subsets single -4:',S[-4][:5].mean().round(2),sd(S[-4][:5]).round(2),' -8:',S[-8][:5].mean().round(2),sd(S[-8][:5]).round(2))
print('pink')
for s in Pk_s: print(s,len(Pk_s[s]),Pk_s[s].mean().round(2),sd(Pk_s[s]).round(2),Pk_d[s].mean().round(2),sd(Pk_d[s]).round(2),'gain',(Pk_d[s]-Pk_s[s]).mean().round(2))
# interaction: gain_AWGN (seeds 0-4) - gain_pink (seeds 0-4), paired by seed index (independent runs; permutation test on seed-wise difference of gains)
print('interaction (AWGN gain - pink gain), seeds 0-4')
out={}
for s in [-4,-6,-8]:
    gA=(Dd[s]-S[s])[:5]; gP=Pk_d[s]-Pk_s[s]
    # unpaired: bootstrap CI of difference in mean gains, Mann-Whitney
    B=[rng.choice(gA,5).mean()-rng.choice(gP,5).mean() for _ in range(20000)]
    mw=stats.mannwhitneyu(gA,gP,alternative='two-sided')
    out[s]=dict(gA=gA.mean(),gP=gP.mean(),diff=gA.mean()-gP.mean(),ci=np.percentile(B,[2.5,97.5]).tolist(),mw_p=mw.pvalue)
    print(s,{k:np.round(v,3) for k,v in out[s].items()})
# pooled over SNR: per-seed mean gain across three SNR
gA=np.mean([(Dd[s]-S[s])[:5] for s in [-4,-6,-8]],0);gP=np.mean([Pk_d[s]-Pk_s[s] for s in [-4,-6,-8]],0)
B=[rng.choice(gA,5).mean()-rng.choice(gP,5).mean() for _ in range(20000)]
print('pooled', gA.mean().round(2),gP.mean().round(2),(gA.mean()-gP.mean()).round(2),np.percentile(B,[2.5,97.5]).round(2),stats.mannwhitneyu(gA,gP).pvalue)
# difficulty-matched
print('matched difficulty: AWGN -4 dB single %.2f (gain %.2f n8; n5 gain %.2f) vs pink -8 single %.2f gain %.2f'%(S[-4].mean(),(Dd[-4]-S[-4]).mean(),(Dd[-4]-S[-4])[:5].mean(),Pk_s[-8].mean(),(Pk_d[-8]-Pk_s[-8]).mean()))
print('pink -6 single %.2f AWGN -2 single %.2f gain(-2) %.2f'%(Pk_s[-6].mean(),S[-2].mean(),(Dd[-2]-S[-2]).mean()))
# Table III/V reconciliation
b=acc(['exp02_snr-4_nokin']);print('BM3 single CE -4dB n=5',b.mean().round(2),sd(b).round(2),' n=8',S[-4].mean().round(2),sd(S[-4]).round(2))
for d in ['exp02_snr0_nokin','exp05_lk1em3','exp05_lk1em2','exp05_lk1em1','exp05_lk1p0','exp02_snr0_kin']:
    a=acc([d]);print(d,a.mean().round(2),sd(a).round(2))
np.savez('cwru_arrays.npz',**{f'S{s}':S[s] for s in S},**{f'D{s}':Dd[s] for s in Dd},**{f'PS{s}':Pk_s[s] for s in Pk_s},**{f'PD{s}':Pk_d[s] for s in Pk_d})
json.dump(out,open('interaction.json','w'),default=float)
