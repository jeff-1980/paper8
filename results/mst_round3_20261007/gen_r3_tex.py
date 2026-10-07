"""Generate the round-3 LaTeX tables from result files (no hand-typed numbers)."""
import json, glob, os, numpy as np
exec(open('an_r3.py').read().split('# ---------------- E1')[0])   # helpers: xc, pb, paired, lobo, ms
rng=np.random.default_rng(12345)
def cw(d):
    o={}
    for f in glob.glob(d+'/seed_*.json'): j=json.load(open(f)); o[j['seed']]=j['test_acc_at_best_val']*100
    return o
def m_sd(o,S): v=np.array([o[s] for s in S]); return f'${v.mean():.1f}\\pm{v.std(ddof=1):.1f}$'
def m1(o,S): return f'${np.mean([o[s] for s in S]):.1f}$'
def dci(a,b,S):
    r=paired(a,b,S); return f'${r[0]:+.1f}$ [${r[1][0]:+.1f}$, ${r[1][1]:+.1f}$]', r
def sgn(r): return f'{r[3]}/{r[4]}'
T={}
# ---- E2 onset table
S=[0,1,2]; rows=[]
for rule,lab in [('H','H channel (main)'),('V','V channel'),('last15','last 15\\% of life')]:
    if rule=='H': A={a:xc(R+f'xjtu_cross/xc_bm3_{a}') for a in ['H','V','HV']}; L={a:lobo(R+f'xjtu_lobo/lobo_bm3_{a}')[0] for a in ['H','V','HV']}
    else: A={a:xc(O+f'e2_{rule}_xc_bm3_{a}') for a in ['H','V','HV']}; L={a:lobo(O+f'e2_{rule}_lobo_bm3_{a}')[0] for a in ['H','V','HV']}
    vh=paired(A['V'],A['H'],S); dv=paired(A['HV'],A['V'],S); ldh=paired(L['HV'],L['H'],S); ldv=paired(L['HV'],L['V'],S)
    rows.append(f"{lab} & {m1(A['H'],S)} & {m1(A['V'],S)} & {m1(A['HV'],S)} & ${vh[0]:+.1f}$ ({sgn(vh)}) & ${dv[0]:+.1f}$ ({sgn(dv)}) & {m1(L['H'],S)} & {m1(L['V'],S)} & {m1(L['HV'],S)} & ${ldh[0]:+.1f}$ ({sgn(ldh)}) & ${ldv[0]:+.1f}$ ({sgn(ldv)}) \\\\")
T['onset']='\n'.join(rows)
# ---- E3 gated table (cross, seeds 0-4) and LOBO (seeds 0-2)
S5=[0,1,2,3,4]; rows=[]
for L,base,ns in [('1D-CNN','xc_cnn',[('window gate','e3_cnn_gated'),('token gate$^{\\dagger}$','e3t_cnn_tgated')]),('Mamba-3','xc_bm3',[('window gate','e3_bm3_gated'),('token gate$^{\\dagger}$','e3t_bm3_tgated')])]:
    H,V,D=xc(R+f'xjtu_cross/{base}_H'),xc(R+f'xjtu_cross/{base}_V'),xc(R+f'xjtu_cross/{base}_HV')
    rows.append(f"{L} & concatenation & {m_sd(D,S5)} & --- & {dci(D,V,S5)[0]} & \\\\")
    for nm,n in ns:
        G=xc(O+n); a,r1=dci(G,D,S5); b,r2=dci(G,V,S5); gs=json.load(open(O+n+'/gate_stats.json'))
        sd=max(v['sd'] for k,v in gs.items() if isinstance(v,dict))
        rows.append(f" & {nm} & {m_sd(G,S5)} & {a} ({sgn(r1)}) & {b} ({sgn(r2)}) & ${sd:.3f}$ \\\\")
    rows.append(f" & V only & {m_sd(V,S5)} & & & \\\\")
    if L=='1D-CNN': rows.append('\\mr')
T['gated_xc']='\n'.join(rows)
rows=[]; B={a:lobo(R+f'xjtu_lobo/lobo_bm3_{a}')[0] for a in ['H','V','HV']}
for nm,n in [('concatenation (H+V)',None),('window gate','e3_lobo_bm3_gated'),('token gate$^{\\dagger}$','e3t_lobo_bm3_tgated'),('H only','H'),('V only','V')]:
    if n is None: G=B['HV']
    elif n in ('H','V'): G=B[n]
    else: G=lobo(O+n)[0]
    pf=None
    if n and n not in ('H','V'):
        o=lobo(O+n)[1]; pf=' / '.join(f"{np.mean([o[s][f] for s in S]):.0f}" for f in (0,3))
    elif n in ('H','V'): o=lobo(R+f'xjtu_lobo/lobo_bm3_{n}')[1]; pf=' / '.join(f"{np.mean([o[s][f] for s in S]):.0f}" for f in (0,3))
    else: o=lobo(R+'xjtu_lobo/lobo_bm3_HV')[1]; pf=' / '.join(f"{np.mean([o[s][f] for s in S]):.0f}" for f in (0,3))
    rows.append(f"{nm} & {m_sd(G,S)} & {pf} \\\\")
T['gated_lobo']='\n'.join(rows)
# ---- E4 CWRU Mamba-3 rows (env 2)
rows=[]
for snr in [-4,-6,-8]:
    DE,DF,FE=cw(O+f'e4_bm3_DE_snr{snr}'),cw(O+f'e4_bm3_DEFE_snr{snr}'),cw(R+f'cwru/bm3_FE_snr{snr}'); S=[0,1,2,3,4]
    f2=lambda o: f'${np.mean([o[s] for s in S]):.2f}\\pm{np.std([o[s] for s in S],ddof=1):.2f}$'
    rows.append(('Mamba-3' if snr==-4 else '       ')+f" & ${snr}$~dB & {f2(FE)} & {f2(DE)} & {f2(DF)} \\\\")
T['cwru_m3']='\n'.join(rows)
# ---- E5
rows=[]
for snr in [-6,-8]:
    DE=cw(O+f'e4_bm3_DE_snr{snr}'); S=[0,1,2,3,4]
    for rho,n in [(0.0,f'e4_bm3_DEFE_snr{snr}'),(0.5,f'e5_bm3_DEFE_snr{snr}_rho0.5'),(0.9,f'e5_bm3_DEFE_snr{snr}_rho0.9')]:
        G=cw(O+n); a,r=dci(G,DE,S)
        rows.append((f'${snr}$~dB' if rho==0 else '')+f" & ${rho}$ & {m_sd(DE,S) if rho==0 else ''} & {m_sd(G,S)} & {a} ({sgn(r)}) \\\\")
    if snr==-6: rows.append('\\mr')
T['corr']='\n'.join(rows)
json.dump(T,open('r3_tex.json','w'),indent=1)
for k,v in T.items(): print('##',k); print(v)
