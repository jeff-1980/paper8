"""Per-seed supplementary tables for the round-3 experiments, from result files."""
import json, glob, os, csv, numpy as np
exec(open('an_r3.py').read().split('# ---------------- E1')[0])
def cw(d):
    o={}
    for f in glob.glob(d+'/seed_*.json'): j=json.load(open(f)); o[j['seed']]=j['test_acc_at_best_val']*100
    return o
def row(lab,o,S,fmt='{:.1f}'): return lab+' & '+' & '.join('$'+fmt.format(o[s])+'$' for s in S)+' \\\\'
def tab(label,cap,S,rows,first='Arm'):
    return (f"\\begin{{table}}[htbp]\n\\caption{{\\label{{{label}}}{cap}}}\n\\footnotesize\\begin{{indented}}\\item[]\n"
            f"\\begin{{tabular}}{{@{{}}l{'c'*len(S)}}}\n\\br\n{first} & "+' & '.join(f'seed {s}' for s in S)+" \\\\\n\\mr\n"+'\n'.join(rows)+"\n\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
out=[]
S8=list(range(8))
out.append(tab('tab:r3_e1','Mamba-2, condition~2$\\to$3, macro-F1 (\\%) per seed, environment~2 (H and V: arms of supplementary section~S6; H+V: trained for this revision, configuration identical except \\texttt{n\\_sensors}).',S8,
  [row('H',xc(R+'xjtu_cross/r2_bm2_H'),S8),row('V',xc(R+'xjtu_cross/r2_bm2_V'),S8),row('H+V',xc(O+'e1_bm2_HV'),S8)]))
S3=[0,1,2]; rows=[]
for rule,lab in [('H','H rule'),('V','V rule'),('last15','last 15\\%')]:
    for a,an in [('H','H'),('V','V'),('HV','H+V')]:
        d=R+f'xjtu_cross/xc_bm3_{a}' if rule=='H' else O+f'e2_{rule}_xc_bm3_{a}'
        rows.append(row(f'{lab}, {an}',xc(d),S3))
    rows.append('\\mr')
out.append(tab('tab:r3_e2xc','Degraded-phase sensitivity, Mamba-3, condition~2$\\to$3, macro-F1 (\\%) per seed (environment~2). ``H rule\'\': the original arms.',S3,rows[:-1],'Onset rule, input'))
rows=[]
for rule,lab in [('H','H rule'),('V','V rule'),('last15','last 15\\%')]:
    for a,an in [('H','H'),('V','V'),('HV','H+V')]:
        d=R+f'xjtu_lobo/lobo_bm3_{a}' if rule=='H' else O+f'e2_{rule}_lobo_bm3_{a}'
        o=lobo(d)[1]
        rows.append(f'{lab}, {an} & '+' & '.join(f'${o[s][0]:.1f}$ / ${o[s][3]:.1f}$' for s in S3)+' \\\\')
    rows.append('\\mr')
out.append(tab('tab:r3_e2lobo','Degraded-phase sensitivity, Mamba-3, condition~3 LOBO, OR recall (\\%) on test bearings 3\\_1 / 3\\_5 per seed.',S3,rows[:-1],'Onset rule, input'))
# per-bearing cross for E2
rows=[]
for rule,lab in [('H','H rule'),('V','V rule'),('last15','last 15\\%')]:
    for a,an in [('H','H'),('V','V'),('HV','H+V')]:
        d=R+f'xjtu_cross/xc_bm3_{a}' if rule=='H' else O+f'e2_{rule}_xc_bm3_{a}'
        j=json.load(open(d+'/per_bearing.json')); tb=j['test_bearings']; ss=[s for s in j['seeds'] if int(s)<3]
        rows.append(f'{lab}, {an} & '+' & '.join(f"${np.mean([j['seeds'][s]['per_bearing'][b]['acc'] for s in ss])*100:.0f}$" for b in tb)+' \\\\')
    rows.append('\\mr')
n_by={rule:json.load(open((R+'xjtu_cross/xc_bm3_H' if rule=='H' else O+f'e2_{rule}_xc_bm3_H')+'/per_bearing.json')) for rule in ['H','V','last15']}
LABS=[('H','H rule'),('V','V rule'),('last15','last 15\\%')]
def _cnt(r): k=list(n_by[r]['seeds'])[0]; return ', '.join(str(n_by[r]['seeds'][k]['per_bearing'][b]['n']) for b in tb)
cnt='; '.join(lab+': '+_cnt(r) for r,lab in LABS)
out.append("\\begin{table}[htbp]\n\\caption{\\label{tab:r3_e2pb}Degraded-phase sensitivity, Mamba-3, condition~2$\\to$3: accuracy (\\%) per test bearing, mean over seeds 0--2. Test windows per bearing (3\\_1, 3\\_3, 3\\_4, 3\\_5): "+cnt+".}\n\\footnotesize\\begin{indented}\\item[]\n\\begin{tabular}{@{}lcccc}\n\\br\nOnset rule, input & 3\\_1 (OR) & 3\\_3 (IR) & 3\\_4 (IR) & 3\\_5 (OR) \\\\\n\\mr\n"+'\n'.join(rows[:-1])+"\n\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
# onset segments
rows=[]
for r in csv.DictReader(open(os.path.join(ROOT,'audit','onset_audit.csv'))):
    n=int(r['n_files']); l=int(r['onset_last15']); h=int(r['onset_H']); v=int(r['onset_V'])
    pre=max(0,min(h,v)-l)/(n-l)*100
    bn=r['bearing'].replace('_','\\_')
    rows.append(f"{bn} & {r['cls']} & ${n}$ & ${h}$ & ${v}$ & ${l}$ & ${pre:.0f}$ \\\\")
out.append("\\begin{table}[htbp]\n\\caption{\\label{tab:r3_seg}Degraded-phase start (file index) under the three rules, and the share (\\%) of the last-15\\% segment that lies before both signal-defined onsets.}\n\\footnotesize\\begin{indented}\\item[]\n\\begin{tabular}{@{}llccccc}\n\\br\nBearing & Class & Files & H onset & V onset & Last-15\\% start & Pre-onset share \\\\\n\\mr\n"+'\n'.join(rows)+"\n\\br\n\\end{tabular}\n\\end{indented}\n\\end{table}\n")
S5=list(range(5)); rows=[]
for nm,d in [('1D-CNN window gate',O+'e3_cnn_gated'),('1D-CNN token gate',O+'e3t_cnn_tgated'),('Mamba-3 window gate',O+'e3_bm3_gated'),('Mamba-3 token gate',O+'e3t_bm3_tgated')]:
    rows.append(row(nm,xc(d),S5))
    gs=json.load(open(d+'/gate_stats.json'))
    rows.append('\\quad gate SD over windows & '+' & '.join(f"${gs[str(s)]['sd']:.4f}$" for s in S5)+' \\\\')
    if gs[str(0)].get('token_sd_mean'): rows.append('\\quad mean token-level SD & '+' & '.join(f"${gs[str(s)]['token_sd_mean']:.3f}$" for s in S5)+' \\\\')
out.append(tab('tab:r3_e3','Gated fusion, condition~2$\\to$3: macro-F1 (\\%) per seed, and the standard deviation of the channel-2 (V) gate weight over the test windows (window-mean for the token gate) and the mean within-window standard deviation of the token gate. Adaptivity rule fixed in advance: window SD $>0.01$ in every seed.',S5,rows))
rows=[]
for nm,d in [('window gate',O+'e3_lobo_bm3_gated'),('token gate',O+'e3t_lobo_bm3_tgated')]:
    o=lobo(d)[1]; gs=json.load(open(d+'/gate_stats.json'))
    rows.append(f'{nm} & '+' & '.join(f'${o[s][0]:.1f}$ / ${o[s][3]:.1f}$' for s in S3)+' \\\\')
    rows.append('\\quad gate SD (3\\_1 / 3\\_5) & '+' & '.join('$%.4f$ / $%.4f$'%(gs['fold0_seed%d'%s]['sd'],gs['fold3_seed%d'%s]['sd']) for s in S3)+' \\\\')
out.append(tab('tab:r3_e3lobo','Gated fusion, Mamba-3, condition~3 LOBO: OR recall (\\%) on test bearings 3\\_1 / 3\\_5 per seed and gate SD over test windows.',S3,rows))
rows=[]
for snr in [-4,-6,-8]:
    for lab,d in [('FE',R+f'cwru/bm3_FE_snr{snr}'),('DE',O+f'e4_bm3_DE_snr{snr}'),('DE+FE',O+f'e4_bm3_DEFE_snr{snr}')]:
        rows.append(row(f'${snr}$~dB, {lab}',cw(d),S5,'{:.2f}'))
    for rho in ([0.5,0.9] if snr!=-4 else []):
        rows.append(row(f'${snr}$~dB, DE+FE, $\\rho={rho}$',cw(O+f'e5_bm3_DEFE_snr{snr}_rho{rho}'),S5,'{:.2f}'))
    rows.append('\\mr')
out.append(tab('tab:r3_cwru','CWRU, Mamba-3, environment~2: test accuracy (\\%) at the validation-selected epoch per seed. FE-only: arms of supplementary section~S6; all other arms trained for this revision. $\\rho$: correlation of the injected noise between the two channels (no $\\rho$: independent noise).',S5,rows[:-1]))
sec=r'''\section{Additional experiments of this revision: per-seed results}
\label{sec:round3}
The experiments of this section were specified in a written protocol committed
to the repository before any of them was run (\texttt{prereg\_mst\_round3.md}),
with one amendment written after the first gated-fusion result and before the
token-gate runs (main text, section~3.5). All were run in environment~2. The
correlated-noise model draws, for every window, $z_0,z_1\sim\mathcal N(0,1)$
independently (the same draw as the independent AWGN), sets
$z_1'=\rho z_0+\sqrt{1-\rho^2}\,z_1$, and scales each channel to its nominal
SNR; $\rho=0$ reproduces the independent AWGN exactly and single-channel inputs
are unaffected (both unit-tested). Over 100 windows the empirical noise
correlation was $0.002$, $0.501$ and $0.900$ for $\rho=0$, $0.5$ and $0.9$, with
per-channel SNR within $0.02$~dB of the nominal value. For the degraded-phase
rules, the window count of every condition-2 and condition-3 bearing was checked
against the onset audit before training.

'''
s=sec+'\n'.join(out)
open('r3_supp.tex','w').write(s); print(len(s))
