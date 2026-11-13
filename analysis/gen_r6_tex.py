"""Generate the round-6 LaTeX tables and text numbers from analysis/an_r6_results.json and the raw r6eval files.
Every number in the round-6 tables of the rewritten manuscript comes from this script (no hand edits).
Output: analysis/r6_tex/*.tex and analysis/r6_numbers.json."""
import json, csv, numpy as np
from pathlib import Path
_W = Path(__file__).resolve().parents[1]
_REL = _W / 'results' / 'mst_round6_20261106'
RUN = _REL if (_REL / 'A_m3_H_H').exists() else _W / 'out_r6'   # released results, else local run directory
CFG = RUN / 'cfg_r6' if (RUN / 'cfg_r6').exists() else _W / 'cfg_r6'
ENV_CSV = (RUN / 'envelope' / 'envelope_onset.csv') if (RUN / 'envelope').exists() else _W / 'r6analysis' / 'envelope_onset.csv'
A = Path(__file__).resolve().parent; W = A.parent; O = RUN
J = json.load(open(A / 'an_r6_results.json')); T = A / 'r6_tex'; T.mkdir(exist_ok=True)
N = {}
def pp(x, d=1): return f'{x*100:+.{d}f}'
def ci(r, d=1): return f"[${r['ci'][0]*100:+.{d}f}$, ${r['ci'][1]*100:+.{d}f}$]"
def cnt(r): return f"({r['pos']}/{r['neg']})"
def full(r, d=1): return "\\stk{" + f"${pp(r['mean'], d)}$ {cnt(r)}" + "}{" + ci(r, d) + "}"
def full1(r, d=1): return f"${pp(r['mean'], d)}$ {ci(r, d)} {cnt(r)}"
def ms(v, d=1): v = np.asarray(v) * 100; return f'${v.mean():.{d}f}\\pm{v.std(ddof=1):.{d}f}$'
def mean(v, d=1): return f'${np.mean(v)*100:.{d}f}$'
def put(k, v): N[k] = v
def write(name, s): open(T / f'{name}.tex', 'w').write(s.strip() + '\n')
LN = {'cnn': '1D-CNN', 'm3': 'Mamba-3', 'm2': 'Mamba-2'}; SEEDS = {'cnn': 8, 'm3': 8, 'm2': 5}

# ---- table: condition 2->3, H rule (block A)
rows = []
for l in ['cnn', 'm3', 'm2']:
    a = J['A'][f'{l}_H']
    rows.append(f"{LN[l]} ({SEEDS[l]}) & {ms(a['H']['macro_f1'])} & {ms(a['V']['macro_f1'])} & {ms(a['HV']['macro_f1'])} & "
                f"{full(a['V-H|macro_f1|own'])} & {full(a['HV-H|macro_f1|own'])} & {full(a['HV-V|macro_f1|own'])} \\\\")
    for m in ['macro_f1', 'beq', 'macro_recall']:
        for x in ['V-H', 'HV-H', 'HV-V']:
            for v in ['own', 'common']:
                r = a[f'{x}|{m}|{v}']; put(f'A.{l}.H.{x}.{m}.{v}', dict(mean=pp(r['mean']), ci=f"{r['ci'][0]*100:+.1f} to {r['ci'][1]*100:+.1f}", pos=r['pos'], neg=r['neg'], p=round(r['p'], 4)))
    for arm in ['H', 'V', 'HV']:
        put(f'A.{l}.H.{arm}.f1', f"{np.mean(a[arm]['macro_f1'])*100:.1f}"); put(f'A.{l}.H.{arm}.beq', f"{np.mean(a[arm]['beq'])*100:.1f}")
        put(f'A.{l}.H.{arm}.pb', {b[7:]: round(v, 2) for b, v in a[arm]['per_bearing'].items()})
rowsB = []
for l in ['cnn', 'm3', 'm2']:
    a = J['A'][f'{l}_H']
    rowsB.append(f"{LN[l]} & {mean(a['H']['beq'])} & {mean(a['V']['beq'])} & {mean(a['HV']['beq'])} & {full(a['V-H|beq|own'])} & {full(a['HV-H|beq|own'])} & {full(a['HV-V|beq|own'])} \\\\")
write('tab_xc', r"""
\begin{table}[t]
\caption{\label{tab:xc}XJTU-SY condition~2$\to$3, degraded phase defined on the
horizontal channel (H rule); all arms trained in round~6 (one software
environment, deterministic kernels; seeds 0--7, Mamba-2 seeds 0--4). Panel~A:
macro-F1 (\%), mean $\pm$ SD, and seed-paired differences (pp) with bootstrap
95\% interval and positive/negative seed counts. Panel~B: bearing-equal balanced
accuracy (\%), the mean over the two classes of the mean accuracy of that class's
two test bearings. The V $-$ H differences of panel~B are the primary contrasts
P1--P3 (table~\ref{tab:primary}). Per-bearing accuracies: supplementary
table~\ref{tab:S_r6_bearing}.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}lcccccc}
\br
\multicolumn{7}{@{}l}{A. Macro-F1} \\
\mr
Learner (seeds) & H & V & H+V & V $-$ H & H+V $-$ H & H+V $-$ V \\
\mr
""" + '\n'.join(rows) + r"""
\mr
\multicolumn{7}{@{}l}{B. Bearing-equal balanced accuracy} \\
\mr
Learner & H & V & H+V & V $-$ H & H+V $-$ H & H+V $-$ V \\
\mr
""" + '\n'.join(rowsB) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: onset rules (blocks A, C)
rows = []
RN = {'H': 'H channel (main)', 'V': 'V channel', 'last15': 'last 15\\% of life'}
for l in ['m3', 'cnn', 'm2']:
    first = True
    for rule in ['H', 'V', 'last15']:
        a = J['A'][f'{l}_{rule}']
        lab = f"{LN[l]} ({SEEDS[l]})" if first else ''; first = False
        rows.append(f"{lab} & {RN[rule]} & {mean(a['H']['macro_f1'])} & {mean(a['V']['macro_f1'])} & {mean(a['HV']['macro_f1'])} & "
                    f"${pp(a['V-H|macro_f1|own']['mean'])}$ {cnt(a['V-H|macro_f1|own'])} & ${pp(a['HV-V|macro_f1|own']['mean'])}$ {cnt(a['HV-V|macro_f1|own'])} & "
                    f"${pp(a['V-H|macro_f1|common']['mean'])}$ {cnt(a['V-H|macro_f1|common'])} \\\\")
        for x in ['V-H', 'HV-H', 'HV-V']:
            for v in ['own', 'common']:
                r = a[f'{x}|macro_f1|{v}']; put(f'A.{l}.{rule}.{x}.f1.{v}', dict(mean=pp(r['mean']), ci=f"{r['ci'][0]*100:+.1f} to {r['ci'][1]*100:+.1f}", pos=r['pos'], neg=r['neg'], p=round(r['p'], 4)))
        for arm in ['H', 'V', 'HV']: put(f'A.{l}.{rule}.{arm}.f1', f"{np.mean(a[arm]['macro_f1'])*100:.1f}"); put(f'A.{l}.{rule}.{arm}.f1c', f"{np.mean(a[arm]['common_f1'])*100:.1f}")
    rows.append('\\mr')
rowsC = []
for rule in ['H', 'last15']:
    c = J['C'][rule]
    rowsC.append(f"{RN[rule]} & ${c['H']['f0']*100:.1f}$ / ${c['H']['f3']*100:.1f}$ & ${c['V']['f0']*100:.1f}$ / ${c['V']['f3']*100:.1f}$ & ${c['HV']['f0']*100:.1f}$ / ${c['HV']['f3']*100:.1f}$ & {full(c['HV-H'])} & {full(c['HV-V'])} \\\\")
    for arm in ['H', 'V', 'HV']: put(f'C.{rule}.{arm}.OR', f"{(c[arm]['f0']+c[arm]['f3'])/2*100:.1f}")
    for x in ['HV-H', 'HV-V', 'V-H']: put(f'C.{rule}.{x}', dict(mean=pp(c[x]['mean']), ci=f"{c[x]['ci'][0]*100:+.1f} to {c[x]['ci'][1]*100:+.1f}", pos=c[x]['pos'], neg=c[x]['neg']))
write('tab_onset', r"""
\begin{table}[t]
\caption{\label{tab:onset}Sensitivity to the degraded-phase definition, round~6.
Panel~A: XJTU-SY condition~2$\to$3, macro-F1 (\%) on each rule's own test set
and seed-paired differences (pp) with positive/negative seed counts; last
column: V $-$ H on the common test set (windows every rule labels faulty).
Panel~B: condition~3 LOBO, Mamba-3, seeds 0--4, outer-race recall (\%) on test
bearings 3\_1 / 3\_5 and differences of the OR-fold mean (pp) with bootstrap
95\% interval. Intervals for panel~A: supplementary table~\ref{tab:S_r6_onset}.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llcccccc}
\br
\multicolumn{8}{@{}l}{A. Condition 2$\to$3} \\
\mr
Learner (seeds) & Onset rule & H & V & H+V & V $-$ H & H+V $-$ V & V $-$ H (common) \\
\mr
""" + '\n'.join(rows[:-1]) + r"""
\br
\end{tabular}}
\par\vspace{4pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llllll}
\multicolumn{6}{@{}l}{B. LOBO, OR folds, Mamba-3 (seeds 0--4)} \\
\mr
Onset rule & H & V & H+V & H+V $-$ H & H+V $-$ V \\
\mr
""" + '\n'.join(rowsC) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: decomposition (block B)
rows = []
PN = [('training_segmentation', 'training segmentation'), ('bearing_weighting', 'bearing weighting'), ('sample_size', 'sample size'), ('test_composition', 'test composition')]
for l in ['m3', 'cnn']:
    first = True
    for x, xl in [('V-H', 'V $-$ H'), ('HV-H', 'H+V $-$ H'), ('HV-V', 'H+V $-$ V')]:
        b = J['B'][l][x]
        lab = f"{LN[l]} ({5 if l=='m3' else 8})" if first else ''; first = False
        rows.append(f"{lab} & {xl} & ${pp(b['C_common_Hrule']['mean'])}$ & " + ' & '.join('\\stk{$' + pp(b[k]['mean']) + '$}{' + ci(b[k]) + '}' for k, _ in PN) + f" & ${pp(b['C_own_last15']['mean'])}$ \\\\")
        for k, _ in PN + [('C_common_Hrule', ''), ('C_own_last15', ''), ('C_common_last15', '')]:
            put(f'B.{l}.{x}.{k}', dict(mean=pp(b[k]['mean']), ci=f"{b[k]['ci'][0]*100:+.1f} to {b[k]['ci'][1]*100:+.1f}", pos=b[k]['pos'], neg=b[k]['neg']))
    rows.append('\\mr')
write('tab_decomp', r"""
\begin{table}[t]
\caption{\label{tab:decomp}Decomposition of the change of each contrast between
the H rule and the last-15\% rule (macro-F1, pp; bootstrap 95\% interval over
seeds). Column~3: the contrast under the H rule on the common test set. The
three training-side terms add up to the change on the common test set; test
composition is the difference between the last-15\% contrast on its own test
set (last column) and on the common test set. Definitions:
section~\ref{subsec:onset}.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llcccccc}
\br
Learner & Contrast & H rule & Training & Bearing & Sample & Test & last 15\% \\
(seeds) & & (common) & segmentation & weighting & size & composition & (own) \\
\mr
""" + '\n'.join(rows[:-1]) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: gated (block E)
GS = {n: json.load(open(O / n / 'r6eval' / 'gate_stats.json')) for n in ['E_m3_gated', 'E_m3_tgated', 'E_cnn_gated', 'E_cnn_tgated', 'E_lobo_m3_gated', 'E_lobo_m3_tgated']}
def gsd(n): return max(v['sd'] for v in GS[n].values() if isinstance(v, dict))
def tsd(n):
    t = [v['token_sd_mean'] for v in GS[n].values() if isinstance(v, dict) and v.get('token_sd_mean') is not None]; return np.mean(t) if t else None
rows = []
for l, base in [('cnn', 'cnn'), ('m3', 'm3')]:
    a = J['A'][f'{base}_H']
    rows.append(f"{LN[l]} & concatenation & {ms(a['HV']['macro_f1'])} & --- & {full(a['HV-V|macro_f1|own'])} & & \\\\")
    for g, gl in [('gated', 'window gate'), ('tgated', 'token gate')]:
        e = J['E'][f'E_{l}_{g}']; f1 = [x for x in json.load(open(O / f'E_{l}_{g}' / 'r6eval' / 'summary.json')).values()]
        f1 = [x['own']['macro_f1'] for x in f1]
        rows.append(f" & {gl} & {ms(f1)} & {full(e['gated-HV'])} & {full(e['gated-V'])} & {full(e['adaptive_minus_fixed'])} & ${gsd(f'E_{l}_{g}'):.3f}$ \\\\")
        put(f'E.{l}.{g}', dict(f1=f"{np.mean(f1)*100:.1f}", minus_concat=pp(e['gated-HV']['mean']), minus_concat_ci=f"{e['gated-HV']['ci'][0]*100:+.1f} to {e['gated-HV']['ci'][1]*100:+.1f}",
                               minus_V=pp(e['gated-V']['mean']), minus_V_ci=f"{e['gated-V']['ci'][0]*100:+.1f} to {e['gated-V']['ci'][1]*100:+.1f}", minus_V_cnt=f"{e['gated-V']['pos']}/{e['gated-V']['neg']}",
                               abl=pp(e['adaptive_minus_fixed']['mean']), abl_ci=f"{e['adaptive_minus_fixed']['ci'][0]*100:+.1f} to {e['adaptive_minus_fixed']['ci'][1]*100:+.1f}",
                               abl_cnt=f"{e['adaptive_minus_fixed']['pos']}/{e['adaptive_minus_fixed']['neg']}", functional=e['functional'], gate_sd=round(gsd(f'E_{l}_{g}'), 4),
                               token_sd=None if tsd(f'E_{l}_{g}') is None else round(tsd(f'E_{l}_{g}'), 3)))
    rows.append(f" & V only & {ms(a['V']['macro_f1'])} & & & & \\\\"); rows.append('\\mr')
rowsL = []
c = J['C']['H']
for arm, al in [('HV', 'concatenation (H+V)'), ('H', 'H only'), ('V', 'V only')]:
    rowsL.append(f"{al} & ${(c[arm]['f0']+c[arm]['f3'])/2*100:.1f}$ & ${c[arm]['f0']*100:.0f}$ / ${c[arm]['f3']*100:.0f}$ & & \\\\")
for g, gl in [('gated', 'window gate'), ('tgated', 'token gate')]:
    e = J['E'][f'E_lobo_m3_{g}']; R = json.load(open(O / f'E_lobo_m3_{g}' / 'r6eval' / 'summary.json'))
    f0 = np.mean([list(v['own']['per_bearing'].values())[0]['acc'] for k, v in R.items() if k.startswith('fold0')]); f3 = np.mean([list(v['own']['per_bearing'].values())[0]['acc'] for k, v in R.items() if k.startswith('fold3')])
    rowsL.append(f"{gl} & ${np.mean(e['OR_mean'])*100:.1f}$ & ${f0*100:.0f}$ / ${f3*100:.0f}$ & {full(e['adaptive_minus_fixed'])} & ${gsd(f'E_lobo_m3_{g}'):.3f}$ \\\\")
    put(f'E.lobo.{g}', dict(OR=f"{np.mean(e['OR_mean'])*100:.1f}", abl=pp(e['adaptive_minus_fixed']['mean']), functional=e['functional'],
                            minus_H=pp(e['gated-H']['mean']), minus_V=pp(e['gated-V']['mean']), minus_HV=pp(e['gated-HV']['mean']),
                            minus_H_cnt=f"{e['gated-H']['pos']}/{e['gated-H']['neg']}", minus_V_cnt=f"{e['gated-V']['pos']}/{e['gated-V']['neg']}", minus_HV_cnt=f"{e['gated-HV']['pos']}/{e['gated-HV']['neg']}",
                            gate_sd=round(gsd(f'E_lobo_m3_{g}'), 4), token_sd=None if tsd(f'E_lobo_m3_{g}') is None else round(tsd(f'E_lobo_m3_{g}'), 3)))
write('tab_gated', r"""
\begin{table}[t]
\caption{\label{tab:gated}Gated fusion and fixed-gate ablation, round~6. Panel~A:
XJTU-SY condition~2$\to$3, macro-F1 (\%), seeds 0--7; differences (pp) with
bootstrap 95\% interval and positive/negative seed counts. ``Adaptive $-$
fixed'': the trained model minus the same model with its gate replaced by the
mean gate over the training windows. Last column: largest standard deviation of
the window gate weight over test windows across seeds (window mean for the
token gate). Panel~B: condition~3 LOBO, Mamba-3, seeds 0--4, OR-fold recall
(\%), mean and test bearings 3\_1 / 3\_5. Concatenation and single-sensor rows
are the arms of tables~\ref{tab:xc} and~\ref{tab:onset}.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llccccc}
\br
\multicolumn{7}{@{}l}{A. Condition 2$\to$3} \\
\mr
Learner & Fusion & macro-F1 & $-$ concatenation & $-$ V only & adaptive $-$ fixed & gate SD \\
\mr
""" + '\n'.join(rows[:-1]) + r"""
\br
\end{tabular}}
\par\vspace{4pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}lcccc}
\multicolumn{5}{@{}l}{B. LOBO, OR folds, Mamba-3} \\
\mr
System & OR recall & 3\_1 / 3\_5 & adaptive $-$ fixed & gate SD \\
\mr
""" + '\n'.join(rowsL) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: common-noise controls (block D)
D = J['D']; acc = D['acc']; C = D['contrasts']
def am(k): v = list(acc[k].values()); return f'${np.mean(v)*100:.1f}\\pm{np.std(v, ddof=1)*100:.1f}$'
for k in acc: put(f'D.acc.{k}', f"{np.mean(list(acc[k].values()))*100:.1f}")
for k, r in C.items(): put(f'D.{k}', dict(mean=pp(r['mean']), ci=f"{r['ci'][0]*100:+.1f} to {r['ci'][1]*100:+.1f}", pos=r['pos'], neg=r['neg'], p=round(r['p'], 4)))
rows = [
 r"\multicolumn{4}{@{}l}{A. Input to the learner} \\", r"\mr",
 f"DE only & & {am('D0_DE')} & --- \\\\",
 f"DE+FE (two channels) & $0$ & {am('D0_DEFE_rho0')} & {full(C['gain_rho0'])} \\\\",
 f" & $0.9$ & {am('D0_DEFE_rho0.9')} & {full(C['gain_rho0.9'])} \\\\",
 f"SUM $=(x_\\mathrm{{DE}}+x_\\mathrm{{FE}})/\\sqrt2$ & $0$ & {am('D2_SUM_rho0')} & {full(C['SUM0-DE'])} \\\\",
 f" & $0.9$ & {am('D2_SUM_rho0.9')} & {full(C['SUM0.9-DE'])} \\\\",
 f"DIFF $=(x_\\mathrm{{DE}}-x_\\mathrm{{FE}})/\\sqrt2$ & $0$ & {am('D2_DIFF_rho0')} & {full(C['DIFF0-DE'])} \\\\",
 f" & $0.9$ & {am('D2_DIFF_rho0.9')} & {full(C['P6_DIFF0.9-DE'])} \\\\",
 r"\mr", r"\multicolumn{4}{@{}l}{B. Common component delayed in the fan-end channel ($\rho=0.9$)} \\", r"\mr",
 f"delay 4 samples & $0.9$ & {am('D3_DEFE_rho0.9_d4')} & {full(C['gain_rho0.9_d4'])} \\\\",
 f"delay 64 samples & $0.9$ & {am('D3_DEFE_rho0.9_d64')} & {full(C['gain_rho0.9_d64'])} \\\\",
 r"\mr", r"\multicolumn{4}{@{}l}{C. Other noise realisations (gain DE+FE $-$ DE)} \\", r"\mr",
 f"realisation 1 & $0$ / $0.9$ & & {full(C['gain_rho0_r1'])} / {full(C['gain_rho0.9_r1'])} \\\\",
 f"realisation 2 & $0$ / $0.9$ & & {full(C['gain_rho0_r2'])} / {full(C['gain_rho0.9_r2'])} \\\\",
 r"\mr", r"\multicolumn{4}{@{}l}{D. Trained at one $\rho$, tested at the other (DE+FE; accuracy drop, pp)} \\", r"\mr",
 f"trained $\\rho=0.9$, tested $\\rho=0$ & & & {full(C['mismatch_D0_DEFE_rho0.9_same_minus_awgn_corr0.0'])} \\\\",
 f"trained $\\rho=0$, tested $\\rho=0.9$ & & & {full(C['mismatch_D0_DEFE_rho0_same_minus_awgn_corr0.9'])} \\\\",
]
write('tab_common', r"""
\begin{table}[t]
\caption{\label{tab:common}CWRU, Mamba-3, $-8$~dB, corrected noise
implementation, round~6, seeds 0--4: controls for the correlated-noise gain.
Accuracy (\%), mean $\pm$ SD, and seed-paired differences from the DE-only
system (pp) with bootstrap 95\% interval and positive/negative seed counts;
SUM and DIFF are single-channel inputs formed from the two $z$-scored noisy
channels. Panel~D: accuracy on the training $\rho$ minus accuracy on the other
$\rho$ for the same checkpoints. With five seeds the smallest two-sided exact
$p$ is $0.0625$.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}lccc}
\br
Input & $\rho$ & Accuracy & $-$ DE only \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: controlled coverage (block F)
F = [r for r in J['F'] if not r.get('missing')]
rows = []
for k in [1, 2, 3]:
    u = [r for r in F if r['k'] == k]
    rows.append(f"$k={k}$ & ${np.mean([r['H'] for r in u])*100:.1f}$ & ${np.mean([r['HV'] for r in u])*100:.1f}$ & ${np.mean([r['diff'] for r in u])*100:+.1f}$ & "
                + ' & '.join(f"${np.mean([r['diff'] for r in u if r['test']==t])*100:+.1f}$" for t in ['Bearing1_1', 'Bearing2_2', 'Bearing3_1']) + f" & {sum(r['diff'] < -0.02 for r in u)}/{len(u)} \\\\")
    put(f'F.k{k}', dict(H=f"{np.mean([r['H'] for r in u])*100:.1f}", HV=f"{np.mean([r['HV'] for r in u])*100:.1f}", diff=f"{np.mean([r['diff'] for r in u])*100:+.1f}", n_neg=int(sum(r['diff'] < -0.02 for r in u)), n=len(u)))
p7 = J['primary'][[k for k in J['primary'] if k.startswith('P7')][0]]
put('F.P7', dict(mean=pp(p7['mean']), ci=f"{p7['ci'][0]*100:+.1f} to {p7['ci'][1]*100:+.1f}", p=round(p7['p'], 4), holm=round(p7['holm'], 4)))
write('tab_cov', r"""
\begin{table}[t]
\caption{\label{tab:cov}Controlled outer-race coverage (Mamba-3, H rule,
round~6, seeds 0--1): own-class recall (\%) on the OR test bearing with $k$ OR
training bearings drawn at random from the seven OR bearings other than the
test bearing (three subsets per test bearing and $k$), fixed IR training
bearings (2\_1, 3\_3, 3\_4) and a fixed training-set size (2160 windows) and
number of optimiser steps. Columns 2--4: mean over the nine (test bearing,
subset) units; columns 5--7: mean H+V $-$ H per test bearing; last column:
units with H+V more than 2~pp below H. Per unit: supplementary
table~\ref{tab:S_r6_cov}.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}lccccccc}
\br
 & H & H+V & H+V $-$ H & 1\_1 & 2\_2 & 3\_1 & units below \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- table: primary contrasts
PR = [('P1', 'Mamba-3 V $-$ H, bearing-equal balanced accuracy, H rule', 'V $>$ H', 'pp'),
      ('P2', '1D-CNN V $-$ H, same', 'V $>$ H', 'pp'),
      ('P3', 'Mamba-2 V $-$ H, same', 'V $>$ H', 'pp'),
      ('P4', 'Mamba-3 V $-$ H, macro-F1, common test set, last-15\\% rule', 'H-rule sign ($>0$)', 'pp'),
      ('P5', 'CWRU DE+FE trained $\\rho=0.9$: tested $0.9$ $-$ tested $0$', '$>2$~pp, 5/5 seeds', 'pp'),
      ('P6', 'CWRU DIFF $-$ DE only, $\\rho=0.9$', 'DIFF $>$ DE', 'pp'),
      ('P7', 'Coverage: (H+V $-$ H) at $k=3$ minus at $k=1$', '$>0$', 'pp')]
rows = []
for pid, lab, pred, _ in PR:
    k = [x for x in J['primary'] if x.startswith(pid + ':')][0]; r = J['primary'][k]
    met = {'P4': r['mean'] > 0, 'P5': r['mean'] > 0.02 and r['pos'] == 5}.get(pid, r['ci'][0] > 0)
    n = r['n']; cn = f"({r['pos']}/{r['neg']})" if pid != 'P7' else f"(9 + 9 units, 3 test bearings)"
    rows.append(f"{pid} & {lab} & {pred} & \\stk{{${pp(r['mean'])}$ {cn}}}{{{ci(r)}}} & ${r['p']:.4f}$ & ${r['holm']:.3f}$ & {'yes' if met else 'no'} \\\\")
    put(f'PR.{pid}', dict(mean=pp(r['mean']), ci=f"{r['ci'][0]*100:+.1f} to {r['ci'][1]*100:+.1f}", p=round(r['p'], 4), holm=round(r['holm'], 3), met=bool(met), pos=r['pos'], neg=r['neg']))
write('tab_primary', r"""
\begin{table}[t]
\caption{\label{tab:primary}Primary contrasts fixed before the round-6 runs, with
the predicted direction, the result (pp; bootstrap 95\% interval; seed or unit
counts), the unadjusted two-sided $p$ (exact Wilcoxon over seeds; P7:
permutation of the $k$ labels over the 18 (test bearing, subset) units,
not stratified by test bearing; a stratified exact permutation gives $0.038$,
supplementary section~\ref{sec:S_p7}) and the Holm-adjusted $p$ over the seven
contrasts. No contrast reaches $\alpha=0.05$ after adjustment; with five seeds
(P3, P5, P6) the smallest attainable unadjusted $p$ is $0.0625$. ``Direction as
predicted'' refers to the interval (P5: the stated threshold), not to
significance.}
\footnotesize\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}lp{0.36\linewidth}p{0.13\linewidth}cccc}
\br
 & Contrast & Prediction & Result & $p$ & Holm $p$ & As predicted \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- G
put('G', dict(max_abs_xjtu=max([abs(x) for k in ['G_m3_H_HV_rep', 'G_m2_H_H_rep'] for x in J['G']['diffs'][k]]), max_abs_cwru=max(abs(x) for x in J['G']['diffs']['G_DEFE_rho0.9_rep'])))

# ---- envelope indicator (supplement)
E = list(csv.DictReader(open(ENV_CSV)))
rows = []
for b in dict.fromkeys(r['bearing'] for r in E):
    rr = {r['channel']: r for r in E if r['bearing'] == b}; h = rr['H']
    def io(r): return '---' if r['indicator_onset'] in ('', 'None') else r['indicator_onset']
    def fr(r, rule):
        v = r[f'frac_{rule}_segment_before_indicator']; return '---' if v in ('', 'None') else f"{float(v)*100:.0f}"
    def fboth(rule):
        ons = [int(x['indicator_onset']) for x in rr.values() if x['indicator_onset'] not in ('', 'None')]
        if not ons: return '---'
        r0, n = int(h[f'onset_{rule}']), int(h['n_files'])
        return f"{max(0, min(ons) - r0) / (n - r0) * 100:.0f}"
    rows.append(f"{b.replace('_', chr(92)+'_')} & {h['cls']} & {h['n_files']} & {h['onset_H']} & {h['onset_V']} & {h['onset_last15']} & {io(rr['H'])} / {io(rr['V'])} & "
                + ' & '.join(f"{fr(rr['H'], ru)} / {fr(rr['V'], ru)} / {fboth(ru)}" for ru in ['H', 'V', 'last15']) + ' \\\\')
write('tab_S_envelope', r"""
\begin{table}[htbp]
\caption{\label{tab:S_envelope}Envelope-spectrum fault indicator: a
complementary signal-based check of the degraded-phase rules, computed from the
same H and V recordings, not ground truth for physical fault onset. Indicator:
squared envelope of the 1--10~kHz band-passed signal (4th-order Butterworth,
zero phase); sum of the spectrum peaks within $\pm2\%$ of 1--3$\times$ the
characteristic fault frequency (BPFO for OR, BPFI for IR bearings) divided by
the median spectrum level in 10--500~Hz. Indicator onset: first file whose
5-file rolling mean exceeds the baseline mean $+3$ SD, the baseline being the
first $\min(20,\lfloor n/5\rfloor)$ files; no other threshold was tried. Columns:
file indices of the three detection rules and of the indicator onset on the H /
V channel, and the percentage of each rule's faulty segment that lies before
the H-channel onset / before the V-channel onset / before both onsets. An early
onset on one channel (for example V of Bearing3\_1 at file 121) may be a false
alarm; the table does not decide which. ``---'': the indicator did not cross the
threshold in any file of that bearing (cause not established).}
\footnotesize
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llcccccccc}
\br
Bearing & Class & Files & H rule & V rule & last 15\% & Indicator (H / V) & \% before (H / V / both), H rule & same, V rule & same, last 15\% \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}}
\end{table}""")

# ---- supplement: per-bearing block A and onset intervals and coverage units
rows = []
for l in ['m3', 'cnn', 'm2']:
    for rule in ['H', 'V', 'last15']:
        a = J['A'][f'{l}_{rule}']
        for arm in ['H', 'V', 'HV']:
            pb = a[arm]['per_bearing']
            rows.append(f"{LN[l]} & {RN[rule]} & {arm.replace('HV','H+V')} & " + ' & '.join(f"{pb[b]*100:.0f}" for b in ['Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']) + f" & {np.mean(a[arm]['macro_recall'])*100:.1f} & {np.mean(a[arm]['common_f1'])*100:.1f} \\\\")
        rows.append('\\mr')
write('tab_S_r6_bearing', r"""
\begin{table}[htbp]
\caption{\label{tab:S_r6_bearing}Round~6, XJTU-SY condition~2$\to$3: accuracy (\%) on
each test bearing (own-rule test set, mean over seeds; true class OR for 3\_1
and 3\_5, IR for 3\_3 and 3\_4), macro-recall (\%) and macro-F1 on the common
test set (\%).}
\footnotesize
\begin{tabular}{@{}lllcccccc}
\br
Learner & Onset rule & Input & 3\_1 & 3\_3 & 3\_4 & 3\_5 & macro-recall & macro-F1 (common) \\
\mr
""" + '\n'.join(rows[:-1]) + r"""
\br
\end{tabular}
\end{table}""")
rows = []
for l in ['m3', 'cnn', 'm2']:
    for rule in ['H', 'V', 'last15']:
        a = J['A'][f'{l}_{rule}']
        rows.append(f"{LN[l]} & {RN[rule]} & " + ' & '.join(f"${pp(a[k]['mean'])}$ {ci(a[k])}" for k in ['V-H|macro_f1|own', 'HV-H|macro_f1|own', 'HV-V|macro_f1|own', 'V-H|macro_f1|common']) + ' \\\\')
write('tab_S_r6_onset', r"""
\begin{table}[htbp]
\caption{\label{tab:S_r6_onset}Round~6, condition~2$\to$3: seed-paired macro-F1
differences (pp) with bootstrap 95\% intervals for every learner and onset
rule (main-text table~8).}
\footnotesize
\adjustbox{max width=\linewidth}{%
\begin{tabular}{@{}llcccc}
\br
Learner & Onset rule & V $-$ H & H+V $-$ H & H+V $-$ V & V $-$ H (common test set) \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}}
\end{table}""")
rows = [f"{r['test'][7:].replace('_', chr(92)+'_')} & {r['k']} & {r['subset']} & {', '.join(b[7:].replace('_', chr(92)+'_') for b in r['or_bearings'])} & {r['n_conditions']} & {r['H']*100:.0f} & {r['HV']*100:.0f} & ${r['diff']*100:+.0f}$ \\\\" for r in F]
write('tab_S_r6_cov', r"""
\begin{table}[htbp]
\caption{\label{tab:S_r6_cov}Round~6 controlled coverage, per unit: OR training
bearings, number of distinct operating conditions among them, and own-class
recall (\%) on the test bearing, mean over seeds 0--1.}
\footnotesize
\begin{tabular}{@{}llllcccc}
\br
Test & $k$ & Subset & OR training bearings & Conditions & H & H+V & H+V $-$ H \\
\mr
""" + '\n'.join(rows) + r"""
\br
\end{tabular}
\end{table}""")
json.dump(N, open(A / 'r6_numbers.json', 'w'), indent=1, default=str)
print(len(N), 'numbers;', sorted(p.name for p in T.glob('*.tex')))
