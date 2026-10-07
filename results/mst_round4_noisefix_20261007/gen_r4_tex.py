"""Generate the corrected CWRU LaTeX (main-text section 4.5, tables 9-10, supplementary S9) from an_r4_results.json."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, 'an_r4_results.json')))
A, C = R['arms'], R['contrasts']
def ms(k): return f"${A[k]['mean']:.2f}\\pm{A[k]['sd']:.2f}$"
def m1(k): return f"${A[k]['mean']:.1f}\\pm{A[k]['sd']:.1f}$"
def ci(k, d=1): c = C[k]; return f"${c['mean']:+.{d}f}$ [${c['lo']:+.{d}f}$, ${c['hi']:+.{d}f}$]"
def sg(k): return f"{C[k]['pos']}/{C[k]['neg']}"
def p(k): return f"{C[k]['p']:.3f}"
T = {}
T['tab9'] = '\n'.join(f"{'Mamba-3' if s==-4 else '       '} & ${s}$~dB & {ms(f'FE@{s}')} & {ms(f'DE@{s}')} & {ms(f'DEFE0@{s}')} & {ci(f'DEFE0-DE@{s}')} ({sg(f'DEFE0-DE@{s}')}) \\\\" for s in [-4, -6, -8])
rows = []
for s in [-6, -8]:
    for r, key in [('0', 'DEFE0'), ('0.5', 'DEFE0.5'), ('0.9', 'DEFE0.9')]:
        dg = '---' if r == '0' else f"{ci(f'dG{r}@{s}')} ({sg(f'dG{r}@{s}')})"
        rows.append((f"${s}$~dB" if r == '0' else '') + f" & ${r}$ & {m1(f'DE@{s}') if r=='0' else ''} & {m1(f'{key}@{s}')} & {ci(f'{key}-DE@{s}')} ({sg(f'{key}-DE@{s}')}) & {dg} \\\\")
    if s == -6: rows.append('\\mr')
T['tab10'] = '\n'.join(rows)
# per-seed supplementary table
lines = []
for s in [-8, -6, -4]:
    for k, lab in [('DE', 'DE-only'), ('FE', 'FE-only'), ('DEFE0', 'DE+FE, $\\rho=0$'), ('DEFE0.5', 'DE+FE, $\\rho=0.5$'), ('DEFE0.9', 'DE+FE, $\\rho=0.9$')]:
        if f'{k}@{s}' in A: lines.append(f"${s}$~dB, {lab} & " + ' & '.join(f"${v:.2f}$" for v in A[f'{k}@{s}']['seeds']) + ' \\\\')
    lines.append('\\mr')
T['supp'] = '\n'.join(lines[:-1])
# numbers for prose
N = {k: f"{C[k]['mean']:+.1f}" for k in C}
N.update({k + '_ci': f"[{C[k]['lo']:+.1f}, {C[k]['hi']:+.1f}]" for k in C})
N.update({k + '_p': f"{C[k]['p']:.3f}" for k in C}); N.update({k + '_sg': sg(k) for k in C})
N.update({k + '_m': f"{A[k]['mean']:.1f}" for k in A})
fe = [C[f'FE-DE@{s}']['mean'] for s in [-4, -6, -8]]; N['fe_range'] = f"{-max(fe):.1f}--{-min(fe):.1f}"
N['fe_neg'] = sum(C[f'FE-DE@{s}']['neg'] for s in [-4, -6, -8])
T['N'] = N
json.dump(T, open(os.path.join(D, 'r4_tex.json'), 'w'), indent=1)
print(T['tab9']); print(T['tab10']); print({k: N[k] for k in ['fe_range', 'fe_neg']})
