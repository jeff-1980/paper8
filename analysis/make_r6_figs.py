"""Figures 2 and 3 of the rewritten manuscript from released files (no training).
Figure 2 (forest plot): analysis/fig2_forest_data_r6.csv (condition-2->3 rows = round 6, from an_r6_results.json;
other rows from earlier environment-2 runs, see the 'source' column). Figure 3 (controlled coverage): block F
of analysis/an_r6_results.json. Output: analysis/fig/fig2_forest.{pdf,png}, analysis/fig/fig3_coverage.{pdf,png}."""
import csv, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
A = Path(__file__).resolve().parent; FIG = A / 'fig'; FIG.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlelocation": "left", "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='left')

# ---- figure 2
rows = list(csv.DictReader(open(A / 'fig2_forest_data_r6.csv', encoding='utf-8')))
C = {'H': '#4C72B0', 'V': '#DD8452'}
fig, ax = plt.subplots(figsize=(6.0, 4.6))
groups = list(dict.fromkeys(r['setting'] for r in rows))
lab = {'Condition 2$\\to$3': 'Train condition 2, test condition 3 (round 6)'}
y = 0; yt = []; yl = []; gy = []
for g in groups:
    y += 0.9; gy.append((y, g)); y += 0.6
    for L in dict.fromkeys(r['learner'] for r in rows if r['setting'] == g):
        for k, dy in [('H', -0.17), ('V', 0.17)]:
            rr = [r for r in rows if r['setting'] == g and r['learner'] == L and r['dual_minus'] == k]
            if not rr: continue
            m, lo, hi = float(rr[0]['mean_pp']), float(rr[0]['ci_lo']), float(rr[0]['ci_hi'])
            ax.plot([lo, hi], [y + dy, y + dy], color=C[k], lw=1.8, solid_capstyle='butt'); ax.plot(m, y + dy, 'o', color=C[k], ms=4.5)
        yt.append(y); yl.append(L); y += 0.8
ax.axvline(0, color='0.45', lw=0.8)
for yy, g in gy: ax.text(-58, yy, lab.get(g, g), fontsize=8, fontweight='bold', va='center', ha='left')
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlim(-58, 45); ax.set_ylim(y + 0.1, -0.3)
ax.set_xlabel('Seed-paired difference, dual (H+V) minus single sensor (pp)')
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.legend(handles=[Line2D([], [], color=C['H'], marker='o', ms=4, lw=1.6, label='dual $-$ H only'),
                   Line2D([], [], color=C['V'], marker='o', ms=4, lw=1.6, label='dual $-$ V only')], loc='lower right', frameon=False, fontsize=7)
ax.text(44, -0.25, 'dual better $\\rightarrow$', ha='right', va='bottom', fontsize=7, color='0.35')
ax.text(-1, -0.25, '$\\leftarrow$ single better', ha='right', va='bottom', fontsize=7, color='0.35')
fig.tight_layout(); fig.savefig(FIG / 'fig2_forest.pdf', metadata={'CreationDate': None, 'ModDate': None}); fig.savefig(FIG / 'fig2_forest.png', dpi=200); plt.close(fig)

# ---- figure 3
J = json.load(open(A / 'an_r6_results.json'))
F = [r for r in J['F'] if not r.get('missing')]
TB = ['Bearing1_1', 'Bearing2_2', 'Bearing3_1']
col = {'Bearing1_1': '#4C72B0', 'Bearing2_2': '#55A868', 'Bearing3_1': '#8172B2'}; mk = {'Bearing1_1': 'o', 'Bearing2_2': 's', 'Bearing3_1': '^'}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.32))
rng = np.random.default_rng(0)
ax = axs[0]
for arm, c, dx, lb in [('H', '#4C72B0', -0.12, 'H only'), ('HV', '#C44E52', 0.12, 'H+V')]:
    for k in [1, 2, 3]:
        v = [r[arm] * 100 for r in F if r['k'] == k]
        ax.scatter(np.full(len(v), k + dx) + rng.uniform(-0.04, 0.04, len(v)), v, s=10, color=c, alpha=0.45, lw=0, zorder=2)
    m = [np.mean([r[arm] * 100 for r in F if r['k'] == k]) for k in [1, 2, 3]]
    ax.plot(np.array([1, 2, 3]) + dx, m, '-', color=c, lw=1.6, zorder=3); ax.plot(np.array([1, 2, 3]) + dx, m, 'D', color=c, ms=4.5, mec='white', mew=0.6, zorder=4)
    ax.text(3.25, m[-1], lb, color=c, fontsize=7, va='center')
ax.set_xticks([1, 2, 3]); ax.set_xlim(0.6, 3.75); ax.set_ylim(-3, 105)
ax.set_xlabel('OR training bearings, $k$'); ax.set_ylabel('Recall on OR test bearing (%)')
ax.set_title('Dual recovers as $k$ grows', loc='left', fontsize=8)
ax = axs[1]
for t in TB:
    for k in [1, 2, 3]:
        v = [r['diff'] * 100 for r in F if r['k'] == k and r['test'] == t]
        ax.scatter(np.full(len(v), k) + {'Bearing1_1': -0.14, 'Bearing2_2': 0, 'Bearing3_1': 0.14}[t], v, s=14, color=col[t], marker=mk[t], alpha=0.85, lw=0, zorder=3)
m = [np.mean([r['diff'] * 100 for r in F if r['k'] == k]) for k in [1, 2, 3]]
ax.plot([1, 2, 3], m, '-', color='0.2', lw=1.4, zorder=2); ax.plot([1, 2, 3], m, '_', color='0.2', ms=14, mew=2, zorder=4)
ax.axhline(0, color='0.45', lw=0.8)
ax.set_xticks([1, 2, 3]); ax.set_xlim(0.6, 3.4); ax.set_xlabel('OR training bearings, $k$'); ax.set_ylabel('H+V $-$ H (pp)')
ax.set_title('Dual penalty shrinks with $k$', loc='left', fontsize=8)
ax.legend(handles=[Line2D([], [], color=col[t], marker=mk[t], ls='', ms=4, label='test ' + t.replace('Bearing', '')) for t in TB]
          + [Line2D([], [], color='0.2', lw=1.4, label='mean over 9 units')], frameon=False, fontsize=6, loc='lower right')
for a, l in zip(axs, 'ab'): panel_letter(a, l)
fig.subplots_adjust(wspace=0.42)
fig.savefig(FIG / 'fig3_coverage.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None}); fig.savefig(FIG / 'fig3_coverage.png', dpi=200, bbox_inches='tight')
print('written', sorted(p.name for p in FIG.glob('*')))
