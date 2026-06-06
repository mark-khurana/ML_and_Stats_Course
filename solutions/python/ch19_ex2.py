# =============================================================================
# Chapter 19 - Exercise 2: Publication-Quality Multi-Panel Figure in Python
# Survival analysis of the lung dataset
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.datasets import load_lung
from lifelines.statistics import logrank_test

# --- Publication settings ---
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colour-blind friendly palette
cb_palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

# --- Load and prepare data ---
lung = load_lung()
lung['status'] = lung['status'] - 1  # Convert to 0/1
lung = lung.dropna(subset=['ph.ecog', 'ph.karno', 'wt.loss', 'age', 'sex'])

# =============================================================================
# Create the multi-panel figure
# =============================================================================
fig = plt.figure(figsize=(7.2, 9.0))  # ~183mm x ~230mm
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.35)

# =============================================================================
# Panel A: Kaplan-Meier Curve by Sex
# =============================================================================
ax_a = fig.add_subplot(gs[0, :])  # Top row, full width

kmf = KaplanMeierFitter()

for sex_val, label, color in [(1, 'Male', cb_palette[0]),
                               (2, 'Female', cb_palette[1])]:
    mask = lung['sex'] == sex_val
    kmf.fit(lung.loc[mask, 'T'], lung.loc[mask, 'status'], label=label)
    kmf.plot_survival_function(ax=ax_a, color=color, ci_show=True, ci_alpha=0.15)

# Log-rank test
lr = logrank_test(
    lung.loc[lung['sex'] == 1, 'T'], lung.loc[lung['sex'] == 2, 'T'],
    lung.loc[lung['sex'] == 1, 'status'], lung.loc[lung['sex'] == 2, 'status']
)
ax_a.text(0.7, 0.85, f'Log-rank p = {lr.p_value:.3f}',
          transform=ax_a.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey'))

ax_a.set_xlabel('Time (days)')
ax_a.set_ylabel('Survival Probability')
ax_a.set_title('A', loc='left', fontweight='bold', fontsize=14)
ax_a.set_ylim(0, 1.05)
ax_a.legend(loc='lower left', frameon=True, framealpha=0.9)

# Number at risk
n_at_risk_times = [0, 200, 400, 600, 800, 1000]
risk_text = "At risk:\n"
for sex_val, label in [(1, 'Male'), (2, 'Female')]:
    mask = lung['sex'] == sex_val
    times = lung.loc[mask, 'T']
    events = lung.loc[mask, 'status']
    counts = []
    for t in n_at_risk_times:
        counts.append(str(sum(times >= t)))
    risk_text += f"  {label}: " + "  ".join(counts) + "\n"

# =============================================================================
# Panel B: Forest Plot of Hazard Ratios
# =============================================================================
ax_b = fig.add_subplot(gs[1, 0])

# Fit Cox model
cph = CoxPHFitter()
cox_data = lung[['T', 'status', 'age', 'sex', 'ph.ecog', 'ph.karno', 'wt.loss']].copy()
cph.fit(cox_data, duration_col='T', event_col='status')

# Extract results
summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].copy()
summary_df.columns = ['HR', 'CI_lo', 'CI_hi']

# Rename for display
name_map = {
    'age': 'Age (per year)',
    'sex': 'Sex (Female vs Male)',
    'ph.ecog': 'ECOG PS (per level)',
    'ph.karno': 'Karnofsky (per point)',
    'wt.loss': 'Weight loss (per kg)'
}
summary_df.index = summary_df.index.map(lambda x: name_map.get(x, x))

# Plot forest
y_pos = range(len(summary_df) - 1, -1, -1)
labels = list(summary_df.index)

ax_b.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)

for i, (label, row) in enumerate(summary_df.iterrows()):
    y = list(y_pos)[i]
    ax_b.errorbarh(y, row['HR'], xerr=[[row['HR'] - row['CI_lo']],
                                         [row['CI_hi'] - row['HR']]],
                   fmt='o', color=cb_palette[0], capsize=4, markersize=6,
                   linewidth=1.5)
    # Annotate with HR (CI)
    ax_b.text(max(summary_df['CI_hi']) * 1.15, y,
              f"{row['HR']:.2f} ({row['CI_lo']:.2f}-{row['CI_hi']:.2f})",
              va='center', fontsize=7.5)

ax_b.set_yticks(list(y_pos))
ax_b.set_yticklabels(labels, fontsize=9)
ax_b.set_xlabel('Hazard Ratio (95% CI)')
ax_b.set_xscale('log')
ax_b.set_title('B', loc='left', fontweight='bold', fontsize=14)

# =============================================================================
# Panel C: Calibration Plot for 1-Year Survival
# =============================================================================
ax_c = fig.add_subplot(gs[1, 1])

# Predict survival at 1 year (365 days) using the Cox model
lung_pred = cox_data.copy()
lung_pred['lp'] = cph.predict_partial_hazard(lung_pred)

# Create decile groups based on linear predictor
lung_pred['risk_decile'] = pd.qcut(lung_pred['lp'], q=10, labels=False,
                                    duplicates='drop')

# Predicted and observed 1-year mortality for each decile
cal_results = []
for decile in sorted(lung_pred['risk_decile'].unique()):
    subset = lung_pred[lung_pred['risk_decile'] == decile]

    # Predicted: use Cox model survival function
    predicted_surv = cph.predict_survival_function(subset, times=[365])
    pred_mort = 1 - predicted_surv.mean(axis=1).values[0]

    # Observed: Kaplan-Meier estimate at 365 days
    kmf_cal = KaplanMeierFitter()
    kmf_cal.fit(subset['T'], subset['status'])
    try:
        obs_surv = kmf_cal.predict(365)
        obs_mort = 1 - obs_surv
    except Exception:
        obs_mort = np.nan

    cal_results.append({
        'decile': decile,
        'predicted': pred_mort,
        'observed': obs_mort,
        'n': len(subset)
    })

cal_df = pd.DataFrame(cal_results).dropna()

# Plot calibration
ax_c.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=0.8, label='Perfect calibration')
ax_c.scatter(cal_df['predicted'], cal_df['observed'],
             s=cal_df['n'] * 2, c=cb_palette[0], alpha=0.7,
             edgecolors='darkblue', linewidth=0.5)

# Fit LOESS-like smoothing
from numpy.polynomial import polynomial as P
if len(cal_df) > 3:
    z = np.polyfit(cal_df['predicted'], cal_df['observed'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(cal_df['predicted'].min(), cal_df['predicted'].max(), 100)
    ax_c.plot(x_smooth, p(x_smooth), color=cb_palette[1], linewidth=1.5,
              label='Observed (smoothed)')

ax_c.set_xlabel('Predicted 1-year mortality')
ax_c.set_ylabel('Observed 1-year mortality')
ax_c.set_title('C', loc='left', fontweight='bold', fontsize=14)
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_aspect('equal')
ax_c.legend(loc='lower right', fontsize=8, frameon=True)

# =============================================================================
# Save figure
# =============================================================================
plt.tight_layout()
plt.savefig('figure_survival_analysis.tiff', dpi=300, bbox_inches='tight')
plt.savefig('figure_survival_analysis.pdf', bbox_inches='tight')
plt.show()

print("Multi-panel figure saved as:")
print("  figure_survival_analysis.tiff (300 DPI, raster)")
print("  figure_survival_analysis.pdf (vector)")
print()
print("Specifications:")
print("  Width: ~183 mm (double-column)")
print("  Resolution: 300 DPI")
print("  Colour palette: colour-blind accessible")
print("  Font: Arial")
print()
print("Panel A: Kaplan-Meier survival curves by sex with log-rank p-value")
print("Panel B: Forest plot of hazard ratios from multivariable Cox model")
print("Panel C: Calibration plot for 1-year mortality prediction")
