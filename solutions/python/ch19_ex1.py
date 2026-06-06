# =============================================================================
# Chapter 19 - Exercise 1: Table 1 in Python
# Using the lung dataset, create a publication-quality Table 1
# =============================================================================

import pandas as pd
import numpy as np

# --- Load data ---
# Use lifelines' lung dataset
from lifelines.datasets import load_lung
lung = load_lung()

# Prepare data
lung['sex_label'] = lung['sex'].map({1: 'Male', 2: 'Female'})
lung['ph.ecog_label'] = lung['ph.ecog'].map({
    0: 'Fully active', 1: 'Restricted', 2: 'Ambulatory',
    3: 'Limited self-care', 4: 'Completely disabled'
})

# --- Part (a): Create Table 1 stratified by sex with SMDs ---
try:
    from tableone import TableOne

    # Define variable types
    columns = ['age', 'ph.ecog', 'ph.karno', 'meal.cal', 'wt.loss']
    categorical = ['ph.ecog']
    nonnormal = []  # all continuous reported as mean (SD)

    # Create Table 1 with SMDs instead of p-values
    table1 = TableOne(
        lung,
        columns=columns,
        categorical=categorical,
        groupby='sex_label',
        nonnormal=nonnormal,
        pval=False,  # No p-values
        smd=True,    # Include SMDs
        rename={
            'age': 'Age, years',
            'ph.ecog': 'ECOG performance status',
            'ph.karno': 'Karnofsky performance score',
            'meal.cal': 'Meal calories, kcal/day',
            'wt.loss': 'Weight loss, kg (last 6 months)'
        },
        missing=True,
        overall=True
    )

    print("=" * 70)
    print("Table 1. Baseline Characteristics of the Study Population")
    print("NCCTG Lung Cancer Dataset, Stratified by Sex")
    print("=" * 70)
    print(table1.tabulate(tablefmt="github"))

    # --- Part (c): Export ---
    # table1.to_excel('table1_lung.xlsx')
    # table1.to_latex('table1_lung.tex')
    # table1.to_html('table1_lung.html')
    print("\nExport options:")
    print("  table1.to_excel('table1_lung.xlsx')")
    print("  table1.to_latex('table1_lung.tex')")
    print("  table1.to_html('table1_lung.html')")

except ImportError:
    print("tableone package not installed. Building Table 1 manually.\n")

    # --- Manual Table 1 construction ---
    def compute_smd_cont(data, var, group_col):
        """Compute SMD for a continuous variable."""
        g1 = data[data[group_col] == 'Male'][var].dropna()
        g2 = data[data[group_col] == 'Female'][var].dropna()
        pooled_sd = np.sqrt((g1.var() + g2.var()) / 2)
        if pooled_sd == 0:
            return 0
        return (g1.mean() - g2.mean()) / pooled_sd

    def summarise_continuous(data, var, group_col):
        """Summarise continuous variable by group."""
        results = {}
        for group in ['Overall'] + list(data[group_col].unique()):
            if group == 'Overall':
                subset = data[var].dropna()
            else:
                subset = data[data[group_col] == group][var].dropna()
            results[group] = f"{subset.mean():.1f} ({subset.std():.1f})"
        return results

    def summarise_categorical(data, var, group_col):
        """Summarise categorical variable by group."""
        results = {}
        categories = sorted(data[var].dropna().unique())
        for group in ['Overall'] + list(data[group_col].unique()):
            if group == 'Overall':
                subset = data[var].dropna()
            else:
                subset = data[data[group_col] == group][var].dropna()
            counts = subset.value_counts()
            total = len(subset)
            group_results = {}
            for cat in categories:
                n = counts.get(cat, 0)
                pct = n / total * 100
                group_results[cat] = f"{n} ({pct:.1f}%)"
            results[group] = group_results
        return results

    # Build table
    cont_vars = {
        'age': 'Age, years',
        'ph.karno': 'Karnofsky performance score',
        'meal.cal': 'Meal calories, kcal/day',
        'wt.loss': 'Weight loss, kg (last 6 months)'
    }

    print(f"{'Variable':<40} {'Overall':>18} {'Male':>18} {'Female':>18} {'SMD':>8}")
    print("-" * 104)

    for var, label in cont_vars.items():
        summ = summarise_continuous(lung, var, 'sex_label')
        smd = compute_smd_cont(lung, var, 'sex_label')
        n_miss = lung[var].isna().sum()
        miss_str = f" [missing: {n_miss}]" if n_miss > 0 else ""
        print(f"{label + miss_str:<40} {summ['Overall']:>18} "
              f"{summ.get('Male', 'N/A'):>18} {summ.get('Female', 'N/A'):>18} "
              f"{abs(smd):>7.3f}")

    # ECOG as categorical
    print(f"\n{'ECOG performance status':<40}")
    ecog_summ = summarise_categorical(lung, 'ph.ecog', 'sex_label')
    for cat in sorted(lung['ph.ecog'].dropna().unique()):
        label_map = {0: '  Fully active', 1: '  Restricted',
                     2: '  Ambulatory', 3: '  Limited self-care',
                     4: '  Completely disabled'}
        label = label_map.get(cat, f'  {cat}')
        overall = ecog_summ['Overall'].get(cat, '0 (0.0%)')
        male = ecog_summ.get('Male', {}).get(cat, '0 (0.0%)')
        female = ecog_summ.get('Female', {}).get(cat, '0 (0.0%)')
        print(f"{label:<40} {overall:>18} {male:>18} {female:>18}")

# --- Part (b): SMD interpretation ---
print("\n" + "=" * 70)
print("Standardised Mean Differences (SMD)")
print("=" * 70)

def compute_smd(data, var, group_col):
    g1 = data[data[group_col] == 'Male'][var].dropna()
    g2 = data[data[group_col] == 'Female'][var].dropna()
    pooled_sd = np.sqrt((g1.var() + g2.var()) / 2)
    if pooled_sd == 0:
        return 0
    return (g1.mean() - g2.mean()) / pooled_sd

for var, label in [('age', 'Age'), ('ph.karno', 'Karnofsky score'),
                   ('meal.cal', 'Meal calories'), ('wt.loss', 'Weight loss')]:
    smd = compute_smd(lung, var, 'sex_label')
    balanced = "Balanced" if abs(smd) < 0.1 else "Imbalanced"
    print(f"  {label:<20}: SMD = {smd:>7.3f}  [{balanced}]")

print("\nNote: SMD < 0.1 indicates good balance between groups.")
print("In observational studies, SMDs are preferred over p-values because")
print("they are independent of sample size, whereas p-values from large")
print("samples can be significant for clinically trivial differences.")
