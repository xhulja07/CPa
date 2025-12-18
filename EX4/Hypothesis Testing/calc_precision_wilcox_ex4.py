import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import numpy as np
from math import sqrt

# Load Excel file
file_name_without = 'EX4_stat_test'
file_name = file_name_without + '.xlsx'
wilcox_alternative = "greater"
dataset = "os"
# Read all sheet names
xl = pd.ExcelFile(file_name)
sheet_names = xl.sheet_names

results = []
for sheet in sheet_names:

    # Read each sheet
    df = pd.read_excel(file_name, sheet_name=sheet)
    print(sheet)
    print(list(df.columns))
    # print(df)
    # Extract required columns
    c1_precision = df[dataset + '_precision_C1']
    c1_precision = c1_precision.replace('nan', np.nan)
    c1_precision = c1_precision.dropna()
    c2_precision = df[dataset + '_precision_C2']
    c2_precision = c2_precision.replace('nan', np.nan)
    c2_precision = c2_precision.dropna()
    c3_precision = df[dataset + '_precision_C3']
    c3_precision = c3_precision.replace('nan', np.nan)
    c3_precision = c3_precision.dropna()
    c4_precision = df[dataset + '_precision_C4']
    c4_precision = c4_precision.replace('nan', np.nan)
    c4_precision = c4_precision.dropna()

    # Perform Wilcoxon signed-rank test
    wilcoxon_c1_c2 = wilcoxon(c1_precision, c2_precision, alternative=wilcox_alternative)
    wilcoxon_c2_c3 = wilcoxon(c2_precision, c3_precision, alternative=wilcox_alternative)
    wilcoxon_c3_c4 = wilcoxon(c3_precision, c4_precision, alternative=wilcox_alternative)

    # Determine significance
    pvalue_c1_c2 = wilcoxon_c1_c2.pvalue
    significant_c1_c2 = "Yes" if pvalue_c1_c2 < 0.05 else "No"

    pvalue_c2_c3 = wilcoxon_c2_c3.pvalue
    significant_c2_c3 = "Yes" if pvalue_c2_c3 < 0.05 else "No"

    pvalue_c3_c4 = wilcoxon_c3_c4.pvalue
    significant_c3_c4 = "Yes" if pvalue_c3_c4 < 0.05 else "No"

    # Calculate Z-value and effect size
    # Note: Compute Z-value using normal approximation formula
    statistic_c1_c2 = wilcoxon_c1_c2.statistic
    statistic_c2_c3 = wilcoxon_c2_c3.statistic
    statistic_c3_c4 = wilcoxon_c3_c4.statistic

    n_cX = len(c1_precision)

    z_c1_c2 = (statistic_c1_c2 - (n_cX * (n_cX + 1) / 4)) / sqrt(n_cX * (n_cX + 1) * (2 * n_cX + 1) / 24)
    z_c2_c3 = (statistic_c2_c3 - (n_cX * (n_cX + 1) / 4)) / sqrt(n_cX * (n_cX + 1) * (2 * n_cX + 1) / 24)
    z_c3_c4 = (statistic_c3_c4 - (n_cX * (n_cX + 1) / 4)) / sqrt(n_cX * (n_cX + 1) * (2 * n_cX + 1) / 24)

    effect_size_c1_c2 = abs(z_c1_c2) / sqrt(n_cX)
    effect_size_c2_c3 = abs(z_c2_c3) / sqrt(n_cX)
    effect_size_c3_c4 = abs(z_c3_c4) / sqrt(n_cX)

    results.append((
        sheet,
        pvalue_c1_c2, significant_c1_c2, z_c1_c2, effect_size_c1_c2,
        pvalue_c2_c3, significant_c2_c3, z_c2_c3, effect_size_c2_c3,
        pvalue_c3_c4, significant_c3_c4, z_c3_c4, effect_size_c3_c4
    ))

# Write the results to a text file
with open(dataset + '_' + wilcox_alternative + '_wilcox_precision_ex4.txt', 'w') as f:
    for (sheet,
         pvalue_c1_c2, significant_c1_c2, z_c1_c2, effect_size_c1_c2,
         pvalue_c2_c3, significant_c2_c3, z_c2_c3, effect_size_c2_c3,
         pvalue_c3_c4, significant_c3_c4, z_c3_c4, effect_size_c3_c4
         ) in results:
        f.write(f'Sheet: {sheet}, wilcoxon alternative: {wilcox_alternative} \n')
        f.write(
            f'C1-C2: p-value = {pvalue_c1_c2}, Statistically Significant: {significant_c1_c2}, Z-value = {z_c1_c2}, Effect Size = {effect_size_c1_c2}\n')
        f.write(
            f'C2-C3: p-value = {pvalue_c2_c3}, Statistically Significant: {significant_c2_c3}, Z-value = {z_c2_c3}, Effect Size = {effect_size_c2_c3}\n')
        f.write(
            f'C3-C4: p-value = {pvalue_c3_c4}, Statistically Significant: {significant_c3_c4}, Z-value = {z_c3_c4}, Effect Size = {effect_size_c3_c4}\n')
        f.write('\n')