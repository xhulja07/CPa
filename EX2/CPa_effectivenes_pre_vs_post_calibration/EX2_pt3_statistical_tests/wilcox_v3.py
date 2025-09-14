import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
import numpy as np
from math import sqrt

# Load Excel file
file_name_without = 'LA_predict_precision_recall_before and after Platt'
model='LAPredict'
file_name = file_name_without + '.xlsx'
wilcox_alternative = "two-sided"
# Read all sheet names
xl = pd.ExcelFile(file_name)
sheet_names = xl.sheet_names

results = []
print(sheet_names)
for sheet in sheet_names:
    # if sheet != 'LApredict':
    #     continue

    # Read each sheet
    df = pd.read_excel(file_name, sheet_name=sheet)
    print(sheet)
    print(list(df.columns))
    # print(df)
    # Extract required columns
    c2_before = df['Precision']
    c2_before = c2_before.replace('nan', np.nan)
    c2_before = c2_before.dropna()
    c2_after = df['Precision_platt']
    c2_after = c2_after.replace('nan', np.nan)
    c2_after = c2_after.dropna()

    c3_before = df['Recall']
    c3_before = c3_before.replace('nan', np.nan)
    c3_before = c3_before.dropna()
    c3_after = df['Recall_platt']
    c3_after = c3_after.replace('nan', np.nan)
    c3_after = c3_after.dropna()

    # Perform Wilcoxon signed-rank test
    wilcoxon_c2 = wilcoxon(c2_before, c2_after, alternative=wilcox_alternative)
    wilcoxon_c3 = wilcoxon(c3_before, c3_after, alternative=wilcox_alternative)

    # Determine significance
    p_c2 = wilcoxon_c2.pvalue
    significant_c2 = "Yes" if p_c2 < 0.05 else "No"

    p_c3 = wilcoxon_c3.pvalue
    significant_c3 = "Yes" if p_c3 < 0.05 else "No"

    # Calculate Z-value and effect size
    # Note: Compute Z-value using normal approximation formula
    T_c2 = wilcoxon_c2.statistic
    T_c3 = wilcoxon_c3.statistic

    n_c2 = len(c2_before)
    n_c3 = len(c3_before)

    z_c2 = (T_c2 - (n_c2 * (n_c2 + 1) / 4)) / sqrt(n_c2 * (n_c2 + 1) * (2 * n_c2 + 1) / 24)
    z_c3 = (T_c3 - (n_c3 * (n_c3 + 1) / 4)) / sqrt(n_c3 * (n_c3 + 1) * (2 * n_c3 + 1) / 24)

    effect_size_c2 = abs(z_c2) / sqrt(n_c2)
    effect_size_c3 = abs(z_c3) / sqrt(n_c3)

    results.append((
        sheet, p_c2, significant_c2, z_c2, effect_size_c2,
        p_c3, significant_c3, z_c3, effect_size_c3
    ))

# Write the results to a text file
with open(model+'_precision_recall.txt', 'w') as f:
    for (sheet, p_c2, significant_c2, z_c2, effect_size_c2,
         p_c3, significant_c3, z_c3, effect_size_c3
         ) in results:
        f.write(f'Sheet: {sheet}, wilcox alternative: {wilcox_alternative} \n')
        f.write(
            f'Precision: p-value = {p_c2}, Statistically Significant: {significant_c2}, Z-value = {z_c2}, Effect Size = {effect_size_c2}\n')
        f.write(
            f'Recall: p-value = {p_c3}, Statistically Significant: {significant_c3}, Z-value = {z_c3}, Effect Size = {effect_size_c3}\n')
        f.write('\n')