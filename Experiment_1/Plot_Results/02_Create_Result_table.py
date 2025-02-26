import pandas as pd
from numpy.polynomial.polynomial import Polynomial
import numpy as np


def calculate_stabilization_slope_with_index(labeled_data, f1_scores, slope_threshold=0.001):
    coefs = Polynomial.fit(labeled_data, f1_scores, deg=3).convert().coef
    trendline_values = np.polyval(coefs[::-1], labeled_data)
    trendline_derivative = np.gradient(trendline_values, labeled_data)
    stabilization_index = np.argmax(np.abs(trendline_derivative) < slope_threshold)
    return stabilization_index if stabilization_index < len(labeled_data) else None, coefs

input_csv_path = './02_Active Leanring/active_learning_metrics_detailed_ImageNet_v1.csv'
input_data = pd.read_csv(input_csv_path)

results = []
for (clf, avg_type, batch_size), group in input_data.groupby(['classifier', 'average_type', 'batch']):
    group = group.sort_values('labeled_sample_size').groupby('labeled_sample_size').agg(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0]).reset_index()

    x = (group['labeled_sample_size'].values / 10000) * 100
    f1 = group['f1-score'].values
    precision = group['precision'].values
    recall = group['recall'].values

    f1_max = f1.max()
    p_max = precision.max()
    r_max = recall.max()

    f1_std = f1.std()
    p_std = precision.std()
    r_std = recall.std()

    if len(x) > 2:
        stab_index, coefs = calculate_stabilization_slope_with_index(x, f1, slope_threshold=0.001)
        if stab_index is not None:
            x_stab = x[stab_index]
            f1_stab = f1[stab_index]
            p_stab = precision[stab_index]
            r_stab = recall[stab_index]
        else:
            x_stab = f1_stab = p_stab = r_stab = np.nan

        try:
            a0, a1, a2, a3 = coefs
        except ValueError:
            a0 = a1 = a2 = a3 = 0
    else:
        f1_max = p_max = r_max = f1_std = p_std = r_std = x_stab = f1_stab = p_stab = r_stab = np.nan
        a0 = a1 = a2 = a3 = np.nan

    results.append({
        'Classifier': clf,
        'Average Type': avg_type,
        'Batch Size': batch_size,
        'F1max': f1_max,
        'Pmax': p_max,
        'Rmax': r_max,
        '\u03c3F1': f1_std,
        '\u03c3P': p_std,
        '\u03c3R': r_std,
        'xstab (%)': x_stab,
        'F1stab': f1_stab,
        'Pstab': p_stab,
        'Rstab': r_stab,
        'a0': a0,
        'a1': a1,
        'a2': a2,
        'a3': a3
    })


results_df = pd.DataFrame(results)
grouped_results = []

for (clf, avg_type), group in results_df.groupby(['Classifier', 'Average Type']):
    group_summary = {
        'Classifier': clf,
        'Average Type': avg_type,
        'Batch Size': 'median',
        'F1': group['F1max'].median(),
        'P': group['Pmax'].median(),
        'R': group['Rmax'].median(),
        '\u03c3F1': group['\u03c3F1'].median(),
        '\u03c3P': group['\u03c3P'].median(),
        '\u03c3R': group['\u03c3R'].median(),
        'xstab (%)': group['xstab (%)'].median(),
        'F1stab': group['F1stab'].median(),
        'Pstab': group['Pstab'].median(),
        'Rstab': group['Rstab'].median(),
        'a0': None, 
        'a1': None,
        'a2': None,
        'a3': None
    }

    for _, row in group.iterrows():
        grouped_results.append({
            'Classifier': clf,
            'Average Type': avg_type,
            'Batch Size': row['Batch Size'],
            'F1': row['F1max'],
            'P': row['Pmax'],
            'R': row['Rmax'],
            '\u03c3F1': row['\u03c3F1'],
            '\u03c3P': row['\u03c3P'],
            '\u03c3R': row['\u03c3R'],
            'xstab (%)': row['xstab (%)'],
            'F1stab': row['F1stab'],
            'Pstab': row['Pstab'],
            'Rstab': row['Rstab'],
            'a0': row['a0'],
            'a1': row['a1'],
            'a2': row['a2'],
            'a3': row['a3']
        })

    grouped_results.append(group_summary)

grouped_results_df = pd.DataFrame(grouped_results)

grouped_results_path = './02_Active Leanring/AL_Imagenet_SL/Resuls_table.csv'
grouped_results_df.to_csv(grouped_results_path, index=False)


