import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

fixed_line_styles = {
    'resnet18': '-', 'rf_ldp': '--', 'xgb_bow': '--', 
    'rf_hog': '-', 'rf_bow': '--', 'svm_bow': '-'
}
classifier_colors = {
    'resnet18': '#1F77B4', 'rf_ldp': '#FF7F0E', 'xgb_bow': '#2CA02C',
    'rf_hog': '#D62728', 'rf_bow': '#9467BD', 'svm_bow': '#8C564B'
}

def load_partial_results(file_path):
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data

def calculate_stabilization_slope(labeled_data, f1_scores, slope_threshold=0.001):
    coefs = Polynomial.fit(labeled_data, f1_scores, deg=3).convert().coef
    trendline_derivative = np.gradient(np.polyval(coefs[::-1], labeled_data))
    stabilization_point = np.argmax(np.abs(trendline_derivative) < slope_threshold)
    return stabilization_point

def plot_fixed_ci_trendlines_with_x_limit(data, metric, ax):
    added_classifiers = set()
    color_mapping = {clf: classifier_colors.get(clf, '#7F7F7F') for clf in data['classifier'].unique()}

    for clf, clf_data in data.groupby('classifier'):
        for batch, subset in clf_data.groupby('batch'):
            subset = subset.sort_values('labeled_sample_size').groupby('labeled_sample_size').mean(numeric_only=True).reset_index()
            x = (subset['labeled_sample_size'].values / 10000) * 100 
            y = subset[metric].values

            ax.scatter(x, y, color='lightgray', s=3, alpha=0.6)

            if len(x) > 2:
                coefs = Polynomial.fit(x, y, deg=3).convert().coef
                y_fit = np.polyval(coefs[::-1], x)
                residuals = y - y_fit
                ci_lower = y_fit + np.percentile(residuals, 2.5)
                ci_upper = y_fit + np.percentile(residuals, 97.5)

                for xi, ci_low, ci_high in zip(x, ci_lower, ci_upper):
                    ax.vlines(xi, ci_low, ci_high, color='lightgrey', alpha=0.4, linewidth=0.7)

                color = color_mapping.get(clf, '#7F7F7F')
                x_trend = np.linspace(x.min(), x.max(), 200)
                ax.plot(x_trend, np.polyval(coefs[::-1], x_trend),
                        color=color, linestyle=fixed_line_styles.get(clf, '-'), linewidth=0.8)

                stabilization_point = calculate_stabilization_slope(x, y)
                stabilization_x = x[stabilization_point]
                stabilization_y = y[stabilization_point]
                ax.hlines(stabilization_y, stabilization_x - 2, stabilization_x + 2,
                          color='black', linestyle='--', linewidth=0.8, alpha=0.8)

            if clf not in added_classifiers:
                ax.plot([], [], color=color, linestyle=fixed_line_styles.get(clf, '-'), label=f"{clf}")
                added_classifiers.add(clf)

    ax.set_xlabel('Labeled Data (%)')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)  
    ax.grid(True)


def f1_plot(file_path, dataset_name, output_dir):

    data = pd.read_csv(file_path)

    average_types = data['average_type'].unique()

    for average_type in average_types:
        filtered_data = data[data['average_type'] == average_type]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fixed_ci_trendlines_with_x_limit(filtered_data, 'f1-score', ax)

        ax.set_title(f'F1-Score vs. Labeled Data for {dataset_name} ({average_type})')
        ax.set_ylabel('F1-Score')
        ax.set_xlabel('Labeled Data (%)')
        ax.legend(title="Classifier", loc='upper right')
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{average_type}_F1_score.png')
        plt.savefig(output_path)
        plt.close(fig)


def plot_combined_density_scatter_no_negative_axes(data, dataset_name, output_dir):
    avg_types = data['average_type'].unique()
    colors = sns.color_palette("tab10", len(data['classifier'].unique()))

    for avg_type in avg_types:
        avg_type_data = data[data['average_type'] == avg_type]

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, classifier in enumerate(avg_type_data['classifier'].unique()):
            classifier_data = avg_type_data[avg_type_data['classifier'] == classifier]

            ax.scatter(
                classifier_data['recall'],
                classifier_data['precision'],
                s=20,
                label=classifier,
                alpha=0.8,
                color=colors[i]
            )

            sns.kdeplot(
                x=classifier_data['recall'],
                y=classifier_data['precision'],
                color=colors[i],
                fill=True,
                alpha=0.2,
                levels=10,
                ax=ax
            )

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Combined Density Scatter Plot for {dataset_name} ({avg_type})')
        ax.legend(title='Classifier', loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{avg_type}_combined_density.png')
        plt.savefig(output_path)
        plt.close(fig)

"""
def plot_dynamic_grid(folder_path):
    data = load_partial_results(folder_path)

    avg_types = data['average_type'].unique()
    metrics = ['f1-score', 'precision', 'recall']
    titles = ['F1-Score', 'Precision', 'Recall']

    num_rows = len(avg_types)
    num_cols = len(metrics)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=True, sharey=True)

    axes = axes.flatten() if num_rows > 1 else [axes]

    for i, avg_type in enumerate(avg_types):
        avg_data = data[data['average_type'] == avg_type]
        for j, metric in enumerate(metrics):
            ax = axes[i * num_cols + j]
            plot_fixed_ci_trendlines(avg_data, metric, ax)

            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel(f"{avg_type.replace('_', ' ').title()}")

    handles = [plt.Line2D([0], [0], color=classifier_colors.get(clf, '#7F7F7F'),
                          linestyle=fixed_line_styles.get(clf, '-'), linewidth=1.5, label=clf) 
               for clf in data['classifier'].unique()]
    fig.legend(handles=handles, title='Classifier', loc='lower center', ncol=6, fontsize='small', bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
"""
def process_dataset(file_path, output_dir):
    data = load_partial_results(file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f1_plot(file_path, "COCO", output_dir= output_dir)
    plot_combined_density_scatter_no_negative_axes(data, "COCO", output_dir)

file_path = './02_Active Leanring/AL_Results_COCO_updated.csv'
output_dir = './02_Active Leanring/AL_Results_Split_Reports_COCO'
process_dataset(file_path, output_dir)
