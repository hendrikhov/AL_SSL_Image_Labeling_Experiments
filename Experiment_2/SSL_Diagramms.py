import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adjustText import adjust_text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import combinations

def generate_ssl_diagram(file1, file2, file3, file4, dataset_name, data_point, save_path, exclude_strategies=None):

    data_1000 = pd.read_csv(file1)
    data_500 = pd.read_csv(file2)
    data_2000 = pd.read_csv(file3)
    data_resnet = pd.read_csv(file4)

    data_1000['Batch Size'] = 1000
    data_500['Batch Size'] = 500
    data_2000['Batch Size'] = 2000
    data_resnet.rename(columns={"Strategy": "Model", "Model": "Strategy"}, inplace=True)
    data_resnet['Batch Size'] = data_resnet['Feature Size'].apply(
        lambda x: min([500, 1000, 2000], key=lambda y: abs(y - x))
    )

    data_resnet['Feature'] = 'BoVW'

    combined_data = pd.concat([data_1000, data_500, data_2000, data_resnet], ignore_index=True)
    grouped_data = combined_data.groupby(['Model', 'Strategy', 'Feature', 'Batch Size'])[data_point].mean().reset_index()
    grouped_data = grouped_data[grouped_data[data_point] > 0]

    if exclude_strategies:
        grouped_data = grouped_data[~grouped_data['Strategy'].isin(exclude_strategies)]

    feature_order = {"Raw": 0, "BoVW": 1, "HOG": 2, "LBP": 3}
    grouped_data['Feature_Order'] = grouped_data['Feature'].map(feature_order)
    grouped_data = grouped_data.sort_values(by=['Model', 'Strategy', 'Feature_Order', 'Batch Size'])

    resnet_data = grouped_data[grouped_data['Model'] == 'ResNet']
    grouped_data = grouped_data[grouped_data['Model'] != 'ResNet']

    x_labels = []
    x_positions = []
    model_positions = {}

    current_x = 0
    for model in grouped_data['Model'].unique():
        model_data = grouped_data[grouped_data['Model'] == model]
        model_start = current_x

        for _, row in model_data.iterrows():
            x_positions.append(current_x)
            current_x += 1

        model_positions[model] = (model_start + current_x - 1) / 2
        current_x += 0.3 

    # Apply feature order to ResNet data
    resnet_data = resnet_data.sort_values(by=['Strategy', 'Feature_Order'])
    model_start = current_x
    for _, row in resnet_data.iterrows():
        x_positions.append(current_x)
        current_x += 1
    model_positions['ResNet'] = (model_start + current_x - 1) / 2

    x_positions = np.array(x_positions)
    x_start = x_positions.min() - 0.5
    x_end = x_positions.max() + 0.5

    strategies = grouped_data['Strategy'].unique()
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(strategies)]
    strategy_to_color = {strategy: colors[idx] for idx, strategy in enumerate(strategies)}

    features = ['Raw', 'BoVW', 'HOG', 'LBP']
    feature_patterns = {"Raw": '/', "BoVW": '', "HOG": '...', "LBP": 'o'}

    if not set(grouped_data['Feature'].unique()).issubset(set(feature_patterns.keys())):
        print("Warning: Feature names in data do not match feature_patterns keys.")
        print("Features in data:", grouped_data['Feature'].unique())
        print("Keys in feature_patterns:", feature_patterns.keys())

    fig, ax = plt.subplots(figsize=(20, 10))

    for pos, (_, row) in zip(x_positions, grouped_data.iterrows()):
        bar_color = strategy_to_color[row['Strategy']]
        bar_pattern = feature_patterns[row['Feature']]
        bar = ax.bar(
            pos, row[data_point], 0.8, color=bar_color, alpha=1.0 if row['Batch Size'] == 500 else 0.66 if row['Batch Size'] == 1000 else 0.33, edgecolor='black', hatch=bar_pattern
        )

    for pos, (_, row) in zip(x_positions[len(x_positions) - len(resnet_data):], resnet_data.iterrows()):
        bar_color = 'grey'  # ResNet is always grey
        bar_pattern = feature_patterns[row['Feature']]
        bar = ax.bar(
            pos, row[data_point], 0.8, color=bar_color, alpha=1.0 if row['Batch Size'] == 500 else 0.66 if row['Batch Size'] == 1000 else 0.33, edgecolor='black', hatch=bar_pattern
        )
      
    ax.set_xlim(x_start, x_end)
    ax.set_xticks(list(model_positions.values()))
    ax.set_xticklabels(list(model_positions.keys()), rotation=0, ha='center', fontsize=12)
    ax.set_title(f'{data_point} by Model, Strategy, Feature, and Batch Size for {dataset_name}', fontsize=16)
    ax.set_ylabel(data_point, fontsize=12)
    if data_point in ["Pseudo-Labeled Samples", "Training Time (s)", "Validation Time (s)","Log-Loss",]:
        max_value = max(grouped_data[data_point].max(), resnet_data[data_point].max())
        ax.set_ylim(0, max_value * 1.1)
    else:
        max_value = grouped_data[data_point].max()
        ax.set_ylim(0, 1)

    if max_value < 1.0:
        ax.axhline(y=max_value, color='black', linestyle='-', linewidth=1.2)
        ax.text(x_start, max_value, f'{max_value:.2f}', ha='left', va='bottom', fontsize=12, color='black')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    strategy_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=strategy_to_color[strategy], edgecolor='black', label=strategy)
        for strategy in strategies
    ]
    feature_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch=feature_patterns[feature], label=feature)
        for feature in features
    ]
    batch_size_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='grey', alpha=1.0 if batch_size == 500 else 0.66 if batch_size == 1000 else 0.33, edgecolor='black', label=f'Batch {batch_size}')
        for batch_size in [500, 1000, 2000]
    ]

    strategy_legend = ax.legend(
        strategy_handles, [handle.get_label() for handle in strategy_handles],
        title='Strategy', fontsize=12, loc='upper left'
    )
    feature_legend = ax.legend(
        feature_handles, [handle.get_label() for handle in feature_handles],
        title='Feature', fontsize=12, loc='upper center'
    )
    
    batch_size_legend = ax.legend(
        batch_size_handles, [handle.get_label() for handle in batch_size_handles],
        title='Batch Size', fontsize=12, loc='upper right'
    )
    ax.add_artist(strategy_legend)
    ax.add_artist(feature_legend)

    plt.tight_layout()

    file_name = f"SSL_{dataset_name}_{data_point.replace(' ', '_')}.png"
    save_file_path = f"{save_path}/{file_name}"
    plt.savefig(save_file_path)
    plt.close()

    print(f"Diagram saved at: {save_file_path}")
  
def generate_recall_vs_precision_scatter(file1, file2, file3, file4, dataset_name, save_path):

    data_1000 = pd.read_csv(file1)
    data_500 = pd.read_csv(file2)
    data_2000 = pd.read_csv(file3)
    data_resnet = pd.read_csv(file4)

    data_1000['Batch Size'] = 1000
    data_500['Batch Size'] = 500
    data_2000['Batch Size'] = 2000
    if 'Feature Size' in data_resnet.columns:
        data_resnet.rename(columns={"Strategy": "Model", "Model": "Strategy"}, inplace=True)
        data_resnet['Batch Size'] = data_resnet['Feature Size'].apply(
            lambda x: min([500, 1000, 2000], key=lambda y: abs(y - x))
        )

    # Add empty feature for ResNet
    data_resnet['Feature'] = ''

    combined_data = pd.concat([data_1000, data_500, data_2000, data_resnet], ignore_index=True)

    strategy_shortcuts = {
        "Binary Relevance": "BR",
        "Classifier Chains": "CC",
        "Label Powerset": "LP",
        "Self-Training" : "ST",
        "Pseudo-Labeling" : "PL",
        "Graph-Based Propagation" : "GBP",
        "Co-Training" : "CT",
        "Multi-View Learning" : "MVT"
    }

    feature_shortcuts = {
        "Raw": "RAW",
        "BoVW": "BVW",
        "HOG": "HOG",
        "LBP": "LBP",
        "": ""
    }

    batch_size_shortcuts = {
        500: "5",
        1000: "10",
        2000: "20"
    }

    model_shortcuts = {
        "CatBoost": "CB",
        "LightGBM": "LGBM",
        "Random Forest": "RF",
        "SVM": "SVM",
        "XGBoost": "XGB",
        "kNN": "kNN",
        "ExtraTrees" : "ET",
        "ResNet": "ResNet"
    }
    models = [key for key in model_shortcuts if key != "ResNet"]
    model_combinations = { " & ".join(combo): " & ".join(model_shortcuts[m] for m in combo) for i in range(2, len(models) + 1) for combo in combinations(models, i)}
    model_shortcuts.update(model_combinations)

    recall_precision_types = [
        #("Recall (Micro)", "Precision (Micro)"),
        ("Recall (Macro)", "Precision (Macro)"),
        ("Recall (Weighted)", "Precision (Weighted)"),
        #("Recall (Samples)", "Precision (Samples)")
    ]

    for recall_type, precision_type in recall_precision_types:
        recall_precision_data = combined_data.groupby(['Model', 'Strategy', 'Feature', 'Batch Size']).agg(
            {recall_type: 'mean', precision_type: 'mean'}).reset_index()
        
        recall_precision_data['F1-Score'] = 2 * (recall_precision_data[recall_type] * recall_precision_data[precision_type]) / (recall_precision_data[recall_type] + recall_precision_data[precision_type])

        top_precision = recall_precision_data.nlargest(3, precision_type)
        top_recall = recall_precision_data[recall_precision_data['Model'] != 'ResNet'].nlargest(3, recall_type)
        top_f1 = recall_precision_data.nlargest(3, 'F1-Score')

        bold_points = pd.concat([top_precision, top_recall, top_f1]).drop_duplicates()

        cluster_x = recall_precision_data[recall_type]
        cluster_y = recall_precision_data[precision_type]
        x_min, x_max = np.percentile(cluster_x, [25, 75])
        y_min, y_max = np.percentile(cluster_y, [5, 95])

        fig, ax = plt.subplots(figsize=(10, 6))
        texts_main = []
        texts_inset_actual = []

        plotted_labels = set()

        for _, row in recall_precision_data.iterrows():
            in_zoom_range = x_min <= row[recall_type] <= x_max and y_min <= row[precision_type] <= y_max
            color = 'orange' if in_zoom_range else 'blue'
            feature = feature_shortcuts.get(row['Feature'], row['Feature'])
            batch_size = batch_size_shortcuts.get(row['Batch Size'], row['Batch Size'])
            model = model_shortcuts.get(row['Model'], row['Model'][:3])
            strategy = strategy_shortcuts.get(row['Strategy'], row['Strategy'])

            combined_label = f"{model}-{strategy}-{feature}"
            full_label = f"{combined_label}-{batch_size}"

            close_points = recall_precision_data[
                (np.sqrt((recall_precision_data[recall_type] - row[recall_type]) ** 2 +
                        (recall_precision_data[precision_type] - row[precision_type]) ** 2) < 0.15) &
                (recall_precision_data['Model'] == row['Model']) &
                (recall_precision_data['Strategy'] == row['Strategy']) &
                (recall_precision_data['Feature'] == row['Feature'])
            ]
            
            label_fontweight = 'bold' if ((bold_points['Model'] == row['Model']) &
                              (bold_points['Strategy'] == row['Strategy']) &
                              (bold_points['Feature'] == row['Feature']) &
                              (bold_points['Batch Size'] == row['Batch Size'])).any() or \
                             ((bold_points['Model'].isin(close_points['Model'])) &
                              (bold_points['Strategy'].isin(close_points['Strategy'])) &
                              (bold_points['Feature'].isin(close_points['Feature']))).any() else 'normal'

            if in_zoom_range:
                texts_inset_actual.append((row[recall_type], row[precision_type], full_label))
                ax.scatter(
                    row[recall_type],
                    row[precision_type],
                    marker='o',
                    s=10,
                    color='orange',
                    alpha=0.6
                )
            else:
                if row['Model'] == 'ResNet':
                    ax.scatter(
                        row[recall_type],
                        row[precision_type],
                        marker='x',
                        s=30,
                        color='red'
                    )
                    texts_main.append(ax.text(
                        row[recall_type],
                        row[precision_type],
                        f"{model}-{batch_size}",
                        fontsize=8
                    ))
                else:
                    if close_points.shape[0] > 1:
                        label_to_display = combined_label
                    else:
                        label_to_display = full_label
                    ax.scatter(
                        row[recall_type],
                        row[precision_type],
                        marker='o',
                        s=10,
                        color=color
                    )
                    if close_points.shape[0] > 1 and combined_label in plotted_labels:
                        continue
                    texts_main.append(ax.text(
                        row[recall_type],
                        row[precision_type],
                        label_to_display,
                        fontsize=8,
                        fontweight=label_fontweight
                    ))

            plotted_labels.add(combined_label)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        adjust_text(
            texts_main,
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5)
        )


        ax_inset = create_zoomed_view(ax, recall_precision_data, x_min, x_max, y_min, y_max, texts_inset_actual, recall_type, precision_type, batch_size, strategy_shortcuts , feature_shortcuts,  batch_size_shortcuts , model_shortcuts)

        ax.set_title(f'{recall_type} vs {precision_type} for {dataset_name}', fontsize=16)
        ax.set_xlabel(recall_type, fontsize=14)
        ax.set_ylabel(precision_type, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        file_name = f"SSL_{dataset_name}_{recall_type.replace(' ', '_')}_vs_{precision_type.replace(' ', '_')}.png"
        save_file_path = f"{save_path}/{file_name}"
        plt.savefig(save_file_path)
        plt.close()

        print(f"Scatterplot saved at: {save_file_path}")


def create_zoomed_view(ax, recall_precision_data, x_min, x_max, y_min, y_max, texts_inset_actual, recall_type, precision_type, batch_size, strategy_shortcuts , feature_shortcuts,  batch_size_shortcuts , model_shortcuts):

    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=1)
    ax_inset.set_xlim(x_min * 0.8, x_max * 1.2)
    ax_inset.set_ylim(y_min * 0.8, y_max * 1.2)

    inset_texts = []
    plotted_labels = set()

    recall_precision_data['Feature'] = recall_precision_data['Feature'].map(feature_shortcuts).fillna(recall_precision_data['Feature'])
    recall_precision_data['Batch Size'] = recall_precision_data['Batch Size'].map(batch_size_shortcuts).fillna(recall_precision_data['Batch Size'])
    recall_precision_data['Model'] = recall_precision_data['Model'].map(model_shortcuts).fillna(recall_precision_data['Model'])
    recall_precision_data['Strategy'] = recall_precision_data['Strategy'].map(strategy_shortcuts).fillna(recall_precision_data['Strategy'])

    for x, y, label in texts_inset_actual:
        # Extract components from label
        label_parts = label.split('-')
        model = label_parts[0]
        strategy = label_parts[1]
        feature = label_parts[2] if len(label_parts) > 2 else ''

        if not all(col in recall_precision_data.columns for col in [recall_type, precision_type, 'Model', 'Strategy', 'Feature']):
            raise ValueError("Missing required columns in recall_precision_data.")

        distances = np.sqrt((recall_precision_data[recall_type] - x) ** 2 + (recall_precision_data[precision_type] - y) ** 2)
        is_close = distances < 0.15
        is_same_model = recall_precision_data['Model'] == model
        is_same_strategy = recall_precision_data['Strategy'] == strategy
        is_same_feature = recall_precision_data['Feature'] == feature

        close_points = recall_precision_data[is_close & is_same_model & is_same_strategy & is_same_feature]
      
        combined_label = f"{model}-{strategy}-{feature}"
        full_label = f"{combined_label}-{batch_size}"
        
        if close_points.shape[0] > 1:
            label_to_display = combined_label
        else:
            label_to_display = full_label
          
        ax_inset.scatter(x, y, marker='o', s=10, color='blue')
        if close_points.shape[0] > 1 and combined_label in plotted_labels:
            continue
        inset_texts.append(ax_inset.text(x, y, label_to_display, fontsize=8))
        plotted_labels.add(label_to_display)

    adjust_text(
        inset_texts,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
        ax=ax_inset
    )

    ax_inset.grid(True, linestyle='--', alpha=0.5)
    ax_inset.xaxis.tick_top()
    ax_inset.xaxis.set_tick_params(pad=0)
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_tick_params(pad=0)
    ax_inset.set_title("", fontsize=10)
    ax_inset.text(
        1.0, -0.1, "Zoomed View", fontsize=10, ha="right", transform=ax_inset.transAxes
    )

    return ax_inset

def create_combined_heatmap(csv_file, data_point, dataset_name, save_path):

    data = pd.read_csv(csv_file)

    data = data[data['Strategy'].isin(['Co-Training', 'Multi-View Learning'])]

    unique_features = data['Feature'].unique()
    unique_models = sorted(
        set(model for models in data['Model'].unique() for model in models.split(' & '))
    )
    n_models = len(unique_models)

    co_training_matrices = {feature: np.full((n_models, n_models), np.nan) for feature in unique_features}
    multi_view_matrices = {feature: np.full((n_models, n_models), np.nan) for feature in unique_features}

    for _, row in data.iterrows():
        model1, model2 = row['Model'].split(' & ')
        feature = row['Feature']
        strategy = row['Strategy']
        value = row[data_point]
        i, j = unique_models.index(model1), unique_models.index(model2)

        if strategy == 'Co-Training' and feature in co_training_matrices:
            if i > j:
                i, j = j, i  # Swap to ensure upper triangle consistency
            co_training_matrices[feature][i, j] = value
        elif strategy == 'Multi-View Learning' and feature in multi_view_matrices:
            if i < j:
                i, j = j, i  # Swap to ensure lower triangle consistency
            multi_view_matrices[feature][i, j] = value

    global_min = np.nanmin([np.nanmin(matrix) for matrix in co_training_matrices.values()] +
                           [np.nanmin(matrix) for matrix in multi_view_matrices.values()])
    global_max = np.nanmax([np.nanmax(matrix) for matrix in co_training_matrices.values()] +
                           [np.nanmax(matrix) for matrix in multi_view_matrices.values()])

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.matshow(np.zeros((n_models, n_models)), cmap="Greys", alpha=0.1)

    # Overlay Co-Training (upper triangle)
    for feature, quadrant_matrix in co_training_matrices.items():
        for i in range(n_models):
            for j in range(n_models):
                if i < j:  # Upper triangle
                    value = quadrant_matrix[i, j]
                    if not np.isnan(value):
                        if feature == "Raw":
                            x_offset, y_offset = -0.3, -0.3
                        elif feature == "BoVW":
                            x_offset, y_offset = 0.1, -0.3
                        elif feature == "HOG":
                            x_offset, y_offset = -0.3, 0.1
                        elif feature == "LBP":
                            x_offset, y_offset = 0.1, 0.1
                        color = plt.cm.Reds((value - global_min) / (global_max - global_min))
                        rect = plt.Rectangle(
                            (j + x_offset - 0.1, i + y_offset - 0.1), 0.4, 0.4,
                            facecolor=color, edgecolor="black", linewidth=0.5
                        )
                        ax.add_patch(rect)
                        ax.text(j + x_offset + 0.1, i + y_offset + 0.1, f"{value:.2f}",
                                va="center", ha="center", fontsize=12)

    # Overlay Multi-View Learning (lower triangle)
    for feature, quadrant_matrix in multi_view_matrices.items():
        for i in range(n_models):
            for j in range(n_models):
                if i > j:  # Lower triangle
                    value = quadrant_matrix[i, j]
                    if not np.isnan(value):
                        if feature == "Raw":
                            x_offset, y_offset = -0.3, -0.3
                        elif feature == "BoVW":
                            x_offset, y_offset = 0.1, -0.3
                        elif feature == "HOG":
                            x_offset, y_offset = -0.3, 0.1
                        elif feature == "LBP":
                            x_offset, y_offset = 0.1, 0.1
                        color = plt.cm.Reds((value - global_min) / (global_max - global_min))
                        rect = plt.Rectangle(
                            (j + x_offset - 0.1, i + y_offset - 0.1), 0.4, 0.4,
                            facecolor=color, edgecolor="black", linewidth=0.5
                        )
                        ax.add_patch(rect)
                        ax.text(j + x_offset + 0.1, i + y_offset + 0.1, f"{value:.2f}",
                                va="center", ha="center", fontsize=12)

    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(unique_models, fontsize=12)
    ax.set_yticklabels(unique_models, fontsize=12)
    ax.set_title(f"Co-Training (Upper Triangle) and Multi-View Learning (Lower Triangle) for {dataset_name}", fontsize=14, pad=35)
    ax.text(0.5, 1.1, f"{data_point}", fontsize=16, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

    # Ledgend - Feature
    legend_x = n_models - 1.5  
    legend_y = n_models - 1.5  
    background = plt.Rectangle(
        (legend_x, legend_y), 1, 1, color="lightgrey", zorder=0, transform=ax.transData
    )
    ax.add_patch(background)
    for feature, (x_offset, y_offset) in zip(
        ["Raw", "BoVW", "HOG", "LBP"],
        [(0.1, 0.1), (0.5, 0.1), (0.1, 0.5), (0.5, 0.5)]  
    ):
        rect = plt.Rectangle(
            (legend_x + x_offset, legend_y + y_offset), 0.4, 0.4, 
            facecolor="white", edgecolor="black", linewidth=1, transform=ax.transData
        )
        ax.add_patch(rect)
        ax.text(
            legend_x + x_offset + 0.2, legend_y + y_offset + 0.2, feature, 
            fontsize=12, ha="center", va="center", transform=ax.transData
        )
    ax.text(
        legend_x + 0.5, legend_y + 0.97, "Features", fontsize=12, fontweight="bold",
        ha="center", va="center", transform=ax.transData
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=data_point)

    plt.tight_layout()
        
    file_name = f"SSL_Heatmap_{dataset_name}_{data_point.replace(' ', '_')}.png"
    save_file_path = f"{save_path}/{file_name}"
    plt.savefig(save_file_path)
    plt.close()

    print(f"Diagram saved at: {save_file_path}")




if __name__ == "__main__":
    file1000 = '03_ SSL/ImageNet/ssl_detailed_results_ImageNet_1000.csv'
    file500 = '03_ SSL/ImageNet/ssl_detailed_results_ImageNet_500.csv'
    file2000 = '03_ SSL/ImageNet/ssl_detailed_results_ImageNet_2000.csv'
    fileResNet = '03_ SSL/ImageNet/ssl_resnet_results_ImageNet.csv'

    dataset_name = "Tiny ImageNet"
    save_path = '03_ SSL/ImageNet'

    data_points = ["Hamming Loss", "Accuracy", "Jaccard Score", "Matthews Corrcoef", "F1-Score (Micro)", "F1-Score (Macro)", "F1-Score (Weighted)", "F1-Score (Samples)", "Training Time (s)", "Validation Time (s)", "Pseudo-Labeled Samples"]
    data_points_I = ["Accuracy", "Jaccard Score", "Matthews Corrcoef", "ROC-AUC", "PR-AUC", "Log-Loss", "Sensitivity", "Specificity", "Cohen's Kappa", "F1-Score (Macro)", "F1-Score (Weighted)", "Training Time (s)", "Validation Time (s)", "Pseudo-Labeled Samples"]


    for data_point in data_points_I:
        #generate_ssl_diagram(file1000, file500, file2000, fileResNet, dataset_name, data_point, save_path, exclude_strategies=["Co-Training", "Multi-View Learning"])
        create_combined_heatmap(file1000, data_point, dataset_name, save_path)

    #generate_recall_vs_precision_scatter(file1000, file500, file2000, fileResNet, dataset_name, save_path)



