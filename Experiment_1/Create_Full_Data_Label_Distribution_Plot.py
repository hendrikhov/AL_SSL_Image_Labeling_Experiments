import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import json
import os

def plot_multilabel_distribution(label_matrix, class_names, figsize=(12, 8)):

    if isinstance(label_matrix, np.ndarray):
        label_matrix = pd.DataFrame(label_matrix, columns=class_names)

    class_occurrence = (label_matrix > 0).sum(axis=0)
    class_occurrence_sorted = class_occurrence.sort_values(ascending=False)
    sorted_class_names = class_occurrence_sorted.index.tolist()
    label_matrix_sorted = label_matrix[sorted_class_names]

    co_occurrence = label_matrix_sorted.T.dot(label_matrix_sorted)
    np.fill_diagonal(co_occurrence.values, 0)

    base_colors = list(mcolors.TABLEAU_COLORS.values())
    patterns = ["", "//", "\\", "--", "++", "xx", "oo"]
    color_patterns = []
    for i in range(len(sorted_class_names)):
        color = base_colors[i % len(base_colors)]
        pattern = patterns[(i // len(base_colors)) % len(patterns)] if i >= len(base_colors) else ""
        color_patterns.append((color, pattern))

    def calculate_non_shared_count(label_matrix, class_index):
        non_shared_count = 0
        for _, row in label_matrix.iterrows():
            active_labels = row[row > 0].index.tolist()
            if len(active_labels) == 1 and active_labels[0] == class_index:
                non_shared_count += 1
        return non_shared_count

    fig, ax = plt.subplots(figsize=figsize)

    for i, class_name in enumerate(sorted_class_names):
        total_count = int(class_occurrence_sorted[class_name])
        remaining_height = total_count  # Initialize remaining height
        bottom = 0  # Initialize bottom position

        print(f"\nClass: {class_name}, Total Count: {total_count}")

        non_shared_count = calculate_non_shared_count(label_matrix_sorted, class_name)
        print(f"  Non-Shared Count: {non_shared_count}")
        ax.bar(i, non_shared_count, bottom=bottom, color=color_patterns[i][0])
        bottom += non_shared_count
        remaining_height -= non_shared_count

        shared_counts = [int(co_occurrence.loc[class_name, other_class]) for other_class in sorted_class_names]
        total_shared_count = sum(shared_counts)
        print(f"  Total Shared Count: {total_shared_count}")

        if total_shared_count > remaining_height:
            scaling_factor = remaining_height / total_shared_count
        else:
            scaling_factor = 1

        cumulative_height = 0
        for j, other_class in enumerate(sorted_class_names):
            shared_count = int(co_occurrence.loc[class_name, other_class] * scaling_factor)
            shared_count = min(shared_count, remaining_height - cumulative_height)
            if shared_count > 0:
                print(f"    Shared Increment with {other_class}: {shared_count}")
                ax.bar(
                    i, shared_count, bottom=bottom + cumulative_height, color=color_patterns[j][0],
                    hatch=color_patterns[j][1] if j >= len(base_colors) else "", linewidth=0
                )
                cumulative_height += shared_count
        print(f"  Final Cumulative Height: {bottom + cumulative_height}")

    for idx, total in enumerate(class_occurrence_sorted):
        ax.text(idx, total + 1, f'{int(total)}', ha='center', va='bottom')

    ax.set_ylabel('Total Occurrences')
    ax.set_xlabel('Classes')
    ax.set_title('Multilabel Class Distribution with Shared Co-occurrences')
    ax.set_xticks(range(len(sorted_class_names)))
    ax.set_xticklabels(sorted_class_names, rotation=45, ha='right')

    legend_patches = [
        mpatches.Patch(facecolor=color_patterns[i][0], hatch=color_patterns[i][1], label=sorted_class_names[i])
        for i in range(len(sorted_class_names))
    ]
    ax.legend(handles=legend_patches, title="Classes", loc='upper right', frameon=False)

    plt.tight_layout()
    plt.show()

def plot_single_label_distribution(label_matrix, class_names, figsize=(12, 8)):

    if isinstance(label_matrix, np.ndarray):
        label_matrix = np.array(label_matrix)

    single_label_counts = np.zeros(len(class_names), dtype=int)
    single_label_rows = np.sum(label_matrix, axis=1) == 1  # Identify rows with a single label

    for idx in range(len(class_names)):
        single_label_counts[idx] = np.sum(label_matrix[single_label_rows, idx])

    sorted_indices = np.argsort(single_label_counts)[::-1]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_single_label_counts = single_label_counts[sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(sorted_class_names, sorted_single_label_counts, color='skyblue')

    for i, count in enumerate(sorted_single_label_counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Number of Single-Label Images', fontsize=12)
    ax.set_title('Single-Label Images Distribution per Class', fontsize=14)
    ax.set_xticks(range(len(sorted_class_names)))
    ax.set_xticklabels(sorted_class_names, rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    plt.show()

def load_coco_annotations(annotations_path):

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    class_names = list(categories.values())
    class_ids = list(categories.keys())

    image_to_labels = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        if image_id not in image_to_labels:
            image_to_labels[image_id] = set()
        image_to_labels[image_id].add(category_id)  # Use a set to ensure uniqueness

    image_ids = sorted(image_to_labels.keys())
    label_matrix = np.zeros((len(image_ids), len(class_ids)), dtype=int)

    for i, img_id in enumerate(image_ids):
        for cat_id in image_to_labels[img_id]:
            class_index = class_ids.index(cat_id)
            label_matrix[i, class_index] = 1

    return label_matrix, class_names

def load_pascalvoc_annotations(voc_annotations_path):
    with open(voc_annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"Keys in the JSON file: {list(data.keys())}")
    
    class_names = data.get("class_names")
    label_matrix = data.get("label_matrix")
    
    if class_names is None or label_matrix is None:
        raise ValueError("The JSON file does not contain the expected 'class_names' or 'label_matrix' keys.")
    
    return np.array(label_matrix), class_names



if __name__ == "__main__":
    annotations_path = './coco_subset_10Kv2/annotations/coco_subset_annotations.json'
    vocannotations_path = './04_Evaluation/voc_processed.json'
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")

    label_matrix, class_names = load_coco_annotations(annotations_path)
    #label_matrix, class_names = load_pascalvoc_annotations(vocannotations_path)

    #plot_multilabel_distribution(label_matrix, class_names, figsize=(12, 8))
    plot_single_label_distribution(label_matrix, class_names, figsize=(12, 8))
