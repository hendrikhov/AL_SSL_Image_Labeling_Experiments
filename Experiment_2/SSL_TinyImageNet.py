import torch
import numpy as np
import cv2
import csv
import torchvision.transforms as T
import random
import os
import pickle
import json
import time
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset, Subset, ConcatDataset, TensorDataset
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, confusion_matrix, jaccard_score, matthews_corrcoef,roc_auc_score, auc, log_loss, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter, defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.base import clone
from sklearn.utils import shuffle
from memory_profiler import profile
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from itertools import combinations

@profile
class TinyImageNetDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.cat_id_to_idx = {data['class_name']: data['class_index'] for data in self.annotations.values()}

        if len(self.cat_id_to_idx) != 20:
            raise ValueError(f"Expected 20 categories, but found {len(self.cat_id_to_idx)}.")

        self.image_files = []
        for subfolder in os.listdir(img_dir):
            subfolder_path = os.path.join(img_dir, subfolder)
            print(f"Processing subfolder: {subfolder_path}")
            if os.path.isdir(subfolder_path):
                for img_file in os.listdir(subfolder_path):
                #print(f"Found file: {img_file}")
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.jpeg')):
                        self.image_files.append(os.path.join(subfolder_path, img_file))
   
        if not self.image_files:
            raise ValueError("No image files found in the dataset directory.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        class_name = os.path.basename(os.path.dirname(img_path))
        label_idx = self.cat_id_to_idx[class_name]

        label = [0] * 20
        label[label_idx] = 1

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
    
@profile    
def count_labels(dataset):
    all_labels = [label.numpy() for _, label in dataset]
    all_labels = np.vstack(all_labels)
    label_counts = np.sum(all_labels, axis=0)
    return label_counts

augmentation_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])
@profile
def augment_dataset(dataset, underrepresented_labels, min_samples, transforms):
    augmented_data = []
    augmented_labels = []

    label_counts = Counter()
    
    for img, label in dataset:
        label_np = label.numpy()
        if any(label_np[idx] == 1 for idx in underrepresented_labels):
            count = max([min_samples - label_counts[i] for i, v in enumerate(label_np) if v == 1])
            for _ in range(count):
                augmented_img = transforms(img)
                augmented_data.append(augmented_img)
                augmented_labels.append(label.numpy())
            for i, v in enumerate(label_np):
                if v == 1:
                    label_counts[i] += count

    augmented_features = np.array([img.numpy() for img in augmented_data])
    augmented_labels = np.array(augmented_labels, dtype=np.float32)
    augmented_features = torch.tensor(augmented_features, dtype=torch.float32)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.float32)
    print(f"Augmented dataset size: {len(augmented_features)}")
    augmented_dataset = TensorDataset(augmented_features, augmented_labels)
    original_features = torch.stack([img for img, _ in dataset])
    original_labels = torch.stack([label for _, label in dataset])
    original_dataset = TensorDataset(original_features, original_labels)
    full_dataset = ConcatDataset([original_dataset, augmented_dataset])

    return full_dataset

@profile
def stratified_train_val_test_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, min_per_label=10):
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.")

    labels = np.array([label.numpy() for _, label in dataset])

    def prioritize_rare_labels(labels):
 
        label_counts = Counter(np.argmax(labels, axis=1))
        for label in range(labels.shape[1]):
            label_counts[label] = np.sum(labels[:, label])

        sorted_labels = [label for label, _ in sorted(label_counts.items(), key=lambda x: x[1])]
        label_to_priority = {label: rank for rank, label in enumerate(sorted_labels)}

        prioritized_labels = np.full(labels.shape[0], -1)
        for i, row in enumerate(labels):
            available_labels = np.where(row == 1)[0]
            if len(available_labels) > 0:
                prioritized_labels[i] = min(available_labels, key=lambda x: label_to_priority[x])

        return prioritized_labels

    flat_labels = prioritize_rare_labels(labels)
    label_counts = Counter(flat_labels)

    oversampled_indices = []
    for label, count in label_counts.items():
        if count < min_per_label:
            label_indices = [i for i, lbl in enumerate(flat_labels) if lbl == label]
            additional_samples_needed = min_per_label - count
            oversampled_indices.extend(np.random.choice(label_indices, additional_samples_needed, replace=True))

    oversampled_features = [dataset[idx][0] for idx in oversampled_indices]
    oversampled_labels = [dataset[idx][1] for idx in oversampled_indices]
    oversampled_dataset = list(zip(oversampled_features, oversampled_labels))
    combined_dataset = ConcatDataset([dataset, oversampled_dataset])
    updated_labels = np.array([label.numpy() for _, label in combined_dataset])
    updated_flat_labels = prioritize_rare_labels(updated_labels)
    print(f"Augmented dataset size: {len(combined_dataset)}")
    print(f"Updated label distribution: {Counter(updated_flat_labels)}")

    train_indices, temp_indices = train_test_split(
        range(len(combined_dataset)),
        test_size=(val_size + test_size),
        stratify=updated_flat_labels,
        random_state=42
    )
    temp_labels = np.array(updated_flat_labels)[temp_indices]
    val_size_relative = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size_relative),
        stratify=temp_labels,
        random_state=42
    )
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    test_dataset = Subset(combined_dataset, test_indices)

    def display_label_counts(split_name, indices, flat_labels):
        label_counts = Counter(flat_labels[indices])
        print(f"{split_name} Label Counts:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

    display_label_counts("Train", train_indices, updated_flat_labels)
    display_label_counts("Validation", val_indices, updated_flat_labels)
    display_label_counts("Test", test_indices, updated_flat_labels)

    return train_dataset, val_dataset, test_dataset
    
@profile
def stratified_split(dataset, num_labeled):
    label_to_indices = defaultdict(list)

    for i in range(len(dataset)):
        _, label = dataset[i]
        for cls_idx, val in enumerate(label.numpy()):
            if val == 1:
                label_to_indices[cls_idx].append(i)

    samples_per_class = num_labeled // 20
    if samples_per_class * 20 != num_labeled:
        raise ValueError(f"num_labeled ({num_labeled}) must be divisible by the number of classes (20).")

    labeled_indices = []
    for cls_idx, indices in label_to_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(f"Not enough samples in class {cls_idx} to allocate {samples_per_class} labeled samples.")
        labeled_indices.extend(random.sample(indices, samples_per_class))

    unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))

    return labeled_indices, unlabeled_indices


@profile
def display_labeled_classes_after(split_name, indices, dataset):   
        label_counts = Counter()
        for idx in indices:
            _, label = dataset[idx]
            label_counts.update(label.numpy().nonzero()[0])

        print(f"{split_name} Classes Present:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

@profile
def extract_hog_features(dataset):
    features, labels = [], []
    for img, lbl in dataset:
        if len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3)
        elif len(img.shape) == 3:
            img = img.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img
        hog_features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
        labels.append(lbl.numpy())

    return np.array(features), np.array(labels)

@profile
def extract_lbp_features(dataset, radius=3, n_points=24):
    features, labels = [], []
    for img, lbl in dataset:
        if len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3)
        elif len(img.shape) == 3:
            img = img.permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img

        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.append(hist)
        labels.append(lbl.numpy())

    return np.array(features), np.array(labels)

@profile
def extract_bovw_features(dataset, kmeans=None, n_clusters=50):
    sift = cv2.SIFT_create()
    descriptors = []
    labels = []

    for img, lbl in dataset:

        if len(img.shape) == 3:
            img = img.permute(1, 2, 0).numpy()
        elif len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img

        if img_gray.dtype != np.uint8:
            img_gray = (img_gray * 255).astype(np.uint8)

        if img_gray is None or img_gray.size == 0:
            print("Warning: Skipping empty or invalid image.")
            continue

        _, desc = sift.detectAndCompute(img_gray, None)
        if desc is not None:
            descriptors.append(desc)
            labels.append(lbl.numpy())

    if not descriptors:
        raise ValueError("No valid descriptors found in the dataset.")
    descriptors = np.vstack(descriptors)
    if kmeans is None:
        print("Training KMeans for BoVW...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(descriptors)

    histograms = []
    for img, _ in dataset:
        if len(img.shape) == 3:
            img = img.permute(1, 2, 0).numpy()
        elif len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3)
        else:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img
        if img_gray.dtype != np.uint8:
            img_gray = (img_gray * 255).astype(np.uint8)

        if img_gray is None or img_gray.size == 0:
            histograms.append(np.zeros(n_clusters))
            continue

        _, desc = sift.detectAndCompute(img_gray, None)
        if desc is not None:
            hist = np.zeros(n_clusters)
            cluster_assignments = kmeans.predict(desc)
            for cluster_id in cluster_assignments:
                hist[cluster_id] += 1
            histograms.append(hist / np.sum(hist))
        else:
            histograms.append(np.zeros(n_clusters))

    return np.array(histograms), np.array(labels), kmeans

@profile
def ml_smote(X, y, k=5):
    smote = SMOTE(k_neighbors=k, random_state=42)
    balanced_features, balanced_labels = smote.fit_resample(X, y)
    return balanced_features, balanced_labels

@profile
def load_or_extract_features(dataset, feature_name, split_name, extractor, *args):
    feature_path = f"{split_name}_{feature_name}_features_IN.npy"
    label_path = f"{split_name}_{feature_name}_labels_IN.npy"

    if os.path.exists(feature_path) and os.path.exists(label_path):
        print(f"Loading precomputed {feature_name} features for {split_name}...")
        features = np.load(feature_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)
        return features, labels
    else:
        print(f"Extracting {feature_name} features for {split_name}...")
        results = extractor(dataset, *args)
        if feature_name == "BoVW":
            features, labels, kmeans = results
            kmeans_path = f"{split_name}_{feature_name}_kmeans_IN.pkl"
            with open(kmeans_path, "wb") as f:
                pickle.dump(kmeans, f)
        else:
            features, labels = results

        np.save(feature_path, features)
        np.save(label_path, labels)
        return features, labels

@profile
def save_or_load_split(dataset, split_name, num_labeled, stratified_split_fn, ml_smote_fn):
    labeled_indices_path = os.path.join(split_cache_dir, f"labeled_indices_{split_name}_{num_labeled}_IN.npy")
    unlabeled_indices_path = os.path.join(split_cache_dir, f"unlabeled_indices_{split_name}_{num_labeled}_IN.npy")
    labeled_features_path = os.path.join(split_cache_dir, f"labeled_features_{split_name}_{num_labeled}_IN.npz")
    labeled_labels_path = os.path.join(split_cache_dir, f"labeled_labels_{split_name}_{num_labeled}_IN.npz")

    if (
        os.path.exists(labeled_indices_path) and
        os.path.exists(unlabeled_indices_path) and
        os.path.exists(labeled_features_path) and
        os.path.exists(labeled_labels_path)
    ):
        print(f"Loading cached split and balanced data for {num_labeled} labeled samples...")
        labeled_indices = np.load(labeled_indices_path, allow_pickle=True)
        unlabeled_indices = np.load(unlabeled_indices_path, allow_pickle=True)
        labeled_features_balanced = np.load(labeled_features_path, allow_pickle=True)["features"]
        labeled_labels_balanced = np.load(labeled_labels_path, allow_pickle=True)["labels"]
    else:
        print(f"Processing splits and balancing for {num_labeled} labeled samples...")
        labeled_indices, unlabeled_indices = stratified_split_fn(dataset, num_labeled=num_labeled)
        np.save(labeled_indices_path, labeled_indices)
        np.save(unlabeled_indices_path, unlabeled_indices)
        labeled_features_raw = np.array([dataset[i][0].numpy().flatten() for i in labeled_indices])
        labeled_labels_raw = np.array([dataset[i][1].numpy() for i in labeled_indices])
        labeled_features_balanced, labeled_labels_balanced = ml_smote_fn(labeled_features_raw, labeled_labels_raw, k=5)
        np.savez_compressed(labeled_features_path, features=labeled_features_balanced)
        np.savez_compressed(labeled_labels_path, labels=labeled_labels_balanced)

    return labeled_indices, unlabeled_indices, labeled_features_balanced, labeled_labels_balanced

@profile
def extract_features(train_dataset, val_dataset, test_dataset):

    kmeans_path = "kmeans_model_imageNet.pkl"
    kmeans = None

    if os.path.exists(kmeans_path):
        print("Loading precomputed KMeans model...")
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    feature_methods = {
        "Raw": lambda dataset: (
            np.array([img.numpy().flatten() for img, _ in dataset]),
            np.array([lbl.numpy() for _, lbl in dataset])
        ),
        "BoVW": lambda dataset: extract_bovw_features(dataset, kmeans=kmeans),
        "HOG": extract_hog_features,
        "LBP": extract_lbp_features
    }
    features = {}
    for feature_name, extractor in feature_methods.items():
        for split_name, dataset in datasets.items():
            if feature_name == "BoVW" and kmeans is None and split_name == "train":
                print("Training KMeans for BoVW features...")
                features_train, labels_train, kmeans = extractor(dataset)
                features[f"{split_name}_{feature_name}"] = (features_train, labels_train)
                with open(kmeans_path, "wb") as f:
                    pickle.dump(kmeans, f)
            else:
                features[f"{split_name}_{feature_name}"] = load_or_extract_features(
                    dataset, feature_name, split_name, extractor
                )

    return features

@profile
def save_or_load_features(feature_name, subset_name, dataset, labeled_features_balanced=None, labeled_labels_balanced=None, kmeans=None):
    feature_file_path = os.path.join(feature_cache_dir, f"{feature_name}_{subset_name}_IN.npz")

    if os.path.exists(feature_file_path):
        print(f"Loading cached features for {feature_name} ({subset_name})...")
        data = np.load(feature_file_path, allow_pickle=True)
        return data["features"], data.get("labels"), data.get("kmeans")

    print(f"Extracting {feature_name} features for {subset_name}...")

    if feature_name == "BoVW":
        if subset_name == "labeled":
            dataset = TensorDataset(
                torch.tensor(labeled_features_balanced, dtype=torch.float32),
                torch.tensor(labeled_labels_balanced, dtype=torch.float32)
            )
        features, labels, kmeans = extract_bovw_features(dataset, kmeans=kmeans)
        np.savez_compressed(feature_file_path, features=features, labels=labels, kmeans=kmeans)
    elif feature_name == "HOG":
        if subset_name == "labeled":
            dataset = TensorDataset(
                torch.tensor(labeled_features_balanced, dtype=torch.float32),
                torch.tensor(labeled_labels_balanced, dtype=torch.float32)
            )
        features, labels = extract_hog_features(dataset)
        np.savez_compressed(feature_file_path, features=features, labels=labels)
    elif feature_name == "LBP":
        if subset_name == "labeled":
            dataset = TensorDataset(
                torch.tensor(labeled_features_balanced, dtype=torch.float32),
                torch.tensor(labeled_labels_balanced, dtype=torch.float32)
            )
        features, labels = extract_lbp_features(dataset)
        np.savez_compressed(feature_file_path, features=features, labels=labels)
    elif feature_name == "Raw":
        if subset_name == "labeled":
            features = labeled_features_balanced
            labels = labeled_labels_balanced
        else:
            features = np.array([dataset[i][0].numpy().flatten() for i in range(len(dataset))])
            labels = None
        np.savez_compressed(feature_file_path, features=features, labels=labels)

    return features, labels, kmeans if feature_name == "BoVW" else None


@profile
def self_training_classical(clf, X_unlabeled, threshold=0.9, max_pseudo_labels=1000):

    probabilities = clf.predict_proba(X_unlabeled)
    probabilities = np.array(probabilities)
    print("Shape of probabilities:", probabilities.shape)
    confidence_scores = probabilities.max(axis=1)
    print(f"Confidence scores (first 10): {confidence_scores[:10]}")
    print(f"Threshold: {threshold}")

    high_confidence_indices = confidence_scores > threshold
    num_high_confidence = np.sum(high_confidence_indices)
    print(f"High-confidence samples: {num_high_confidence}")

    if num_high_confidence > 0:

        pseudo_features = X_unlabeled[high_confidence_indices]
        pseudo_labels = probabilities[high_confidence_indices].argmax(axis=1)

        if max_pseudo_labels is not None and num_high_confidence > max_pseudo_labels:
            selected_indices = np.random.choice(
                np.where(high_confidence_indices)[0], size=max_pseudo_labels, replace=False
            )
            pseudo_features = X_unlabeled[selected_indices]
            pseudo_labels = probabilities[selected_indices].argmax(axis=1)
    else:
        pseudo_features = np.empty((0, X_unlabeled.shape[1]))
        pseudo_labels = np.empty((0,), dtype=int)
        print("No high-confidence samples found for pseudo-labeling.")

    return pseudo_labels, pseudo_features

@profile
def co_training_classical(model1, model2, X_labeled1, X_labeled2, y_labeled, X_unlabeled1, X_unlabeled2, max_iterations=10):

    for iteration in range(max_iterations):
        print(f"Co-Training Iteration {iteration + 1}/{max_iterations}")

        model1.fit(X_labeled1, y_labeled)
        model2.fit(X_labeled2, y_labeled)
        pseudo_labels1 = model1.predict(X_unlabeled1)
        pseudo_labels2 = model2.predict(X_unlabeled2)
        agreement_indices = np.where(pseudo_labels1 == pseudo_labels2)[0]

        if len(agreement_indices) == 0:
            print("No agreement between models. Stopping co-training.")
            break

        X_labeled1 = np.vstack([X_labeled1, X_unlabeled1[agreement_indices]])
        X_labeled2 = np.vstack([X_labeled2, X_unlabeled2[agreement_indices]])
        y_labeled = np.concatenate([y_labeled, pseudo_labels1[agreement_indices]])
        X_unlabeled1 = np.delete(X_unlabeled1, agreement_indices, axis=0)
        X_unlabeled2 = np.delete(X_unlabeled2, agreement_indices, axis=0)

    return model1, model2

@profile
def self_training(model, X_labeled, y_labeled, X_unlabeled, max_iterations=10, threshold=0.8):
    for iteration in range(max_iterations):
        print(f"Self-Training Iteration {iteration + 1}/{max_iterations}")

        model.fit(X_labeled, y_labeled)
        probabilities = model.predict_proba(X_unlabeled)
        high_confidence_indices = np.max(probabilities, axis=1) > threshold
        pseudo_labels = model.predict(X_unlabeled[high_confidence_indices])

        if not high_confidence_indices.any():
            print("No high-confidence pseudo-labels found. Stopping self-training.")
            break

        X_labeled = np.vstack([X_labeled, X_unlabeled[high_confidence_indices]])
        y_labeled = np.concatenate([y_labeled, pseudo_labels])
        X_unlabeled = X_unlabeled[~high_confidence_indices]

    return model

@profile
def graph_based_label_propagation(X_labeled, y_labeled, X_unlabeled, k_neighbors=10, max_iterations=10, tolerance=1e-3):
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])

    distances = pairwise_distances(X_combined)
    neighbors = np.argsort(distances, axis=1)[:, 1:k_neighbors + 1]
    graph = nx.Graph()

    for i, neigh in enumerate(neighbors):
        for n in neigh:
            graph.add_edge(i, n, weight=np.exp(-distances[i, n]))

    y_prev = y_combined.copy()
    for iteration in range(max_iterations):
        print(f"Graph-Based Propagation Iteration {iteration + 1}/{max_iterations}")
        changes = 0

        for i in range(len(X_combined)):
            if y_combined[i] == -1:
                neighbor_labels = y_combined[neighbors[i]]
                valid_labels = neighbor_labels[neighbor_labels != -1]
                if len(valid_labels) > 0:
                    new_label = np.bincount(valid_labels.astype(int)).argmax()
                    if new_label != y_combined[i]:
                        y_combined[i] = new_label
                        changes += 1

        if changes / len(X_unlabeled) < tolerance:
            print("Convergence reached.")
            break

        y_prev = y_combined.copy()

    return y_combined[len(X_labeled):]

@profile
def pseudo_label_classical(model, X_unlabeled, threshold=0.9, max_pseudo_labels=None):

    probabilities = model.predict_proba(X_unlabeled)
    print(f"Shape of probabilities: {probabilities.shape}")
    if hasattr(probabilities, "toarray"):
        probabilities = probabilities.toarray()
    high_confidence_mask = probabilities.max(axis=1) >= threshold
    pseudo_labels = probabilities.argmax(axis=1)
    filtered_pseudo_labels = pseudo_labels[high_confidence_mask]
    filtered_X_unlabeled = X_unlabeled[high_confidence_mask]
    if max_pseudo_labels is not None and len(filtered_X_unlabeled) > max_pseudo_labels:
        indices = np.random.choice(len(filtered_X_unlabeled), max_pseudo_labels, replace=False)
        filtered_pseudo_labels = filtered_pseudo_labels[indices]
        filtered_X_unlabeled = filtered_X_unlabeled[indices]

    print(f"Selected {len(filtered_X_unlabeled)} high-confidence pseudo-labeled samples out of {len(X_unlabeled)} total.")

    return filtered_pseudo_labels, filtered_X_unlabeled

@profile
def multi_view_learning(models, views, y_labeled, X_test_views, max_iterations=10, tolerance=1e-3):

    predictions = np.zeros((len(models), len(y_labeled)), dtype=int)

    for i, (model, view) in enumerate(zip(models, views)):
        model.fit(view, y_labeled)
        predictions[i] = model.predict(view)

    for iteration in range(max_iterations):
        print(f"Multi-View Learning Iteration {iteration + 1}/{max_iterations}")
        previous_predictions = predictions.copy()

        for i, (model, view) in enumerate(zip(models, views)):

            for i, (model, view) in enumerate(zip(models, views)):
                other_views_predictions = np.delete(predictions, i, axis=0).mean(axis=0).round().astype(int)
                valid_classes = np.unique(y_labeled)
                corrected_predictions = np.array([
                    pred if pred in valid_classes else min(valid_classes, key=lambda x: abs(x - pred))
                    for pred in other_views_predictions
                ])
                print(f"Valid classes: {valid_classes}")
                print(f"Corrected predictions: {np.unique(corrected_predictions)}")
                model.fit(view, corrected_predictions)

            predictions[i] = model.predict(view)

        aggregated_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.astype(int))
        if np.mean(previous_predictions != predictions) < tolerance:
            print("Convergence reached.")
            break

    test_predictions = []
    test_probabilities = []

    for i, (model, X_test_view) in enumerate(zip(models, X_test_views)):
        test_predictions.append(model.predict(X_test_view))
        if hasattr(model, "predict_proba"):
            test_probabilities.append(model.predict_proba(X_test_view))

    y_test_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(test_predictions).astype(int))

    if test_probabilities:
        num_classes = 20
        aligned_probabilities = []
        for proba in test_probabilities:
            aligned_proba = np.zeros((proba.shape[0], num_classes))
            min_classes = min(proba.shape[1], num_classes)
            aligned_proba[:, :min_classes] = proba[:, :min_classes]
            aligned_probabilities.append(aligned_proba)

        y_test_pred_proba = np.mean(np.array(aligned_probabilities), axis=0)
    else:
        y_test_pred_proba = None

    return aggregated_predictions, y_test_pred, y_test_pred_proba

@profile
def reduce_features(features, method="PCA", n_components=100):

    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "SVD":
        reducer = TruncatedSVD(n_components=n_components)
    else:
        raise ValueError("Unsupported reduction method. Use 'PCA' or 'SVD'.")
    
    reduced_features = reducer.fit_transform(features)
    return reduced_features, reducer

class CombinedClassifierSingleInput:
    def __init__(self, model1, model2, voting="soft"):
        self.model1 = model1
        self.model2 = model2
        self.voting = voting

    def predict(self, X):
        if self.voting == "soft" and hasattr(self.model1, "predict_proba") and hasattr(self.model2, "predict_proba"):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        elif self.voting == "hard":
            pred1 = self.model1.predict(X)
            pred2 = self.model2.predict(X)
            combined_pred = np.vstack([pred1, pred2]).T
            return np.array([np.bincount(row).argmax() for row in combined_pred])
        else:
            raise ValueError("Invalid voting method or models lack predict_proba.")

    def predict_proba(self, X):
        if self.voting == "soft" and hasattr(self.model1, "predict_proba") and hasattr(self.model2, "predict_proba"):
            proba1 = self.model1.predict_proba(X)
            proba2 = self.model2.predict_proba(X)
            return (proba1 + proba2) / 2
        else:
            raise ValueError("Soft voting requires both models to support predict_proba.")
        

@profile
def ssl_train_and_evaluate_classical_methods(features, y_train, y_val, y_test, labeled_indices, unlabeled_indices, num_labeled, max_epochs=20, patience=5):
    methods = {
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "kNN": KNeighborsClassifier(),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42)
    }

    strategy_to_function = {
        "Self-Training": self_training_classical,
        "Co-Training": co_training_classical,
        "Graph-Based Propagation": graph_based_label_propagation,
        "Pseudo-Labeling": pseudo_label_classical,
        "Multi-View Learning": multi_view_learning,
    }

    reduction_methods = {
        "Raw": ("PCA", 100),
        "BoVW": ("PCA", 50),
        "HOG": ("SVD", 75),
        "LBP": ("PCA", 30)
    }

    methods_excluding_catboost = {k: v for k, v in methods.items() if k != "CatBoost"}

    strategy_model_feature_pairs = {
        "Raw": [
            ("Self-Training", [methods[met] for met in methods.keys()]),
            ("Pseudo-Labeling", [methods[met] for met in methods.keys()]),
            ("Graph-Based Propagation", [methods[met] for met in methods.keys()]),
            ("Co-Training", [(methods_excluding_catboost[met1], methods_excluding_catboost[met2]) 
                            for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
            ("Multi-View Learning", [[methods_excluding_catboost[met1], methods_excluding_catboost[met2]] 
                                    for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
        ],

        "BoVW": [
            ("Self-Training", [methods[met] for met in methods.keys()]),
            ("Pseudo-Labeling", [methods[met] for met in methods.keys()]),
            ("Graph-Based Propagation", [methods[met] for met in methods.keys()]),
            ("Co-Training", [(methods_excluding_catboost[met1], methods_excluding_catboost[met2]) 
                            for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
            ("Multi-View Learning", [[methods_excluding_catboost[met1], methods_excluding_catboost[met2]] 
                                    for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
        ],

        "HOG": [
            ("Self-Training", [methods[met] for met in methods.keys()]),
            ("Pseudo-Labeling", [methods[met] for met in methods.keys()]),
            ("Graph-Based Propagation", [methods[met] for met in methods.keys()]),
            ("Co-Training", [(methods_excluding_catboost[met1], methods_excluding_catboost[met2]) 
                            for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
            ("Multi-View Learning", [[methods_excluding_catboost[met1], methods_excluding_catboost[met2]] 
                                    for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
        ],

        "LBP": [
            ("Self-Training", [methods[met] for met in methods.keys()]),
            ("Pseudo-Labeling", [methods[met] for met in methods.keys()]),
            ("Graph-Based Propagation", [methods[met] for met in methods.keys()]),
            ("Co-Training", [(methods_excluding_catboost[met1], methods_excluding_catboost[met2]) 
                            for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
            ("Multi-View Learning", [[methods_excluding_catboost[met1], methods_excluding_catboost[met2]] 
                                    for met1, met2 in combinations(methods_excluding_catboost.keys(), 2)]),
        ],
    }

    results = []
    results_csv_file = "ssl_detailed_results_ImageNet_2000.csv"

    if not os.path.exists(results_csv_file):
        with open(results_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Strategy", "Model", "Feature",
                "Accuracy", "Jaccard Score", "Matthews Corrcoef", "ROC-AUC", "PR-AUC",
                "Log-Loss", "Sensitivity", "Specificity", "Cohen's Kappa",
                "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
                "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)",
                "Training Time (s)", "Validation Time (s)",
                "Labeled Samples", "Pseudo-Labeled Samples",
                "True Positives", "False Positives", "True Negatives", "False Negatives"
            ])

    def save_result_to_csv(result, file_path):
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Error writing result to {file_path}: {e}")

    for feature_name, strategies in strategy_model_feature_pairs.items():
        print(f"\nUsing feature method: {feature_name}")

        reduction_method, n_components = reduction_methods[feature_name]
        labeled_key = f"train_{feature_name}_labeled_{num_labeled}"
        labeled_labels_key = f"train_{feature_name}_labeled_{num_labeled}_labels"
        unlabeled_key = f"train_{feature_name}_unlabeled_{num_labeled}"
        val_key = f"val_{feature_name}"
        test_key = f"test_{feature_name}"
        X_train_labeled = features[labeled_key].astype(np.float32)
        X_train_unlabeled = features[unlabeled_key].astype(np.float32)
        X_val = features[val_key].astype(np.float32)
        X_test = features[test_key].astype(np.float32)
        #initial_X_train_unlabeled = X_train_unlabeled.copy()
        y_train_labeled = features[labeled_labels_key]
        label_datasets = {"y_train_labeled": y_train_labeled, "y_val": y_val, "y_test": y_test}

        for name, labels in label_datasets.items():
            if labels.ndim == 2 and labels.shape[1] == 20:
                print(f"Converting {name} with shape {labels.shape} to single-label format...")
                labels = np.argmax(labels, axis=1)
                label_datasets[name] = labels
                print(f"Converted {name} to shape {labels.shape}.")
            else:
                print(f"No conversion needed for {name} with shape {labels.shape}.")

        y_train_labeled = label_datasets["y_train_labeled"]
        y_val = label_datasets["y_val"]
        y_test = label_datasets["y_test"]

        print(f"Scaling and reducing features for {feature_name}...")
        scaler = StandardScaler()
        X_train_labeled_scaled = scaler.fit_transform(X_train_labeled)

        if reduction_method == "PCA":
            n_components = min(n_components, X_train_labeled_scaled.shape[0], X_train_labeled_scaled.shape[1])
            reducer = PCA(n_components=n_components)
        elif reduction_method == "SVD":
            n_components = min(n_components, X_train_labeled_scaled.shape[1])
            reducer = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError(f"Unsupported reduction method: {reduction_method}")

        X_train_labeled_reduced = reducer.fit_transform(X_train_labeled_scaled)
        X_train_unlabeled_reduced = reducer.transform(scaler.transform(X_train_unlabeled))
        X_val_reduced = reducer.transform(scaler.transform(X_val))
        X_test_reduced = reducer.transform(scaler.transform(X_test))
        """
        X_train_unlabeled_reduced = X_train_unlabeled
        X_train_labeled_reduced = X_train_labeled
        X_val_reduced = X_val
        X_test_reduced = X_test
        """
        initial_X_train_unlabeled_reduced = X_train_unlabeled_reduced

        if len(X_train_labeled) != len(y_train_labeled):
            raise ValueError(
                f"Mismatch between features and labels for {feature_name}: "
                f"{len(X_train_labeled)} features vs {len(y_train_labeled)} labels"
            )

        for strategy_name, model_list in strategies:
            strategy_function = strategy_to_function[strategy_name]
            
            for model in model_list:
                if isinstance(model, tuple) and len(model) == 2:
                    model_name = f"{type(model[0]).__name__} & {type(model[1]).__name__}"
                elif isinstance(model, list) and all(hasattr(m, "fit") for m in model):
                    model_name = " & ".join([type(m).__name__ for m in model])
                elif hasattr(model, "fit"):
                    model_name = type(model).__name__
                else:
                    model_name = "Unknown"

                print(f"Running {strategy_name} with model(s): {model_name}")

                if (
                    (completed_experiments["Labeled_Samples"] == num_labeled) &
                    (completed_experiments["Feature"] == feature_name) &
                    (completed_experiments["Model"] == model_name) &
                    (completed_experiments["Strategy"] == strategy_name)
                ).any():
                    print(f"Skipping already completed experiment: {strategy_name} with {model_name} using {feature_name}")
                    continue

                print(f"Running {strategy_name} with {model_name} on {feature_name}")

                if strategy_name in ["Self-Training", "Pseudo-Labeling", "Graph-Based Propagation"]:
                    clf = clone(model)
                elif strategy_name == "Co-Training":
                    model1, model2 = model
                    clf1 = clone(model1)
                    clf2 = clone(model2)
                elif strategy_name == "Multi-View Learning":
                    clf = [clone(m) for m in model]

                else:
                    raise ValueError(f"Unsupported strategy: {strategy_name}")

                training_start_time = time.time()
                current_X_train = X_train_labeled_reduced
                current_y_train = y_train_labeled
                X_train_unlabeled_reduced = initial_X_train_unlabeled_reduced.copy()   
                combined_y_train = y_train_labeled
                best_clf = None

                if strategy_name in ["Self-Training", "Pseudo-Labeling", "Co-Training", "Graph-Based Propagation", "Multi-View Learning"]:
                    best_val_loss = float('inf')
                    no_improvement_epochs = 0       

                    for epoch in range(max_epochs):
                        print(f"{strategy_name}: Epoch {epoch + 1}/{max_epochs}")
                        if strategy_name in ["Self-Training", "Pseudo-Labeling"]:
                            clf.fit(current_X_train, current_y_train)

                            pseudo_labels, pseudo_features = strategy_function(
                                clf, X_train_unlabeled_reduced, threshold=0.7, max_pseudo_labels=1000
                            ) # 500:0.35; #1000:0.5
                            if len(pseudo_labels) > 0:
                                combined_X_train = np.vstack([current_X_train, pseudo_features])
                                combined_y_train = np.concatenate([current_y_train, pseudo_labels])
                                high_confidence_mask = np.isin(X_train_unlabeled_reduced, pseudo_features).all(axis=1)
                                X_train_unlabeled_reduced = X_train_unlabeled_reduced[~high_confidence_mask]
                                print(f"Remaining unlabeled samples: {len(X_train_unlabeled_reduced)}")
                                current_X_train, current_y_train = shuffle(combined_X_train, combined_y_train, random_state=42)
                            else:
                                print("No pseudo-labels added in this epoch.")

                        elif strategy_name == "Co-Training":
                            strategy_function(
                                clf1, clf2,
                                current_X_train, current_X_train,
                                current_y_train, X_train_unlabeled_reduced, X_train_unlabeled_reduced
                            )
                            best_clf = CombinedClassifierSingleInput(clf1, clf2, voting="soft")
                            training_end_time = time.time()
                            training_time = training_end_time - training_start_time
                            validation_start_time = time.time()
                            break

                        elif strategy_name == "Graph-Based Propagation":
                            propagated_labels = strategy_function(
                                current_X_train, current_y_train, X_train_unlabeled_reduced
                            )
                            if len(propagated_labels) > 0:
                                combined_X_train = np.vstack([current_X_train, X_train_unlabeled_reduced])
                                combined_y_train = np.concatenate([current_y_train, propagated_labels])
                                X_train_unlabeled_reduced = np.empty((0, X_train_unlabeled_reduced.shape[1]))
                                current_X_train, current_y_train = shuffle(combined_X_train, combined_y_train, random_state=42)
                                valid_indices = current_y_train != -1
                                X_train_filtered = current_X_train[valid_indices]
                                y_train_filtered = current_y_train[valid_indices]
                                clf.fit(X_train_filtered, y_train_filtered)
                                best_clf = clf
                                training_end_time = time.time()
                                training_time = training_end_time - training_start_time
                                validation_start_time = time.time()
                                break
                            else:
                                print("No labels propagated in this epoch.")

                        elif strategy_name == "Multi-View Learning":
                            y_pred_views, y_test_pred, y_test_pred_proba = strategy_function(clf, [current_X_train] * len(clf), current_y_train, [X_test_reduced] * len(clf))
                            print(f"Multi-view predictions updated in epoch {epoch + 1}.")
                            training_end_time = time.time()
                            training_time = training_end_time - training_start_time
                            validation_start_time = time.time()
                            break
                                        
                        training_end_time = time.time()
                        training_time = training_end_time - training_start_time
                        validation_start_time = time.time()
                        y_val_pred = clf.predict(X_val_reduced) if strategy_name != "Multi-View Learning" else np.mean(y_pred_views, axis=0)
                        val_loss = np.mean(np.abs(y_val - y_val_pred))

                        if val_loss <= best_val_loss:
                            best_val_loss = val_loss
                            no_improvement_epochs = 0
                            best_clf = clf
                            print("Validation loss improved. Saving model...")
                        else:
                            no_improvement_epochs += 1
                            print(f"No improvement for {no_improvement_epochs} epoch(s).")

                        if no_improvement_epochs >= patience:
                            print("Early stopping triggered.")
                            break

                        if len(X_train_unlabeled_reduced) == 0:
                            print("No unlabeled data remaining. Stopping training.")
                            break               
                
                if strategy_name != "Multi-View Learning":
                    if best_clf is None:
                        print(f"Skipping evaluation for {strategy_name} with {model_name} using {feature_name} due to training failure.")
                        continue
                    else:
                        print(f"Training completed for {strategy_name} with {model_name}. Proceeding to evaluation.")

                    y_test_pred = best_clf.predict(X_test_reduced)
                    y_test_pred_proba = best_clf.predict_proba(X_test_reduced) if hasattr(best_clf, "predict_proba") else None
                
                pseudo_labeled_count = len(current_y_train) - len(labeled_indices)
                accuracy = accuracy_score(y_test, y_test_pred)
                jaccard = jaccard_score(y_test, y_test_pred, average="macro")
                matthews = matthews_corrcoef(y_test, y_test_pred)
                #report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro') if y_test_pred_proba is not None else None
                except Exception as e:
                    print(f"Error calculating ROC-AUC: {e}")
                    roc_auc = ""
                if y_test_pred_proba is not None:
                    valid_classes = np.arange(20)
                    num_classes = len(valid_classes)
                    aligned_proba = np.zeros((y_test_pred_proba.shape[0], num_classes))
                    existing_classes = np.unique(y_test)
                    for i, cls in enumerate(existing_classes):
                        if cls < num_classes:
                            aligned_proba[:, cls] = y_test_pred_proba[:, i]
                    y_test_one_hot = np.zeros((len(y_test), num_classes))
                    y_test_one_hot[np.arange(len(y_test)), y_test] = 1
                    pr_auc_per_class = {}
                    for i in range(num_classes):
                        precision, recall, _ = precision_recall_curve(y_test_one_hot[:, i], aligned_proba[:, i])
                        pr_auc_per_class[i] = auc(recall, precision)
                    pr_auc = np.mean(list(pr_auc_per_class.values()))
                    log_loss_value = log_loss(y_test, aligned_proba, labels=valid_classes)
                else:
                    pr_auc = None
                    log_loss_value = None
                    print("Probabilities not available; skipping Precision-Recall AUC and Log-Loss calculations.")

                cm = confusion_matrix(y_test, y_test_pred)
                num_classes = 20
                metrics = {"tp": [], "fp": [], "fn": [], "tn": [], "sensitivity": [], "specificity": []}

                for i in range(num_classes):
                    tp = cm[i, i]
                    fn = cm[i, :].sum() - tp
                    fp = cm[:, i].sum() - tp
                    tn = cm.sum() - (tp + fn + fp)
                    metrics["tp"].append(tp)
                    metrics["fp"].append(fp)
                    metrics["fn"].append(fn)
                    metrics["tn"].append(tn)
                    metrics["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

                mean_tp = np.mean(metrics["tp"])
                mean_fp = np.mean(metrics["fp"])
                mean_fn = np.mean(metrics["fn"])
                mean_tn = np.mean(metrics["tn"])
                mean_sensitivity = np.mean(metrics["sensitivity"])
                mean_specificity = np.mean(metrics["specificity"])

                cohen_kappa = cohen_kappa_score(y_test, y_test_pred)
                report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
                macro_avg = report["macro avg"]
                weighted_avg = report["weighted avg"]
                validation_end_time = time.time()
                validation_time = validation_end_time - validation_start_time
                result_row = [
                    strategy_name, model_name, feature_name,
                    accuracy, jaccard, matthews, roc_auc, pr_auc,
                    log_loss_value, mean_sensitivity, mean_specificity, cohen_kappa,
                    macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"],
                    weighted_avg["precision"], weighted_avg["recall"], weighted_avg["f1-score"],
                    training_time, validation_time,
                    len(labeled_indices), pseudo_labeled_count,
                    mean_tp, mean_fp, mean_tn, mean_fn
                ]
                save_result_to_csv(result_row, results_csv_file)

                results.append({
                    "Strategy": strategy_name,
                    "Model": model_name,
                    "Feature": feature_name,
                    "Accuracy": accuracy,
                    "Jaccard Score": jaccard,
                    "Matthews Corrcoef": matthews,
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc,
                    "Log-Loss": log_loss_value,
                    "Sensitivity": mean_sensitivity,
                    "Specificity": mean_specificity,
                    "Cohen's Kappa": cohen_kappa,
                    "Macro Precision": macro_avg["precision"],
                    "Macro Recall": macro_avg["recall"],
                    "Macro F1-Score": macro_avg["f1-score"],
                    "Weighted Precision": weighted_avg["precision"],
                    "Weighted Recall": weighted_avg["recall"],
                    "Weighted F1-Score": weighted_avg["f1-score"],
                    "Training Time (s)": training_time,
                    "Validation Time (s)": validation_time,
                    "Labeled Samples": len(labeled_indices),
                    "Pseudo-Labeled Samples": pseudo_labeled_count,
                    "True Positives": mean_tp,
                    "False Positives": mean_fp,
                    "True Negatives": mean_tn,
                    "False Negatives": mean_fn,
                })

                log_experiment(num_labeled, feature_name, model_name, strategy_name)

    return results


@profile
def save_results_to_csv(results, csv_file_path):
    file_exists = os.path.exists(csv_file_path)
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        csv_file_path,
        mode='a',
        header=not file_exists,
        index=False
    )
    print(f"Results saved to {csv_file_path}.")

@profile
def log_experiment(labeled_samples, feature_name, model_name, strategy_name):
    global completed_experiments
    try:
        if 'completed_experiments' not in globals() or completed_experiments is None:
            completed_experiments = pd.DataFrame(columns=["Labeled_Samples", "Feature", "Model", "Strategy"])
        
        # Create a new entry for the experiment
        new_entry = pd.DataFrame([{
            "Labeled_Samples": labeled_samples,
            "Feature": feature_name,
            "Model": model_name,
            "Strategy": strategy_name
        }])
        completed_experiments = pd.concat([completed_experiments, new_entry], ignore_index=True)
        completed_experiments.to_csv(progress_file, index=False)
        print(f"Experiment logged: {labeled_samples}, {feature_name}, {model_name}, {strategy_name}")
    
    except Exception as e:
        print(f"Error logging experiment to {progress_file}: {e}")

images_dir = './Image_Net/tiny-imagenet-subset'
annotations_path = './Image_Net/ImageNet_subset_labels.json'
augmented_dataset_path = "./ImageNet_cache/augmented_dataset.pkl"
train_val_test_splits_path = "./ImageNet_cache/train_val_test_splits.pkl"
features_cache_dir = "./ImageNet_cache/features/"
label_cache_dir = "./cache_ImageNet/labels/"
feature_cache_dir = "./cache_ImageNet/features/"
split_cache_dir = "./cache_ImageNet/splits"


max_epochs = 10  # Maximum number of epochs for training
patience = 3  # Number of epochs to wait for improvement

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    os.makedirs(features_cache_dir, exist_ok=True)
    os.makedirs(label_cache_dir, exist_ok=True)
    os.makedirs(feature_cache_dir, exist_ok=True)
    os.makedirs(split_cache_dir, exist_ok=True)

    if os.path.exists(augmented_dataset_path):
        print("Loading augmented dataset from cache...")
        with open(augmented_dataset_path, "rb") as f:
            full_dataset = pickle.load(f)
    else:
        print("Generating augmented dataset...")
        full_dataset = TinyImageNetDataset(images_dir, annotations_path, transform=transform)

        label_counts = count_labels(full_dataset)
        print("Label counts before augmentation:")
        for idx, count in enumerate(label_counts):
            print(f"Label {idx}: {count} samples")

        min_samples = 50
        underrepresented_labels = np.where(label_counts < min_samples)[0]
        print("Underrepresented labels:", underrepresented_labels)
        org_dataset_size = len(full_dataset)

        full_dataset = augment_dataset(full_dataset, underrepresented_labels, min_samples=50, transforms=augmentation_transforms)
        label_counts = count_labels(full_dataset)
        print("Label counts after augmentation:")
        for idx, count in enumerate(label_counts):
            print(f"Label {idx}: {count} samples")

        with open(augmented_dataset_path, "wb") as f:
            pickle.dump(full_dataset, f)

    if os.path.exists(train_val_test_splits_path):
        print("Loading train/val/test splits from cache...")
        with open(train_val_test_splits_path, "rb") as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)
    else:
        print("Performing train/val/test split...")
        train_dataset, val_dataset, test_dataset = stratified_train_val_test_split(
            full_dataset, train_size=0.7, val_size=0.15, test_size=0.15, min_per_label=30
        )
    test_labels_path = os.path.join(label_cache_dir, "test_labels_IN.npy")
    if os.path.exists(test_labels_path):
        print("Loading test labels from cache...")
        test_labels = np.load(test_labels_path, allow_pickle=True)
    else:
        print("Extracting test labels...")
        test_labels = np.array([label.numpy() for _, label in test_dataset])
        np.save(test_labels_path, test_labels)

    val_labels_path = os.path.join(label_cache_dir, "val_labels_IN.npy")
    if os.path.exists(val_labels_path):
        print("Loading validation labels from cache...")
        y_val = np.load(val_labels_path, allow_pickle=True)
    else:
        print("Extracting validation labels...")
        y_val = np.array([label.numpy() for _, label in val_dataset])
        np.save(val_labels_path, y_val)
    y_test = test_labels

    print("Extracting or loading features...")
    def extract_raw_features(dataset):
        return np.array([img.numpy().flatten() for img, _ in dataset]), np.array([label.numpy() for _, label in dataset])

    extractors = {
        "Raw": extract_raw_features,
        "BoVW": extract_bovw_features,
        "HOG": extract_hog_features,
        "LBP": extract_lbp_features
    }

    features = {}
    for feature_name in ["Raw", "BoVW", "HOG", "LBP"]:
        for split_name, dataset in [("val", val_dataset), ("test", test_dataset)]:
            feature_file_path = os.path.join(feature_cache_dir, f"{split_name}_{feature_name}_IN.npz")
            
            if os.path.exists(feature_file_path):
                print(f"Loading precomputed {feature_name} features for {split_name}...")
                data = np.load(feature_file_path, allow_pickle=True)
                features[f"{split_name}_{feature_name}"] = data["features"]
            else:
                print(f"Extracting {feature_name} features for {split_name}...")
                extracted_features, _ = load_or_extract_features(dataset, feature_name, split_name, extractors[feature_name])
                np.savez_compressed(feature_file_path, features=extracted_features)
                features[f"{split_name}_{feature_name}"] = extracted_features

    progress_file = "ssl_experiment_progress_ImageNet_2000.csv"

    if os.path.exists(progress_file):
        completed_experiments = pd.read_csv(progress_file)
    else:
        completed_experiments = pd.DataFrame(columns=["Labeled_Samples", "Feature", "Model", "Strategy"])

    labeled_data_sizes = [2000] #1500, 500, 1000
    all_results = []
    for num_labeled in labeled_data_sizes:
        print(f"\nProcessing with labeled data size: {num_labeled}")

        labeled_indices, unlabeled_indices, labeled_features_balanced, labeled_labels_balanced = save_or_load_split(
            train_dataset, "train", num_labeled, stratified_split, ml_smote
        )

        display_labeled_classes_after("Labeled Dataset", labeled_indices, train_dataset)
        unique_classes_after_split = set()
        for idx in labeled_indices:
            _, label = train_dataset[idx]
            unique_classes_after_split.update(label.numpy().nonzero()[0])

        print("Unique classes after stratified split:", sorted(unique_classes_after_split))
        print(f"\nRunning SSL experiment with {num_labeled} labeled samples...")
        features_labeled = {}
        features_unlabeled = {}
        for feature_name in ["Raw", "BoVW", "HOG", "LBP"]:
            print(f"Processing {feature_name} features...")

            labeled_key = f"train_{feature_name}_labeled_{num_labeled}"
            unlabeled_key = f"train_{feature_name}_unlabeled_{num_labeled}"

            features_balanced, labels_balanced, kmeans = save_or_load_features(
                feature_name, "labeled", None, labeled_features_balanced, labeled_labels_balanced
            )
            features_labeled[feature_name] = features_balanced
            unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
            unlabeled_features, _, _ = save_or_load_features(feature_name, "unlabeled", unlabeled_dataset, kmeans=kmeans)
            features_unlabeled[feature_name] = unlabeled_features
            features[labeled_key] = features_balanced
            features[f"{labeled_key}_labels"] = labels_balanced
            features[unlabeled_key] = unlabeled_features

        y_train = np.array([label.numpy() for _, label in train_dataset])

        for feature_name in ["Raw", "BoVW", "HOG", "LBP"]:
            print(f"Running SSL experiment for {feature_name} with {num_labeled} labeled samples...")

            results = ssl_train_and_evaluate_classical_methods(
                features=features.copy(),
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                num_labeled=num_labeled,
                max_epochs=max_epochs,
                patience=patience
            )
            all_results.extend(results)

    print("\nAll Results:")
    for result in all_results:
        print(result)

    save_results_to_csv(all_results, "ssl_results_ImageNet_2000.csv")