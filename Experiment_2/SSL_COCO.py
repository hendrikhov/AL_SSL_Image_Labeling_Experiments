import torch
import torch.nn as nn
import numpy as np
import cv2
import csv
import torchvision.transforms as T
import random
import os
import pickle
import gc
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, TensorDataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, precision_recall_curve
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from pycocotools.coco import COCO
from PIL import Image
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.base import clone
from sklearn.utils import shuffle
from memory_profiler import profile
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Manager


@profile
class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.coco = COCO(annotation_path)
        self.img_dir = img_dir
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.coco.getCatIds())}
        if len(self.cat_id_to_idx) != 16:
            raise ValueError(f"Expected 16 categories, but found {len(self.cat_id_to_idx)}.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        img_path = self.coco.loadImgs(image_id)[0]['file_name']
        img = Image.open(f"{self.img_dir}/{img_path}").convert("RGB")

        label = [0] * 16
        for annotation in annotations:
            if annotation['category_id'] in self.cat_id_to_idx:
                mapped_idx = self.cat_id_to_idx[annotation['category_id']]
                label[mapped_idx] = 1

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
def stratified_train_val_test_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, min_per_label=30):
    
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.")

    # Extract labels for stratification
    labels = np.array([label.numpy() for _, label in dataset])

    # Prioritize rare labels in the dataset
    def prioritize_rare_labels(labels):
        label_counts = Counter(np.argmax(labels, axis=1))
        for label in range(labels.shape[1]):
            label_counts[label] = np.sum(labels[:, label])

        # Sort labels by frequency (lowest to highest)
        sorted_labels = [label for label, _ in sorted(label_counts.items(), key=lambda x: x[1])]
        label_to_priority = {label: rank for rank, label in enumerate(sorted_labels)}

        prioritized_labels = np.full(labels.shape[0], -1)
        for i, row in enumerate(labels):
            available_labels = np.where(row == 1)[0]
            if len(available_labels) > 0:
                prioritized_labels[i] = min(available_labels, key=lambda x: label_to_priority[x])

        return prioritized_labels

    # Prioritize labels and count occurrences
    flat_labels = prioritize_rare_labels(labels)
    label_counts = Counter(flat_labels)
    
    # Oversample underrepresented labels
    oversampled_indices = []
    for label, count in label_counts.items():
        if count < min_per_label:
            label_indices = [i for i, lbl in enumerate(flat_labels) if lbl == label]
            additional_samples_needed = min_per_label - count
            oversampled_indices.extend(np.random.choice(label_indices, additional_samples_needed, replace=True))

    # Create oversampled dataset
    oversampled_features = [dataset[idx][0] for idx in oversampled_indices]
    oversampled_labels = [dataset[idx][1] for idx in oversampled_indices]
    oversampled_dataset = list(zip(oversampled_features, oversampled_labels))

    # Combine original and oversampled datasets
    combined_dataset = ConcatDataset([dataset, oversampled_dataset])

    # Extract updated labels and dataset
    updated_labels = np.array([label.numpy() for _, label in combined_dataset])
    updated_flat_labels = prioritize_rare_labels(updated_labels)

    print(f"Oversampled dataset size: {len(combined_dataset)}")
    print(f"Updated label distribution: {Counter(updated_flat_labels)}")

    # Stratified split into train + temp (val+test)
    train_indices, temp_indices = train_test_split(
        range(len(combined_dataset)),
        test_size=(val_size + test_size),
        stratify=updated_flat_labels,
        random_state=42
    )

    # Stratify the temp (val+test) split further
    temp_labels = np.array(updated_flat_labels)[temp_indices]
    val_size_relative = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size_relative),
        stratify=temp_labels,
        random_state=42
    )

    # Create datasets using the indices
    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)
    test_dataset = Subset(combined_dataset, test_indices)

    # Display label counts for each split
    def display_label_counts(split_name, indices, flat_labels):
        label_counts = Counter(flat_labels[indices])
        print(f"{split_name} Label Counts:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

    display_label_counts("Train", train_indices, updated_flat_labels)
    display_label_counts("Validation", val_indices, updated_flat_labels)
    display_label_counts("Test", test_indices, updated_flat_labels)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
    

@profile
def stratified_split(dataset, num_labeled):
    label_to_indices = defaultdict(list)

    for i in range(len(dataset)):
        _, label = dataset[i]
        for cls_idx, val in enumerate(label.numpy()):
            if val == 1:
                label_to_indices[cls_idx].append(i)

    labeled_indices = []
    for cls_idx, indices in label_to_indices.items():
        if len(indices) >= 10:
            labeled_indices.extend(indices[:10])
            label_to_indices[cls_idx] = indices[10:]
        else:
            labeled_indices.extend(indices)
            label_to_indices[cls_idx] = []

    remaining_samples_needed = num_labeled - len(labeled_indices)
    all_remaining_indices = [idx for indices in label_to_indices.values() for idx in indices]

    if remaining_samples_needed > len(all_remaining_indices):
        raise ValueError(
            f"Not enough samples to meet the requested number of labeled samples ({num_labeled}). "
            f"Consider increasing the dataset size or reducing `num_labeled`."
        )

    additional_indices = random.sample(all_remaining_indices, remaining_samples_needed)
    labeled_indices.extend(additional_indices)

    remaining_samples_needed = num_labeled - len(labeled_indices)
    if remaining_samples_needed > 0:
        all_remaining_indices = [idx for indices in label_to_indices.values() for idx in indices]
        
        labeled_class_counts = Counter()
        for idx in labeled_indices:
            _, label = dataset[idx]
            labeled_class_counts.update(label.numpy().nonzero()[0])
        
        sorted_classes = sorted(label_to_indices.keys(), key=lambda cls: labeled_class_counts[cls])
        
        additional_indices = []
        for cls in sorted_classes:
            if remaining_samples_needed <= 0:
                break
            if label_to_indices[cls]:
                to_sample = min(len(label_to_indices[cls]), remaining_samples_needed)
                sampled_indices = random.sample(label_to_indices[cls], to_sample)
                additional_indices.extend(sampled_indices)
                label_to_indices[cls] = [idx for idx in label_to_indices[cls] if idx not in sampled_indices]
                remaining_samples_needed -= to_sample

        labeled_indices.extend(additional_indices)    
    labeled_indices = list(set(labeled_indices))

    all_indices = set(range(len(dataset)))
    unlabeled_indices = list(all_indices - set(labeled_indices))

    class_counts = defaultdict(int)
    for idx in labeled_indices:
        _, label = dataset[idx]
        for cls_idx, val in enumerate(label.numpy()):
            if val == 1:
                class_counts[cls_idx] += 1

    for cls, count in class_counts.items():
        if count < 10:
            print(f"Warning: Class {cls} has fewer than 10 samples in the labeled set (count: {count}).")

    print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")

    return labeled_indices, unlabeled_indices

@profile
def display_labeled_classes_after(split_name, indices, dataset):
        label_counts = Counter()
        for idx in indices:
            _, label = dataset[idx]
            label_counts.update(label.numpy().nonzero()[0])  # Count nonzero labels (classes)

        print(f"{split_name} Classes Present:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

@profile
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

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
            histograms.append(hist / np.sum(hist))  # Normalize histogram
        else:
            histograms.append(np.zeros(n_clusters))  # Empty histogram if no descriptors

    return np.array(histograms), np.array(labels), kmeans

@profile
def ml_smote(X, y, k=5):

    X = np.array(X)
    y = np.array(y, dtype=np.int32)

    if len(X) != len(y):
        raise ValueError("Features and labels must have the same number of samples.")

    X_resampled = list(X)  # Avoid converting back and forth
    y_resampled = list(y)

    minority_indices = np.where(np.sum(y, axis=1) < y.shape[1] / 2)[0]

    if len(minority_indices) == 0:
        print("No minority samples detected; returning original dataset.")
        return np.array(X), np.array(y)

    nn = NearestNeighbors(n_neighbors=min(len(minority_indices), k + 1))
    nn.fit(X[minority_indices])

    for idx in minority_indices:
        neighbors = nn.kneighbors([X[idx]], return_distance=False)[0][1:]
        if len(neighbors) == 0:
            continue

        for _ in range(k):
            neighbor_idx = neighbors[np.random.randint(0, len(neighbors))]
            neighbor = X[minority_indices[neighbor_idx]]
            synthetic_sample = X[idx] + np.random.random(X[idx].shape) * (neighbor - X[idx])
            synthetic_label = y[idx] | y[minority_indices[neighbor_idx]]  # Union of labels

            X_resampled.append(synthetic_sample)
            y_resampled.append(synthetic_label)

    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)
    label_counts = np.sum(y_resampled, axis=0)
    missing_labels = [label for label, count in enumerate(label_counts) if count == 0]

    for label in missing_labels:
        label_samples = [i for i, label_row in enumerate(y) if label_row[label] == 1]
        if not label_samples:
            print(f"Warning: No original samples found for label {label}.")
            continue

        while label_counts[label] == 0:
            sample_idx = np.random.choice(label_samples)
            neighbors = nn.kneighbors([X[sample_idx]], return_distance=False)[0][1:]
            if len(neighbors) == 0:
                continue

            neighbor_idx = neighbors[np.random.randint(0, len(neighbors))]
            neighbor = X[neighbor_idx]
            synthetic_sample = X[sample_idx] + np.random.random(X[sample_idx].shape) * (neighbor - X[sample_idx])
            synthetic_label = y[sample_idx] | y[neighbor_idx]

            X_resampled = np.vstack((X_resampled, synthetic_sample))
            y_resampled = np.vstack((y_resampled, synthetic_label))
            label_counts[label] += 1

    return X_resampled, y_resampled

@profile
def load_or_extract_features(dataset, feature_name, split_name, extractor, *args):
    feature_path = f"{split_name}_{feature_name}_features.npy"
    label_path = f"{split_name}_{feature_name}_labels.npy"

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
            kmeans_path = f"{split_name}_{feature_name}_kmeans.pkl"
            with open(kmeans_path, "wb") as f:
                pickle.dump(kmeans, f)
        else:
            features, labels = results

        np.save(feature_path, features)
        np.save(label_path, labels)
        return features, labels

@profile
def save_or_load_split(dataset, split_name, num_labeled, stratified_split_fn, ml_smote_fn):
    labeled_indices_path = os.path.join(split_cache_dir, f"labeled_indices_{split_name}_{num_labeled}.npy")
    unlabeled_indices_path = os.path.join(split_cache_dir, f"unlabeled_indices_{split_name}_{num_labeled}.npy")
    labeled_features_path = os.path.join(split_cache_dir, f"labeled_features_{split_name}_{num_labeled}.npz")
    labeled_labels_path = os.path.join(split_cache_dir, f"labeled_labels_{split_name}_{num_labeled}.npz")

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

    kmeans_path = "kmeans_model.pkl"
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

    feature_file_path = os.path.join(feature_cache_dir, f"{feature_name}_{subset_name}.npz")

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

# SSL Methods and Strategies
@profile
class SemiSupervisedTrainer:
    def __init__(self, model, device, num_classes=16):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_supervised(self, labeled_loader):
        self.model.train()
        for images, labels in labeled_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def pseudo_labeling(self, unlabeled_loader, threshold=0.7):
        self.model.eval()
        pseudo_labels, pseudo_images = [], []
        with torch.no_grad():
            for images, _ in unlabeled_loader:
                images = images.to(self.device)
                outputs = torch.sigmoid(self.model(images))
                mask = outputs > threshold
                pseudo_labels.append(mask.float().cpu())
                pseudo_images.append(images.cpu())

        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        pseudo_images = torch.cat(pseudo_images, dim=0)
        return TensorDataset(pseudo_images, pseudo_labels)

    def consistency_regularization(self, unlabeled_loader, augmentation_transforms):
        self.model.train()
        consistency_loss = 0.0
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            augmented_images = augmentation_transforms(images)
            original_outputs = self.model(images)
            augmented_outputs = self.model(augmented_images)
            loss = nn.MSELoss()(original_outputs, augmented_outputs)
            consistency_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return consistency_loss / len(unlabeled_loader)

    def entropy_minimization(self, unlabeled_loader):
        self.model.train()
        entropy_loss = 0.0
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            outputs = torch.sigmoid(self.model(images))
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-8), dim=1).mean()
            self.optimizer.zero_grad()
            entropy.backward()
            self.optimizer.step()
            entropy_loss += entropy.item()
        return entropy_loss / len(unlabeled_loader)

@profile
def dynamic_thresholding(epoch, max_epochs):
    return max(0.9 - (epoch / max_epochs) * 0.4, 0.5)

@profile
def self_training_classical(model, X_labeled, y_labeled, X_unlabeled, max_iterations=10, threshold=0.8):

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
        y_labeled = np.vstack([y_labeled, pseudo_labels])

        X_unlabeled = X_unlabeled[~high_confidence_indices]

    return model

@profile
def co_training_classical(model1, model2, X_labeled1, X_labeled2, y_labeled, X_unlabeled1, X_unlabeled2, max_iterations=10):

    for iteration in range(max_iterations):
        print(f"Co-Training Iteration {iteration + 1}/{max_iterations}")

        model1.fit(X_labeled1, y_labeled)
        model2.fit(X_labeled2, y_labeled)
        pseudo_labels1 = model1.predict(X_unlabeled1)
        pseudo_labels2 = model2.predict(X_unlabeled2)
        agreement_indices = np.where(pseudo_labels1 == pseudo_labels2)[0]

        if not agreement_indices.any():
            print("No agreement between models. Stopping co-training.")
            break

        X_labeled1 = np.vstack([X_labeled1, X_unlabeled1[agreement_indices]])
        X_labeled2 = np.vstack([X_labeled2, X_unlabeled2[agreement_indices]])
        y_labeled = np.vstack([y_labeled, pseudo_labels1[agreement_indices]])
        X_unlabeled1 = np.delete(X_unlabeled1, agreement_indices, axis=0)
        X_unlabeled2 = np.delete(X_unlabeled2, agreement_indices, axis=0)

    return model1, model2

# Graph-Based Propagation Implementation
@profile
def graph_based_label_propagation(X_labeled, y_labeled, X_unlabeled, k_neighbors=10):


    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.vstack([y_labeled, np.zeros((X_unlabeled.shape[0], y_labeled.shape[1]))])
    distances = pairwise_distances(X_combined)
    neighbors = np.argsort(distances, axis=1)[:, 1:k_neighbors + 1]
    graph = nx.Graph()

    for i, neigh in enumerate(neighbors):
        for n in neigh:
            graph.add_edge(i, n, weight=np.exp(-distances[i, n]))

    propagated_labels = np.zeros_like(y_combined)
    for i, label in enumerate(y_combined):
        if label.sum() > 0:
            propagated_labels[i] = label
        else:
            neighbor_labels = y_combined[neighbors[i]]
            propagated_labels[i] = np.mean(neighbor_labels, axis=0)

    return propagated_labels[len(X_labeled):]

@profile
def pseudo_label_classical(model, X_unlabeled, threshold=0.9, max_pseudo_labels=None):

    probabilities = model.predict_proba(X_unlabeled)
    if hasattr(probabilities, "toarray"):
        probabilities = probabilities.toarray()
    high_confidence_mask = probabilities.max(axis=1) >= threshold
    pseudo_labels = (probabilities > 0.5).astype(int)
    filtered_pseudo_labels = pseudo_labels[high_confidence_mask]
    filtered_X_unlabeled = X_unlabeled[high_confidence_mask]

    if max_pseudo_labels and len(filtered_X_unlabeled) > max_pseudo_labels:
        indices = np.random.choice(len(filtered_X_unlabeled), max_pseudo_labels, replace=False)
        filtered_pseudo_labels = filtered_pseudo_labels[indices]
        filtered_X_unlabeled = filtered_X_unlabeled[indices]

    print(f"Selected {len(filtered_X_unlabeled)} high-confidence pseudo-labeled samples out of {len(X_unlabeled)} total.")

    return filtered_pseudo_labels, filtered_X_unlabeled

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


@profile
def early_stopping(train_function, validate_function, patience=3, max_epochs=50):
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        train_function() 
        val_loss = validate_function()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break


def fit_with_timeout(clf, X, y, return_dict):

    try:
        clf.fit(X, y)
        return_dict["status"] = "success"
        return_dict["clf"] = clf
    except Exception as e:
        return_dict["status"] = f"failed: {e}"

@profile
# Classical Multilabel Classifiers
def ssl_train_and_evaluate_classical_methods(features, y_train, y_val, y_test, labeled_indices, unlabeled_indices, num_labeled, max_epochs=20, patience=5):
    
    methods = {
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "kNN": KNeighborsClassifier()
    }

    strategies = [
        ("Binary Relevance", BinaryRelevance),
        ("Classifier Chains", ClassifierChain),
        ("Label Powerset", LabelPowerset)
    ]

    feature_compatibility = {
        "Raw": ["Random Forest", "SVM", "XGBoost", "LightGBM", "CatBoost", "kNN"],
        "BoVW": ["Random Forest", "SVM", "XGBoost", "LightGBM", "CatBoost"],
        "HOG": ["Random Forest", "SVM", "XGBoost", "LightGBM", "CatBoost"],
        "LBP": ["Random Forest", "SVM", "XGBoost", "LightGBM", "CatBoost"]
    }


    reduction_methods = {
        "Raw": ("PCA", 100), 
        "BoVW": ("PCA", 50),
        "HOG": ("SVD", 75),
        "LBP": ("PCA", 30)
    }


    results = []

    import csv
    import time
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        hamming_loss, accuracy_score, jaccard_score, matthews_corrcoef
    )
    import os

    results_csv_file = "ssl_detailed_results_COCO_2000.csv"

    if not os.path.exists(results_csv_file):
        with open(results_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Strategy", "Model", "Feature",
                "Hamming Loss", "Accuracy", "Jaccard Score", "Matthews Corrcoef",
                "Precision (Micro)", "Recall (Micro)", "F1-Score (Micro)",
                "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
                "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)",
                "Precision (Samples)", "Recall (Samples)", "F1-Score (Samples)",
                "Training Time (s)", "Validation Time (s)",
                "Labeled Samples", "Pseudo-Labeled Samples",
                "True Positives", "False Positives", "True Negatives", "False Negatives"
            ])

    def save_result_to_csv(result, file_path):
        """Append a single result to the CSV file."""
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Error writing result to {file_path}: {e}")


    for feature_name in ["Raw", "BoVW", "HOG", "LBP"]:
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

        initial_X_train_unlabeled = X_train_unlabeled.copy()

        y_train_labeled = features[labeled_labels_key]

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

        initial_X_train_unlabeled_reduced = X_train_unlabeled_reduced.copy()

        if len(X_train_labeled) != len(y_train_labeled):
            raise ValueError(
                f"Mismatch between features and labels for {feature_name}: "
                f"{len(X_train_labeled)} features vs {len(y_train_labeled)} labels"
            )

        for strategy_name, strategy in strategies:
            for model_name, model in methods.items():
                if model_name not in feature_compatibility[feature_name]:
                    print(f"Skipping incompatible combination: {model_name} with {feature_name}")
                    continue

                if model_name == "CatBoost" and strategy_name == "Label Powerset":
                    print(f"Skipping combination (timeout): {strategy_name} with {model_name}")
                    continue

                if (
                    (completed_experiments["Labeled_Samples"] == num_labeled) &
                    (completed_experiments["Feature"] == feature_name) &
                    (completed_experiments["Model"] == model_name) &
                    (completed_experiments["Strategy"] == strategy_name)
                ).any():
                    print(f"Skipping already completed experiment: {strategy_name} with {model_name} using {feature_name}")
                    continue

                print(f"Evaluating {strategy_name} with {model_name} using {feature_name}")
                clf = strategy(model)

                training_start_time = time.time()
                best_val_loss = float('inf')
                no_improvement_epochs = 0

                current_X_train = X_train_labeled_reduced
                current_y_train = y_train_labeled
                X_train_unlabeled_reduced = initial_X_train_unlabeled_reduced.copy()                

                combined_y_train = y_train_labeled

                for epoch in range(max_epochs):
                    print(f"Epoch {epoch + 1}/{max_epochs}")

                    #Timeout logic for clf.fit()
                    manager = Manager()
                    return_dict = manager.dict()
                    process = Process(target=fit_with_timeout, args=(clf, current_X_train, current_y_train, return_dict))
                    
                    process.start()
                    process.join(timeout=900)  # 15 minutes

                    if process.is_alive():
                        process.terminate()
                        process.join()
                        print(f"Training for {strategy_name} with {model_name} using {feature_name} exceeded 15 minutes. Aborting.")
                        best_clf = None  # Indicate that this model failed to train
                        break  

                    if return_dict.get("status") != "success":
                        print(f"Training failed for {strategy_name} with {model_name} using {feature_name}: {return_dict.get('status')}")
                        best_clf = None  # Indicate failure
                        break 

                    #clf.fit(current_X_train, current_y_train)
                    clf = return_dict.get("clf", clf)
                    
                    training_end_time = time.time()
                    training_time = training_end_time - training_start_time

                    if X_train_unlabeled_reduced.shape[0] == 0:
                        print("No more unlabeled samples remaining. Skipping pseudo-labeling.")
                        break
                    else:
                        pseudo_labels, pseudo_features = pseudo_label_classical(
                            clf, X_train_unlabeled_reduced, threshold=0.91, max_pseudo_labels=1000
                        )

                        if len(pseudo_labels) > 0:
                            combined_X_train = np.vstack([current_X_train, pseudo_features])
                            combined_y_train = np.vstack([current_y_train, pseudo_labels])
                            high_confidence_mask = np.isin(X_train_unlabeled_reduced, pseudo_features).all(axis=1)
                            X_train_unlabeled_reduced = X_train_unlabeled_reduced[~high_confidence_mask]

                            print(f"Remaining unlabeled samples: {len(X_train_unlabeled_reduced)}")

                            current_X_train, current_y_train = shuffle(combined_X_train, combined_y_train, random_state=42)
                        else: 
                            print("No pseudo-labels added in this epoch.")

                    validation_start_time = time.time()

                    y_val_pred = clf.predict(X_val_reduced)
                    val_loss = hamming_loss(y_val, y_val_pred)
                    print(f"Validation Loss: {val_loss:.4f}")


                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_epochs = 0
                        print("Validation loss improved. Saving model...")
                        best_clf = clf
                    else:
                        no_improvement_epochs += 1
                        print(f"No improvement for {no_improvement_epochs} epoch(s).")

                    if no_improvement_epochs >= patience:
                        print("Early stopping triggered.")
                        break

                if best_clf is None:
                    print(f"Skipping evaluation for {strategy_name} with {model_name} using {feature_name} due to training failure.")
                    continue
                else:
                    print(f"Training completed for {strategy_name} with {model_name}. Proceeding to evaluation.")
                    
                if 'combined_y_train' not in locals():
                    combined_y_train = current_y_train

                pseudo_labeled_count = len(combined_y_train) - len(labeled_indices)
                                
                y_test_pred = best_clf.predict(X_test_reduced)

                y_test_dense = y_test.toarray() if hasattr(y_test, "toarray") else y_test
                y_test_pred_dense = y_test_pred.toarray() if hasattr(y_test_pred, "toarray") else y_test_pred
                y_test_flat = y_test_dense.ravel()
                y_test_pred_flat = y_test_pred_dense.ravel()

                hamming = hamming_loss(y_test_dense, y_test_pred_dense)
                accuracy = accuracy_score(y_test_dense, y_test_pred_dense)
                jaccard = jaccard_score(y_test_dense, y_test_pred_dense, average="samples")
                matthews = matthews_corrcoef(y_test_flat, y_test_pred_flat)
                report = classification_report(y_test_dense, y_test_pred_dense, output_dict=True, zero_division=0)

                # Confusion matrix values
                try:
                    cm = confusion_matrix(y_test_dense.ravel(), y_test_pred_dense.ravel())
                    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)
                except ValueError:
                    tn, fp, fn, tp = None, None, None, None
                    print("Confusion matrix not applicable for multilabel.")

                micro_avg = report["micro avg"]
                macro_avg = report["macro avg"]
                weighted_avg = report["weighted avg"]
                samples_avg = report.get("samples avg", {"precision": 0, "recall": 0, "f1-score": 0})

                validation_end_time = time.time()
                validation_time = validation_end_time - validation_start_time

                result_row = [
                    strategy_name, model_name, feature_name,
                    hamming, accuracy, jaccard, matthews,
                    micro_avg["precision"], micro_avg["recall"], micro_avg["f1-score"],
                    macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"],
                    weighted_avg["precision"], weighted_avg["recall"], weighted_avg["f1-score"],
                    samples_avg["precision"], samples_avg["recall"], samples_avg["f1-score"],
                    training_time, validation_time,
                    len(labeled_indices), pseudo_labeled_count,
                    tp, fp, tn, fn
                ]

                save_result_to_csv(result_row, results_csv_file)

                results.append({
                    "Strategy": strategy_name,
                    "Model": model_name,
                    "Feature": feature_name,
                    "Hamming Loss": hamming,
                    "Accuracy": accuracy,
                    "Labeled Samples": len(labeled_indices),
                    "Pseudo-Labeled Samples": pseudo_labeled_count
                })

                log_experiment(num_labeled, feature_name, model_name, strategy_name)

    return results


@profile
#Evalutation
def evaluate_resnet(model, val_loader, test_loader, device):
    model.eval()

    print("\nComputing optimal thresholds from validation set...")
    val_labels = []
    val_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            val_predictions.append(outputs.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_predictions = np.vstack(val_predictions)
    val_labels = np.vstack(val_labels)

    optimal_thresholds = []
    for i in range(val_predictions.shape[1]):
        precision, recall, thresholds = precision_recall_curve(val_labels[:, i], val_predictions[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds.append(thresholds[optimal_idx])

    optimal_thresholds = np.array(optimal_thresholds)
    print("Optimal thresholds computed:", optimal_thresholds)

    print("\nEvaluating on the test set...")
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            test_predictions.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_predictions = np.vstack(test_predictions)
    test_labels = np.vstack(test_labels)

    binary_predictions = (test_predictions > optimal_thresholds).astype(int)

    print("Classification Report:")
    print(classification_report(test_labels, binary_predictions, zero_division=0))
    print(f"Hamming Loss: {hamming_loss(test_labels, binary_predictions):.4f}")
    print(f"Accuracy: {accuracy_score(test_labels, binary_predictions):.4f}")
    print("Evaluation completed.")

@profile
def save_results_to_csv(results, csv_file_path):
    file_exists = os.path.exists(csv_file_path)
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        csv_file_path,
        mode='a',  # Append mode
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


images_dir = './coco_subset_10Kv2/images'
annotations_path = './coco_subset_10Kv2/annotations/coco_subset_annotations.json'

augmented_dataset_path = "./coco_cache/augmented_dataset.pkl"
train_val_test_splits_path = "./coco_cache/train_val_test_splits.pkl"
features_cache_dir = "./coco_cache/features/"
label_cache_dir = "./cache_coco/labels/"
feature_cache_dir = "./cache_coco/features/"
split_cache_dir = "./cache_coco/splits"


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
        full_dataset = COCODataset(images_dir, annotations_path, transform=transform)

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

    test_labels_path = os.path.join(label_cache_dir, "test_labels.npy")
    if os.path.exists(test_labels_path):
        print("Loading test labels from cache...")
        test_labels = np.load(test_labels_path, allow_pickle=True)
    else:
        print("Extracting test labels...")
        test_labels = np.array([label.numpy() for _, label in test_dataset])
        np.save(test_labels_path, test_labels)

    val_labels_path = os.path.join(label_cache_dir, "val_labels.npy")
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
            feature_file_path = os.path.join(feature_cache_dir, f"{split_name}_{feature_name}.npz")
            
            if os.path.exists(feature_file_path):
                print(f"Loading precomputed {feature_name} features for {split_name}...")
                data = np.load(feature_file_path, allow_pickle=True)
                features[f"{split_name}_{feature_name}"] = data["features"]
            else:
                print(f"Extracting {feature_name} features for {split_name}...")
                extracted_features, _ = load_or_extract_features(dataset, feature_name, split_name, extractors[feature_name])
                
                np.savez_compressed(feature_file_path, features=extracted_features)
                features[f"{split_name}_{feature_name}"] = extracted_features

    progress_file = "ssl_experiment_progress_COCO_2000.csv"

    if os.path.exists(progress_file):
        completed_experiments = pd.read_csv(progress_file)
    else:
        completed_experiments = pd.DataFrame(columns=["Labeled_Samples", "Feature", "Model", "Strategy"])


    labeled_data_sizes = [2000] # 1500, , 500, 1000
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

    save_results_to_csv(all_results, "ssl_results_COCO_2000.csv")
