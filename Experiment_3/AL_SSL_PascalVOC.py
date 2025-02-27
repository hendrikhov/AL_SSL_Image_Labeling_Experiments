import torch
import numpy as np
import cv2
import json
import torchvision.transforms as T
import os
import pickle
import pandas as pd
import re
import hashlib
import warnings
import shutil
from itertools import combinations
from sklearn.utils import shuffle
from collections import Counter
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from PIL import Image
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score, log_loss, roc_auc_score, average_precision_score, roc_curve,confusion_matrix
from scipy.stats import f_oneway, ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.exceptions import UndefinedMetricWarning
from collections import defaultdict

class PascalVOCDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.img_dir = img_dir
        self.annotation_path = annotation_path
        self.transform = transform
        
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.labels = list(set([label for annotation in self.annotations for label in annotation['labels']]))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.labels)}
        
        expected_classes = 20
        if len(self.labels)!= expected_classes:
            raise ValueError(f"Expected {expected_classes} categories, but found {len(self.labels)}.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_filename = annotation['filename']
        labels = annotation['labels']
        
        img_path = f"{self.img_dir}/{image_filename}"
        img = Image.open(img_path).convert("RGB")
        
        label = [0] * len(self.labels)
        for label_name in labels:
            label[self.label_to_idx[label_name]] = 1

        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32)


class FeatureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def save_processed_data(file_path, images, labels):
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    np.savez_compressed(file_path, images=images, labels=labels)

def load_processed_data(file_path):
    data = np.load(file_path)
    return data['images'], data['labels']

def save_training_stats(file_path, stats):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=4)

def process_coco_output(dataset, target_size=(224, 224), save_path=None, stats_path=None):
    if save_path and os.path.exists(save_path):
        print(f"Loading processed data from {save_path}...")
        images, labels = load_processed_data(save_path)
        if stats_path and os.path.exists(stats_path):
            print(f"Loading stats from {stats_path}...")
        return images, labels

    images = []
    labels = []
    stats = {
        "total_images": 0,
        "positive_single_label": 0,
        "positive_dual_label": 0,
        "negative_samples": 0
    }

    for img, label in dataset:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.numpy()

        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        stats["total_images"] += 1
        num_labels = np.sum(label)
        if num_labels == 1:
            stats["positive_single_label"] += 1
        elif num_labels == 2:
            stats["positive_dual_label"] += 1
        else:
            stats["negative_samples"] += 1        

        images.append(img)
        labels.append(label)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32)
    
    if save_path:
        print(f"Saving processed data to {save_path}...")
        save_processed_data(save_path, images, labels)
    
    if stats_path:
        print(f"Saving stats to {stats_path}...")
        save_training_stats(stats_path, stats)

    return images, labels

augmentation_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=5),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])

def augment_data(images, base_augmentations=5):
    num_augments = int(base_augmentations)
    augmented_images = []
    for img in images:
        img_pil = T.ToPILImage()(img)
        for _ in range(num_augments):
            augmented_img = augmentation_transforms(img_pil)
            augmented_images.append(np.array(augmented_img))
    return np.array(augmented_images)

def extract_hog_features(dataset):
    
    if len(dataset) == 0:
        raise ValueError("The provided dataset is empty. Ensure valid data is passed for HOG extraction.")
    
    features, labels = [], []
    for img, lbl in dataset:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor): 
            img = img.numpy()
        elif len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3) 
        
        if img.ndim == 3 and img.shape[0] in [3, 1]:
            img = np.transpose(img, (1, 2, 0)) #img = img.permute(1, 2, 0).numpy() 
        #else:
        #    raise ValueError(f"Unexpected image shape: {img.shape}")

        if img.ndim == 3 and img.shape[-1] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            img_gray = img
        else:
            raise ValueError(f"Unexpected image shape for grayscale conversion: {img.shape}")

        hog_features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.numpy()
        labels.append(lbl.flatten() if isinstance(lbl, (np.ndarray, list)) else lbl)
        features = np.array(features)
        labels = np.array(labels)

        if labels.ndim > 1:
            labels = np.squeeze(labels)

        return features, labels

def extract_lbp_features(dataset, radius=3, n_points=24):

    if len(dataset) == 0:
        raise ValueError("The provided dataset is empty. Ensure valid data is passed for LBP extraction.")
    
    features, labels = [], []
    for img, lbl in dataset:
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor): 
            img = img.numpy()
        elif len(img.shape) == 1:
            img = img.numpy().reshape(224, 224, 3) 
        
        if img.ndim == 3 and img.shape[0] in [3, 1]:
            img = np.transpose(img, (1, 2, 0)) #img = img.permute(1, 2, 0).numpy() 
        #else:
        #    raise ValueError(f"Unexpected image shape: {img.shape}")

        if img.ndim == 3 and img.shape[-1] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            img_gray = img
        else:
            raise ValueError(f"Unexpected image shape for grayscale conversion: {img.shape}")

        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.append(hist)
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.numpy()
        elif isinstance(lbl, np.ndarray):
            lbl = lbl
        elif isinstance(lbl, (float, int)):
            lbl = np.array([lbl])
        else:
            raise ValueError(f"Unexpected label type: {type(lbl)}")
        labels.append(lbl.flatten() if isinstance(lbl, np.ndarray) else lbl)

    return np.array(features), np.array(labels)

def extract_bovw_features(dataset, kmeans=None, n_clusters=50):
    sift = cv2.SIFT_create()
    descriptors = []
    labels = []

    if len(dataset) == 0:
        raise ValueError("The provided dataset is empty. Ensure valid data is passed for BoVW extraction.")

    print("Extracting SIFT descriptors...")
    for idx, (img, lbl) in enumerate(dataset):
        try:
            if isinstance(img, Image.Image):
                img = np.array(img)
            elif isinstance(img, torch.Tensor): 
                img = img.numpy()

            if img.ndim == 3 and img.shape[0] in [3, 1]:
                img = np.transpose(img, (1, 2, 0)) 
            
            if img.ndim == 3 and img.shape[-1] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 2:
                img_gray = img
            else:
                print(f"Index {idx}: Skipping invalid image shape {img.shape}.")
                continue

            if img_gray.dtype != np.uint8:
                img_gray = (img_gray * 255).astype(np.uint8)

            _, desc = sift.detectAndCompute(img_gray, None)
            if desc is not None:
                descriptors.append(desc)
                labels.append(lbl)
            else:
                print(f"Index {idx}: No descriptors found.")
        except Exception as e:
            print(f"Index {idx}: Error extracting descriptors: {e}")

    if len(descriptors) == 0:
        raise ValueError("No valid descriptors found in the dataset.")

    descriptors = np.vstack(descriptors)
    print(f"Total descriptors shape: {descriptors.shape}")

    if kmeans is None:
        print("Training KMeans for BoVW...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(descriptors)
    else:
        print("Using pre-trained KMeans model for BoVW.")
        if not hasattr(kmeans, "cluster_centers_") or kmeans.cluster_centers_ is None:
            raise ValueError("KMeans model does not have valid cluster centers.")

        print(f"KMeans cluster centers shape: {kmeans.cluster_centers_.shape}")
        if kmeans.cluster_centers_.shape[1] != descriptors.shape[1]:
            raise ValueError(
                f"KMeans cluster centers have dimension {kmeans.cluster_centers_.shape[1]} "
                f"but descriptors have dimension {descriptors.shape[1]}."
            )

        if kmeans.cluster_centers_.shape[0] != n_clusters:
            raise ValueError(
                f"KMeans was initialized with {n_clusters} clusters, but has "
                f"{kmeans.cluster_centers_.shape[0]} clusters."
            )

        if hasattr(kmeans, "inertia_"):
            print(f"KMeans inertia: {kmeans.inertia_}")
            if kmeans.inertia_ == 0 or np.isnan(kmeans.inertia_):
                raise ValueError("KMeans inertia is zero or NaN, indicating a potential problem with clustering.")

    try:
        test_sample = descriptors[:10]
        test_clusters = kmeans.predict(test_sample)
        print(f"Test prediction successful. Predicted clusters: {test_clusters}")
    except Exception as e:
        raise RuntimeError(f"KMeans predict function failed: {e}")

    cluster_assignments = kmeans.predict(descriptors)
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Cluster assignment distribution: {distribution}")
    if len(distribution) < n_clusters:
        print("Warning: Not all clusters have been assigned descriptors.")

    print("Generating BoVW histograms for the dataset...")
    histograms = []
    for idx, (img, _) in enumerate(dataset):
        try:
            if isinstance(img, Image.Image):
                img = np.array(img)
            elif isinstance(img, torch.Tensor):
                img = img.numpy()

            if img.ndim == 3 and img.shape[0] in [3, 1]:
                img = np.transpose(img, (1, 2, 0))

            if img.ndim == 3 and img.shape[-1] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 2:
                img_gray = img
            else:
                print(f"Index {idx}: Skipping invalid image shape {img.shape} during histogram creation.")
                histograms.append(np.zeros(kmeans.n_clusters))
                continue

            if img_gray.dtype != np.uint8:
                img_gray = (img_gray * 255).astype(np.uint8)

            _, desc = sift.detectAndCompute(img_gray, None)
            if desc is not None:
                hist = np.zeros(n_clusters)
                cluster_assignments = kmeans.predict(desc)
                for cluster_id in cluster_assignments:
                    hist[cluster_id] += 1
                histograms.append(hist / np.sum(hist))  # Normalize histogram
                #print(f"Image {idx}: Histogram successfully generated.")
            else:
                print(f"Image {idx}: No descriptors available.")
                histograms.append(np.zeros(n_clusters))
        except Exception as e:
            print(f"Error generating histogram for image at index {idx}: {e}")
            histograms.append(np.zeros(n_clusters))

    if len(histograms) == 0:
        raise ValueError("No BoVW histograms were generated. Check input images and descriptor computation.")

    print(f"Generated {len(histograms)} histograms for the dataset.")
    return np.array(histograms), np.array(labels), kmeans

def extract_features_for_dataset(dataset, save_dir, n_clusters=50):
    os.makedirs(save_dir, exist_ok=True)

    hog_path = os.path.join(save_dir, "hog_features.npz")
    lbp_path = os.path.join(save_dir, "lbp_features.npz")
    bovw_path = os.path.join(save_dir, "bovw_features.npz")
    kmeans_path = os.path.join(save_dir, "kmeans.pkl")

    if not os.path.exists(hog_path):
        print("Extracting HOG features...")
        hog_features, hog_labels = extract_hog_features(dataset)
        np.savez_compressed(hog_path, features=hog_features, labels=hog_labels)
    else:
        print("HOG features already exist. Skipping extraction.")

    if not os.path.exists(lbp_path):
        print("Extracting LBP features...")
        lbp_features, lbp_labels = extract_lbp_features(dataset)
        np.savez_compressed(lbp_path, features=lbp_features, labels=lbp_labels)
    else:
        print("LBP features already exist. Skipping extraction.")

    if not os.path.exists(bovw_path) or not os.path.exists(kmeans_path):
        print("Extracting BoVW features...")
        bovw_features, bovw_labels, kmeans = extract_bovw_features(dataset, n_clusters=n_clusters)
        np.savez_compressed(bovw_path, features=bovw_features, labels=bovw_labels)
        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
    else:
        print("BoVW features and KMeans model already exist. Skipping extraction.")

    print("Feature extraction for the dataset completed.")

def save_training_set(file_path, train_data, train_labels):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(file_path, train_data=train_data, train_labels=train_labels)

def load_training_set(file_path):
    data = np.load(file_path)
    return data['train_data'], data['train_labels']

def process_class(class_label, dataset, labels, target_labeled_count, augment_threshold, base_augmentations, save_dir, kmeans=None, shared_pool=None):
    print(f"Processing class {class_label}...")

    file_path = os.path.join(save_dir, f"class_{class_label}_training_set.npz")
    lbp_path = os.path.join(save_dir, f"class_{class_label}_lbp_features.npz")
    bovw_path = os.path.join(save_dir, f"class_{class_label}_bovw_features.npz")
    hog_path = os.path.join(save_dir, f"class_{class_label}_hog_features.npz")

    single_label_mask = (labels.sum(axis=1) == 1) & (labels[:, class_label] == 1)
    single_label_indices = np.where(single_label_mask)[0]
    dual_label_mask = (labels.sum(axis=1) == 2) & (labels[:, class_label] == 1)
    dual_label_indices = np.where(dual_label_mask)[0]

    total_label_count = labels.sum()
    class_label_count = labels[:, class_label].sum()
    num_positives = max(3, int(target_labeled_count * (class_label_count / total_label_count)))
    num_negatives = target_labeled_count - num_positives # hand over num negative for the pool

    selected_positives = set(single_label_indices[:min(len(single_label_indices), num_positives)])
    remaining_positives_needed = num_positives - len(selected_positives)
    if remaining_positives_needed > 0:
        dual_shared_counts = Counter(np.where(labels[dual_label_indices])[1])
        sorted_dual_indices = sorted(
            dual_label_indices,
            key=lambda idx: dual_shared_counts.most_common(1)[0][1],
            reverse=True,
        )
        selected_positives.update(sorted_dual_indices[:remaining_positives_needed])

    positive_data = dataset[list(selected_positives)]
    augmented_positives = []

    num_positives_selected = len([idx for idx in selected_positives if idx in single_label_indices])
    print(f"Class {class_label} -> {num_positives} positive samples selected.")

    if shared_pool is not None:
        shared_pool.update(selected_positives)

    if os.path.exists(file_path) and os.path.exists(lbp_path) and os.path.exists(bovw_path):
        try:
            train_data, train_labels = load_training_set(file_path)
            lbp_features = np.load(lbp_path)["features"]
            bovw_features = np.load(bovw_path)["features"]
            hog_features = np.load(hog_path)["features"]
            """
            if len(train_data) != len(train_labels):
                raise ValueError(f"Mismatch between training data and labels for class {class_label}.")
            if len(train_data) != lbp_features.shape[0] or len(train_data) != bovw_features.shape[0] or len(train_data) != hog_features.shape[0]:
                raise ValueError(f"Mismatch between training data and feature shapes for class {class_label}.")
            """
            class_stats = {
                "single_label_before_augmentation": len(single_label_indices),
                "dual_label_before_augmentation": len(dual_label_indices),
                "single_labels_selected_for_training": len([idx for idx in selected_positives if idx in single_label_indices]),
                "dual_labels_selected_for_training": len([idx for idx in selected_positives if idx in dual_label_indices]),
                "single_label_after_augmentation": len(train_labels[train_labels == 1]),
                "dual_label_after_augmentation": 0,
                "negative_samples_added": num_negatives
            }  

            print(f"Validation successful for class {class_label}. Using existing files.")
            return class_label, (train_data, train_labels), class_stats, num_negatives, shared_pool

        except Exception as e:
            print(f"Error during validation of existing files for class {class_label}: {e}")
            print("Reprocessing the class from scratch...")

    class_stats = {
        "single_label_before_augmentation": len(single_label_indices),
        "dual_label_before_augmentation": len(dual_label_indices),
        "single_labels_selected_for_training": len([idx for idx in selected_positives if idx in single_label_indices]),
        "dual_labels_selected_for_training": len([idx for idx in selected_positives if idx in dual_label_indices]),
        "single_label_after_augmentation": 0,
        "dual_label_after_augmentation": 0,
        "negative_samples_added": 0
    }    

    for idx in selected_positives:
        is_single_label = idx in single_label_indices

        if is_single_label:
            if num_positives_selected < 20:
                augment_factor = base_augmentations * 4
            elif num_positives_selected < augment_threshold:
                augment_factor = base_augmentations * 2
            else:
                augment_factor = base_augmentations
        else:  # Dual-label case
            if len(single_label_indices) < augment_threshold:
                augment_factor = base_augmentations
            else:
                augment_factor = max(1, base_augmentations // 2)

        augmented_images = augment_data([dataset[idx]], base_augmentations=augment_factor)
        #print(f"Index {idx}: Augmented {len(augmented_images)} images.")
        augmented_positives.extend(augmented_images)

        if is_single_label:
            class_stats["single_label_after_augmentation"] += len(augmented_images)
        else:
            class_stats["dual_label_after_augmentation"] += len(augmented_images)

    if augmented_positives:
        augmented_positives = np.array(augmented_positives)
        print(f"Augmented positives shape: {augmented_positives.shape}, dtype: {augmented_positives.dtype}")
        print(f"Original positive data shape before stacking: {positive_data.shape}")
        positive_data = np.vstack([positive_data, augmented_positives])
        train_labels = np.hstack([np.ones(len(selected_positives)), np.ones(len(augmented_positives))])
    else:
        train_labels = np.ones(len(selected_positives))

    if len(positive_data) == 0:
        raise ValueError(f"No positive data found for class {class_label}.")

    expected_length = len(selected_positives) + len(augmented_positives)
    actual_length = len(positive_data)
    if actual_length != expected_length:
        raise ValueError(
            f"Dataset length mismatch for class {class_label}. "
            f"Expected {expected_length}, but got {actual_length}."
        )
    else:
        print(f"Dataset length check passed for class {class_label}: {actual_length} samples.")

    train_data = positive_data
    print(f"Final dataset size for class {class_label}: {train_data.shape[0]} samples.")

    #selected_negatives = set(negative_indices[:min(len(negative_indices), num_negatives)])
    #negative_data = dataset[list(selected_negatives)]
    #class_stats["negative_samples_added"] = len(negative_data)

    #train_data = np.vstack([positive_data, negative_data])
    #train_labels = np.hstack([np.ones(len(positive_data)), np.zeros(len(negative_data))])

    train_labels = np.ones(len(positive_data))

    print(f"Extracting features for class {class_label} training set...")
    train_dataset = FeatureDataset(train_data, train_labels)
    hog_features, _ = extract_hog_features(train_dataset)
    lbp_features, _ = extract_lbp_features(train_dataset)
    bovw_features, _, _ = extract_bovw_features(train_dataset, kmeans=kmeans)

    os.makedirs(save_dir, exist_ok=True)
    save_training_set(file_path, train_data, train_labels)

    np.savez_compressed(os.path.join(save_dir, f"class_{class_label}_hog_features.npz"), features=hog_features)
    np.savez_compressed(os.path.join(save_dir, f"class_{class_label}_lbp_features.npz"), features=lbp_features)
    np.savez_compressed(os.path.join(save_dir, f"class_{class_label}_bovw_features.npz"), features=bovw_features)

    print(f"Saving training set for class {class_label} to {file_path}...")
    
    return class_label, (train_data, train_labels), class_stats, num_negatives, shared_pool

def create_binary_training_sets_with_augmentation(dataset, labels, target_fraction=0.05, augment_threshold=100, base_augmentations=6, save_dir="training_sets", stats_path=None, n_jobs=-1, kmeans = None, precomputed_features=None):
    num_classes = labels.shape[1]
    total_images = labels.shape[0]
    target_labeled_count = int(total_images * target_fraction)

    shared_labeled_pool = set()
    training_sets = {}

    # Parallel processing for each class
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_class)(
            class_label, dataset, labels, target_labeled_count, augment_threshold, base_augmentations, save_dir, kmeans, shared_labeled_pool
        ) for class_label in range(num_classes)
    )
    training_sets = {class_label: training_set for class_label, training_set, _, _, _ in results}
    stats = {class_label: class_stats for class_label, _, class_stats, _, _ in results}
    num_negatives_per_class = {class_label: remaining_negatives for class_label, _, _, remaining_negatives, _ in results}
    for _, _, _, _, pool in results:
        shared_labeled_pool.update(pool) # Update the positive pool globally

    print(f"Global positive labeled pool size: {len(shared_labeled_pool)} unique labeled images.")

    precomputed_lbp_features = precomputed_features["lbp"]
    precomputed_bovw_features = precomputed_features["bovw"]
    precomputed_hog_features = precomputed_features["hog"]

    print("Adding negative samples for all classes...")
    for class_label in range(num_classes):
        file_path = os.path.join(save_dir, f"class_{class_label}_training_set.npz")
        lbp_path = os.path.join(save_dir, f"class_{class_label}_lbp_features.npz")
        bovw_path = os.path.join(save_dir, f"class_{class_label}_bovw_features.npz")        
        hog_path = os.path.join(save_dir, f"class_{class_label}_hog_features.npz")
        
        train_data, train_labels = load_training_set(file_path)
        num_existing_negatives = np.sum(train_labels == 0)
        num_needed_negatives = num_negatives_per_class[class_label]
        if num_existing_negatives >= num_needed_negatives:
            print(f"Skipping negative selection for class {class_label} (already {num_existing_negatives} negatives).")
            continue

        print(f"Adding negatives for class {class_label}...")

        lbp_features = np.load(lbp_path)["features"]
        bovw_features = np.load(bovw_path)["features"]
        hog_features = np.load(hog_path)["features"]

        negative_indices = [idx for idx in shared_labeled_pool if labels[idx, class_label] == 0]
        num_negatives = min(len(negative_indices), num_negatives_per_class[class_label])
        selected_negatives = shuffle(negative_indices)[:num_negatives]

        negative_data = dataset[selected_negatives]
        train_data = np.vstack([train_data, negative_data])
        train_labels = np.hstack([train_labels, np.zeros(len(negative_data))])
        
        stats[class_label]["negative_samples_added"] = len(negative_data)

        valid_lbp_negatives = [idx for idx in selected_negatives if idx < precomputed_lbp_features.shape[0]]
        valid_bovw_negatives = [idx for idx in selected_negatives if idx < precomputed_bovw_features.shape[0]]
        valid_hog_negatives = [idx for idx in selected_negatives if idx < precomputed_hog_features.shape[0]]

        invalid_lbp_indices = [idx for idx in selected_negatives if idx not in valid_lbp_negatives]
        invalid_bovw_indices = [idx for idx in selected_negatives if idx not in valid_bovw_negatives]
        invalid_hog_indices = [idx for idx in selected_negatives if idx not in valid_hog_negatives]

        if len(valid_lbp_negatives) < len(selected_negatives):
            print(f"Warning: Skipping {len(selected_negatives) - len(valid_lbp_negatives)} invalid indices for LBP features.")
        if len(valid_bovw_negatives) < len(selected_negatives):
            print(f"Warning: Skipping {len(selected_negatives) - len(valid_bovw_negatives)} invalid indices for BoVW features.")
        if len(valid_hog_negatives) < len(selected_negatives):
            print(f"Warning: Skipping {len(selected_negatives) - len(valid_hog_negatives)} invalid indices for HOG features.")

        lbp_negative_features = precomputed_lbp_features[valid_lbp_negatives]
        bovw_negative_features = precomputed_bovw_features[valid_bovw_negatives]
        hog_negative_features = precomputed_hog_features[valid_hog_negatives]

        if invalid_lbp_indices:
            print("Manually extracting LBP features for invalid indices...")
            invalid_lbp_data = dataset[invalid_lbp_indices]
            manual_lbp_features, _ = extract_lbp_features(FeatureDataset(invalid_lbp_data, np.zeros(len(invalid_lbp_data))))
            if len(manual_lbp_features) > 0:
                lbp_negative_features = np.vstack([lbp_negative_features, manual_lbp_features])
                print(f"Manually extracted LBP features added. New shape: {lbp_negative_features.shape}")
            else:
                print("Warning: No valid LBP features could be manually extracted.")

        if invalid_bovw_indices:
            print("Manually extracting BoVW features for invalid indices...")
            invalid_bovw_data = dataset[invalid_bovw_indices]
            manual_bovw_features, _, _ = extract_bovw_features(FeatureDataset(invalid_bovw_data, np.zeros(len(invalid_bovw_data))), kmeans=kmeans)
            if len(manual_bovw_features) > 0:
                bovw_negative_features = np.vstack([bovw_negative_features, manual_bovw_features])
                print(f"Manually extracted BoVW features added. New shape: {bovw_negative_features.shape}")
            else:
                print("Warning: No valid BoVW features could be manually extracted.")

        if invalid_hog_indices:
            print("Manually extracting HOG features for invalid indices...")
            invalid_hog_data = dataset[invalid_hog_indices]
            manual_hog_features, _ = extract_hog_features(FeatureDataset(invalid_hog_data, np.zeros(len(invalid_hog_data))))

            if manual_hog_features.size > 0:
                if hog_negative_features.size > 0:
                    if hog_negative_features.shape[1] == manual_hog_features.shape[1]:
                        hog_negative_features = np.concatenate([hog_negative_features, manual_hog_features], axis=0)
                        print(f"Manually extracted HOG features added. New shape: {hog_negative_features.shape}")
                    else:
                        print("Warning: HOG feature dimension mismatch. Expected {hog_negative_features.shape[1]}, got {manual_hog_features.shape[1]}. Re-extracting HOG features for ALL positive labels.")
                        positive_indices = np.where(train_labels == 1)[0]
                        positive_hog_data = train_data[positive_indices]
                        new_hog_features, _ = extract_hog_features(FeatureDataset(positive_hog_data, np.ones(len(positive_hog_data))))

                        if new_hog_features.shape[1] == manual_hog_features.shape[1]:
                            hog_features = new_hog_features  # Replace all HOG features with the new extracted ones
                            hog_negative_features = manual_hog_features
                            print(f"HOG features re-extracted successfully. New shape: {hog_features.shape}")
                        else:
                            print(f"Critical Error: Re-extracted HOG features still don't match. Skipping addition.")
                else:
                    hog_negative_features = manual_hog_features
                    print(f"HOG negative features initialized with manually extracted features. New shape: {hog_negative_features.shape}")
            else:
                print("Warning: No valid HOG features could be manually extracted.")

        lbp_features = np.vstack([lbp_features, lbp_negative_features])
        bovw_features = np.vstack([bovw_features, bovw_negative_features])
        hog_features = np.vstack([hog_features, hog_negative_features])

        print(f"LBP Features: {lbp_features.shape}")
        print(f"BoVW Features: {bovw_features.shape}")
        print(f"HOG Features: {hog_features.shape}")

        print(f"Saving updated training set and features for class {class_label}...")
        save_training_set(file_path, train_data, train_labels)
        np.savez_compressed(lbp_path, features=lbp_features)
        np.savez_compressed(bovw_path, features=bovw_features)
        np.savez_compressed(hog_path, features=hog_features)

    if stats_path:
        print(f"Saving training stats to {stats_path}...")
        save_training_stats(stats_path, stats)

    shared_pool_path = os.path.join(save_dir, "shared_labeled_pool.npz")
    np.savez_compressed(shared_pool_path, labeled_pool=np.array(list(shared_labeled_pool), dtype=np.int64))
    print(f"Shared labeled pool saved at: {shared_pool_path}")

    return training_sets, shared_labeled_pool

def main_active_learning_with_voting_dynamic(dataset, labels, save_dir, bovw_features_path=None, lbp_features_path=None, stats_path= None, AL_results_dir = None, target_fraction = 0.1, global_labeled_pool=None):

    if bovw_features_path and os.path.exists(bovw_features_path):
        print(f"Loading dataset-wide BoVW features from {bovw_features_path}...")
        dataset_bovw_features = np.load(bovw_features_path)['features']
    else:
        raise FileNotFoundError(f"Dataset-wide BoVW features not found at {bovw_features_path}.")

    if lbp_features_path and os.path.exists(lbp_features_path):
        print(f"Loading dataset-wide LBP features from {lbp_features_path}...")
        dataset_lbp_features = np.load(lbp_features_path)['features']
    else:
        raise FileNotFoundError(f"Dataset-wide LBP features not found at {lbp_features_path}.")

    updated_training_dir = os.path.join(AL_results_dir)
    os.makedirs(updated_training_dir, exist_ok=True)
    
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            training_stats = json.load(f)
    else:
        print(f"Warning: Stats file not found at {stats_path}. Initializing an empty stats dictionary.")
        training_stats = {}

    total_classes = labels.shape[1]
    total_samples = labels.shape[0]
    class_performance = {}
    max_labeled_samples = int(target_fraction * total_samples)
    
    for class_label in range(total_classes):
        training_set_path = os.path.join(save_dir, f"class_{class_label}_training_set.npz")
        if str(class_label) in training_stats and isinstance(training_stats[str(class_label)], list) and len(training_stats[str(class_label)]) > 0:
            class_performance[class_label] = training_stats[str(class_label)][-1].get("f1", 0.0)
        else:
            print(f"Warning: No previous F1-score found for class {class_label}. Defaulting to 0.0.")
            class_performance[class_label] = 0.0

    shared_pool_path = os.path.join(save_dir, "shared_labeled_pool.npz")

    if global_labeled_pool is None:
        if os.path.exists(shared_pool_path):
            loaded_data = np.load(shared_pool_path)
            global_labeled_pool = set(loaded_data["labeled_pool"])
            print(f"Loaded shared labeled pool from {shared_pool_path}, size: {len(global_labeled_pool)}")
        else:
            raise FileNotFoundError(f"Error: No existing shared labeled pool found at {shared_pool_path}.")
    else:
        global_labeled_pool = global_labeled_pool.copy()

    print(f"Initialized labeled pool with {len(global_labeled_pool)} samples before Active Learning.")
    new_labels_added = 1
    iteration = 0
    while len(global_labeled_pool) < max_labeled_samples: #new_labels_added > 0: 
        print(f"\n========== Active Learning Iteration {iteration + 1} ==========")

        sorted_classes = sorted(class_performance.items(), key=lambda x: x[1])
        new_labels_added = 0

        for class_label, _ in sorted_classes:

            print(f"Processing class {class_label} for Active Learning...")

            training_set_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz") \
                if os.path.exists(os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz")) \
                else os.path.join(save_dir, f"class_{class_label}_training_set.npz")

            bovw_features_path = os.path.join(AL_results_dir, f"class_{class_label}_bovw_features.npz") \
                if os.path.exists(os.path.join(AL_results_dir, f"class_{class_label}_bovw_features.npz")) \
                else os.path.join(save_dir, f"class_{class_label}_bovw_features.npz")

            lbp_features_path = os.path.join(AL_results_dir, f"class_{class_label}_lbp_features.npz") \
                if os.path.exists(os.path.join(AL_results_dir, f"class_{class_label}_lbp_features.npz")) \
                else os.path.join(save_dir, f"class_{class_label}_lbp_features.npz")
            
            if not (os.path.exists(training_set_path) and os.path.exists(bovw_features_path) and os.path.exists(lbp_features_path)):
                print(f"Skipping class {class_label}, missing training set.")
                continue

            data = np.load(training_set_path)
            train_data = data['train_data']
            train_labels = data['train_labels']
            train_bovw_features = np.load(bovw_features_path)['features']
            train_lbp_features = np.load(lbp_features_path)['features']

            positive_indices = {idx for idx in range(len(train_labels)) if train_labels[idx] == 1}
            negative_indices = {idx for idx in range(len(train_labels)) if train_labels[idx] == 0}
            training_indices = set(np.arange(train_data.shape[0])) #set(np.where(np.isin(np.arange(len(dataset)), train_data, assume_unique=True))[0])
            valid_training_indices = {idx for idx in training_indices if idx < len(dataset)}
            #valid_training_indices = [idx for idx in train_data if idx in dataset]

            labeled_count = len(valid_training_indices) 
            labeled_percentage = (labeled_count / total_samples) * 100

            print(f"Total labeled images for class {class_label}: {labeled_count}, size trainingset {len(train_data)}, positives: {len(positive_indices)}, negatives: {len(negative_indices)} ")
            print(f"Percentage of dataset labeled for class {class_label}: {labeled_percentage:.2f}%")

            if len(global_labeled_pool) >= max_labeled_samples:
                print("\nGlobal labeled budget reached. Selecting only from the labeled pool.")
                break
                train_indices = set(np.arange(train_data.shape[0]))
                available_indices = global_labeled_pool - train_indices

                if not available_indices:
                    print(f"No new samples available for class {class_label} from labeled pool.")
                    continue

                available_indices = np.array(list(available_indices))
            else:
                all_indices = set(range(len(dataset)))
                available_indices = np.array(list(all_indices - valid_training_indices))

            if len(available_indices) == 0:
                print(f"No more samples available for active learning in class {class_label}.")
                continue

            unlabelled_data = dataset[available_indices]
            unlabelled_labels = labels[available_indices]
            unlabelled_bovw_features = dataset_bovw_features[available_indices]
            unlabelled_lbp_features = dataset_lbp_features[available_indices]

            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            rf_model = RandomForestClassifier()

            xgb_model, rf_model, train_data, train_labels, train_bovw_features, train_lbp_features, new_samples, results = voting_active_learning_binary_dynamic(
                dataset= dataset, 
                labels= labels,
                bovw_features=dataset_bovw_features,
                lbp_features=dataset_lbp_features,
                class_label=class_label,
                train_data=train_data,
                train_bovw_features=train_bovw_features,
                train_lbp_features=train_lbp_features,
                train_labels=train_labels,
                unlabelled_data=unlabelled_data,
                unlabelled_bovw_features=unlabelled_bovw_features,
                unlabelled_lbp_features=unlabelled_lbp_features,
                unlabelled_labels=unlabelled_labels,
                xgb_model=xgb_model,
                rf_model=rf_model,
                max_iterations=1, 
                target_fraction=target_fraction,
                max_labeled_samples=max_labeled_samples,
                global_labeled_pool=global_labeled_pool,
                iteration_main= iteration
            )

            global_labeled_pool.update(new_samples)
            new_samples_added = len(new_samples)
            new_labels_added = new_labels_added + new_samples_added
            print(f"Added {new_samples_added} new samples for class {class_label}")

            updated_training_set_path = os.path.join(updated_training_dir, f"class_{class_label}_training_set.npz")
            updated_bovw_features_path = os.path.join(updated_training_dir, f"class_{class_label}_bovw_features.npz")
            updated_lbp_features_path = os.path.join(updated_training_dir, f"class_{class_label}_lbp_features.npz")

            np.savez_compressed(updated_training_set_path, train_data=train_data, train_labels=train_labels)
            np.savez_compressed(updated_bovw_features_path, features=train_bovw_features)
            np.savez_compressed(updated_lbp_features_path, features=train_lbp_features)

            last_result = results[-1]
            if str(class_label) not in training_stats:
                training_stats[str(class_label)] = []
            training_stats[str(class_label)].append(last_result)

            print(f"Total labeled samples so far: {len(global_labeled_pool)}/{max_labeled_samples}")

        if new_labels_added == 0:
            print("No new labeled samples were added in this iteration. Stopping Active Learning.")
            break

        iteration += 1
    
    #Add all labels from the Data_Pool

    for class_label, _ in sorted_classes:

        print(f"+++++++++++++++++++++Adding Data_Pool to {class_label} to finalize Active Learning+++++++++++++++++++++++++++++++++")

        training_set_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz") 
        bovw_features_path = os.path.join(AL_results_dir, f"class_{class_label}_bovw_features.npz") 
        lbp_features_path = os.path.join(AL_results_dir, f"class_{class_label}_lbp_features.npz") 

        data = np.load(training_set_path)
        train_data = data['train_data']
        train_labels = data['train_labels']
        train_bovw_features = np.load(bovw_features_path)['features']
        train_lbp_features = np.load(lbp_features_path)['features']

        positive_indices = {idx for idx in range(len(train_labels)) if train_labels[idx] == 1}
        negative_indices = {idx for idx in range(len(train_labels)) if train_labels[idx] == 0}
        training_indices = set(np.arange(train_data.shape[0])) 
        valid_training_indices = {idx for idx in training_indices if idx < len(dataset)}

        labeled_count = len(valid_training_indices) 
        labeled_percentage = (labeled_count / total_samples) * 100

        print(f"Total labeled images for class {class_label}: {labeled_count}, size trainingset {len(train_data)}, positives: {len(positive_indices)}, negatives: {len(negative_indices)} ")
        print(f"Percentage of dataset labeled for class {class_label}: {labeled_percentage:.2f}%")

        train_indices = set(np.arange(train_data.shape[0]))
        available_indices = global_labeled_pool - train_indices

        if not available_indices:
            print(f"No new samples available for class {class_label} from labeled pool.")
            continue

        available_indices = np.array(list(available_indices))

        if len(available_indices) == 0:
            print(f"No more samples available for active learning in class {class_label}.")
            continue

        unlabelled_data = dataset[available_indices]
        unlabelled_labels = labels[available_indices]
        unlabelled_bovw_features = dataset_bovw_features[available_indices]
        unlabelled_lbp_features = dataset_lbp_features[available_indices]

        if len(available_indices) != len(unlabelled_data):
            raise ValueError(f"Mismatch: available_indices length ({len(available_indices)}) does not match unlabeled_data length ({len(unlabelled_data)})")

        xgb_model, rf_model, train_data, train_labels, train_bovw_features, train_lbp_features, new_samples, results = voting_active_learning_binary_dynamic(
            dataset= dataset, 
            labels= labels,
            bovw_features=dataset_bovw_features,
            lbp_features=dataset_lbp_features,
            class_label=class_label,
            train_data=train_data,
            train_bovw_features=train_bovw_features,
            train_lbp_features=train_lbp_features,
            train_labels=train_labels,
            unlabelled_data=unlabelled_data,
            unlabelled_bovw_features=unlabelled_bovw_features,
            unlabelled_lbp_features=unlabelled_lbp_features,
            unlabelled_labels=unlabelled_labels,
            xgb_model=xgb_model,
            rf_model=rf_model,
            max_iterations=1, 
            target_fraction=target_fraction,
            max_labeled_samples=max_labeled_samples,
            global_labeled_pool=global_labeled_pool,
            iteration_main= iteration
        )

        new_labels_added = new_labels_added + new_samples_added
        print(f"Added {new_samples_added} new samples for class {class_label}")

        updated_training_set_path = os.path.join(updated_training_dir, f"class_{class_label}_training_set.npz")
        updated_bovw_features_path = os.path.join(updated_training_dir, f"class_{class_label}_bovw_features.npz")
        updated_lbp_features_path = os.path.join(updated_training_dir, f"class_{class_label}_lbp_features.npz")

        np.savez_compressed(updated_training_set_path, train_data=train_data, train_labels=train_labels)
        np.savez_compressed(updated_bovw_features_path, features=train_bovw_features)
        np.savez_compressed(updated_lbp_features_path, features=train_lbp_features)

        last_result = results[-1]
        if str(class_label) not in training_stats:
            training_stats[str(class_label)] = []
        training_stats[str(class_label)].append(last_result)

        print(f"Total labeled samples so far: {len(global_labeled_pool)}/{max_labeled_samples}")

    updated_stats_path = os.path.join(updated_training_dir, "training_stats.json")
    with open(updated_stats_path, "w") as stats_file:
        json.dump(convert_numpy_types(training_stats), stats_file, indent=4)

    print("Active learning pipeline completed.")
    print(f"Updated training sets and stats saved in {updated_training_dir}.")

def voting_active_learning_binary_dynamic(dataset, labels, bovw_features, lbp_features, class_label, train_data, train_bovw_features, train_lbp_features, train_labels, unlabelled_data, unlabelled_bovw_features, unlabelled_lbp_features, unlabelled_labels, xgb_model, rf_model, max_iterations=10, target_fraction=0.1, max_labeled_samples=None, global_labeled_pool=None, iteration_main= 1):
    global previous_weights
    global prev_xgb_auroc
    global prev_rf_auroc
    results = []
    newly_labeled_samples = set()

    def compute_dynamic_weights_roc_auc(true_labels, xgb_probs, rf_probs, prev_xgb_auroc=0.5, prev_rf_auroc=0.5, previous_weights=np.array([0.5, 0.5]), lambda_factor=3):
        from sklearn.metrics import roc_auc_score      

        try:
            xgb_auroc = roc_auc_score(true_labels, xgb_probs)
        except ValueError:
            xgb_auroc = 0.5
        try:
            rf_auroc = roc_auc_score(true_labels, rf_probs)
        except ValueError:
            rf_auroc = 0.5  

        delta_xgb = abs(xgb_auroc - prev_xgb_auroc)
        delta_rf = abs(rf_auroc - prev_rf_auroc)
        delta_auroc = max(delta_xgb, delta_rf) 

        alpha = min(1, max(0.3, 1 - lambda_factor * delta_auroc))

        total = xgb_auroc + rf_auroc
        new_weights = np.array([xgb_auroc / total, rf_auroc / total]) if total > 0 else np.array([0.5, 0.5])

        smoothed_weights = alpha * new_weights + (1 - alpha) * previous_weights
        previous_weights = smoothed_weights

        print(f"AUROC XGB: {xgb_auroc:.3f}, AUROC RF: {rf_auroc:.3f}, ΔAUROC: {delta_auroc:.3f}, Alpha: {alpha:.3f}, Weights: {smoothed_weights}")

        return smoothed_weights, xgb_auroc, rf_auroc

    for iteration in range(max_iterations):
        print(f"Active Learning Iteration {iteration + 1} for class {class_label}...")

        xgb_model.fit(train_bovw_features, train_labels)
        rf_model.fit(train_lbp_features, train_labels)

        xgb_probabilities = xgb_model.predict_proba(unlabelled_bovw_features)[:, 1]
        rf_probabilities = rf_model.predict_proba(unlabelled_lbp_features)[:, 1]

        xgb_entropy = -xgb_probabilities * np.log2(xgb_probabilities + 1e-9) - (1 - xgb_probabilities) * np.log2(1 - xgb_probabilities + 1e-9)
        rf_entropy = -rf_probabilities * np.log2(rf_probabilities + 1e-9) - (1 - rf_probabilities) * np.log2(1 - rf_probabilities + 1e-9)

        prev_xgb_auroc = prev_xgb_auroc if prev_xgb_auroc is not None else 0.5
        prev_rf_auroc = prev_rf_auroc if prev_rf_auroc is not None else 0.5
        previous_weights = previous_weights if previous_weights is not None else np.array([0.5, 0.5])

        combined_weights, prev_xgb_auroc, prev_rf_auroc = compute_dynamic_weights_roc_auc(train_labels, xgb_probabilities, rf_probabilities, prev_xgb_auroc, prev_rf_auroc)
        combined_entropy = combined_weights[0] * xgb_entropy + combined_weights[1] * rf_entropy

        if len(global_labeled_pool) >= max_labeled_samples:
            print("Global labeled budget reached. Switching to selecting only from labeled pool.")

            valid_indices = set(range(len(unlabelled_data)))
            available_pool_indices = list(valid_indices)
            max_possible_samples = len(available_pool_indices)

            if max_possible_samples <= 0:
                print(f"No new samples available in global pool for class {class_label}. Stopping iteration.")
                break

            top_k = min(max(5, max_possible_samples), len(available_pool_indices))
            uncertain_indices = np.random.choice(available_pool_indices, size=top_k, replace=False)

            uncertain_indices = [idx for idx in uncertain_indices if idx in valid_indices]
            if not uncertain_indices:
                print(f"No valid indices found in unlabelled_data for class {class_label}. Skipping selection.")
                break

            selected_samples = unlabelled_data[uncertain_indices]
            selected_bovw_features = unlabelled_bovw_features[uncertain_indices]
            selected_lbp_features = unlabelled_lbp_features[uncertain_indices]
            selected_labels = unlabelled_labels[uncertain_indices, class_label]
        else:
            max_possible_samples = max_labeled_samples - len(global_labeled_pool)
            max_selectable_samples = int(0.02 * len(unlabelled_data))
            top_k = min(max(3, min(10, max_selectable_samples)), max_possible_samples)

            if top_k <= 0:
                print("No more budget to add samples from the unlabeled dataset.")
                break

            uncertain_indices = np.argsort(combined_entropy)[:top_k]
            selected_samples = unlabelled_data[uncertain_indices]
            selected_bovw_features = unlabelled_bovw_features[uncertain_indices]
            selected_lbp_features = unlabelled_lbp_features[uncertain_indices]
            selected_labels = unlabelled_labels[uncertain_indices, class_label]

        train_data = np.vstack([train_data, selected_samples])
        train_bovw_features = np.vstack([train_bovw_features, selected_bovw_features])
        train_lbp_features = np.vstack([train_lbp_features, selected_lbp_features])
        train_labels = np.hstack([train_labels, selected_labels])

        newly_labeled_samples.update(set(uncertain_indices))
        global_labeled_pool.update(set(uncertain_indices))

        print(f"Added {len(uncertain_indices)} new samples for class {class_label}")

        positive_indices = np.where(selected_labels == 1)[0]
        negative_indices = np.where(selected_labels == 0)[0]

        print(f"Adding {len(positive_indices)} positive and {len(negative_indices)} negative samples.")

        augmented_positives = []
        augmented_negatives = []

        augment_threshold = 100
        base_augmentations = 6

        for idx in positive_indices:
            if len(train_labels[train_labels == 1]) < 20:
                augment_factor = base_augmentations * 4
            elif len(train_labels[train_labels == 1]) < augment_threshold:
                augment_factor = base_augmentations * 2
            else:
                augment_factor = base_augmentations

            augmented_images = augment_data([selected_samples[idx]], base_augmentations=augment_factor)
            augmented_positives.extend(augmented_images)

        for idx in negative_indices:
            if len(train_labels[train_labels == 0]) < 20:
                augment_factor = max(1, base_augmentations // 2)
            elif len(train_labels[train_labels == 0]) < augment_threshold:
                augment_factor = max(1, base_augmentations // 3)
            else:
                augment_factor = max(1, base_augmentations // 6)

            augmented_images = augment_data([selected_samples[idx]], base_augmentations=augment_factor)
            augmented_negatives.extend(augmented_images)

        if augmented_positives:
            augmented_positives = np.array(augmented_positives)
        if augmented_negatives:
            augmented_negatives = np.array(augmented_negatives)

        if len(augmented_positives) > 0:
            print(f"Augmented {len(augmented_positives)} positive samples.")
            augmented_pos_dataset = FeatureDataset(augmented_positives, np.ones(len(augmented_positives)))
            augmented_bovw_features, _, _ = extract_bovw_features(augmented_pos_dataset, kmeans=kmeans)
            augmented_lbp_features, _ = extract_lbp_features(augmented_pos_dataset)

            if augmented_bovw_features.shape[0] == len(augmented_positives) and augmented_lbp_features.shape[0] == len(augmented_positives):
                train_bovw_features = np.vstack([train_bovw_features, augmented_bovw_features])
                train_lbp_features = np.vstack([train_lbp_features, augmented_lbp_features])
                train_data = np.vstack([train_data, augmented_positives])
                train_labels = np.hstack([train_labels, np.ones(len(augmented_positives))])
            else:
                print("Warning: Mismatch in BoVW or LBP features for augmented positives. Skipping feature addition.")

        if len(augmented_negatives) > 0:
            print(f"Augmented {len(augmented_negatives)} negative samples.")
            augmented_neg_dataset = FeatureDataset(augmented_negatives, np.zeros(len(augmented_negatives)))
            augmented_bovw_features, _, _ = extract_bovw_features(augmented_neg_dataset, kmeans=kmeans)
            augmented_lbp_features, _ = extract_lbp_features(augmented_neg_dataset)

            if augmented_bovw_features.shape[0] == len(augmented_negatives) and augmented_lbp_features.shape[0] == len(augmented_negatives):
                train_bovw_features = np.vstack([train_bovw_features, augmented_bovw_features])
                train_lbp_features = np.vstack([train_lbp_features, augmented_lbp_features])
                train_data = np.vstack([train_data, augmented_negatives])
                train_labels = np.hstack([train_labels, np.zeros(len(augmented_negatives))])
            else:
                print("Warning: Mismatch in BoVW or LBP features for augmented negatives. Skipping feature addition.")
        """
        unlabelled_data = np.delete(unlabelled_data, selected_samples, axis=0)
        unlabelled_bovw_features = np.delete(unlabelled_bovw_features, selected_samples, axis=0)
        unlabelled_lbp_features = np.delete(unlabelled_lbp_features, selected_samples, axis=0)
        unlabelled_labels = np.delete(unlabelled_labels, selected_samples, axis=0)
        """
        labels_1d = labels[:, class_label]

        xgb_predictions_train = xgb_model.predict_proba(bovw_features)[:, 1]
        rf_predictions_train = rf_model.predict_proba(lbp_features)[:, 1]

        combined_probabilities_train = (combined_weights[0] * xgb_predictions_train + combined_weights[1] * rf_predictions_train)

        precision_vals, recall_vals, thresholds = precision_recall_curve(labels_1d, combined_probabilities_train)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        dynamic_threshold = thresholds[optimal_idx]

        #combined_predictions_train = (combined_probabilities_train >= dynamic_threshold).astype(int)

        xgb_train_preds = xgb_model.predict(bovw_features)
        xgb_train_probs = xgb_model.predict_proba(bovw_features)[:, 1]
        rf_train_preds = rf_model.predict(lbp_features)
        rf_train_probs = rf_model.predict_proba(lbp_features)[:, 1]
        ensemble_probs = combined_weights[0] * xgb_train_probs + combined_weights[1] * rf_train_probs
        ensemble_preds = (ensemble_probs >= dynamic_threshold).astype(int)

        total_accuracy = accuracy_score(labels_1d, ensemble_preds)
        total_precision = precision_score(labels_1d, ensemble_preds)
        total_recall = recall_score(labels_1d, ensemble_preds)
        total_f1 = f1_score(labels_1d, ensemble_preds)
        total_loss = log_loss(labels_1d, ensemble_probs)

        xgb_train_accuracy = accuracy_score(labels_1d, xgb_train_preds)
        xgb_train_precision = precision_score(labels_1d, xgb_train_preds)
        xgb_train_recall = recall_score(labels_1d, xgb_train_preds)
        xgb_train_f1 = f1_score(labels_1d, xgb_train_preds)
        xgb_train_loss = log_loss(labels_1d, xgb_train_probs)
        xgb_auroc = roc_auc_score(labels_1d, xgb_train_probs)

        rf_train_accuracy = accuracy_score(labels_1d, rf_train_preds)
        rf_train_precision = precision_score(labels_1d, rf_train_preds)
        rf_train_recall = recall_score(labels_1d, rf_train_preds)
        rf_train_f1 = f1_score(labels_1d, rf_train_preds)
        rf_train_loss = log_loss(labels_1d, rf_train_probs)
        rf_auroc = roc_auc_score(labels_1d, rf_train_probs)

        #xgb_feature_importance = xgb_model.feature_importances_.tolist()
        #rf_feature_importance = rf_model.feature_importances_.tolist()

        num_added_samples = len(selected_labels)
        num_positive_added = np.sum(selected_labels)
        num_negative_added = num_added_samples - num_positive_added

        xgb_confidence = np.mean(np.abs(xgb_train_probs - 0.5))
        rf_confidence = np.mean(np.abs(rf_train_probs - 0.5))

        entropy_mean = np.mean(combined_entropy)
        entropy_std = np.std(combined_entropy)

        results.append({
            "iteration": iteration_main,
            "num_added_samples": num_added_samples,
            "num_positive_added": num_positive_added,
            "num_negative_added": num_negative_added,

            "total_accuracy": total_accuracy,
            "total_precision": total_precision,
            "total_recall": total_recall,
            "total_f1": total_f1,
            "total_loss": total_loss,

            "xgb_train_accuracy": xgb_train_accuracy,
            "xgb_train_precision": xgb_train_precision,
            "xgb_train_recall": xgb_train_recall,
            "xgb_train_f1": xgb_train_f1,
            "xgb_train_loss": xgb_train_loss,
            "xgb_auroc": xgb_auroc,

            "rf_train_accuracy": rf_train_accuracy,
            "rf_train_precision": rf_train_precision,
            "rf_train_recall": rf_train_recall,
            "rf_train_f1": rf_train_f1,
            "rf_train_loss": rf_train_loss,
            "rf_auroc": rf_auroc,

            "xgb_confidence": xgb_confidence,
            "rf_confidence": rf_confidence,
            "entropy_mean": entropy_mean,
            "entropy_std": entropy_std,

            #"xgb_feature_importance": xgb_feature_importance,
            #"rf_feature_importance": rf_feature_importance,

            "auroc_weights": combined_weights.tolist(),
            "dynamic_threshold": dynamic_threshold
        })

        print(
            f"\n===== Iteration {iteration_main} Summary for class {class_label}====="
            f"\n Added {num_added_samples} samples (Pos: {num_positive_added}, Neg: {num_negative_added})"
            f"\n Accuracy: {total_accuracy:.4f} | Precision: {total_precision:.4f} | Recall: {total_recall:.4f} | F1: {total_f1:.4f}"
            f"\n XGB AUROC: {xgb_auroc:.4f} | RF AUROC: {rf_auroc:.4f}"
            f"\n Dynamic Weights: XGB={combined_weights[0]:.4f}, RF={combined_weights[1]:.4f}"
            f"\n Dynamic Threshold: {dynamic_threshold:.4f}"
            f"\n Confidence: XGB={xgb_confidence:.4f} | RF={rf_confidence:.4f}"
            f"\n Entropy - Mean: {entropy_mean:.4f} | Std: {entropy_std:.4f}"
            f"\n==============================="
        )

        available_pool_indices = available_pool_indices if 'available_pool_indices' in locals() else []

        if (len(global_labeled_pool) if global_labeled_pool is not None else 0) >= \
        (max_labeled_samples if max_labeled_samples is not None else 0) and \
        len(available_pool_indices) == 0:
            print("No more samples available for selection. Stopping active learning.")
            break

    return xgb_model, rf_model, train_data, train_labels, train_bovw_features, train_lbp_features, newly_labeled_samples, results

def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, (np.float32, np.float64)):  
        return float(obj) 
    elif isinstance(obj, (np.int32, np.int64)):  
        return int(obj)
    elif isinstance(obj, dict):  
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):  
        return [convert_numpy_types(i) for i in obj]
    return obj

def validate_training_sets(save_dir, original_dataset, output_stats_path="training_set_validation.json"):

    validation_results = {}  

    total_classes = len([f for f in os.listdir(save_dir) if f.endswith("_training_set.npz")])

    for class_label in range(total_classes):
        training_set_path = os.path.join(save_dir, f"class_{class_label}_training_set.npz")
        bovw_features_path = os.path.join(save_dir, f"class_{class_label}_bovw_features.npz")

        if not os.path.exists(training_set_path):
            print(f"Warning: Training set for class {class_label} not found. Skipping...")
            validation_results[class_label] = {
                "total_samples": 0,
                "positive_samples": 0,
                "negative_samples": 0,
                "non_augmented_samples": 0,
                "augmented_samples": 0,
                "bovw_feature_count": 0
            }
            continue

        data = np.load(training_set_path)
        train_data = data["train_data"]
        train_labels = data["train_labels"]

        num_total_samples = len(train_labels)
        num_positive = len({idx for idx in range(len(train_labels)) if train_labels[idx] == 1})
        num_negative = len({idx for idx in range(len(train_labels)) if train_labels[idx] == 0})

        training_indices = set(np.where(np.isin(np.arange(len(dataset)), train_data, assume_unique=True))[0])
        num_non_augmented = len({idx for idx in training_indices if idx < len(dataset)})
        
        bovw_feature_count = 0
        if os.path.exists(bovw_features_path):
            bovw_features = np.load(bovw_features_path)["features"]
            bovw_feature_count = bovw_features.shape[0]

        validation_results[class_label] = {
            "total_samples": num_total_samples,
            "positive_samples": int(num_positive),
            "negative_samples": int(num_negative),
            "non_augmented_samples": num_non_augmented,
            "augmented_samples": num_total_samples - num_non_augmented,
            "bovw_feature_count": bovw_feature_count
        }

        print(f"\n--- Validation Report for Class {class_label} ---")
        print(f"Total samples: {num_total_samples}")
        print(f"Positive samples: {num_positive}")
        print(f"Negative samples: {num_negative}")
        print(f"Non-augmented samples: {num_non_augmented}")
        print(f"Augmented samples: {num_total_samples - num_non_augmented}")
        print(f"BoVW Feature Count: {bovw_feature_count}")

    with open(output_stats_path, "w") as json_file:
        json.dump(validation_results, json_file, indent=4)

    print(f"\nValidation completed. Stats saved to {output_stats_path}")

def reshape_and_save_datasets(input_dir, output_dir, dataset, reshape_threshold=2, dataset_filename="reshaped_dataset.npz", n_components=100, batch_size=2048):

    os.makedirs(output_dir, exist_ok=True)

    reshaped_dataset_path = os.path.join(output_dir, dataset_filename)
    if os.path.exists(reshaped_dataset_path):
        print(f"Loading existing reshaped dataset from {reshaped_dataset_path}...")
        reshaped_dataset = np.load(reshaped_dataset_path)['dataset']
    else:
        print("Reshaping the main dataset...")
        reshaped_dataset = dataset.reshape(dataset.shape[0], -1)

        print("Applying PCA to the main dataset...")
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        reshaped_dataset = ipca.fit_transform(reshaped_dataset)

        np.savez_compressed(reshaped_dataset_path, dataset=reshaped_dataset)
        print(f"Reshaped dataset saved to {reshaped_dataset_path}")

    pattern = re.compile(r'class_\d+_training_set\.npz')
    reducer = PCA(n_components=n_components)
    for filename in os.listdir(input_dir):
        if pattern.match(filename):
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"Skipping {filename}, already reshaped.")
                continue

            file_path = os.path.join(input_dir, filename)
            data = np.load(file_path)

            train_data = data['train_data']
            train_labels = data['train_labels']

            print(f"Reshaping {filename}...")
            train_data = train_data.reshape(train_data.shape[0], -1)

            print(f"Applying PCA to {filename}...")
            train_data = reducer.fit_transform(train_data)

            np.savez_compressed(output_path, train_data=train_data, train_labels=train_labels)
            print(f"Reshaped data saved to {output_path}")
        else:
            print(f"Skipping {filename}, does not match the class pattern.")

    print("All datasets reshaped and saved successfully.")
    return reshaped_dataset

def generate_image_ids_from_data(dataset):
    image_ids = [hashlib.md5(img.tobytes()).hexdigest()[:12] for img in dataset]
    return image_ids

def main_ssl_pipeline(models, AL_results_dir, SSL_results_dir, dataset, labels, iterations=10):
    #### Change COndition for Iterations (normal and cov!!!)
    
    performance_log = []
    #results_matrix = pd.DataFrame()
    results_matrix = initialize_results_matrix(dataset, labels)
    image_ids = generate_image_ids_from_data(dataset)
    image_ids = np.array(image_ids) 
    all_trained_models = {} 
    best_f1_scores = {class_label:0.0 for class_label in range(labels.shape[1])}

    patience = 30
    best_micro_f1 = 0
    best_macro_f1 = 0
    no_improvement_count = 0

    for iteration in range(1, iterations + 1):
        print(f"Iteration {iteration}/{iterations}")####!!!!!
        stop_counter = 0 
        all_pseudo_labels = {} 

        for class_label in range(labels.shape[1]):
            #print(f"Processing Class {class_label} for SSL (co-training)")
            training_set_path = os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz")
            if os.path.exists(training_set_path):
                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                train_image_ids = data['image_ids']
            else:
                training_set_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz")              
                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                if "image_ids" in data:
                    train_image_ids = data['image_ids']
                else:
                    train_image_ids = generate_image_ids_from_data(train_data)
            
            if len(train_data) > len(train_labels):
                print("Mismatch detected: Extra data point in train_data!")
                extra_index = None
                if len(train_image_ids) == len(train_data):
                    image_id_counts = {img_id: 0 for img_id in train_image_ids}
                    for img_id in train_image_ids:
                        image_id_counts[img_id] += 1

                    for i, img_id in enumerate(train_image_ids):
                        if image_id_counts[img_id] > 1 or i >= len(train_labels):
                            extra_index = i
                            print(f"Identified extra sample at index: {extra_index} (Image ID: {img_id})")
                            break

                if extra_index is None:
                    print("No clear extra image ID found, falling back to feature comparison...")
                    label_set = set(train_labels)

                    for i, row in enumerate(train_data):
                        if i >= len(train_labels) or row[0] not in label_set:  # Assumes first column is indicative
                            extra_index = i
                            print(f"Identified extra sample at index: {extra_index} using feature comparison")
                            break

                if extra_index is not None:
                    print(f"Removing extra data point at index: {extra_index}")
                    train_data = np.delete(train_data, extra_index, axis=0)
                    train_image_ids = np.delete(train_image_ids, extra_index, axis=0)  # Keep image IDs aligned
                    np.savez(training_set_path, train_data=train_data, train_labels=train_labels, image_ids=train_image_ids)
                    print(f"Corrected dataset saved at: {training_set_path}")

                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                train_image_ids = data['image_ids']

            assert len(train_data) == len(train_labels), "Error: Mismatch still exists after fixing!"

            #training_indices = set(np.arange(train_data.shape[0]))
            #valid_training_indices = {idx for idx in training_indices if idx < len(dataset)}
            #all_indices = set(range(len(dataset)))
            #available_indices = np.array(list(all_indices - valid_training_indices))
            #unlabeled_data = dataset[available_indices]
            #unlabeled_labels = labels[available_indices, class_label]
            labeled_image_ids = set(train_image_ids)
            available_mask = np.array([img_id not in labeled_image_ids for img_id in image_ids])
            unlabeled_data = dataset[available_mask]
            unlabeled_labels = labels[available_mask, class_label]
            unlabeled_image_ids = image_ids[available_mask]
            
            if unlabeled_data.shape[0] == 0:
                print(f"Skipping Class {class_label} as there is no unlabeled data left.")
                stop_counter += 1
                if stop_counter == labels.shape[1]:
                    print("No new data added for any class. Stopping SSL pipeline.")
                    return all_trained_models, performance_log, results_matrix
                continue

            pseudo_labels, trained_models, results_matrix = co_training(models, dataset, image_ids, train_data, train_labels, train_image_ids, unlabeled_data, unlabeled_labels, unlabeled_image_ids, class_label, results_matrix, main_loop=iteration, max_iterations=3)

            if class_label not in all_pseudo_labels:
                all_pseudo_labels[class_label] = {}
            all_pseudo_labels[class_label].update(pseudo_labels[class_label])
            all_trained_models[class_label] = trained_models
            print(f"Class {class_label}: Assigned {len(all_pseudo_labels[class_label])} pseudo-labels.")

        print("Applying Co-Occurrence Voting across all classes...")
        full_train_labels, train_image_ids = build_train_labels(SSL_results_dir, AL_results_dir, labels.shape[1])
        co_occurrence_matrix = build_co_occurrence_matrix(full_train_labels)
        all_pseudo_labels = co_occurrence_voting(all_pseudo_labels, co_occurrence_matrix, alpha=0.7, beta=0.3)

        os.makedirs(SSL_results_dir, exist_ok=True)
        for class_label, pseudo_label_data in all_pseudo_labels.items():
            training_set_path = os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz")
            if os.path.exists(training_set_path):
                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                train_image_ids = data['image_ids']
            else:
                training_set_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz")              
                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                if "image_ids" in data:
                    train_image_ids = data['image_ids']
                else:
                    train_image_ids = generate_image_ids_from_data(train_data)
            new_image_ids = list(pseudo_label_data.keys())
            new_labels = np.array(list(pseudo_label_data.values()))
            pseudo_labeled_data = dataset[np.where(np.isin(image_ids, new_image_ids))[0]]
            #new_labels = np.array(pseudo_label_data)
            #available_indices = np.arange(len(dataset))
            #pseudo_labeled_data = dataset[available_indices[:len(new_labels)]]
            updated_data = np.vstack((train_data, pseudo_labeled_data))
            updated_labels = np.hstack((train_labels, new_labels))
            updated_image_ids = np.hstack((train_image_ids, new_image_ids))

            updated_path = os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz")
            np.savez_compressed(updated_path, train_data=updated_data, train_labels=updated_labels, image_ids=updated_image_ids)

            class_frequency = np.mean(train_labels)
            minority_threshold = 0.2
            macro_weight = 0.6 if class_frequency < minority_threshold else 0.4
            micro_weight = 0.4 if class_frequency < minority_threshold else 0.6

            performance = analyze_performance(all_trained_models[class_label], dataset, labels[:, class_label], macro_weight, micro_weight)
            performance_log.append(performance)

        serializable_performance_log = [convert_to_serializable2(entry) for entry in performance_log]
        with open('performance_log.json', 'w') as f:
            json.dump(serializable_performance_log, f, indent=2)

        micro_f1_scores = []
        macro_f1_scores = []
        class_f1_scores = {}
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
 
        best_micro_f1_dir = os.path.join(SSL_results_dir, "best_micro_f1")
        best_macro_f1_dir = os.path.join(SSL_results_dir, "best_macro_f1")
        best_per_class_dir = os.path.join(SSL_results_dir, "best_per_class")
 
        os.makedirs(best_micro_f1_dir, exist_ok=True)
        os.makedirs(best_macro_f1_dir, exist_ok=True)
        os.makedirs(best_per_class_dir, exist_ok=True)
 
        for class_label in range(labels.shape[1]):
            ssl_model = all_trained_models[class_label]
 
            ssl_data = np.load(os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz"))
            train_labels = ssl_data['train_labels']
            train_data = ssl_data['train_data']

            if len(train_data) > len(train_labels):
                print("Mismatch detected: Extra data point in train_data!")
                extra_index = None
                if len(train_image_ids) == len(train_data):
                    image_id_counts = {img_id: 0 for img_id in train_image_ids}
                    for img_id in train_image_ids:
                        image_id_counts[img_id] += 1

                    for i, img_id in enumerate(train_image_ids):
                        if image_id_counts[img_id] > 1 or i >= len(train_labels):
                            extra_index = i
                            print(f"Identified extra sample at index: {extra_index} (Image ID: {img_id})")
                            break

                if extra_index is None:
                    print("No clear extra image ID found, falling back to feature comparison...")
                    label_set = set(train_labels)

                    for i, row in enumerate(train_data):
                        if i >= len(train_labels) or row[0] not in label_set:  # Assumes first column is indicative
                            extra_index = i
                            print(f"Identified extra sample at index: {extra_index} using feature comparison")
                            break

                if extra_index is not None:
                    print(f"Removing extra data point at index: {extra_index}")
                    train_data = np.delete(train_data, extra_index, axis=0)
                    train_image_ids = np.delete(train_image_ids, extra_index, axis=0)  # Keep image IDs aligned
                    np.savez(training_set_path, train_data=train_data, train_labels=train_labels, image_ids=train_image_ids)
                    print(f"Corrected dataset saved at: {training_set_path}")

                data = np.load(training_set_path)
                train_data = data['train_data']
                train_labels = data['train_labels']
                train_image_ids = data['image_ids']

            assert len(train_data) == len(train_labels), "Error: Mismatch still exists after fixing!"
            
            ssl_labels = train_labels
            ssl_features = train_data

            model_preds = {}
            auroc_scores = {}
            for model_name, model in ssl_model.items():
                proba_preds = model.predict_proba(ssl_features)[:, 1]
                auroc_scores[model_name] = roc_auc_score(ssl_labels, proba_preds)
                
            total_auroc = sum(auroc_scores.values())
            model_weights = {model: score / total_auroc for model, score in auroc_scores.items()}
 
            for model_name, weight in model_weights.items():
                model_preds[model_name] = model.predict_proba(dataset)[:, 1]
            ssl_combined_prob = sum(model_preds[m] * w for m, w in model_weights.items())
            fpr, tpr, thresholds = roc_curve(labels[:, class_label], ssl_combined_prob)
            youden_index = np.argmax(tpr - fpr) 
            optimal_threshold = thresholds[youden_index]
            y_pred = (ssl_combined_prob > optimal_threshold).astype(int)
 
            y_true = labels[:, class_label]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn            
            micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            micro_f1_scores.append(micro_f1)
            macro_f1_scores.append(macro_f1)
            class_f1_scores[class_label] = {
                "micro_f1": micro_f1,
                "macro_f1": macro_f1
            }
            best_f1_path = os.path.join(best_per_class_dir, f"class_{class_label}_training_set.npz")
            if class_label not in best_f1_scores or isinstance(best_f1_scores[class_label],(int, float)) and micro_f1 > best_f1_scores[class_label]:
                best_f1_scores[class_label] = micro_f1
                shutil.copy2(os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz"), best_f1_path)
                np.savez_compressed(best_f1_path, f1_score=micro_f1)
            print(f"Class {class_label}: Micro F1-Score: {micro_f1:.4f}, Macro F1-Score: {macro_f1:.4f}, TP: {tp:.4f}, TN: {tn:.4f}, FP: {fp:.4f}, FN: {fn:.4f}" )
       
        mean_micro_f1 = np.mean(micro_f1_scores)
        mean_macro_f1 = np.mean(macro_f1_scores)
 
        performance_log.append({
            "iteration": iteration,
            "micro_f1": mean_micro_f1,
            "macro_f1": mean_macro_f1,
            "class_f1_scores": class_f1_scores
        })
 
        print(f"Micro F1-Score: {mean_micro_f1:.4f}, Macro F1-Score: {mean_macro_f1:.4f}")       
        
        if mean_micro_f1 <= best_micro_f1 and mean_macro_f1 <= best_macro_f1:
            no_improvement_count += 1
            print(f"No F1 Improvement ({no_improvement_count}/{patience} patience)")
        else:
            no_improvement_count = 0
            best_micro_f1 = max(best_micro_f1, micro_f1)
            best_macro_f1 = max(best_macro_f1, mean_macro_f1)
 
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            for file in os.listdir(SSL_results_dir):
                if file.endswith(".npz"):
                    shutil.copy2(os.path.join(SSL_results_dir, file), best_micro_f1_dir)
 
        if mean_macro_f1 > best_macro_f1:
            best_macro_f1 = mean_macro_f1
            for file in os.listdir(SSL_results_dir):
                if file.endswith(".npz"):
                    shutil.copy2(os.path.join(SSL_results_dir, file), best_macro_f1_dir)        
 
        if no_improvement_count >= patience:
            print(f"Stopping Early: F1-score plateaued for {patience} iterations")
            break
 
        #if len(performance_log) > 3 and performance_log[-1]['weighted_f1'] <= performance_log[-3]['weighted_f1']:
        #    print("No performance improvement detected. Stopping early.")
        #    break
 
    return all_trained_models, performance_log, results_matrix, image_ids

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj

def convert_to_serializable2(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable2(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable2(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable2(v) for v in obj)
    else:
        return obj

def co_training(models, dataset, image_ids, train_data, train_labels, train_image_ids, unlabeled_data, unlabeled_labels, unlabeled_image_ids, class_label, results_matrix, main_loop, max_iterations):
    pseudo_labels = {class_label: {}}
    trained_models = {}
    negative_threshold = 0.1
    class_frequency = np.mean(train_labels)
    minority_threshold = 0.2

    macro_weight = 0.6 if class_frequency < minority_threshold else 0.4
    micro_weight = 0.4 if class_frequency < minority_threshold else 0.6

    for model_name, model in models.items():
        model.fit(train_data, train_labels)
        trained_models[model_name] = model
    
    prev_performance, combined_preds = analyze_performance(trained_models, reshaped_dataset, labels[:, class_label], macro_weight, micro_weight)

    if os.path.exists('all_predictions.json'):
        with open('all_predictions.json', 'r') as f:
            all_predictions = json.load(f)
    else:
        all_predictions = {}

    disagreement_threshold = 0.3
    disagreement_trend = []

    for iteration in range(max_iterations):
        #print(f"Co-Training Iteration {iteration + 1}/{max_iterations}")

        base_dynamic_threshold = max(0.6, min(0.95, class_frequency + 0.7))
        dynamic_threshold = base_dynamic_threshold
        adjustment_step = 0.05
        min_threshold = 0.45 #0.35
        max_negative_threshold = 0.4

        total_positives_in_train = np.sum(train_labels)
        total_train_size = len(train_labels)
        negative_multiplier = total_train_size / max(1, total_positives_in_train)

        while True:
            confident_mask = combined_preds > dynamic_threshold
            confident_image_ids = image_ids[confident_mask]
            confident_image_ids = [img_id for img_id in confident_image_ids if img_id in unlabeled_image_ids]

            if len(confident_image_ids) > 0:
                break
            if iteration == 0 and dynamic_threshold > min_threshold:
                dynamic_threshold -= adjustment_step
                print(f"No confident samples for class {class_label}. Lowering threshold to {dynamic_threshold:.2f}")
            else:
                #print(f"Skipping class {class_label} in iteration {iteration} as no confident samples found.")
                return pseudo_labels, trained_models, results_matrix

        if len(confident_image_ids) > 1000:
            sorted_image_ids = sorted(confident_image_ids, key=lambda img_id: combined_preds[np.where(image_ids == img_id)[0][0]], reverse=True)[:1000]
            confident_image_ids = sorted_image_ids

        if len(confident_image_ids) > 100:
            num_clusters = min(10, len(confident_image_ids) // 15)
            if num_clusters >= 2:
                confident_samples = unlabeled_data[np.where(np.isin(unlabeled_image_ids, confident_image_ids))[0]]
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(confident_samples)
            else:
                cluster_labels = np.zeros(len(confident_image_ids))

            diverse_image_ids = []
            total_allowed = min(250, len(confident_image_ids))
            samples_per_cluster = max(1, total_allowed // num_clusters)

            for cluster in np.unique(cluster_labels):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                cluster_confidences = np.array([combined_preds[np.where(image_ids == confident_image_ids[idx])[0][0]] for idx in cluster_indices])
                top_indices = cluster_indices[np.argsort(cluster_confidences)[-samples_per_cluster:]]
                diverse_image_ids.extend([confident_image_ids[idx] for idx in top_indices])

            if len(diverse_image_ids) > 250:
                sorted_confidences = np.array([combined_preds[np.where(image_ids == img_id)[0][0]] for img_id in diverse_image_ids])
                top_indices = np.argsort(sorted_confidences)[-250:]
                diverse_image_ids = [diverse_image_ids[i] for i in top_indices]

            confident_image_ids = diverse_image_ids
            
        if len(confident_image_ids) == 0:
            num_negatives = 1
        else:
            confident_samples = unlabeled_data[np.where(np.isin(unlabeled_image_ids, confident_image_ids))[0]]
            confident_labels = np.array([1 if combined_preds[np.where(image_ids == img_id)[0][0]] > dynamic_threshold else 0 for img_id in confident_image_ids])
            unlabeled_image_ids = [img_id for img_id in unlabeled_image_ids if img_id not in confident_image_ids]
            unlabeled_data = np.array([sample for img_id, sample in zip(image_ids, dataset) if img_id in unlabeled_image_ids])
            unlabeled_labels = np.array([labels[np.where(image_ids == img_id)[0][0], class_label] for img_id in unlabeled_image_ids])

            temp_train_data = np.vstack((train_data, confident_samples))
            temp_train_labels = np.hstack((train_labels, confident_labels))
            num_negatives = max(1, int(len(confident_image_ids) * negative_multiplier /5))

        negative_candidates = [img_id for img_id, pred in zip(image_ids, combined_preds) if pred < negative_threshold and img_id in unlabeled_image_ids]

        while len(negative_candidates) == 0 and negative_threshold <= max_negative_threshold:
            negative_threshold += adjustment_step
            negative_candidates = [img_id for img_id, pred in zip(image_ids, combined_preds) if pred < negative_threshold and img_id in unlabeled_image_ids]
        
        if len(negative_candidates) == 0:
            print(f"Warning: No negative candidates found even at threshold {negative_threshold:.2f}")
        else:
            sorted_negatives = sorted(negative_candidates, key=lambda img_id: combined_preds[np.where(image_ids == img_id)[0][0]])
            selected_negatives = sorted_negatives[:num_negatives] if len(sorted_negatives) > num_negatives else sorted_negatives
            negative_samples = unlabeled_data[np.where(np.isin(unlabeled_image_ids, selected_negatives))[0]]
            negative_labels = np.zeros(len(selected_negatives))
            unlabeled_image_ids = [img_id for img_id in unlabeled_image_ids if img_id not in selected_negatives]
            unlabeled_data = np.array([sample for img_id, sample in zip(image_ids, dataset) if img_id in unlabeled_image_ids])
            unlabeled_labels = np.array([labels[np.where(image_ids == img_id)[0][0], class_label] for img_id in unlabeled_image_ids])

            temp_train_data = np.vstack((train_data, negative_samples))
            temp_train_labels = np.hstack((train_labels, negative_labels))

            if len(confident_image_ids) == 0:
                confident_image_ids = selected_negatives
            else:
                confident_image_ids.extend(selected_negatives)

        temp_trained_models = {name: model.fit(temp_train_data, temp_train_labels) for name, model in models.items()}
        current_performance, combined_preds_2 = analyze_performance(temp_trained_models, reshaped_dataset, labels[:, class_label], macro_weight, micro_weight)

        if class_label not in all_predictions:
            all_predictions[class_label] = {}
        if main_loop not in all_predictions[class_label]:
            all_predictions[class_label][main_loop] = {}
        all_predictions[class_label][main_loop][iteration] = prev_performance

        with open('all_predictions.json', 'w') as f:
            json.dump(all_predictions, f, default=convert_to_serializable)

        model_predictions = np.array(list(current_performance['model_predictions'].values()))
        disagreement = np.std(model_predictions, axis=0)
        avg_disagreement = np.mean(disagreement)

        disagreement_trend.append(avg_disagreement)
        #print(f"Average Model Disagreement: {avg_disagreement:.4f}")

        #if current_performance['weighted_f1'] > prev_performance['weighted_f1']:
        #else:
        #    print(f"Performance did not improve for class {class_label}, stopping early.")
        #    break
        if avg_disagreement > disagreement_threshold:
            print(f"High model disagreement detected (>{disagreement_threshold}). Stopping pseudo-labeling early.")
            break
        else:
            train_data = temp_train_data
            train_labels = temp_train_labels
            for img_id, pred in zip(confident_image_ids, combined_preds[np.isin(image_ids, confident_image_ids)]):
                pseudo_labels[class_label][img_id] = pred

            trained_models = temp_trained_models

        for img_id, pred in zip(confident_image_ids, combined_preds[np.isin(image_ids, confident_image_ids)]):
            if img_id in results_matrix.index:
                results_matrix.at[img_id, 'Pseudo_Label'] = int(pred > dynamic_threshold)
                results_matrix.at[img_id, 'Confidence_Score'] = pred
                results_matrix.at[img_id, 'Disagreement_Flag'] = (
                    np.std(np.array(list(current_performance['model_predictions'].values())), axis=0)[np.where(image_ids == img_id)[0][0]] > 0.5
                )
            else:
                results_matrix.loc[img_id] = {
                    'Pseudo_Label': int(pred > dynamic_threshold),
                    'Confidence_Score': pred,
                    'Disagreement_Flag': (
                        np.std(np.array(list(current_performance['model_predictions'].values())), axis=0)[np.where(image_ids == img_id)[0][0]] > 0.5
                    )
                }
        current_performance, combined_preds = analyze_performance(temp_trained_models, reshaped_dataset, labels[:, class_label], macro_weight, micro_weight)

    return pseudo_labels, trained_models, results_matrix

def analyze_performance(trained_models, validation_data, validation_labels, macro_weight, micro_weight):
    performance_metrics = {}
    model_preds = {}

    for model_name, model in trained_models.items():
        preds = model.predict(validation_data)

        f1 = f1_score(validation_labels, preds)
        precision = precision_score(validation_labels, preds)
        recall = recall_score(validation_labels, preds)

        performance_metrics[model_name] = {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        if hasattr(model, 'predict_proba'):
            model_preds[model_name] = model.predict_proba(validation_data)[:, 1]

    combined_preds = (
        macro_weight * (model_preds['svm'] + model_preds['knn']) / 2 +
        micro_weight * (model_preds['xgb'] + model_preds['lgbm'] + model_preds['catboost']) / 3
    )
    weighted_f1 = sum(
        performance_metrics[model]['f1'] * (macro_weight / 2 if model in ['svm', 'knn'] else micro_weight / 3)
        for model in performance_metrics
    )
    weighted_precision = sum(
        performance_metrics[model]['precision'] * (macro_weight / 2 if model in ['svm', 'knn'] else micro_weight / 3)
        for model in performance_metrics
    )
    weighted_recall = sum(
        performance_metrics[model]['recall'] * (macro_weight / 2 if model in ['svm', 'knn'] else micro_weight / 3)
        for model in performance_metrics
    )
    return {
        'individual_scores': performance_metrics,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'combined_preds': combined_preds,
        'model_predictions': model_preds
    }, combined_preds

def build_train_labels(SSL_results_dir, AL_results_dir, num_classes):

    train_labels_dict = {}

    for class_label in range(num_classes):
        training_set_path = os.path.join(SSL_results_dir, f"class_{class_label}_training_set.npz")
        if os.path.exists(training_set_path):
            data = np.load(training_set_path)
            class_labels = data['train_labels']
            image_ids = data['image_ids']
        else:
            training_set_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz")              
            data = np.load(training_set_path)
            train_data = data['train_data']
            class_labels = data['train_labels']
            if "image_ids" in data:
                image_ids = data['image_ids']
            else:
                image_ids = generate_image_ids_from_data(train_data)

        for img_id, label in zip(image_ids, class_labels):
            if img_id not in train_labels_dict:
                train_labels_dict[img_id] = np.zeros(num_classes, dtype=int)
            
            train_labels_dict[img_id][class_label] = label

    sorted_image_ids = sorted(train_labels_dict.keys())
    full_train_labels = np.array([train_labels_dict[img_id] for img_id in sorted_image_ids])

    return full_train_labels, sorted_image_ids

def build_co_occurrence_matrix(labels):

    num_classes = labels.shape[1]
    co_occurrence_matrix = {}

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                co_occurrence_count = np.sum((labels[:, i] == 1) & (labels[:, j] == 1))
                total_count_i = np.sum(labels[:, i] == 1)

                if total_count_i > 0:
                    co_occurrence_prob = co_occurrence_count / total_count_i
                else:
                    co_occurrence_prob = 0

                co_occurrence_matrix[(i, j)] = co_occurrence_prob

    return co_occurrence_matrix

def co_occurrence_voting(pseudo_labels, co_occurrence_matrix, alpha=0.9, beta=0.1):
    final_labels = {}
    #max_length = max(len(p) for p in pseudo_labels.values())
    added_label_counts = {}

    for label_i, prob_dict in pseudo_labels.items():
        image_ids = np.array(list(prob_dict.keys()))
        prob_values = np.array(list(prob_dict.values()), dtype=float)
        #prob_i = np.pad(prob_i, (0, max_length - len(prob_i)), mode='constant', constant_values=0)
        negative_mask = prob_values < 0.4
        non_negative_ids = image_ids[~negative_mask]
        original_positive_mask = prob_values > 0.4
        original_positive_count = np.sum(original_positive_mask)
        if original_positive_count > 0: 
            required_positive_count = int(np.ceil(0.75 * original_positive_count)) 
        else: 
            required_positive_count = 0
        filtered_prob_values = prob_values[~negative_mask]

        dynamic_threshold = max(0.3, min(0.85, np.mean(filtered_prob_values) + 0.05))
        score = alpha * prob_values
        co_occurrence_score = np.zeros_like(prob_values)

        for label_j, other_probs_dict in pseudo_labels.items():
            if label_i != label_j:
                other_image_ids = np.array(list(other_probs_dict.keys()))
                #other_probs = np.array(list(other_probs_dict.values()), dtype=float)
                co_prob = co_occurrence_matrix.get((label_i, label_j), 0)
                valid_ids = np.intersect1d(image_ids, other_image_ids)
                valid_mask = np.isin(image_ids, valid_ids)
                matching_other_probs = np.array([other_probs_dict[img_id] for img_id in image_ids if img_id in other_probs_dict])
                co_occurrence_score[valid_mask] += co_prob * (matching_other_probs > dynamic_threshold) * (1 + beta)

        positive_labels = ((score + (1 - alpha) * co_occurrence_score) > dynamic_threshold).astype(int)
        threshold_reduction_step = 0.05
        #min_threshold = 0.3
        while np.sum(positive_labels[original_positive_mask]) < required_positive_count:
            dynamic_threshold -= threshold_reduction_step
            positive_labels = ((score + (1 - alpha) * co_occurrence_score) > dynamic_threshold).astype(int)

        #negative_labels = (prob_values < 0.4).astype(int)
        final_labels[label_i] = {img_id: label for img_id, label in zip(image_ids, positive_labels)}
        for img_id in image_ids[negative_mask]:
            final_labels[label_i][img_id] = 0 

        num_added_labels = np.sum(list(final_labels[label_i].values()))
        num_added_negatives = np.sum(negative_mask)
        total_proposed_labels = len(prob_values)
        total_added = num_added_labels + num_added_negatives
        added_label_counts[label_i] = (num_added_labels, num_added_negatives, total_proposed_labels, total_added)

    for label_i, (added_pos, added_neg, total_proposed, total) in added_label_counts.items():
        print(f"Class {label_i}: {added_pos} pseudo-positive labels and {added_neg} pseudo-negatives ({total}/{total_proposed}) total")

    return final_labels

def initialize_results_matrix(dataset, labels):
    data_ids = np.arange(len(dataset))
    columns = ['Data_ID', 'True_Label', 'Pseudo_Label', 'Confidence_Score', 'Disagreement_Flag']
    results_matrix = pd.DataFrame({'Data_ID': data_ids})
    results_matrix['True_Label'] = [label_row for label_row in labels]
    results_matrix['Pseudo_Label'] = np.nan
    results_matrix['Confidence_Score'] = np.nan
    results_matrix['Disagreement_Flag'] = False
    return results_matrix

def evaluate_ssl_results(ssl_results_dir, pretrained_models_dir, dataset, ground_truth_labels, classifiers, n_bootstrap=1000, p_threshold=0.05, image_ids=None):
 
    evaluation_results = {}
    direct_metrics = {}
    auc_scores = []
    pr_scores = []
    manual_vs_auto = {}
 
    for class_label in range(ground_truth_labels.shape[1]):
        ssl_path = os.path.join(ssl_results_dir, f"class_{class_label}_training_set.npz")
        model_path = os.path.join(pretrained_models_dir, f"trained_model_class_{class_label}.pkl")
 
        if not os.path.exists(ssl_path):
            continue
        
        ssl_data = np.load(ssl_path)
        ssl_labels = ssl_data['train_labels']
        ssl_image_ids = ssl_data['image_ids'] 
        #valid_image_ids = np.intersect1d(ssl_image_ids, image_ids)
        #mask = np.isin(image_ids, valid_image_ids)
        #true_labels = ground_truth_labels[mask, class_label]
        #missing_image_ids = np.setdiff1d(image_ids, ssl_image_ids, assume_unique=True)
        valid_mask = np.isin(image_ids, ssl_image_ids)
        valid_image_ids = image_ids[valid_mask]
        true_labels = ground_truth_labels[valid_mask, class_label]
        missing_image_ids = image_ids[~valid_mask]

        if len(missing_image_ids) > 0:
            print(f"Warning: Class {class_label} missing {len(missing_image_ids)} labels. Predicting missing values...")
 
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    pretrained_model = pickle.load(f)
            else:
                print(f"Skipping Class {class_label} as no trained model is available.")
                continue
 
            model_preds = {}
            auroc_scores = {}
            for model_name, model in pretrained_model.items():
                if hasattr(model, 'predict_proba'):
                    proba_preds = model.predict_proba(dataset)[:, 1]
                    model_preds[model_name] = proba_preds
                    try:
                        auroc_scores[model_name] = roc_auc_score(ground_truth_labels[:, class_label], proba_preds)
                    except ValueError:
                        auroc_scores[model_name] = 0.5
                
            total_auroc = sum(auroc_scores.values())
            if total_auroc > 0:
                model_weights = {model: score / total_auroc for model, score in auroc_scores.items()}
            else:
                model_weights = {model: 1 / len(classifiers) for model in classifiers}
 
            print(f"Class {class_label}: Model Weights (AUROC-based): {model_weights}")
 
            combined_preds = np.zeros_like(list(model_preds.values())[0])
            for model_name, weight in model_weights.items():
                combined_preds += weight * model_preds[model_name]
 
            valid_matching_ids = np.intersect1d(ssl_image_ids, image_ids)
            aligned_preds = np.array([combined_preds[np.where(image_ids == img_id)[0][0]] for img_id in valid_matching_ids])
            aligned_labels = np.array([ground_truth_labels[np.where(image_ids == img_id)[0][0], class_label] for img_id in valid_matching_ids])           
            
            num_classes = ground_truth_labels.shape[1]
            total_samples = ground_truth_labels.shape[0]
            class_occurrence = np.sum(ground_truth_labels[:, class_label])
            class_ratio = class_occurrence / total_samples
            if class_ratio >= (1/num_classes):
                method = "roc"
            else:
                method = "prc"
 
            if method == "roc":
 
                fpr, tpr, thresholds = roc_curve(aligned_labels, aligned_preds)
                youden_index = tpr - fpr
                optimal_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[optimal_idx]
 
            elif method == "prc":
                # F1-maximizing threshold from the Precision-Recall Curve (PRC)
                precisions, recalls, thresholds = precision_recall_curve(aligned_labels, aligned_preds)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
 
            #predicted_labels = (combined_preds > 0.5).astype(int)
            print(f"Optimal Threshold for Class {class_label}: {optimal_threshold:.4f}")
            predicted_labels = (combined_preds > optimal_threshold).astype(int)
            #predicted_labels_dict = {img_id: int(pred > optimal_threshold) for img_id, pred in zip(image_ids, combined_preds)}
            #missing_predictions = np.array([predicted_labels_dict[img_id] for img_id in missing_image_ids])
            #print(f"Before Concatenation: ssl_labels={ssl_labels.shape}, missing_predictions={missing_predictions.shape}")
            ssl_labels_by_id = defaultdict(list)
            for img_id, label in zip(ssl_image_ids, ssl_labels):
                ssl_labels_by_id[img_id].append(label)
            #full_ssl_labels_dict = {img_id: ssl_labels[np.where(ssl_image_ids == img_id)[0][0]] for img_id in ssl_image_ids}
            #for img_id, pred in zip(missing_image_ids, missing_predictions):
            #    full_ssl_labels_dict[img_id] = pred
            #complete_ssl_labels_dict = {img_id: full_ssl_labels_dict.get(img_id, 0) for img_id in image_ids}
            #sorted_ssl_image_ids = np.array(list(complete_ssl_labels_dict.keys()))
            #sorted_ssl_labels = np.array(list(complete_ssl_labels_dict.values()))
            #ssl_image_ids = np.array(sorted_ssl_image_ids)
            #ssl_labels = sorted_ssl_labels
            complete_ssl_labels = []
            for idx, img_id in enumerate(image_ids):
                if ssl_labels_by_id[img_id]:
                    complete_ssl_labels.append(ssl_labels_by_id[img_id].pop(0))
                else:
                    complete_ssl_labels.append(predicted_labels[idx])
            ssl_labels = np.array(complete_ssl_labels)
            ssl_image_ids = np.array(image_ids)
            print(f"After Alignment: ssl_labels={ssl_labels.shape}")

        true_labels = ground_truth_labels[:, class_label]
        predicted_labels = complete_ssl_labels
        tp = np.sum((true_labels == 1) & (predicted_labels == 1))
        tn = np.sum((true_labels == 0) & (predicted_labels == 0))
        fp = np.sum((true_labels == 0) & (predicted_labels == 1))
        fn = np.sum((true_labels == 1) & (predicted_labels == 0))
        
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
 
        direct_metrics[class_label] = {
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
 
        ssl_path = os.path.join(ssl_results_dir, f"class_{class_label}_training_set.npz")
        model_path = os.path.join(pretrained_models_dir, f"trained_model_class_{class_label}.pkl")
 
        if not (os.path.exists(ssl_path) and os.path.exists(model_path)):
            continue
 
        ssl_data = np.load(ssl_path)
        ssl_train_features = ssl_data['train_data']
        ssl_train_labels = ssl_data['train_labels']
 
        with open(model_path, 'rb') as f:
            pretrained_ssl_model = pickle.load(f)
 
        model_preds = {}
        auroc_scores = {}
        for model_name, model in pretrained_ssl_model.items():
            if hasattr(model, 'predict_proba'):
                proba_preds = model.predict_proba(dataset)[:, 1]
                model_preds[model_name] = proba_preds
                try:
                    auroc_scores[model_name] = roc_auc_score(ground_truth_labels[:, class_label], proba_preds)
                except ValueError:
                    auroc_scores[model_name] = 0.5
            
        total_auroc = sum(auroc_scores.values())
        if total_auroc > 0:
            model_weights = {model: score / total_auroc for model, score in auroc_scores.items()}
        else:
            model_weights = {model: 1 / len(classifiers) for model in classifiers}
 
        print(f"Class {class_label}: Model Weights (AUROC-based): {model_weights}")
 
        combined_preds = np.zeros_like(list(model_preds.values())[0])
        for model_name, weight in model_weights.items():
            combined_preds += weight * model_preds[model_name]
        
        #ssl_preds = pretrained_ssl_model.predict_proba(ssl_train_features)[:, 1]
        ssl_preds = combined_preds
        manual_features = dataset
        manual_labels = ground_truth_labels[:, class_label]
 
        model_performance = {}
 
        for clf_name, clf in classifiers.items():
 
            cv_auc_manual = cross_val_score(clf, manual_features, manual_labels, cv=5, scoring='roc_auc')
            cv_pr_manual = cross_val_score(clf, manual_features, manual_labels, cv=5, scoring='average_precision')
 
            auc_auto = roc_auc_score(ssl_labels, ssl_preds)
            pr_auto = average_precision_score(ssl_labels, ssl_preds)
 
            auc_scores.append((auc_auto, np.mean(cv_auc_manual)))
            pr_scores.append((pr_auto, np.mean(cv_pr_manual)))
 
            model_performance[clf_name] = {
                "auc_auto": auc_auto,
                "cv_auc_manual_mean": np.mean(cv_auc_manual),
                "pr_auto": pr_auto,
                "cv_pr_manual_mean": np.mean(cv_pr_manual)
            }
 
        manual_vs_auto[class_label] = model_performance
 
    evaluation_results["direct_comparison"] = direct_metrics
    evaluation_results["indirect_comparison"] = manual_vs_auto
 
    auc_auto_vals, auc_manual_vals = zip(*auc_scores)
    pr_auto_vals, pr_manual_vals = zip(*pr_scores)
 
    anova_auc = f_oneway(auc_auto_vals, auc_manual_vals)
    anova_pr = f_oneway(pr_auto_vals, pr_manual_vals)
 
    if anova_auc.pvalue < p_threshold:
        print(f"Significant AUC difference detected (p = {anova_auc.pvalue})")
 
    ttest_auc = ttest_ind(auc_auto_vals, auc_manual_vals)
    ttest_pr = ttest_ind(pr_auto_vals, pr_manual_vals)
 
    bootstrap_results = {"auc_auto": [], "auc_manual": [], "pr_auto": [], "pr_manual": []}
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(len(auc_auto_vals), len(auc_auto_vals), replace=True)
        bootstrap_results["auc_auto"].append(np.mean(np.array(auc_auto_vals)[sample_indices]))
        bootstrap_results["auc_manual"].append(np.mean(np.array(auc_manual_vals)[sample_indices]))
        bootstrap_results["pr_auto"].append(np.mean(np.array(pr_auto_vals)[sample_indices]))
        bootstrap_results["pr_manual"].append(np.mean(np.array(pr_manual_vals)[sample_indices]))
 
    evaluation_results["statistical_analysis"] = {
        "anova_auc": {"statistic": anova_auc.statistic, "p-value": anova_auc.pvalue},
        "anova_pr": {"statistic": anova_pr.statistic, "p-value": anova_pr.pvalue},
        "ttest_auc": {"statistic": ttest_auc.statistic, "p-value": ttest_auc.pvalue},
        "ttest_pr": {"statistic": ttest_pr.statistic, "p-value": ttest_pr.pvalue},
        "bootstrap_results": bootstrap_results
    }
 
    results_path = os.path.join(ssl_results_dir, "ssl_evaluation_results.json")
    evaluation_results_converted = {key: convert_numpy_types(value) for key, value in evaluation_results.items()}
    with open(results_path, "w") as f:
        json.dump(evaluation_results_converted, f, indent=4)
 
    return evaluation_results

def evaluate_al_ssl_models(dataset, labels, features_dir, AL_results_dir, SSL_training_dir, SSL_model_dir):

    evaluation_results = {
        "AL": {"micro_f1": [], "macro_f1": [], "micro_precision": [], "macro_precision": [],
               "micro_recall": [], "macro_recall": [], "micro_accuracy": [], "macro_accuracy": [],
               "micro_roc_auc": [], "macro_roc_auc": [], 
               "per_class_f1": [], "per_class_precision": [], "per_class_recall": [],
               "per_class_accuracy": [], "per_class_roc_auc": [], "per_class_loss": []},
        "SSL": {"micro_f1": [], "macro_f1": [], "micro_precision": [], "macro_precision": [],
                "micro_recall": [], "macro_recall": [], "micro_accuracy": [], "macro_accuracy": [],
                "micro_roc_auc": [], "macro_roc_auc": [], 
                "per_class_f1": [], "per_class_precision": [], "per_class_recall": [],
                "per_class_accuracy": [], "per_class_roc_auc": [], "per_class_loss": []}
    }

    num_classes = labels.shape[1]
    bovw_path = os.path.join(features_dir, f"bovw_features.npz")
    lbp_path = os.path.join(features_dir, f"lbp_features.npz")
    bovw_features = np.load(bovw_path)['features']
    lbp_features = np.load(lbp_path)['features']

    for class_label in range(num_classes):
        al_training_path = os.path.join(AL_results_dir, f"class_{class_label}_training_set.npz")
        al_bovw_path = os.path.join(AL_results_dir, f"class_{class_label}_bovw_features.npz")
        al_lbp_path = os.path.join(AL_results_dir, f"class_{class_label}_lbp_features.npz")
        ssl_training_path = os.path.join(SSL_training_dir, f"class_{class_label}_training_set.npz")
        ssl_model_path = os.path.join(SSL_model_dir, f"trained_model_class_{class_label}.pkl")

        if not os.path.exists(al_training_path) or not os.path.exists(ssl_model_path):
            print(f"Skipping Class {class_label}: Model not found in AL or SSL results.")
            continue

        y_true = labels[:, class_label]

        al_data = np.load(al_training_path)
        al_labels = al_data['train_labels']
        al_bovw_features = np.load(al_bovw_path)['features']
        al_lbp_features = np.load(al_lbp_path)['features']

        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        rf_model = RandomForestClassifier()
        xgb_model.fit(al_bovw_features, al_labels)
        rf_model.fit(al_lbp_features, al_labels)

        xgb_prob_train = xgb_model.predict_proba(al_bovw_features)[:, 1]
        rf_prob_train = rf_model.predict_proba(al_lbp_features)[:, 1]

        xgb_auroc = roc_auc_score(al_labels, xgb_prob_train)
        rf_auroc = roc_auc_score(al_labels, rf_prob_train)
        total_auroc = xgb_auroc + rf_auroc
        xgb_weight = xgb_auroc / total_auroc
        rf_weight = rf_auroc / total_auroc

        xgb_prob = xgb_model.predict_proba(bovw_features)[:, 1]
        rf_prob = rf_model.predict_proba(lbp_features)[:, 1]

        al_combined_prob = (xgb_weight * xgb_prob) + (rf_weight * rf_prob)
        y_pred_al = (al_combined_prob > 0.5).astype(int)

        with open(ssl_model_path, 'rb') as f:
            ssl_model = pickle.load(f)

        ssl_data = np.load(ssl_training_path)
        ssl_labels = ssl_data['train_labels']
        ssl_features = ssl_data['train_data']

        model_preds = {}
        auroc_scores = {}
        for model_name, model in ssl_model.items():
            proba_preds = model.predict_proba(ssl_features)[:, 1]
            auroc_scores[model_name] = roc_auc_score(ssl_labels, proba_preds)
            
        total_auroc = sum(auroc_scores.values())
        model_weights = {model: score / total_auroc for model, score in auroc_scores.items()}

        for model_name, weight in model_weights.items():
            model_preds[model_name] = model.predict_proba(dataset)[:, 1]
        ssl_combined_prob = sum(model_preds[m] * w for m, w in model_weights.items())
        y_pred_ssl = (ssl_combined_prob > 0.5).astype(int)

        for model_type, y_pred in zip(["AL", "SSL"], [y_pred_al, y_pred_ssl]): 
            class_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
            class_precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
            class_recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
            class_accuracy_scores = accuracy_score(y_true, y_pred)
            class_roc_auc_scores = roc_auc_score(y_true, y_pred, average=None)
            class_loss_scores = log_loss(y_true, y_pred)

            micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
            macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
            macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            micro_accuracy = accuracy_score(y_true, y_pred)
            macro_accuracy = np.mean(class_accuracy_scores)
            micro_roc_auc = roc_auc_score(y_true.ravel(), y_pred.ravel())
            macro_roc_auc = np.mean(class_roc_auc_scores)

            evaluation_results[model_type]["per_class_f1"].append(class_f1_scores.tolist())
            evaluation_results[model_type]["per_class_precision"].append(class_precision_scores.tolist())
            evaluation_results[model_type]["per_class_recall"].append(class_recall_scores.tolist())
            evaluation_results[model_type]["per_class_accuracy"].append(class_accuracy_scores)
            evaluation_results[model_type]["per_class_roc_auc"].append(class_roc_auc_scores)
            evaluation_results[model_type]["per_class_loss"].append(class_loss_scores)
            evaluation_results[model_type]["micro_f1"].append(micro_f1)
            evaluation_results[model_type]["macro_f1"].append(macro_f1)
            evaluation_results[model_type]["micro_precision"].append(micro_precision)
            evaluation_results[model_type]["macro_precision"].append(macro_precision)
            evaluation_results[model_type]["micro_recall"].append(micro_recall)
            evaluation_results[model_type]["macro_recall"].append(macro_recall)
            evaluation_results[model_type]["micro_accuracy"].append(micro_accuracy)
            evaluation_results[model_type]["macro_accuracy"].append(macro_accuracy)
            evaluation_results[model_type]["micro_roc_auc"].append(micro_roc_auc)
            evaluation_results[model_type]["macro_roc_auc"].append(macro_roc_auc)

        for model_type in ["AL", "SSL"]:
            for metric in ["micro_f1", "macro_f1", "micro_precision", "macro_precision", "micro_recall", "macro_recall",
                        "micro_accuracy", "macro_accuracy", "micro_roc_auc", "macro_roc_auc"]:
                evaluation_results[model_type][f"mean_{metric}"] = np.mean(evaluation_results[model_type][metric])

    results_path = os.path.join(SSL_training_dir, "al_ssl_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    return evaluation_results

def final_evaluation(reshaped_dataset, bovw_dataset, lbp_dataset, labels, training_set_dir, SSL_models, kmeans, output_dir):

    trained_models = {class_label: {} for class_label in range(labels.shape[1])}
    #x_train = {class_label: {} for class_label in range(labels.shape[1])}
    #y_train = {class_label: {} for class_label in range(labels.shape[1])}
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    rf_model = RandomForestClassifier()
    reducer = PCA(n_components=100)
    final_labels = {
        "Per Model Metrics": {},
        "SSL Model Combination": {},
        "AL Model Combination": {}
    }
    evaluation_results = {
        "Per Model Metrics": {},
        "SSL Model Combination": {},
        "AL Model Combination": {}
    }
    
def final_evaluation(reshaped_dataset, bovw_dataset, lbp_dataset, labels, training_set_dir, SSL_models, kmeans, output_dir):

    trained_models = {class_label: {} for class_label in range(labels.shape[1])}
    #x_train = {class_label: {} for class_label in range(labels.shape[1])}
    #y_train = {class_label: {} for class_label in range(labels.shape[1])}
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    rf_model = RandomForestClassifier()
    reducer = PCA(n_components=100)
    final_labels = {
        "Per Model Metrics": {},
        "SSL Model Combination": {},
        "AL Model Combination": {}
    }
    evaluation_results = {
        "Per Model Metrics": {},
        "SSL Model Combination": {},
        "AL Model Combination": {}
    }
    
    def apply_threshold(predictions, thresholds, class_label):
        return np.array([(predictions[class_idx] >= thresholds[class_label][model_name]).astype(int) for class_idx, model_name in enumerate(thresholds[class_label].keys())]).T  

    for class_label in range(labels.shape[1]):
        training_set_path = os.path.join(training_set_dir, f"class_{class_label}_training_set.npz")
        bovw_features_path = os.path.join(training_set_dir, f"class_{class_label}_bovw_features.npz")
        lbp_features_path = os.path.join(training_set_dir, f"class_{class_label}_lbp_features.npz")
        data = np.load(training_set_path, allow_pickle=True)
        train_data = data["train_data"]
        train_labels = data["train_labels"]
        train_dataset = FeatureDataset(train_data, train_labels)
        """
        reshaped_path = "{training_set_dir}_reshaped"
        if os.path.exists(reshaped_path):
            training_set_path = os.path.join(reshaped_path, f"class_{class_label}_training_set.npz")
            data = np.load(training_set_path, allow_pickle=True)
            train_data = data["train_data"]
            train_labels = data["train_labels"]
        """
        if train_data.shape[1] > 100:
            if not os.path.exists(bovw_features_path):
                bovw_features, bovw_labels, _ = extract_bovw_features(train_dataset, kmeans=kmeans)
            else:
                bovw_data = np.load(bovw_features_path, allow_pickle=True)
                bovw_features = bovw_data["features"]
                bovw_labels = train_labels
            if not os.path.exists(lbp_features_path):
                lbp_features, lbp_labels = extract_lbp_features(train_dataset)
            else:
                lbp_data = np.load(lbp_features_path, allow_pickle=True)
                lbp_features = lbp_data["features"]
                lbp_labels = train_labels
            train_data = train_data.reshape(train_data.shape[0], -1)
            train_data = reducer.fit_transform(train_data)
            trained_models[class_label] = {}
            for model_name, model in SSL_models.items():
                model.fit(train_data, train_labels)
                trained_models[class_label][model_name] = model
                #x_train[class_label][model_name] = train_data
                #y_train[class_label][model_name] = train_labels

            xgb_model.fit(bovw_features, bovw_labels)
            trained_models[class_label]["xgb_bovw"] = xgb_model
            #x_train[class_label]["xgb_bovw"] = bovw_features
            #y_train[class_label]["xgb_bovw"] = bovw_labels
            rf_model.fit(lbp_features, lbp_labels)
            trained_models[class_label]["rf_lbp"] = rf_model
            #x_train[class_label]["rf_lbp"] = lbp_features
            #y_train[class_label]["rf_lbp"] = lbp_labels
        else:
            trained_models[class_label] = {}
            for model_name, model in SSL_models.items():
                model.fit(train_data, train_labels)
                trained_models[class_label][model_name] = model
                #x_train[class_label][model_name] = train_data
                #y_train[class_label][model_name] = train_labels
       
       
    best_thresholds, best_f1s, best_predictions = {class_label: {} for class_label in range(labels.shape[1])}, {class_label: {} for class_label in range(labels.shape[1])}, {class_label: {} for class_label in range(labels.shape[1])}
    for class_label, models in trained_models.items():
        print(f"Finding best thresholds for class {class_label}...")
        best_thresholds[class_label] = {}
        best_predictions[class_label] = {}
        y_true = labels[:, class_label]

        for model_name, model in models.items():
            if model_name == "xgb_bovw":
                x_test = bovw_dataset
            elif model_name == "rf_lbp":
                x_test = lbp_dataset
            else:
                x_test = reshaped_dataset
             
            y_pred_raw = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_test)

            best_threshold, best_f1 = 0.5, 0  
            for threshold in np.arange(0.05, 0.7, 0.025): 
                y_pred = (y_pred_raw >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            best_thresholds[class_label][model_name] = best_threshold
            best_f1s[class_label][model_name] = best_f1
            best_predictions[class_label][model_name] = (y_pred_raw >= best_threshold).astype(int)

            print(f"Best Threshold for {model_name.upper()} in class {class_label}: {best_threshold:.2f} (F1-score: {best_f1:.4f})")

    AL_micro_f1_scores = []
    AL_macro_f1_scores = []
    AL_micro_recall = []
    AL_macro_recall = []
    AL_micro_precision = []
    AL_macro_precision = []
    SSL_micro_f1_scores = []
    SSL_macro_f1_scores = []
    SSL_micro_recall = []
    SSL_macro_recall = []
    SSL_micro_precision = []
    SSL_macro_precision = []
    optimization_results = {}
    al_results = {}

    for class_label, models in trained_models.items():
        print(f"Finding best model combination for class {class_label}...")
        y_true = labels[:, class_label]
        model_names = list(models.keys())

        best_f1 = 0
        best_combination = None
        best_pred = None
        best_voting_method = None
        best_threshold = None

        optimization_results[class_label] = {
            "model_weights": {},
            "voting_mechanism": None,
            "optimized_thresholds": {},
            "best_combination": None,
            "final_f1_score": None,
            "final_preds":{}
        }

        model_f1_scores = {model: f1_score(y_true, best_predictions[class_label][model], zero_division=0) for model in model_names}

        total_f1 = sum(model_f1_scores.values())
        model_weights = {model: (model_f1_scores[model] / total_f1) if total_f1 > 0 else 1 / len(model_names) for model in model_names}
        optimization_results[class_label]["model_weights"] = model_weights

        for r in range(1, len(model_names) + 1):
            for model_subset in combinations(model_names, r):
                binary_predictions = [(best_predictions[class_label][model] >= best_thresholds[class_label][model]).astype(int) for model in model_subset]

                majority_vote = np.mean(binary_predictions, axis=0)
                weighted_prediction = np.sum([binary_predictions[i] * model_weights[model] for i, model in enumerate(model_subset)], axis=0)

                for threshold in np.arange(0.05, 0.7, 0.025):
                    final_vote_pred = (majority_vote >= threshold).astype(int)
                    final_weighted_pred = (weighted_prediction >= threshold).astype(int)

                    f1_vote = f1_score(y_true, final_vote_pred, average='macro',zero_division=0)
                    f1_weighted = f1_score(y_true, final_weighted_pred, average='macro', zero_division=0)
                    f1_vote_2 = f1_score(y_true, final_vote_pred, average='micro',zero_division=0)
                    f1_weighted_2 = f1_score(y_true, final_weighted_pred, average='micro', zero_division=0)

                    if f1_vote > best_f1:
                        best_f1 = f1_vote
                        best_f1_2 = f1_vote_2
                        best_combination = model_subset
                        best_pred = final_vote_pred
                        best_voting_method = "Majority Voting"
                        best_threshold = threshold

                    if f1_weighted > best_f1:
                        best_f1 = f1_weighted
                        best_f1_2 = f1_weighted_2
                        best_combination = model_subset
                        best_pred = final_weighted_pred
                        best_voting_method = "Weighted Voting"
                        best_threshold = threshold

                        SSL_macro_f1_scores.append(f1_score(y_true, best_pred, average="macro", zero_division=0))
                        SSL_micro_f1_scores.append(f1_score(y_true, best_pred, average="micro", zero_division=0))
                        SSL_micro_recall.append(recall_score(y_true, best_pred, average="micro", zero_division=0))
                        SSL_macro_recall.append(recall_score(y_true, best_pred, average="macro", zero_division=0))
                        SSL_micro_precision.append(precision_score(y_true, best_pred, average="micro", zero_division=0))
                        SSL_macro_precision.append(precision_score(y_true, best_pred, average="macro", zero_division=0))
        
        optimization_results[class_label]["best_combination"] = best_combination
        optimization_results[class_label]["voting_mechanism"] = best_voting_method
        optimization_results[class_label]["optimized_thresholds"] = best_threshold
        optimization_results[class_label]["final_f1_score"] = best_f1_2
        optimization_results[class_label]["final_preds"] = best_pred
        optimization_results[class_label]["voting_mechanism"] = best_voting_method

        print(f"Best SSL Combination for class {class_label}: {best_combination}")
        print(f"SSL Voting Mechanism Used: {best_voting_method}")
        print(f"SSL Optimal Threshold: {best_threshold:.2f} (F1-score: {best_f1_2:.4f})")

        al_models = ["rf_lbp", "xgb_bovw"]
        if all(al_model in best_predictions[class_label] for al_model in al_models):
            y_pred_rf = best_predictions[class_label]["rf_lbp"]
            y_pred_xgb = best_predictions[class_label]["xgb_bovw"]

            recall_rf = f1_score(y_true, y_pred_rf, zero_division=0)
            precision_xgb = f1_score(y_true, y_pred_xgb, zero_division=0)

            best_al_f1 = 0
            best_al_pred = None
            best_al_threshold = None
            best_al_alpha = None

            for alpha in np.arange(0.05, 0.95, 0.05):
                combined_al_prediction = (alpha * y_pred_rf + (1 - alpha) * y_pred_xgb)

                for threshold in np.arange(0.05, 0.95, 0.025):
                    final_al_pred = (combined_al_prediction >= threshold).astype(int)
                    f1_al = f1_score(y_true, final_al_pred, average='macro', zero_division=0)
                    f1_al_2 = f1_score(y_true, final_al_pred, average='micro', zero_division=0)
                    if f1_al > best_al_f1:
                        best_al_f1 = f1_al
                        best_al_f1_2 = f1_al_2
                        best_al_pred = final_al_pred
                        best_al_threshold = threshold
                        best_al_alpha = alpha

                    AL_macro_f1_scores.append(f1_score(y_true, best_al_pred, average="macro", zero_division=0))
                    AL_micro_f1_scores.append(f1_score(y_true, best_al_pred, average="micro", zero_division=0))
                    AL_micro_recall.append(recall_score(y_true, best_al_pred, average="micro", zero_division=0))
                    AL_macro_recall.append(recall_score(y_true, best_al_pred, average="macro", zero_division=0))
                    AL_micro_precision.append(precision_score(y_true, best_al_pred, average="micro", zero_division=0))
                    AL_macro_precision.append(precision_score(y_true, best_al_pred, average="macro", zero_division=0))
            
            al_results[class_label] = {
                "optimized_alpha": best_al_alpha,
                "optimized_threshold": best_al_threshold,
                "final_f1_score": best_al_f1_2,
                "final_prediction": best_al_pred
            }

        try:
            print(f"AL Model Optimization for class {class_label}: alpha={best_al_alpha:.2f}")
            print(f"AL Optimal Threshold: {best_al_threshold:.2f} (F1-score: {best_al_f1:.4f})")
        except Exception as e:
            print(f"File is missing :{e}")

    true_multi_label = labels 
    final_labels["SSL Model Combination"] = np.array([optimization_results[class_label]["final_preds"] for class_label in sorted(optimization_results.keys())]).T
    try:
        final_labels["AL Model Combination"] = np.array([al_results[class_label]["final_prediction"] for class_label in sorted(al_results.keys())]).T
    except Exception as e:
        print(f"File is missing :{e}")

    ssl_final_predictions = final_labels["SSL Model Combination"]
    evaluation_results["SSL Model Combination"] = {
        "F1-score (macro)": np.mean(SSL_macro_f1_scores),
        "Precision (macro)": np.mean(SSL_macro_precision),
        "Recall (macro)": np.mean(SSL_macro_recall),
        "F1-score (micro)": np.mean(SSL_micro_f1_scores),
        "Precision (micro)": np.mean(SSL_micro_precision),
        "Recall (micro)": np.mean(SSL_micro_recall),
    }
    try:
        al_final_predictions = final_labels["AL Model Combination"]
        evaluation_results["AL Model Combination"] = {
            "F1-score (macro)": np.mean(AL_macro_f1_scores),
            "Precision (macro)": np.mean(AL_macro_precision),
            "Recall (macro)": np.mean(AL_macro_recall),
            "F1-score (micro)": np.mean(AL_micro_f1_scores),
            "Precision (micro)": np.mean(AL_micro_precision),
            "Recall (micro)": np.mean(AL_micro_recall),
        }
    except Exception as e:
        print(f"File is missing :{e}")
    
    print("Final evaluation metrics calculated per model and best model combinations successfully!")

    os.makedirs(output_dir, exist_ok=True)
    best_thresholds_flat = []
    best_f1_flat = []
    for class_label, thresholds in best_thresholds.items():
        for model_name, threshold in thresholds.items():
            best_thresholds_flat.append({"Class": class_label, "Model": model_name, "Best Threshold": threshold})
            best_f1_flat.append({"Class": class_label, "Model": model_name, "Best F1-Score": best_f1s[class_label][model_name]})
    df_best_thresholds = pd.DataFrame(best_thresholds_flat)
    df_best_f1 = pd.DataFrame(best_f1_flat)
    df_best_combined = df_best_thresholds.merge(df_best_f1, on=["Class", "Model"])
    df_best_combined.to_csv(os.path.join(output_dir, "indiv_model_results.csv"), index=False)
    
    opt_results_flat = []
    for class_label, opt_result in optimization_results.items():
        opt_results_flat.append({
            "Class": class_label,
            "Best Combination": ", ".join(opt_result["best_combination"]),
            "Voting Mechanism": opt_result["voting_mechanism"],
            "Optimized Thresholds": json.dumps(opt_result["optimized_thresholds"]),
            "Final F1-Score": opt_result["final_f1_score"],
            "Model Weights": json.dumps(opt_result["model_weights"])
        })
    df_optimization_results = pd.DataFrame(opt_results_flat)
    df_optimization_results.to_csv(os.path.join(output_dir, "Comb_model_results.csv"), index=False)

    al_results_flat = []
    try:
        for class_label, al_result in al_results.items():
            al_results_flat.append({
                "Class": class_label,
                "Optimized Alpha": al_result["optimized_alpha"],
                "Optimized Threshold": al_result["optimized_threshold"],
                "Final F1-Score": al_result["final_f1_score"]
            })
    except Exception as e:
        print(f"File is missing :{e}")

    df_al_results = pd.DataFrame(al_results_flat)
    df_al_results.to_csv(os.path.join(output_dir, "RecPrec_results.csv"), index=False)    

    eval_results_flat = []

    for category, metrics in evaluation_results.items():
        for metric_name, metric_value in metrics.items():
            eval_results_flat.append({
                "Category": category,
                "Metric": metric_name,
                "Value": metric_value
            })

    df_evaluation_results = pd.DataFrame(eval_results_flat)
    df_evaluation_results.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    return evaluation_results


images_dir = './Active_Learning/pascal_voc/JPEGImages'
annotations_path = './Active_Learning//pascal_voc/voc_annotations.json'
save_path = './05_Experiment_3/processed_PascalVOC_dataset.npz'
save_dir = './05_Experiment_3/PascalVOC_training_sets'
stats_path = './05_Experiment_3/PascalVOC_training_stats.json'
feature_dir = './05_Experiment_3/PascalVOC_features'
bovw_features_path = './05_Experiment_3/PascalVOC_features/bovw_features.npz'
lbp_features_path = './05_Experiment_3/PascalVOC_features/lbp_features.npz'
hog_features_path = './05_Experiment_3/PascalVOC_features/hog_features.npz'
AL_results_dir= './05_Experiment_3/PascalVOC_training_sets_AL'
reshaped_dir = './05_Experiment_3/PascalVOC_training_sets_AL_reshaped'
SSL_results_dir = './05_Experiment_3/PascalVOC_training_sets_SSL'
output_stats_path = './05_Experiment_3/training_set_validation.json'
output_stats_path_AL = './05_Experiment_3/training_set_validation_AL.json'
output_stats_path_SSL = './05_Experiment_3/training_set_validation_SSL.json'
results_output_dir = os.path.join(SSL_results_dir, 'ssl_output')


models = {
        'svm': SVC(probability=True, kernel='rbf', C=1.0),
        'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'lgbm': LGBMClassifier(boosting_type='gbdt', num_leaves=31, random_state=42, verbose=-1),
        'catboost': CatBoostClassifier(verbose=0, iterations=100, learning_rate=0.1)
    }

# Suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore")

coco_dataset = PascalVOCDataset(images_dir, annotations_path)
dataset, labels = process_coco_output(coco_dataset, target_size=(224, 224), save_path=save_path, stats_path=stats_path)
extract_features_for_dataset(coco_dataset, save_dir=feature_dir, n_clusters=50)

target_fraction = 0.05 
augment_threshold = 100  
base_augmentations = 6  
n_jobs = 1
image_ids = None

with open(os.path.join(feature_dir, "kmeans.pkl"), "rb") as f:
    kmeans = pickle.load(f)

precomputed_features = {
    "bovw": np.load(bovw_features_path)["features"],
    "lbp": np.load(lbp_features_path)["features"],
    "hog": np.load(hog_features_path)["features"]
}

training_sets, shared_label_pool = create_binary_training_sets_with_augmentation(
    dataset=dataset,
    labels=labels,
    target_fraction=target_fraction,
    augment_threshold=augment_threshold,
    base_augmentations=base_augmentations,
    save_dir=save_dir,
    stats_path=stats_path,
    n_jobs=n_jobs,
    kmeans=kmeans,
    precomputed_features=precomputed_features
)
if os.path.exists(output_stats_path):
    print(f"Skipping training set creation pipeline as {output_stats_path} already exists.")
else:
    validate_training_sets(AL_results_dir, dataset, output_stats_path=output_stats_path)

if os.path.exists(output_stats_path_AL):
    print(f"Skipping AL pipeline as {output_stats_path_AL} already exists.")
else:
    prev_xgb_auroc = None
    prev_rf_auroc = None
    previous_weights = None
    main_active_learning_with_voting_dynamic(
        dataset=dataset,
        labels=labels,
        save_dir=save_dir,
        bovw_features_path= bovw_features_path,
        lbp_features_path= lbp_features_path,
        stats_path= './05_Experiment_3/coco_training_sets_AL/training_stats.json',
        AL_results_dir= AL_results_dir,
        target_fraction = 0.1,
        global_labeled_pool= shared_label_pool
    )
    validate_training_sets(AL_results_dir, dataset, output_stats_path=output_stats_path_AL)

reshaped_dataset = reshape_and_save_datasets(input_dir=AL_results_dir, output_dir=reshaped_dir, dataset=dataset)

if os.path.exists(output_stats_path_SSL):
    print(f"Skipping SSL pipeline as {output_stats_path_SSL} already exists.")
else:
    all_trained_models, performance_log, results_matrix, image_ids = main_ssl_pipeline(models, reshaped_dir, SSL_results_dir, reshaped_dataset, labels, iterations=50)

    validate_training_sets(SSL_results_dir, dataset, output_stats_path=output_stats_path_SSL)
    
    os.makedirs(results_output_dir, exist_ok=True)
    for class_label, models in all_trained_models.items():
        model_path = os.path.join(results_output_dir, f'trained_model_class_{class_label}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)

    performance_log_path = os.path.join(results_output_dir, 'performance_log.json')
    
    serializable_performance_log = [convert_to_serializable2(entry) for entry in performance_log]
    with open(performance_log_path, 'w') as f:
        json.dump(serializable_performance_log, f, indent=2)

    results_matrix_path = os.path.join(results_output_dir, 'results_matrix.csv')
    results_matrix.to_csv(results_matrix_path, index=False)
    print(f"All data for SSL locally saved.")

    evaluation_results = evaluate_ssl_results(
        ssl_results_dir=SSL_results_dir,
        pretrained_models_dir= results_output_dir,
        dataset=reshaped_dataset,
        ground_truth_labels=labels,
        classifiers=models,
        image_ids=image_ids
    )

if os.path.exists(os.path.join(SSL_results_dir, "al_ssl_evaluation_results.json")):
    print(f"Skipping SSL Evaluation as al_ssl_evaluation_results.json already exists.")
else:
    if image_ids is None:
        image_ids = generate_image_ids_from_data(reshaped_dataset)
        image_ids = np.array(image_ids) 
        evaluation_results = evaluate_ssl_results(
            ssl_results_dir=SSL_results_dir,
            pretrained_models_dir= results_output_dir,
            dataset=reshaped_dataset,
            ground_truth_labels=labels,
            classifiers=models,
            image_ids=image_ids
        )
    evaluation_results = evaluate_al_ssl_models(reshaped_dataset, labels, feature_dir, AL_results_dir, SSL_results_dir, results_output_dir)

training_set_dirs = [SSL_results_dir, AL_results_dir] 
output_dirs = ["05_Experiment_3/PascalVoc_results/SSL","05_Experiment_3/PascalVoc_results/AL"]
bovw_dataset = precomputed_features["bovw"] 
lbp_dataset = precomputed_features["lbp"]
for train_dir, out_dir in zip(training_set_dirs, output_dirs):
    print(f"Running final_evaluation for Training Set: {train_dir} ; Output: {out_dir}")
    final_evaluation(reshaped_dataset, bovw_dataset, lbp_dataset, labels, train_dir, models, kmeans, out_dir)
