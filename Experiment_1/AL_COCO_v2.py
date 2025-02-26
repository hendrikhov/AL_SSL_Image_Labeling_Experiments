import sympy as sp
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from PIL import Image
from pycocotools.coco import COCO
import random
import cv2
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from imblearn.over_sampling import ADASYN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
from sklearn.metrics import precision_recall_curve
import os
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_sample_weight

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

    return labeled_indices, unlabeled_indices

def create_sampler(dataset, indices, class_weights):
    sample_weights = []
    for idx in indices:
        _, label = dataset[idx]
        weight = sum(class_weights[i] for i, val in enumerate(label.numpy()) if val == 1)
        sample_weights.append(weight)
    return WeightedRandomSampler(sample_weights, len(indices))


def class_aware_sampling(labels, num_samples):
    class_counts = Counter([label for sublist in labels for label in sublist])
    inverse_class_freq = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [sum(inverse_class_freq[cls] for cls in label) for label in labels]
    weights = np.array(weights) / np.sum(weights)
    sampled_indices = np.random.choice(len(labels), size=num_samples, p=weights)
    return sampled_indices

def diversity_sampling(features, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(features)
    cluster_centers = kmeans.cluster_centers_
    closest_samples = []
    for center in cluster_centers:
        closest = np.argmin(np.linalg.norm(features - center, axis=1))
        closest_samples.append(closest)
    return closest_samples

def entropy_sampling(predictions, num_samples, unlabeled_indices):
    
    epsilon = 1e-8
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)

    top_uncertain_indices = np.argsort(entropy)[::-1][:num_samples]
    print("top_uncertain:", top_uncertain_indices, type(top_uncertain_indices))

    selected_indices = [unlabeled_indices[idx] for idx in top_uncertain_indices]

    print("Top uncertain sample indices:", selected_indices)
    return selected_indices

def extract_features(model, data_loader, device='cpu'):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            embeddings = model(images)
            features.append(embeddings.cpu().numpy())
    return np.vstack(features)


def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())
    return np.vstack(predictions), np.vstack(ground_truth)


def find_optimal_threshold(y_true, y_probs):
    thresholds = []
    for i in range(y_probs.shape[1]):
        precision, recall, threshold = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = threshold[np.argmax(f1)]
        thresholds.append(best_threshold)
    return thresholds


def train_model(model, loader, criterion, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

def ml_smote(X, y, k=5):

    X = np.array(X)
    y = np.array(y, dtype=np.int32)

    if len(X) != len(y):
        raise ValueError("Features and labels must have the same number of samples.")

    X_resampled = X.tolist()
    y_resampled = y.tolist()

    minority_indices = np.where(np.sum(y, axis=1) < y.shape[1] / 2)[0]
    minority_indices = [idx for idx in minority_indices if idx < len(X)]

    if len(minority_indices) == 0:
        print("No minority samples detected; returning original dataset.")
        return X, y

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
            synthetic_label = y[idx] | y[minority_indices[neighbor_idx]]

            X_resampled.append(synthetic_sample.tolist())
            y_resampled.append(synthetic_label.tolist())

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
            neighbor = X[minority_indices[neighbor_idx]]
            synthetic_sample = X[sample_idx] + np.random.random(X[sample_idx].shape) * (neighbor - X[sample_idx])
            synthetic_label = y[sample_idx] | y[minority_indices[neighbor_idx]]

            X_resampled.append(synthetic_sample.tolist())
            y_resampled.append(synthetic_label.tolist())
            label_counts[label] += 1

    return np.array(X_resampled), np.array(y_resampled)
    
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

def extract_sift_features(dataset):

    sift = cv2.SIFT_create()
    descriptors_list = []

    for idx, (img, _) in enumerate(dataset):
        try:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if img is None or img.size == 0:
                print(f"Warning: Image {idx} is empty or invalid, skipping.")
                descriptors_list.append(None)
                continue

            keypoints, descriptors = sift.detectAndCompute(img, None)

            if descriptors is not None:
                descriptors_list.append(descriptors)
            else:
                print(f"Warning: No descriptors found for image {idx}.")
                descriptors_list.append(None)

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            descriptors_list.append(None)

    return descriptors_list


def create_visual_vocabulary(descriptors, num_words=1000):

    valid_descriptors = [desc for desc in descriptors if desc is not None and desc.shape[1] == 128]

    if len(valid_descriptors) == 0:
        raise ValueError("No valid descriptors found. Check your dataset and feature extraction.")

    all_descriptors = np.vstack(valid_descriptors)

    #kmeans = KMeans(n_clusters=num_words, random_state=42)
    kmeans = MiniBatchKMeans(n_clusters=num_words, random_state=42, batch_size=1000)
    kmeans.fit(all_descriptors)

    return kmeans


def generate_bow_histograms(dataset, kmeans):
 
    sift = cv2.SIFT_create()
    histograms = []

    for idx, (img, _) in enumerate(dataset):

        img = img.permute(1, 2, 0).numpy()
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.dtype != np.uint8:
            #print(f"Warning: Image {idx} has invalid depth, converting to uint8.")
            img = (img * 255).astype(np.uint8)

        if img is None or img.size == 0:
            print(f"Warning: Image {idx} is empty or invalid, skipping.")
            continue

        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            print(f"Warning: No descriptors found for image {idx}.")
            continue

        histogram = np.zeros(len(kmeans.cluster_centers_))
        if descriptors is not None:
            clusters = kmeans.predict(descriptors)
            for cluster in clusters:
                histogram[cluster] += 1

        #histograms = normalize(histograms, norm='l2')
        histograms.append(histogram)

    histograms = np.array(histograms)
    if histograms.size == 0:
        raise ValueError("Histogram array is empty. No valid histograms were generated.")
    if histograms.ndim != 2:
        raise ValueError(f"Expected histograms to be 2D, but got shape {histograms.shape}. Check the histogram generation process.")
    histograms_norm = np.linalg.norm(histograms, axis=1, keepdims=True)
    histograms = np.divide(histograms, histograms_norm, where=histograms_norm != 0)
    return histograms

def extract_hog_features(dataset):
    features, labels = [], []
    for img, lbl in dataset:
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
        labels.append(lbl.numpy())
    return np.array(features), np.array(labels)

from skimage.feature import local_binary_pattern

def extract_lbp_features(dataset, radius=3, n_points=8 * 3):

    lbp_features = []
    labels = []

    for idx, (img, label) in enumerate(dataset):
        img = img.permute(1, 2, 0).numpy()

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.dtype != np.uint8:
            #print(f"Warning: Converting image {idx} to uint8 for LBP.")
            img = (img * 255).astype(np.uint8)

        lbp = local_binary_pattern(img, n_points, radius, method='uniform')

        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        lbp_features.append(lbp_hist)
        labels.append(label.numpy())

    return np.array(lbp_features), np.array(labels)


def train_random_forest(train_features, train_labels):
    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    rf.fit(train_features, train_labels)
    return rf

def train_svm(train_features, train_labels):
    svm = MultiOutputClassifier(SVC(kernel="linear", probability=True, random_state=42))
    svm.fit(train_features, train_labels)
    return svm

def train_xgboost(train_features, train_labels):
    xgb = MultiOutputClassifier(XGBClassifier(eval_metric='logloss'))
    xgb.fit(train_features, train_labels)
    return xgb

def evaluate_model_2(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print(classification_report(test_labels, predictions, zero_division=0))

def incremental_train(classifier, train_features, train_labels, batch_size=32):

    num_samples = train_features.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    losses = []
    for epoch in range(3):  # Simulate 3 epochs
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_features = train_features[batch_indices]
            batch_labels = train_labels[batch_indices]

            classifier.partial_fit(batch_features, batch_labels, classes=np.unique(train_labels))
            
            predictions = classifier.predict_proba(batch_features)
            batch_loss = log_loss(batch_labels, predictions)
            epoch_loss += batch_loss

        losses.append(epoch_loss / (num_samples // batch_size))
        print(f"Epoch {epoch + 1}: Loss = {losses[-1]:.4f}")
    
    return losses

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


def count_labels(dataset):
    all_labels = [label.numpy() for _, label in dataset]
    all_labels = np.vstack(all_labels)
    label_counts = np.sum(all_labels, axis=0)
    return label_counts

import torchvision.transforms as T

augmentation_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
])

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

    print(f"Augmentation completed. Label counts after augmentation: {label_counts}")
    return augmented_data, augmented_labels

def verify_train_dataset(dataset):

    print(f"Dataset type: {type(dataset)}")

    try:
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error accessing dataset size: {e}")
        return

    for idx in range(min(5, len(dataset))):
        try:
            sample = dataset[idx]
            print(f"Sample {idx}:")
            print(f"  Type: {type(sample)}")
            if isinstance(sample, tuple) and len(sample) == 2:
                img, label = sample
                print(f"  Image type: {type(img)}, Label type: {type(label)}")
                if isinstance(img, torch.Tensor):
                    print(f"  Image shape: {img.shape}")
                elif isinstance(img, np.ndarray):
                    print(f"  Image shape: {img.shape}")
                else:
                    print(f"  Unexpected image type: {type(img)}")
                if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
                    print(f"  Label shape: {label.shape}")
                else:
                    print(f"  Label: {label}")
            else:
                print("  Sample is not a tuple or does not have two elements.")
        except Exception as e:
            print(f"Error accessing sample {idx}: {e}")


def find_optimal_thresholds(y_true, y_probs):

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)    

    try:
        if y_true.ndim == 1:
            num_classes = len(np.unique(y_true))
            y_true = np.eye(num_classes)[y_true]
            print(f"y_true converted to one-hot encoding with shape {y_true.shape}.")
        
        if y_probs.ndim == 1:
            y_probs = y_probs.reshape(-1, y_true.shape[1])
            print(f"y_probs reshaped to match y_true with shape {y_probs.shape}.")

        if y_true.shape != y_probs.shape:
            raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but y_probs has shape {y_probs.shape}.")

        optimal_thresholds = []
        for i in range(y_probs.shape[1]):
            try:
                precision, recall, thresholds_for_class = precision_recall_curve(y_true[:, i], y_probs[:, i])
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_threshold = thresholds_for_class[np.argmax(f1)]
                optimal_thresholds.append(best_threshold)
            except ValueError:
                print(f"Error calculating threshold for class {i}. Defaulting to 0.5.")
                optimal_thresholds.append(0.5)
        
        return np.array(optimal_thresholds)

    except Exception as e:
        print(f"Error in find_optimal_thresholds: {e}")
        print("Ensure y_true and y_probs are in multilabel indicator format with consistent shapes.")
        raise



images_dir = './coco_subset_10Kv2/images'
annotations_path = './coco_subset_10Kv2/annotations/coco_subset_annotations.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),\
    transforms.ToTensor()
])

full_dataset = COCODataset(img_dir=images_dir, annotation_path=annotations_path, transform=transform)

label_counts = count_labels(full_dataset)
print("Label counts before augmentation:")
for idx, count in enumerate(label_counts):
    print(f"Label {idx}: {count} samples")

min_samples = 50
underrepresented_labels = np.where(label_counts < min_samples)[0]
print("Underrepresented labels:", underrepresented_labels)

augmented_features, augmented_labels = augment_dataset(full_dataset, underrepresented_labels, min_samples=50, transforms=augmentation_transforms)
print(f"Augmented dataset size: {len(augmented_features)}")

augmented_features = np.array(augmented_features)
augmented_labels = np.array(augmented_labels, dtype=np.float32)
augmented_features = torch.tensor(augmented_features, dtype=torch.float32)
augmented_labels = torch.tensor(augmented_labels, dtype=torch.float32)
augmented_dataset = TensorDataset(augmented_features, augmented_labels)
original_features = torch.stack([img for img, _ in full_dataset])
original_labels = torch.stack([label for _, label in full_dataset])
original_dataset = TensorDataset(original_features, original_labels)

full_dataset = ConcatDataset([original_dataset, augmented_dataset])

label_counts = count_labels(full_dataset)
label_counts_after = np.sum(augmented_labels.numpy(), axis=0)
print("Label counts after augmentation:")
for idx, count in enumerate(label_counts):
    print(f"Label {idx}: {count} samples")

train_dataset, val_dataset, test_dataset = stratified_train_val_test_split(
    full_dataset, train_size=0.7, val_size=0.15, test_size=0.15, min_per_label=10
)

#verify_train_dataset(train_dataset)
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")


test_labels = np.array([label.numpy() for _, label in test_dataset])

labeled_data_sizes = [500, 1000, 1500, 2000]

for num_labeled in labeled_data_sizes:
    print(f"\nProcessing with labeled data size: {num_labeled}")

    labeled_indices, unlabeled_indices = stratified_split(train_dataset, num_labeled=num_labeled)

    print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")

    def display_labeled_classes_after(split_name, indices, dataset):
        label_counts = Counter()
        for idx in indices:
            _, label = dataset[idx]
            label_counts.update(label.numpy().nonzero()[0])  # Count nonzero labels (classes)

        print(f"{split_name} Classes Present:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

    display_labeled_classes_after("Labeled Dataset", labeled_indices, train_dataset)

    unique_classes_after_split = set()
    for idx in labeled_indices:
        _, label = train_dataset[idx]
        unique_classes_after_split.update(label.numpy().nonzero()[0])

    print("Unique classes after stratified split:", sorted(unique_classes_after_split))

    device = 'cpu'
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 16)
    model = model.to(device)

    feature_extractor = FeatureExtractor(model).to(device)

    print("Extracting features using SIFT + BoVW...")
    sift_descriptors = extract_sift_features(train_dataset)

    valid_descriptor_count = sum(1 for desc in sift_descriptors if desc is not None)
    print(f"Number of valid descriptors: {valid_descriptor_count} out of {len(sift_descriptors)} images.")

    kmeans = create_visual_vocabulary(sift_descriptors, num_words=1000)
    train_histograms = generate_bow_histograms(train_dataset, kmeans)
    val_histograms = generate_bow_histograms(val_dataset, kmeans)
    test_histograms = generate_bow_histograms(test_dataset, kmeans)

    print("Extracting features using HOG...")
    hog_features_train, hog_labels_train = extract_hog_features(train_dataset)
    hog_features_val, hog_labels_val = extract_hog_features(val_dataset)
    hog_features_test, hog_labels_test = extract_hog_features(test_dataset)

    print("Extracting features using LBP...")
    lbp_features_train, lbp_labels_train = extract_lbp_features(train_dataset)
    lbp_features_val, lbp_labels_val = extract_lbp_features(val_dataset)
    lbp_features_test, lbp_labels_test = extract_lbp_features(test_dataset)

    train_labels = np.array([label.numpy() for _, label in train_dataset])

    if train_labels.ndim == 2 and train_labels.shape[1] > 1:
        print("Labels already in binary format.")
    else:
        print("Converting labels to binary format...")
        train_labels = np.eye(16)[train_labels]

    valid_indices = []
    train_histograms_list = []
    sift = cv2.SIFT_create()

    for idx, (img, _) in enumerate(train_dataset):
        try:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if img is None or img.size == 0:
                print(f"Warning: Image {idx} is empty or invalid, skipping.")
                continue

            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is None:
                print(f"Warning: No descriptors found for image {idx}, skipping.")
                continue

            histogram = np.zeros(len(kmeans.cluster_centers_))
            clusters = kmeans.predict(descriptors)
            for cluster in clusters:
                histogram[cluster] += 1

            train_histograms_list.append(histogram)
            valid_indices.append(idx)

        except Exception as e:
            print(f"Error processing image {idx}: {e}, skipping.")

    train_histograms = np.array(train_histograms_list)

    print(f"Generated {len(valid_indices)} valid histograms out of {len(train_dataset)} images.")

    train_labels = np.array([label.numpy() for _, label in train_dataset])
    unique_classes_train = np.unique(train_labels.argmax(axis=1))
    print("Unique classes in train_labels:", unique_classes_train)

    print(f"Shape of train_labels: {np.shape(train_labels)}")

    descriptor_class_counts = Counter()
    for idx in valid_indices:
        _, label = train_dataset[idx]
        descriptor_class_counts.update(label.numpy().nonzero()[0])

    print("Valid descriptors per class:", dict(descriptor_class_counts))

    valid_classes = set(train_labels[valid_indices].argmax(axis=1))
    print(f"Classes in valid_indices: {sorted(valid_classes)}")
    valid_labeled_indices = [valid_indices.index(idx) for idx in labeled_indices if idx in valid_indices]

    labeled_classes = set(train_labels[valid_indices[idx]].argmax() for idx in valid_labeled_indices)
    print(f"Classes in valid_labeled_indices: {sorted(labeled_classes)}")

    class_counts = Counter(train_labels[valid_indices[idx]].argmax() for idx in valid_labeled_indices)
    print("Class counts in valid_labeled_indices:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} samples")

    unique_classes_after_mapping = set()
    for idx in valid_labeled_indices:
        _, label = train_dataset[valid_indices[idx]]
        unique_classes_after_mapping.update(label.numpy().nonzero()[0])

    print("Unique classes after mapping to valid indices:", sorted(unique_classes_after_mapping))

    labeled_histograms = np.array([train_histograms[idx] for idx in valid_labeled_indices])
    labeled_labels = np.array([train_labels[valid_indices[idx]] for idx in valid_labeled_indices])

    unique_classes = set(range(train_labels.shape[1]))
    present_classes = set(train_labels[valid_indices[idx]].argmax() for idx in valid_labeled_indices)
    missing_classes = unique_classes - present_classes

    if missing_classes:
        print(f"Missing classes in labeled data: {missing_classes}")
        for missing_class in missing_classes:
            class_indices = [idx for idx in valid_indices if train_labels[idx].argmax() == missing_class]
            if class_indices:
                valid_labeled_indices.append(class_indices[0])
            else:
                print(f"Warning: No samples found for missing class {missing_class}.")

    labeled_histograms = np.array([train_histograms[idx] for idx in valid_labeled_indices])
    labeled_labels = np.array([train_labels[valid_indices[idx]] for idx in valid_labeled_indices])
    labeled_hog_features = np.array([hog_features_train[valid_indices[idx]] for idx in valid_labeled_indices])
    labeled_lbp_features = np.array([lbp_features_train[valid_indices[idx]] for idx in valid_labeled_indices])
    labeled_hog_labels = np.array([hog_labels_train[valid_indices[idx]] for idx in valid_labeled_indices])
    labeled_lbp_labels = np.array([lbp_labels_train[valid_indices[idx]] for idx in valid_labeled_indices])

    unique_classes_label = np.unique(labeled_labels.argmax(axis=1))
    print("Unique classes in labeled_labels:", unique_classes_label)

    #print(f"Labeled Indices: {labeled_indices[:5]}")  # Print a sample of labeled_indices
    #print(f"Type of labeled_indices: {type(labeled_indices)}")
    #print(f"First label shape: {train_labels[0].shape if len(train_labels) > 0 else 'No labels'}")

    print("Balancing BoVW histograms with custom MLSMOTE...")
    train_histograms_balanced, train_labels_balanced = ml_smote(labeled_histograms, labeled_labels, k=5)

    print("Balancing HOG features with custom MLSMOTE...")
    hog_features_train_balanced, hog_labels_train_balanced = ml_smote(labeled_hog_features, labeled_hog_labels, k=5)

    print("Balancing LBP features with custom MLSMOTE...")
    lbp_features_train_balanced, lbp_labels_train_balanced = ml_smote(labeled_lbp_features, labeled_lbp_labels, k=5)

    print("Calculating class weights...")
    unique_classes = np.unique(train_labels_balanced.argmax(axis=1))
    print("Unique classes in train_labels_balanced:", unique_classes)
    unique_classes_hog = np.unique(hog_labels_train_balanced.argmax(axis=1))
    print("Unique classes in hog_labels_train_balanced:", unique_classes_hog)
    unique_classes_lbp = np.unique(lbp_labels_train_balanced.argmax(axis=1))
    print("Unique classes in lbp_labels_train_balanced:", unique_classes_lbp)

    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=train_labels_balanced.argmax(axis=1)
    )

    weights_dict = dict(enumerate(class_weights))
    print("Weights dictionary before update:", weights_dict)

    for cls in range(train_labels_balanced.shape[1]):
        if cls not in weights_dict:
            weights_dict[cls] = 1.0

    print("Weights dictionary after update:", weights_dict)

    batch_size = 32

    class_supports = [0] * 16
    for _, label in train_dataset:
        class_supports = [a + b for a, b in zip(class_supports, label.numpy())]
    class_weights = torch.tensor([2.0 / support if support > 0 else 5.0 for support in class_supports], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #print(f"Labeled Indices: {labeled_indices[:5]}")
    #print(f"Type of labeled_indices: {type(labeled_indices)}")
    #print(f"Type of train_histograms: {type(train_histograms)}")
    #print(f"Shape of train_histograms: {np.shape(train_histograms)}")
    #print(f"Shape of hog_features_train: {np.shape(hog_features_train)}")
    #print(f"Shape of lbp_features_train: {np.shape(lbp_features_train)}")

    train_sample_weights = compute_sample_weight(class_weight=weights_dict, y=train_labels_balanced.argmax(axis=1))

    print("Training Random Forest on BoVW features with adjusted weights...")
    rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_bow.fit(train_histograms_balanced, train_labels_balanced, sample_weight=train_sample_weights)

    print("Training SVM on BoVW features...")
    svm_base = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    svm_bow = MultiOutputClassifier(svm_base)
    svm_bow.fit(train_histograms_balanced, train_labels_balanced)

    print("Training XGBoost on BoVW features...")
    unique_classes = np.unique(train_labels_balanced.argmax(axis=1))
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    weights_dict_xgb = {class_mapping[old_class]: class_weights[old_class] for old_class in unique_classes}
    print("Class Mapping:", class_mapping)
    print("Updated Weights for XGBoost:", weights_dict_xgb)
    sample_weight = np.array([weights_dict_xgb[np.argmax(label)] for label in train_labels_balanced])


    xgb_base = XGBClassifier(eval_metric='mlogloss')
    xgb_bow = MultiOutputClassifier(xgb_base)
    xgb_bow.fit(train_histograms_balanced, train_labels_balanced, sample_weight=sample_weight)

    print("Training Random Forest on HOG features...")
    hog_sample_weights = compute_sample_weight(class_weight=weights_dict, y=hog_labels_train_balanced.argmax(axis=1))
    rf_hog = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_hog.fit(hog_features_train_balanced, hog_labels_train_balanced, sample_weight=hog_sample_weights)

    print("Training Random Forest on LBP features...")
    lbp_sample_weights = compute_sample_weight(class_weight=weights_dict, y=lbp_labels_train_balanced.argmax(axis=1))
    rf_lbp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_lbp.fit(lbp_features_train_balanced, lbp_labels_train_balanced, sample_weight=lbp_sample_weights)
   
    for warmup_iteration in range(3):
        print(f"Warm-Up Iteration {warmup_iteration + 1}")
        sampler = create_sampler(train_dataset, labeled_indices, class_weights)
        labeled_loader = DataLoader(Subset(train_dataset, labeled_indices), batch_size=batch_size, sampler=sampler)
        train_model(model, labeled_loader, criterion, optimizer, device)
   
    performance_metrics = []

    for iteration in range(22):
        print(f"Active Learning Iteration {iteration + 1}")

        class_supports = [0] * 16
        for idx in labeled_indices:
            if idx < len(train_labels_balanced):
                class_supports = [a + b for a, b in zip(class_supports, train_labels_balanced[idx])]

        class_weights = torch.tensor([2.0 / support if support > 0 else 5.0 for support in class_supports], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        sampler = create_sampler(train_dataset, valid_labeled_indices, class_weights)
        labeled_loader = DataLoader(Subset(train_dataset, valid_labeled_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_model(model, labeled_loader, criterion, optimizer, device)
       
        val_predictions, val_labels = evaluate_model(model, val_loader, device)
        thresholds = find_optimal_threshold(val_labels, val_predictions)
        
        val_binary_predictions = (val_predictions > np.array(thresholds)).astype(int)
        print(classification_report(val_labels, val_binary_predictions, zero_division=0))
       
        print("Training classifiers on updated labeled set...")
        valid_labeled_indices = [idx for idx in labeled_indices if idx < len(train_labels_balanced)]

        print("Classes in valid_labeled_indices:", np.unique(train_labels_balanced[valid_labeled_indices].argmax(axis=1)))

        hog_sample_weights = compute_sample_weight(
            class_weight=weights_dict, 
            y=hog_labels_train_balanced[valid_labeled_indices].argmax(axis=1)
        )
        lbp_sample_weights = compute_sample_weight(
            class_weight=weights_dict, 
            y=lbp_labels_train_balanced[valid_labeled_indices].argmax(axis=1)
        )
        train_sample_weights = compute_sample_weight(
            class_weight=weights_dict, 
            y=train_labels_balanced[valid_labeled_indices].argmax(axis=1)
        )

        rf_bow.fit(
            train_histograms_balanced[valid_labeled_indices], 
            train_labels_balanced[valid_labeled_indices], 
            sample_weight=train_sample_weights
        )
        svm_bow.fit(
            train_histograms_balanced[valid_labeled_indices], 
            train_labels_balanced[valid_labeled_indices]
        )

        train_histograms_balanced_xgb = train_histograms_balanced.copy()
        train_labels_balanced_xgb = train_labels_balanced.copy()
        valid_labeled_indices_xgb = valid_labeled_indices.copy()

        if missing_classes:
            print(f"Warning: Missing classes in labeled set: {missing_classes}")

            for missing_class in missing_classes:
                synthetic_feature = train_histograms_balanced.mean(axis=0)
                synthetic_label = np.zeros_like(train_labels_balanced[0])
                synthetic_label[missing_class] = 1  # Set the missing class label

                train_histograms_balanced_xgb = np.vstack([train_histograms_balanced_xgb, synthetic_feature])
                train_labels_balanced_xgb = np.vstack([train_labels_balanced_xgb, synthetic_label])
                valid_labeled_indices_xgb.append(len(train_labels_balanced_xgb) - 1)

        unique_classes_present = np.unique(train_labels_balanced_xgb.argmax(axis=1))
        print(f"Unique classes after adding synthetic samples: {unique_classes_present}")

        train_labels_mapped_xgb = np.array([
            [1 if cls in np.nonzero(label)[0] else 0 for cls in range(train_labels_balanced_xgb.shape[1])]
            for label in train_labels_balanced_xgb
        ])

        unique_classes, class_counts = np.unique(train_labels_balanced_xgb.argmax(axis=1), return_counts=True)
        class_weights_xgb = {cls: len(train_labels_balanced_xgb) / count for cls, count in zip(unique_classes, class_counts)}

        sample_weights_xgb = np.array([
            class_weights_xgb[label] for label in train_labels_balanced_xgb.argmax(axis=1)
        ])

        xgb_bow.fit(
            train_histograms_balanced_xgb[valid_labeled_indices_xgb],
            train_labels_mapped_xgb[valid_labeled_indices_xgb],
            sample_weight=sample_weights_xgb[valid_labeled_indices_xgb]
        )

        rf_hog.fit(
            hog_features_train_balanced[valid_labeled_indices], 
            hog_labels_train_balanced[valid_labeled_indices], 
            sample_weight=hog_sample_weights
        )
        rf_lbp.fit(
            lbp_features_train_balanced[valid_labeled_indices], 
            lbp_labels_train_balanced[valid_labeled_indices], 
            sample_weight=lbp_sample_weights
        )

        print("Evaluating on validation set...")
        val_predictions_rf = rf_bow.predict(val_histograms)
        val_predictions_svm = svm_bow.predict(val_histograms)
        val_predictions_xgb = xgb_bow.predict_proba(val_histograms)
        val_predictions_rfh = rf_hog.predict(hog_features_val)
        val_predictions_rfl = rf_lbp.predict(lbp_features_val)

        print("Random Forest on BOW on Validation Set:")
        print(classification_report(val_labels, val_predictions_rf, zero_division=0))

        print("SVM on Validation Set:")
        print(classification_report(val_labels, val_predictions_svm, zero_division=0))

        print("XGBoost on Validation Set:")
        print(type(val_predictions_xgb))
        print(len(val_predictions_xgb))
        print(val_predictions_xgb[0].shape)
        val_predictions_xgb = np.array([
            class_probs[:, 1] if class_probs.ndim == 2 else class_probs
            for class_probs in val_predictions_xgb
        ]).T
        print("Processed val_predictions_xgb shape:", val_predictions_xgb.shape)
        try:
            optimal_thresholds = find_optimal_thresholds(val_labels, val_predictions_xgb)
            val_predictions_xgb_multilabel = (val_predictions_xgb > np.array(optimal_thresholds)).astype(int)
            print("Using optimal thresholds for classification report:")
        except Exception as e:
            print(f"Error finding optimal thresholds: {e}. Falling back to default threshold of 0.5.")
            val_predictions_xgb_multilabel = (val_predictions_xgb > 0.5).astype(int)
        print(classification_report(val_labels, val_predictions_xgb_multilabel, zero_division=0))

        print("Random Forest on HOG on Validation Set:")
        print(classification_report(hog_labels_val, val_predictions_rfh, zero_division=0))

        print("Random Forest on LBP on Validation Set:")
        print(classification_report(lbp_labels_val, val_predictions_rfl, zero_division=0))
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_predictions, test_ground_truth = evaluate_model(model, test_loader, device)
        thresholds = find_optimal_threshold(test_ground_truth, test_predictions)
        test_binary_predictions = (test_predictions > np.array(thresholds)).astype(int)
        
        print("Testing Random Forest on BoVW:")
        test_predictions_rf = rf_bow.predict(test_histograms)
        
        print("Testing SVM on BoVW:")
        test_predictions_svm = svm_bow.predict(test_histograms)
        
        print("Testing XGBoost on BoVW:")
        test_predictions_xgb = xgb_bow.predict_proba(test_histograms)
        test_predictions_xgb = np.array([
            class_probs[:, 1] if class_probs.ndim == 2 else class_probs
            for class_probs in test_predictions_xgb
        ]).T
        try:
            optimal_thresholds = find_optimal_thresholds(test_labels, test_predictions_xgb)
            test_predictions_xgb_multilabel = (test_predictions_xgb > np.array(optimal_thresholds)).astype(int)
            print("Using optimal thresholds for classification report:")
        except Exception as e:
            print(f"Error finding optimal thresholds: {e}. Falling back to default threshold of 0.5.")
            test_predictions_xgb_multilabel = (test_predictions_xgb > 0.5).astype(int)

        print("Random Forest on HOG on Validation Set:")
        test_predictions_rfh = rf_hog.predict(hog_features_test)

        print("Random Forest on LBP on Validation Set:")
        test_predictions_rfl = rf_lbp.predict(lbp_features_test)

        classifiers = {
            'xgb_bow': test_predictions_xgb_multilabel,
            'rf_bow': test_predictions_rf,
            'svm_bow': test_predictions_svm,
            'rf_hog': test_predictions_rfh,
            'rf_lbp': test_predictions_rfl,
            'resnet18': test_binary_predictions
        }

        csv_file_path = 'active_learning_metrics_detailed.csv'
        file_exists = os.path.exists(csv_file_path)
        all_metrics = []

        for name, predictions in classifiers.items():
            labeled_data = len(valid_labeled_indices)
            #specific_labels = hog_labels_val if name == 'rf_hog' else (lbp_labels_val if name == 'rf_lbp' else val_labels)
            specific_labels = hog_labels_test if name == 'rf_hog' else (
                lbp_labels_test if name == 'rf_lbp' else test_labels
            )
            try:
                val_metrics = classification_report(specific_labels, predictions, output_dict=True, zero_division=0)

                for avg_type in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
                    metrics_data = {
                        'iteration': iteration,
                        'classifier': name,
                        'average_type': avg_type,
                        'precision': val_metrics[avg_type]['precision'],
                        'recall': val_metrics[avg_type]['recall'],
                        'f1-score': val_metrics[avg_type]['f1-score'],
                        'labeled_sample_size': labeled_data
                    }
                    all_metrics.append(metrics_data)

            except Exception as e:
                print(f"Error computing metrics for classifier {name}: {e}")

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(
            csv_file_path,
            mode='a', 
            header=not file_exists, 
            index=False
        )

        unlabeled_loader = DataLoader(Subset(train_dataset, unlabeled_indices), batch_size=batch_size, shuffle=False)
        unlabeled_features = extract_features(feature_extractor, unlabeled_loader, device)
    
        valid_unlabeled_indices = [
            idx for idx in unlabeled_indices 
            if idx < train_histograms.shape[0] and idx not in valid_labeled_indices
        ]

        unlabeled_histograms = train_histograms[valid_unlabeled_indices]
        hog_features_unlabeled = hog_features_train[valid_unlabeled_indices]
        lbp_features_unlabeled = lbp_features_train[valid_unlabeled_indices]
        model_predictions = []

        predictions_rf_bow = rf_bow.predict_proba(unlabeled_histograms)
        model_predictions.append(predictions_rf_bow)

        predictions_svm_bow = svm_bow.predict_proba(unlabeled_histograms)
        model_predictions.append(predictions_svm_bow)
        """
        predictions_xgb = xgb_bow.predict_proba(unlabeled_histograms)
        predictions_xgb = np.array([
            class_probs[:, 1] if class_probs.ndim == 2 else class_probs  # Handle binary vs multiclass
            for class_probs in predictions_xgb
        ]).T
        if predictions_xgb.ndim == 1:
            predictions_xgb = predictions_xgb.reshape(-1, 1)  # Ensure 2D shape
        model_predictions.append(predictions_xgb)
        """

        predictions_rf_hog = rf_hog.predict_proba(hog_features_unlabeled)
        model_predictions.append(predictions_rf_hog)
        predictions_rf_lbp = rf_lbp.predict_proba(lbp_features_unlabeled)
        model_predictions.append(predictions_rf_lbp)

        #predictions_resnet, _ = evaluate_model(model, unlabeled_loader, device)
        #model_predictions.append(predictions_resnet)

        model_predictions = [
            np.array(pred) if isinstance(pred, list) else pred
            for pred in model_predictions
        ]

        min_samples = min(pred.shape[1] for pred in model_predictions)

        aligned_predictions = []
        for pred in model_predictions:
            if pred.shape[1] > min_samples:
                truncated_pred = pred[:, :min_samples, :]
                aligned_predictions.append(truncated_pred)
            else:
                aligned_predictions.append(pred)

        aligned_predictions = np.array(aligned_predictions)
        aggregated_predictions = np.median(aligned_predictions, axis=0)  # Shape: (16, min_samples, 2)
        probabilities = np.array([array[:, 1] for array in aggregated_predictions]).T  # Transpose to shape (num_samples, num_classes)
        print("Reshaped probabilities:\n", probabilities)
        print("Shape:", probabilities.shape)

        target_new_indices = 250

        uncertainty_indices = entropy_sampling(probabilities, num_samples=100, unlabeled_indices=valid_unlabeled_indices)
        diversity_indices = diversity_sampling(unlabeled_features, num_samples=100)
        class_aware_indices = class_aware_sampling(
            [train_dataset[idx][1].numpy() for idx in valid_unlabeled_indices],
            num_samples=50
        )

        combined_indices = set(uncertainty_indices) | set(diversity_indices) | set(class_aware_indices)
        new_indices = [unlabeled_indices[int(idx)] for idx in combined_indices if int(idx) < len(unlabeled_indices)]

        if len(new_indices) < target_new_indices:
            remaining_needed = target_new_indices - len(new_indices)
            additional_indices = np.random.choice(
                [idx for idx in unlabeled_indices if idx not in new_indices],
                size=remaining_needed,
                replace=False
            )
            new_indices.extend(additional_indices)

        labeled_indices.extend(new_indices)
        unlabeled_indices = [idx for idx in valid_unlabeled_indices if idx not in new_indices]

        print(f"Added {len(new_indices)} new indices to the labeled set.")

    """
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('active_learning_metrics_detailed.csv', index=False)
    """
    #torch.save(model.state_dict(), "active_learning_model.pth")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_predictions, test_ground_truth = evaluate_model(model, test_loader, device)
    print(classification_report(test_ground_truth, (test_predictions > 0.5).astype(int), zero_division=0))

    print("Testing Random Forest on BoVW:")
    test_predictions_rf = rf_bow.predict(test_histograms)
    print(classification_report(test_labels, test_predictions_rf, zero_division=0))

    print("Testing SVM on BoVW:")
    test_predictions_svm = svm_bow.predict(test_histograms)
    print(classification_report(test_labels, test_predictions_svm, zero_division=0))

    print("Testing XGBoost on BoVW:")
    test_predictions_xgb = xgb_bow.predict(test_histograms)
    print(classification_report(test_labels, test_predictions_xgb, zero_division=0))

    print("Random Forest on HOG on Validation Set:")
    test_predictions_rfh = rf_hog.predict(hog_features_test)
    print(classification_report(test_labels, test_predictions_rfh, zero_division=0))

    print("Random Forest on LBP on Validation Set:")
    test_predictions_rfl = rf_lbp.predict(lbp_features_test)
    print(classification_report(test_labels, test_predictions_rfl, zero_division=0))

    #test_binary_predictions = (test_predictions > np.array(thresholds)).astype(int)
    #print(classification_report(test_ground_truth, test_binary_predictions, zero_division=0))
