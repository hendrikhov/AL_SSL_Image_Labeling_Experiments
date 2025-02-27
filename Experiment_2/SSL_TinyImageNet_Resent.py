import torch
import torch.nn as nn
import numpy as np
import csv
import torchvision.transforms as T
import random
import os
import pickle
import json
import time
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, TensorDataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, precision_recall_curve, confusion_matrix, jaccard_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from memory_profiler import profile

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
def stratified_train_val_test_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, min_per_label=30):
    
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
    print(f"Oversampled dataset size: {len(combined_dataset)}")
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
            label_counts.update(label.numpy().nonzero()[0])

        print(f"{split_name} Classes Present:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} samples")
        print()

@profile
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

@profile
def ml_smote(X, y, k=5):

    X = np.array(X)
    y = np.array(y, dtype=np.int32)

    if len(X) != len(y):
        raise ValueError("Features and labels must have the same number of samples.")

    X_resampled = list(X)
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
            synthetic_label = y[idx] | y[minority_indices[neighbor_idx]]
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
class SemiSupervisedTrainer:
    def __init__(self, model, device, num_classes=20):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def compute_validation_loss(self, val_loader):

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        average_loss = total_loss / len(val_loader)
        return average_loss

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
def evaluate_resnet(
    model, 
    val_loader, 
    test_loader, 
    device, 
    labeled_dataset_size, 
    pseudo_labeled_dataset_size, 
    results_csv_file="results.csv", 
    training_time=None, 
    validation_time=None
):
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
    hamming = hamming_loss(test_labels, binary_predictions)
    accuracy = accuracy_score(test_labels, binary_predictions)
    jaccard = jaccard_score(test_labels, binary_predictions, average="samples")
    matthews = matthews_corrcoef(test_labels.ravel(), binary_predictions.ravel())
    report = classification_report(test_labels, binary_predictions, output_dict=True, zero_division=0)

    try:
        cm = confusion_matrix(test_labels.ravel(), binary_predictions.ravel())
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)
    except ValueError:
        tn, fp, fn, tp = None, None, None, None
        print("Confusion matrix not applicable for multilabel.")

    micro_avg = report["micro avg"]
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]
    samples_avg = report.get("samples avg", {"precision": 0, "recall": 0, "f1-score": 0})

    print("Classification Report:")
    print(classification_report(test_labels, binary_predictions, zero_division=0))
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")
    print(f"Matthews Corrcoef: {matthews:.4f}")

    result_row = [
        "ResNet", "Evaluation", labeled_dataset_size,
        hamming, accuracy, jaccard, matthews,
        micro_avg["precision"], micro_avg["recall"], micro_avg["f1-score"],
        macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"],
        weighted_avg["precision"], weighted_avg["recall"], weighted_avg["f1-score"],
        samples_avg["precision"], samples_avg["recall"], samples_avg["f1-score"],
        training_time, validation_time,
        labeled_dataset_size, pseudo_labeled_dataset_size,
        tp, fp, tn, fn
    ]

    if not os.path.exists(results_csv_file):
        with open(results_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Strategy", "Model", "Feature Size",
                "Hamming Loss", "Accuracy", "Jaccard Score", "Matthews Corrcoef",
                "Precision (Micro)", "Recall (Micro)", "F1-Score (Micro)",
                "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)",
                "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)",
                "Precision (Samples)", "Recall (Samples)", "F1-Score (Samples)",
                "Training Time (s)", "Validation Time (s)",
                "Labeled Samples", "Pseudo-Labeled Samples",
                "True Positives", "False Positives", "True Negatives", "False Negatives"
            ])

    with open(results_csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

    print("Evaluation results saved to CSV.")


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
augmented_dataset_path = "./ImageNet2_cache/augmented_dataset.pkl"
train_val_test_splits_path = "./ImageNet2_cache/train_val_test_splits.pkl"
features_cache_dir = "./ImageNet2_cache/features/"
label_cache_dir = "./cache_ImageNet2/labels/"
feature_cache_dir = "./cache_ImageNet2/features/"
split_cache_dir = "./cache_ImageNet2/splits"

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

    progress_file = "ssl_experiment_progress_ImageNet_2000.csv"

    if os.path.exists(progress_file):
        completed_experiments = pd.read_csv(progress_file)
    else:
        completed_experiments = pd.DataFrame(columns=["Labeled_Samples", "Feature", "Model", "Strategy"])


    labeled_data_sizes = [500, 1000, 2000]
    all_results = []
    for num_labeled in labeled_data_sizes:

        print(f"\nProcessing with labeled data size: {num_labeled}")

        labeled_indices, unlabeled_indices, labeled_features_balanced, labeled_labels_balanced = save_or_load_split(
            train_dataset, "train", num_labeled, stratified_split, ml_smote
        )
   
        labeled_loader = DataLoader(Subset(train_dataset, labeled_indices), batch_size=32, shuffle=True)
        unlabeled_loader = DataLoader(Subset(train_dataset, unlabeled_indices), batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        device = 'cpu'
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 20)
        model.to(device)

        feature_extractor = FeatureExtractor(model).to(device)
        trainer = SemiSupervisedTrainer(model, device)

        best_val_loss = float('inf')
        no_improvement_epochs = 0
        labeled_dataset_size = len(labeled_indices)
        unlabeled_dataset_size = len(unlabeled_indices)
        pseudo_labeled_dataset_size = 0
        epoch_start_time = time.time()

        for epoch in range(max_epochs):

            print(f"Epoch {epoch + 1}/{max_epochs}")            
            trainer.train_supervised(labeled_loader)
            threshold = dynamic_thresholding(epoch, max_epochs)
            pseudo_dataset = trainer.pseudo_labeling(unlabeled_loader)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=32, shuffle=True)
            pseudo_labeled_dataset_size = len(pseudo_dataset)
            trainer.train_supervised(pseudo_loader)

            consistency_loss = trainer.consistency_regularization(unlabeled_loader, augmentation_transforms)
            print(f"Consistency Loss: {consistency_loss:.4f}")

            entropy_loss = trainer.entropy_minimization(unlabeled_loader)
            print(f"Entropy Loss: {entropy_loss:.4f}")

            val_start_time = time.time()
            val_loss = trainer.compute_validation_loss(val_loader)  # Replace with actual validation set
            print(f"Validation Loss: {val_loss:.4f}")

            epoch_end_time = time.time()
            training_time = epoch_end_time - epoch_start_time
            print(f"Epoch Training Time: {training_time:.2f} seconds")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                print("Validation loss improved. Saving model...")
                torch.save(model.state_dict(), "best_model.pth")
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epoch(s).")

            if no_improvement_epochs >= patience:
                print("Early stopping triggered.")
                break

        print("\nTesting the best ResNEt SSL model...")
        model.load_state_dict(torch.load("best_model.pth"))

        val_end_time = time.time()

        validation_time = val_end_time - val_start_time
        print(f"Validation Time: {validation_time:.2f} seconds")
        evaluate_resnet(
            model,
            val_loader,
            test_loader,
            device,
            labeled_dataset_size=labeled_dataset_size,
            pseudo_labeled_dataset_size=pseudo_labeled_dataset_size,
            results_csv_file="ssl_resnet_results_ImageNet.csv",
            training_time=training_time,
            validation_time=validation_time
        )
