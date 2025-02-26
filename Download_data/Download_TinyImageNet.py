import os
import requests
import zipfile
import shutil
import json
import random

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {dest_path}")

 

def download_and_extract_tiny_imagenet(destination_dir):
    tiny_imagenet_url = 'https://image-net.org/data/tiny-imagenet-200.zip'
    os.makedirs(destination_dir, exist_ok=True)
    zip_path = os.path.join(destination_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        download_file(tiny_imagenet_url, zip_path)
    else:
        print("Dataset already downloaded.")

    extracted_dir = os.path.join(destination_dir, "tiny-imagenet-200")

    if not os.path.exists(extracted_dir):
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)

        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

    train_folder = os.path.join(extracted_dir, "train")
    validation_folder = os.path.join(extracted_dir, "val")
    subset_folder = os.path.join(destination_dir, "tiny-imagenet-subset")

    os.makedirs(subset_folder, exist_ok=True)

    val_annotations_file = os.path.join(validation_folder, "val_annotations.txt")
    val_label_mapping = {}
    with open(val_annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            val_label_mapping[parts[0]] = parts[1]

    all_images_by_label = {}

    for label_dir in os.listdir(train_folder):
        label_path = os.path.join(train_folder, label_dir, "images")
        if os.path.isdir(label_path):
            all_images_by_label[label_dir] = [os.path.join(label_path, img) for img in os.listdir(label_path)]

    for img_file, label in val_label_mapping.items():
        img_path = os.path.join(validation_folder, "images", img_file)
        if label not in all_images_by_label:
            all_images_by_label[label] = []
        all_images_by_label[label].append(img_path)

    selected_classes = random.sample(all_images_by_label.keys(), 20)
    selected_images = []
    label_map = {}

    for i, label in enumerate(selected_classes):
        images = all_images_by_label[label]
        selected_images.extend(images)
        label_map[label] = {
            "class_index": i,
            "class_name": label  # Label names are typically WordNet IDs (wnids)
        }

    selected_images = selected_images[:10000]

    for img_path in selected_images:
        label = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        label_dir = os.path.join(subset_folder, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(img_path, label_dir)

    label_json_path = os.path.join(subset_folder, "ImageNet_subset_labels.json")
    with open(label_json_path, "w") as json_file:
        json.dump(label_map, json_file, indent=4)

    print(f"Subset created with 20 classes and 10,000 images in {subset_folder}.")
    print(f"Label mapping saved to {label_json_path}.")

destination_dir = './tiny_imagenet'

download_and_extract_tiny_imagenet(destination_dir)