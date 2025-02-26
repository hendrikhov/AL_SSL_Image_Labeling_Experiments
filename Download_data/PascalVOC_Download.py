import os
import requests
import tarfile
import random
import shutil
import xml.etree.ElementTree as ET

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {dest_path}")

def download_and_extract_pascal_voc(destination_dir):
    pascal_voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    os.makedirs(destination_dir, exist_ok=True)
    images_dir = os.path.join(destination_dir, 'JPEGImages')
    annotations_dir = os.path.join(destination_dir, 'Annotations')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    tar_path = os.path.join(destination_dir, "VOCtrainval_11-May-2012.tar")
    print("Downloading Pascal VOC...")
    download_file(pascal_voc_url, tar_path)

    print("Extracting Pascal VOC...")
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(destination_dir)

    print("Extraction complete.")
    
    extracted_folder = os.path.join(destination_dir, 'VOCdevkit', 'VOC2012')
    
    for img_file in os.listdir(os.path.join(extracted_folder, 'JPEGImages')):
        shutil.move(os.path.join(extracted_folder, 'JPEGImages', img_file), os.path.join(images_dir, img_file))

    for ann_file in os.listdir(os.path.join(extracted_folder, 'Annotations')):
        shutil.move(os.path.join(extracted_folder, 'Annotations', ann_file), os.path.join(annotations_dir, ann_file))

    shutil.rmtree(os.path.join(destination_dir, 'VOCdevkit'))

    print("Pascal VOC dataset is organized. Images and annotations saved.")
    select_labeled_images(images_dir, annotations_dir, 10000)

def select_labeled_images(images_dir, annotations_dir, subset_size):
    all_annotations = os.listdir(annotations_dir)
    labeled_images = []

    for ann_file in all_annotations:
        if ann_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, ann_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text
            labeled_images.append(filename)

    if len(labeled_images) < subset_size:
        print("Not enough labeled images available. Selecting all labeled images.")
        selected_images = labeled_images
    else:
        selected_images = random.sample(labeled_images, subset_size)

    selected_images_dir = os.path.join(images_dir, 'Selected_Images')
    os.makedirs(selected_images_dir, exist_ok=True)

    print(f"Copying {len(selected_images)} labeled images...")
    for img_file in selected_images:
        shutil.copy(os.path.join(images_dir, img_file), selected_images_dir)

    print("Selected labeled images copied successfully.")

destination_dir = './pascal_voc'
download_and_extract_pascal_voc(destination_dir)
