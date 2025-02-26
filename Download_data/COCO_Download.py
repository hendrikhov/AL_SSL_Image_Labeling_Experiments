import os
import random
import requests
from pycocotools.coco import COCO
import json
import zipfile

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {dest_path}")

def download_coco_subset(annotation_url, images_dir, annotations_dir, subset_size=10000):
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_zip_path = os.path.join(annotations_dir, "annotations_trainval2017.zip")
    
    print("Downloading COCO annotations...")
    if not os.path.exists(annotation_zip_path):
        download_file(annotation_url, annotation_zip_path)

    with zipfile.ZipFile(annotation_zip_path, 'r') as zip_ref:
        zip_ref.extractall(annotations_dir)
    print("Extracted annotations.")

    coco = COCO(os.path.join(annotations_dir, 'annotations/instances_train2017.json'))

    print("Selecting random subset of images...")
    all_image_ids = coco.getImgIds()
    random_image_ids = random.sample(all_image_ids, subset_size)

    os.makedirs(images_dir, exist_ok=True)

    print("Downloading images...")
    for img_id in random_image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = os.path.join(images_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)
            print(f"Downloaded: {img_info['file_name']}")

    print("Saving subset annotations...")
    subset_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco.loadCats(coco.getCatIds())
    }

    for img_id in random_image_ids:
        img_info = coco.loadImgs(img_id)[0]
        subset_annotations["images"].append(img_info)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        subset_annotations["annotations"].extend(anns)

    subset_annotation_path = os.path.join(annotations_dir, 'coco_subset_10000_annotations.json')
    with open(subset_annotation_path, 'w') as f:
        json.dump(subset_annotations, f)

    print("Download complete. Images and annotations saved.")

annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
images_dir = "./coco_subset_10000/images"
annotations_dir = "./coco_subset_10000/annotations"
subset_size = 10000

download_coco_subset(annotation_url, images_dir, annotations_dir, subset_size)
