# AL_SSL_Image_Labeling_Experiments
A comprehensive implementation of Active Learning (AL) and Semi-Supervised Learning (SSL) strategies for automatic image labeling, optimized for datasets with varying complexity and resource constraints.

# Description:
This repository contains the full implementation of three experiments investigating the effectiveness of Active Learning (AL) and Semi-Supervised Learning (SSL) in automatic image labeling under computational and resource limitations. The experiments systematically compare classical machine learning models, feature extraction techniques, and hybrid AL-SSL approaches to improve label efficiency across different datasets.

The objective of this study is to validate wether classical methodes can be used instead of deepl learning methodes in resource constraints areas, while maintaining competitive classification performance. The experiments analyze how AL selects informative samples, how SSL refines pseudo-labeling strategies, and how combining AL and SSL optimally balances precision and recall.

# Datasets Used:
The experiments were conducted on three widely used datasets, each presenting unique challenges in terms of class balance, resolution, and label dependencies:
The code code for downloading the datasets can be found in the folder "Download_data".
## Microsoft COCO (Common Objects in Context)
- Multi-label dataset with severe class imbalance and high label co-occurrence.
- Varying image resolutions (480×640 to 1024×1024 pixels), requiring robust feature extraction.
- Selected 16 object categories for experimentation and 10.000 images
## PascalVOC 2012
- Multi-label dataset with more balanced class distributions.
- Consistent image size (~500×375 pixels) and structured object annotations.
- Includes 20 object categories and 11,530 images, making class representation per label lower than COCO.
## TinyImageNet
- Single-label classification dataset with highly similar categories.
- Fixed 64×64 pixel resolution, emphasizing the challenges of feature extraction.
- Balanced class distribution but high intra-class variability (e.g., different cat breeds in separate categories).
- Selected 20 object classes of 500 images each.

# Repository Contents:
## Experiment 1 (Active Learning - AL):
- Implementation of AL strategies, including uncertainty sampling and diversity-based selection.
- Performance evaluation of classical machine learning models trained on incrementally labeled datasets.
- Feature extraction methods: LBP, BoVW, HOG for dimensionality reduction.
- Identification of the optimal labled limit for active learning (iteratied over different starting batch sizes: 500, 1000, 1500, 2000)

## Experiment 2 (Semi-Supervised Learning - SSL):
- Implementation of pseudo-labeling, co-training, and label propagation strategies.
- Exploration of SSL performance across different dataset complexities and class distributions.
- Uses different initial labeled set sizes (500, 1000, 2000) to analyze SSL effectiveness.
- Analysis of SSL effectiveness based on initial labeled data proportions and feature representation.

## Experiment 3 (Optimized AL-SSL Pipeline):
- Integrates AL and SSL using optimized weighted model voting and adaptive thresholding.
- Uses AUROC-based threshold optimization for balancing recall and precision.
- Ensures a structured transition from AL to SSL at 10% labeled data, avoiding overfitting.
- Implements binary classification per label to improve minority class representation.
- Demonstrates significant performance improvements over Experiments 1 & 2.

# Key Findings:
- AL alone was highly effective in early stages, with the best feature extraction methods providing strong precision (LBP) and recall (BoVW).
- SSL required sufficient image resolution (minimum ~224×224 pixels) and effective label dependencies to refine pseudo-labels accurately.
- Tiny ImageNet showed the weakest improvements with SSL, likely due to its low resolution (64×64) and fine-grained class similarities.
- Optimizing thresholding and weighted voting in Experiment 3 significantly enhanced performance, demonstrating that hybrid AL-SSL is stronger than either method alone.
- Datasets with high label co-occurrence can benefit from AL but may suffer from SSL overfitting, leading to incorrect pseudo-label propagation.
- The optimal dataset of ca. 10,000 images for AL-SSL contains less than 20 classes with moderate label dependencies, while avoiding excessive class overlap.
