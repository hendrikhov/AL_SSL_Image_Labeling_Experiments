# AL_SSL_Image_Labeling_Experiments
A comprehensive implementation of Active Learning (AL) and Semi-Supervised Learning (SSL) strategies for automatic image labeling, optimized for datasets with varying complexity and resource constraints.

# Description:
This repository contains the complete implementation for three experiments investigating the effectiveness of Active Learning (AL) and Semi-Supervised Learning (SSL) in automatic image labeling under computational and resource limitations. The experiments systematically compare classical machine learning models, feature extraction techniques, and hybrid AL-SSL approaches to improve label efficiency across different datasets.

# Datasets Used:
COCO (Common Objects in Context): Multi-label dataset with high class imbalance and diverse object categories.
Pascal VOC: Multi-label dataset with a more balanced label distribution and structured image compositions.
Tiny ImageNet: Single-label dataset with low resolution (64×64 pixels) and fine-grained class distinctions, testing feature extraction robustness.
Each dataset presents unique challenges in class balance, resolution, and label dependencies, making them ideal for evaluating AL-SSL performance across different complexity levels.

# Repository Contents:
## Experiment 1 (Active Learning - AL):
Implementation of AL strategies, including uncertainty sampling and diversity-based selection.
Performance evaluation of classical machine learning models trained on incrementally labeled datasets.
Feature extraction methods: LBP, BoVW, HOG, and PCA for dimensionality reduction.

## Experiment 2 (Semi-Supervised Learning - SSL):
Implementation of pseudo-labeling, co-training, and label propagation strategies.
Exploration of SSL performance across different dataset complexities and class distributions.
Analysis of SSL effectiveness based on initial labeled data proportions and feature representation.

## Experiment 3 (Optimized AL-SSL Pipeline):
Integration of AL and SSL with threshold optimization and weighted model voting.
Application of binary classification approaches to improve minority class representation.
Dynamic evaluation of AL-SSL transitions to minimize label noise and maximize learning efficiency.

# Key Features:
Preprocessing scripts for dataset preparation, feature extraction, and dimensionality reduction.
Modular functions for AL sample selection, pseudo-labeling, and adaptive thresholding.
Model training and evaluation pipelines, including metrics tracking and performance visualization.
Configuration options for experimenting with different models, datasets, and learning strategies.
