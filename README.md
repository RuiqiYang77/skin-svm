<p align="center">
  <img src="assets/logo.png" alt="Zhejiang University logo" width="100">
</p>

<h1 align="center">Handcrafted Skin Lesion Classification with SVM</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-classical%20ML-orange.svg" alt="scikit-learn"></a>
  <a href="https://huggingface.co/datasets/RuiqiYang77/skin-svm"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg" alt="Dataset"></a>
</p>

<p align="center">
  Official code repository for a biomedical image-processing project on three-class skin lesion classification using handcrafted visual descriptors and classical machine learning.
</p>


## Overview

This project studies whether interpretable handcrafted features can provide a compact and reproducible baseline for skin lesion classification. Given an RGB lesion image and its binary lesion mask, the pipeline extracts color, texture, shape, and optional dermoscopic descriptors, then trains a Support Vector Machine classifier for three diagnostic categories:

- `nv`: melanocytic nevus
- `mel`: melanoma
- `vasc`: vascular lesion

The main experimental pipeline is:

**Image + lesion mask** -> **mask preprocessing and optional image normalization** -> **color / texture / shape feature extraction** -> **standardization and optional PCA** -> **RBF-SVM classification** -> **split-wise evaluation and augmentation-robustness analysis**.

## Highlights

- Reproducible classical machine-learning baseline for skin lesion classification.
- Grouped train/validation/test split to avoid leakage between original and augmented images.
- Handcrafted feature families covering lesion color, texture, border, shape, and asymmetry.
- Saved experiment artifacts, including configuration, features, split file, predictions, metrics, model bundle, and confusion matrix.
- Single-image prediction script for inference with a trained experiment.

## Dataset

The dataset is hosted on Hugging Face: [RuiqiYang77/skin-svm](https://huggingface.co/datasets/RuiqiYang77/skin-svm).

Download the dataset and place it under `data/`.

## Installation

Create a clean Python environment:

```bash
conda create -n skin-svm python=3.10
conda activate skin-svm
pip install -r requirement.txt
```

## Training

Run the default SVM experiment:

```bash
python src/train.py --config config/svm.yaml --experiment_id svm_exp001
```

To reuse a previously extracted feature table:

```bash
python src/train.py \
  --config config/svm.yaml \
  --experiment_id svm_exp001 \
  --reuse_features
```

## Evaluation

Evaluate a trained model on an external test set:

```bash
python src/evaluate.py \
  --config config/svm.yaml \
  --experiment_id svm_exp001 \
  --test_csv data/external/test/label.csv \
  --test_image_dir data/external/test/image \
  --test_mask_dir data/external/test/mask
```

## Prediction

Run inference on a single image and mask:

```bash
python src/predict.py \
  --config config/svm.yaml \
  --experiment_id svm_exp001 \
  --image_path data/image/1.jpg \
  --mask_path data/mask/mask_1.jpg
```

## Method Details

### Feature Extraction

The feature extractor is implemented in `src/dataloader/features.py`. It supports several groups of handcrafted descriptors:

- **Color features:** RGB, HSV, and Lab channel moments; HSV histograms; dark-pixel ratio; hue variation; saturation statistics.
- **Texture features:** grayscale statistics, LBP histograms, and GLCM properties such as contrast, dissimilarity, homogeneity, energy, correlation, and ASM.
- **Shape features:** lesion area ratio, perimeter, circularity, eccentricity, axis lengths, solidity, extent, compactness, convex-area ratio, border irregularity, and asymmetry.
- **Optional dermoscopic features:** configurable descriptors for pigment networks, dots/globules, streaks, regression structures, and composite clinical patterns.

### Model

The default model is an SVM pipeline:

```text
StandardScaler -> optional PCA -> SVC(kernel="rbf", class_weight="balanced")
```

The SVM implementation is in `src/model/svm.py`. Logistic Regression and Random Forest backends are also available for baseline comparison through `config/lr.yaml` and `config/rf.yaml`.


## License

This project is released under the license provided in `LICENSE`.
