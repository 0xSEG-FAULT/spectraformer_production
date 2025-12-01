# Spectraformer: Grain Classification using Hyperspectral NIR Spectroscopy

A comprehensive implementation of the SpectraFormer model for classifying grain varieties (Barley, Chickpea, Sorghum) using near-infrared spectroscopy data with transformer-based deep learning.

## Features

- **Multiple Preprocessing Techniques**: S, SM, SA, SA0, SAM, SA0M, S0M, 0M
- **Three Grain Datasets**: Barley (24 classes), Chickpea (19 classes), Sorghum (10 classes)
- **SpectraFormer Model**: Combines CNN and Transformer for spectral analysis
- **Comprehensive Pipeline**: From preprocessing to model evaluation
- **Visualization Tools**: Before/after preprocessing plots and confusion matrices

## Project Structure


## Quick Start

### 1. Setup Environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2. Preprocess All Datasets
python scripts/01_preprocess_all_datasets.py

### 3. Visualize Preprocessing
python scripts/02_visualize_preprocessing.py

### 4. Train Models
python scripts/04_train_best_model.py

### 5. Evaluate & Generate Results
python scripts/05_evaluate_and_visualize.py


## Preprocessing Methods

| Code | Method | Description |
|------|--------|-------------|
| S | Smoothing | Savitzky-Golay smoothing for noise reduction |
| A | Baseline | airPLS baseline correction |
| 0 | Negatives | Remove negative values |
| M | Normalization | Min-Max normalization |

Combined techniques: SM, SA, SA0, SAM, SA0M, S0M, 0M

## Paper Reference

Chen, Z., Zhou, R., & Ren, P. (2024). Spectraformer deep learning model for grain spectral qualitative analysis based on transformer structure. *RSC Advances*, 14, 8053-8066.

## Results

Expected accuracies (with SA0M preprocessing):
- Barley: ~85%
- Chickpea: ~95%
- Sorghum: ~86%

## Author

Implementation based on the SpectraFormer research paper.
