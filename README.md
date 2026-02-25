# Caltech-101 Image Classification Project

This project implements and compares multiple image classification methods on the Caltech-101 dataset, including classical machine learning approaches and modern deep learning architectures.

## Dataset

**Caltech-101**: 101 object categories with ~9,000 images
- Download link: https://www.kaggle.com/datasets/imbikramsaha/caltech-101

## Project Structure

```
caltech101-classification/
├── data/                    # Dataset storage
├── models/                  # Saved model checkpoints
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Training and evaluation scripts
├── results/                # Output metrics and figures
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   └── utils/             # Evaluation and visualization utilities
├── requirements.txt
└── README.md
```

## Goals

1. **Train and evaluate at least 3 different methods** for image classification:
   - Classical ML: HOG + SVM
   - Deep Learning: ResNet, EfficientNet, ViT

2. **Compare classical machine learning and deep learning methods**

3. **Evaluation Metrics**:
   - Accuracy (overall performance)
   - Per-class accuracy (class imbalance effects)
   - Confusion matrix (visualize misclassification)
   - Precision, Recall, F1-Score (macro & weighted averages)
   - Top-k accuracy (optional, e.g., Top-5)

4. **Ablation Studies** (at least 2):
   - Image size: 64×64 vs 128×128
   - Data augmentation: with vs without
   - Feature extractor: HOG vs CNN features
   - Optimizer: SGD vs Adam for CNN

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset
```bash
python scripts/download_data.py
```

### 2. Train Classical ML Model
```bash
python scripts/train_classical.py --feature_type hog --classifier svm
```

### 3. Train Deep Learning Models
```bash
# ResNet
python scripts/train_deep.py --model resnet50 --image_size 128 --epochs 50

# EfficientNet
python scripts/train_deep.py --model efficientnet_b0 --image_size 128 --epochs 50

# Vision Transformer
python scripts/train_deep.py --model vit_b_16 --image_size 128 --epochs 50
```

### 4. Run Ablation Studies
```bash
python scripts/ablation_studies.py --study image_size
python scripts/ablation_studies.py --study augmentation
python scripts/ablation_studies.py --study optimizer
```

### 5. Evaluate Models
```bash
python scripts/evaluate.py --model_path models/resnet50_best.pth
```

## Results

Results will be saved in the `results/` directory:
- **Metrics**: JSON files with all evaluation metrics
- **Figures**: Confusion matrices, training curves, comparison plots

## Report Deliverables

- Methods description
- Results (metrics + plots)
- Observations and/or Ablation studies
- Interpretations and Lessons learned
- Notebooks, Scripts, and Figures

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## License

MIT License
