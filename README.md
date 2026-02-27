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
# ResNet-50 (30 epochs, 128×128 images)
python scripts/train_deep.py --model resnet50 --data_dir data/101_ObjectCategories \
    --image_size 128 --batch_size 32 --epochs 30 --lr 0.001 --optimizer adam --augmentation

# EfficientNet-B0 (20 epochs, 128×128 images)
python scripts/train_deep.py --model efficientnet_b0 --data_dir data/101_ObjectCategories \
    --image_size 128 --batch_size 32 --epochs 20 --lr 0.001 --optimizer adam --augmentation

# Vision Transformer (15 epochs, 224×224 images)
python scripts/train_deep.py --model vit_b_16 --data_dir data/101_ObjectCategories \
    --image_size 224 --batch_size 16 --epochs 15 --lr 0.0001 --optimizer adam --augmentation
```

### 4. Run Ablation Studies
```bash
# Data augmentation ablation (ViT-B/16)
python scripts/ablation_studies.py --study augmentation --model vit_b_16 \
    --data_dir data/101_ObjectCategories --image_size 224 --batch_size 16 \
    --epochs 10 --lr 0.0001

# Optimizer comparison (SGD vs Adam)
python scripts/ablation_studies.py --study optimizer --model vit_b_16 \
    --data_dir data/101_ObjectCategories --image_size 224 --batch_size 16 \
    --epochs 10 --lr 0.0001
```

### 5. Evaluate Models
```bash
# Generate complete evaluation report with visualizations
python scripts/evaluate.py --model_path models/resnet50_best.pth \
    --data_dir data/101_ObjectCategories --image_size 128 --generate_report

# Evaluate ViT model
python scripts/evaluate.py --model_path models/vit_b_16_best.pth \
    --data_dir data/101_ObjectCategories --image_size 224 --generate_report
```

## Results

### ✅ Completed Experiments

| Model | Test Accuracy | Top-5 Accuracy | Training Time | Parameters |
|-------|---------------|----------------|---------------|------------|
| HOG + SVM | 66.90% | 82.10% | ~2 min | ~1M |
| ResNet-50 | 93.93% | 99.00% | ~3 hours | 25M |
| EfficientNet-B0 | 91.63% | 98.39% | ~10 hours | 5M |
| **ViT-B/16** | **94.32%** | 98.46% | ~2 hours | 86M |

**Best Model**: Vision Transformer (ViT-B/16) with 94.32% test accuracy

### Ablation Studies

1. **Data Augmentation** (ViT-B/16): With augmentation (93.39%) vs Without (92.40%) → +0.99%
2. **Optimizer Comparison** (ViT-B/16): SGD (94.93%) vs Adam (92.86%) → +2.07% for SGD

### Key Findings

- Deep learning models outperform classical HOG+SVM by **~27%**
- Vision Transformers achieve best accuracy with proper hyperparameter tuning
- Transfer learning from ImageNet is critical for small datasets
- SGD with momentum can outperform Adam for ViT architectures

### Detailed Reports

- **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)**: Comprehensive experiment progress and results
- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Complete project report with analysis and insights

Results are saved in the `results/` directory:
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
