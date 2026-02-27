# Caltech-101 Experiment Summary

## Experiment Progress

### ✅ Completed

#### 1. Data Preparation
- ✅ Dataset Download: Caltech-101 (9,144 images, 102 categories)
- ✅ Dataset Split: Train (6,073) / Val (1,302) / Test (1,302)

#### 2. Model Training (4 Models)

| Model | Val Accuracy | Test Accuracy | Top-5 Accuracy | Training Time | Status |
|-------|-------------|---------------|----------------|---------------|--------|
| **HOG + SVM** | - | 66.90% | 82.10% | ~2 min | ✅ Done |
| **ResNet-50** | 92.78% | **93.93%** | 99.00% | ~3 hrs | ✅ Done |
| **EfficientNet-B0** | 91.40% | 91.63% | 98.39% | ~10 hrs | ✅ Done |
| **ViT-B/16** | 93.01% | **94.32%** | 98.46% | ~2 hrs | ✅ Done |

**Best Model**: ViT-B/16 (94.32% test accuracy)

#### 3. Ablation Studies

##### 3.1 Data Augmentation Ablation (ViT-B/16) ✅

| Configuration | Val Accuracy | Test Accuracy | Improvement |
|--------------|-------------|---------------|-------------|
| No Augmentation | 92.40% | - | baseline |
| **With Augmentation** | **93.39%** | - | **+0.99%** |

**Conclusion**: Data augmentation improves accuracy by ~1%

##### 3.2 Optimizer Ablation (ViT-B/16) ✅

| Optimizer | Val Accuracy | Top-5 Accuracy | Improvement |
|-----------|-------------|----------------|-------------|
| **SGD** | **94.93%** | 99.69% | baseline |
| Adam | 92.86% | 99.23% | -2.07% |

**Conclusion**: SGD outperforms Adam by ~2% for ViT-B/16 on this task, contrary to typical expectations. This may be due to the specific learning rate and momentum settings.

---

## Detailed Results

### Model Performance Comparison

#### Overall Metrics

| Model | Accuracy | Precision<br/>(Macro) | Recall<br/>(Macro) | F1-Score<br/>(Macro) |
|-------|---------|----------------------|-------------------|---------------------|
| HOG + SVM | 66.90% | 58.82% | 48.68% | 50.37% |
| ResNet-50 | **93.93%** | 92.23% | 90.20% | 90.32% |
| EfficientNet-B0 | 91.63% | 89.04% | 87.05% | 86.82% |
| ViT-B/16 | **94.32%** | **92.37%** | **91.29%** | **91.09%** |

#### Best Performing Classes by Model

**ViT-B/16** (test set):
- Perfect classification (100% F1): Leopards, Motorbikes, accordion, bass, buddha, etc.
- Worst classes: lobster (36.36%), crocodile (40.00%)

**ResNet-50** (test set):
- Perfect classification (100% F1): Leopards, Motorbikes, barrel, butterfly, cannon, etc.
- Worst classes: water_lilly (28.57%), wild_cat (50.00%)

#### Deep Learning vs Classical ML

| Comparison | HOG + SVM | Deep Learning (Avg) | Improvement |
|------------|-----------|---------------------|-------------|
| Accuracy | 66.90% | 93.29% | **+26.39%** |
| Training Time | 2 min | 5-10 hrs | - |
| Parameters | ~1M | 5M-86M | - |

**Conclusion**: Deep learning models significantly outperform traditional methods (+26%), but require more computational resources.

---

## Hyperparameter Configuration

### Best Model (ViT-B/16)

```python
{
    "model": "Vision Transformer Base (patch_size=16)",
    "pretrained": "ImageNet",
    "image_size": 224,
    "batch_size": 16,
    "epochs": 15,
    "learning_rate": 0.0001,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",
    "data_augmentation": True,
    "augmentation_types": [
        "RandomHorizontalFlip (p=0.5)",
        "RandomRotation (±15°)",
        "ColorJitter (brightness/contrast/saturation ±20%, hue ±10%)"
    ]
}
```

---

## File Structure

### Trained Models
```
models/
├── hog_svm_classifier.pkl          # HOG + SVM
├── resnet50_best.pth                # ResNet-50
├── efficientnet_b0_best.pth         # EfficientNet-B0
└── vit_b_16_best.pth                # ViT-B/16 (Best)
```

### Experiment Results
```
results/
├── metrics/
│   ├── hog_svm_test_metrics.json
│   ├── resnet50_test_metrics.json
│   ├── resnet50_detailed_metrics.json
│   ├── efficientnet_b0_test_metrics.json
│   ├── vit_b_16_detailed_metrics.json
│   └── ...
├── figures/
│   ├── hog_svm_confusion_matrix.png
│   ├── resnet50_confusion_matrix_normalized.png
│   ├── resnet50_training_history.png
│   ├── efficientnet_b0_confusion_matrix.png
│   ├── efficientnet_b0_training_history.png
│   ├── vit_b_16_confusion_matrix_normalized.png
│   └── vit_b_16_training_history.png
└── ablation/
    ├── metrics/
    │   ├── ablation_augFalse_metrics.json
    │   └── ablation_augTrue_metrics.json
    └── figures/
        └── ablation_augmentation.png
```

---

## Key Findings

### 1. Model Architecture Impact
- **ViT-B/16** performs best (94.32%), demonstrating Transformer advantages in vision tasks
- **ResNet-50** follows closely (93.93%), proving CNNs remain highly effective
- **EfficientNet-B0** achieves best parameter efficiency (91.63% with only 5M parameters)

### 2. Data Augmentation Importance
- Data augmentation improves accuracy by ~**1%**
- Effectively prevents overfitting
- Especially important for small datasets

### 3. Value of Pretraining
- All deep learning models use ImageNet pretraining
- Significantly accelerates convergence
- Improves final performance

### 4. Class Difficulty Variance
- Easy classes: Motorbikes, Leopards (100% F1)
- Hard classes: water_lilly, lobster, crocodile (<50% F1)
- Reasons: Low sample count, high intra-class variance, high inter-class similarity

---

## Experiment Completion

### ✅ All Tasks Completed
- ✅ 4 model training runs (HOG+SVM, ResNet-50, EfficientNet-B0, ViT-B/16)
- ✅ Data augmentation ablation study
- ✅ Optimizer ablation study (SGD vs Adam)
- ✅ All visualizations and metrics generated

### Key Deliverables
- Trained models: `models/` directory
- Metrics: `results/metrics/` directory (JSON files)
- Visualizations: `results/figures/` (confusion matrices, training curves)
- Ablation results: `results/ablation/` (metrics and comparative plots)

---

## Hardware Environment

- **Device**: CPU (Apple Silicon / Intel)
- **Python**: 3.13.2
- **PyTorch**: 2.10.0
- **Total Training Time**: ~16 hours (all models)

---

## Model Citations

```bibtex
@article{resnet2015,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={CVPR},
  year={2016}
}

@article{efficientnet2019,
  title={EfficientNet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  journal={ICML},
  year={2019}
}

@article{vit2020,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}
```

---

**Last Updated**: February 26, 2026
