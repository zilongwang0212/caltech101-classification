# Trained Models

This directory contains trained model checkpoints.

## Model Files

| Model | Size | Status | Notes |
|-------|------|--------|-------|
| `hog_svm_classifier.pkl` | 76 MB | ✅ Included | Classical ML baseline |
| `efficientnet_b0_best.pth` | 48 MB | ✅ Included | Most parameter-efficient model |
| `resnet50_best.pth` | 272 MB | ⚠️ Not included | Exceeds GitHub 100MB limit |
| `vit_b_16_best.pth` | 983 MB | ⚠️ Not included | Exceeds GitHub 100MB limit |

## Large Model Files

**ResNet-50** and **ViT-B/16** models exceed GitHub's 100MB file size limit.

### Download Instructions

To download these large models, you have two options:

#### Option 1: Re-train the models
```bash
# ResNet-50 (~3 hours on CPU)
python scripts/train_deep.py --model resnet50 --data_dir data/101_ObjectCategories \
    --image_size 128 --batch_size 32 --epochs 30 --lr 0.001 --optimizer adam --augmentation

# ViT-B/16 (~2 hours on CPU)
python scripts/train_deep.py --model vit_b_16 --data_dir data/101_ObjectCategories \
    --image_size 224 --batch_size 16 --epochs 15 --lr 0.0001 --optimizer adam --augmentation
```

#### Option 2: Download from external source
Contact the repository maintainer or check the releases page for download links to pre-trained models.

## Model Performance Summary

| Model | Test Accuracy | Top-5 Accuracy | Parameters |
|-------|---------------|----------------|------------|
| HOG+SVM | 66.90% | 82.10% | ~1M |
| EfficientNet-B0 | 91.63% | 98.39% | 5M |
| ResNet-50 | 93.93% | 99.00% | 25M |
| ViT-B/16 | **94.32%** | 98.46% | 86M |

**Best Model**: ViT-B/16 achieves the highest test accuracy at 94.32%.

## Ablation Study Models

The `ablation/` subdirectory contains model checkpoints from ablation studies:
- Data augmentation experiments (with/without augmentation)
- Optimizer comparison (SGD vs Adam)

Note: Some ablation models may also exceed file size limits and are not included in the repository.
