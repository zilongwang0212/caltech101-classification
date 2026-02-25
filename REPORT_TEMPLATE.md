# Caltech-101 Image Classification Project - Report Template

## 1. Introduction

### Project Overview
This project implements and compares multiple image classification methods on the Caltech-101 dataset, which contains 101 object categories with approximately 9,000 images.

### Objectives
- Train and evaluate at least three different methods for image classification
- Compare classical machine learning and deep learning approaches
- Conduct ablation studies to understand model behavior
- Provide comprehensive analysis and insights

---

## 2. Methods

### 2.1 Dataset
- **Dataset**: Caltech-101
- **Classes**: 101 object categories
- **Images**: ~9,000 total images
- **Split**: 70% Train, 15% Validation, 15% Test (stratified)

### 2.2 Data Preprocessing
- Image resizing to fixed dimensions (64×64, 128×128)
- Normalization using ImageNet statistics
- Data augmentation (optional):
  - Random cropping
  - Horizontal flipping
  - Random rotation (±15°)
  - Color jittering

### 2.3 Models Implemented

#### 2.3.1 Classical Machine Learning
**HOG + SVM**
- Feature extraction using Histogram of Oriented Gradients (HOG)
- Classification using Support Vector Machine (SVM)
- Parameters:
  - Image size: 64×64
  - Orientations: 9
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - SVM kernel: RBF
  - C: 10.0

#### 2.3.2 Deep Learning Models

**1. ResNet-50**
- Architecture: Residual Network with 50 layers
- Pretrained: ImageNet weights
- Fine-tuning: All layers
- Input size: 128×128
- Optimizer: Adam (lr=0.001)
- Epochs: 50

**2. EfficientNet-B0**
- Architecture: EfficientNet Base model
- Pretrained: ImageNet weights
- Compound scaling approach
- Input size: 128×128
- Optimizer: Adam (lr=0.001)
- Epochs: 50

**3. Vision Transformer (ViT-B/16)**
- Architecture: Transformer-based architecture
- Pretrained: ImageNet weights
- Patch size: 16×16
- Input size: 128×128
- Optimizer: Adam (lr=0.001)
- Epochs: 50

### 2.4 Training Configuration
- Batch size: 32
- Learning rate: 0.001
- Scheduler: Cosine annealing
- Loss function: Cross-entropy
- Early stopping: Based on validation accuracy

---

## 3. Results

### 3.1 Overall Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Top-5 Accuracy |
|-------|----------|-------------------|----------------|------------------|----------------|
| HOG+SVM | [FILL] | [FILL] | [FILL] | [FILL] | N/A |
| ResNet-50 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| EfficientNet-B0 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| ViT-B/16 | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

**Best Model**: [FILL]

### 3.2 Training Curves

[Insert training/validation loss and accuracy curves]

### 3.3 Confusion Matrices

[Insert confusion matrix visualizations for best model]

### 3.4 Per-Class Performance

**Top 5 Best Performing Classes**:
1. [FILL]: [Accuracy/F1-Score]
2. [FILL]: [Accuracy/F1-Score]
3. [FILL]: [Accuracy/F1-Score]
4. [FILL]: [Accuracy/F1-Score]
5. [FILL]: [Accuracy/F1-Score]

**Top 5 Worst Performing Classes**:
1. [FILL]: [Accuracy/F1-Score]
2. [FILL]: [Accuracy/F1-Score]
3. [FILL]: [Accuracy/F1-Score]
4. [FILL]: [Accuracy/F1-Score]
5. [FILL]: [Accuracy/F1-Score]

---

## 4. Ablation Studies

### 4.1 Image Size Comparison (64×64 vs 128×128)

| Image Size | Accuracy | Improvement |
|------------|----------|-------------|
| 64×64 | [FILL] | Baseline |
| 128×128 | [FILL] | [FILL]% |

**Findings**: [Describe the impact of image size on performance]

### 4.2 Data Augmentation (With vs Without)

| Configuration | Accuracy | Improvement |
|---------------|----------|-------------|
| Without Augmentation | [FILL] | Baseline |
| With Augmentation | [FILL] | [FILL]% |

**Findings**: [Describe the impact of data augmentation]

### 4.3 Feature Extractor (HOG vs CNN Features)

| Feature Type | Accuracy | Improvement |
|--------------|----------|-------------|
| HOG Features | [FILL] | Baseline |
| CNN Features (ResNet) | [FILL] | [FILL]% |

**Findings**: [Compare traditional vs deep learning features]

### 4.4 Optimizer Comparison (SGD vs Adam)

| Optimizer | Accuracy | Training Time |
|-----------|----------|---------------|
| SGD | [FILL] | [FILL] |
| Adam | [FILL] | [FILL] |

**Findings**: [Compare optimizer performance]

---

## 5. Observations and Analysis

### 5.1 Model Comparison
- **Classical ML vs Deep Learning**: [Discuss performance gap and reasons]
- **Transfer Learning Impact**: [Analyze the benefit of pretrained weights]
- **Model Complexity**: [Compare model sizes and inference times]

### 5.2 Common Misclassifications
[Identify patterns in errors - which classes are commonly confused]

### 5.3 Class Imbalance Effects
[Discuss how class imbalance affects model performance]

### 5.4 Computational Efficiency
| Model | Training Time | Inference Time | Model Size |
|-------|---------------|----------------|------------|
| HOG+SVM | [FILL] | [FILL] | [FILL] |
| ResNet-50 | [FILL] | [FILL] | [FILL] |
| EfficientNet-B0 | [FILL] | [FILL] | [FILL] |
| ViT-B/16 | [FILL] | [FILL] | [FILL] |

---

## 6. Interpretations and Lessons Learned

### 6.1 Key Findings
1. **[Finding 1]**: [Description]
2. **[Finding 2]**: [Description]
3. **[Finding 3]**: [Description]

### 6.2 Challenges Encountered
- [Challenge 1 and how it was addressed]
- [Challenge 2 and how it was addressed]
- [Challenge 3 and how it was addressed]

### 6.3 Lessons Learned
1. **Data Preprocessing**: [Insights about data preparation]
2. **Model Selection**: [Insights about choosing appropriate models]
3. **Hyperparameter Tuning**: [Insights about optimization]
4. **Evaluation Metrics**: [Insights about metric selection]

### 6.4 Future Improvements
- **Model Enhancements**: [Suggestions for better models or architectures]
- **Data Augmentation**: [Additional augmentation strategies]
- **Ensemble Methods**: [Potential for model ensembling]
- **Class Imbalance**: [Strategies to handle imbalanced classes]

---

## 7. Conclusion

### Summary
[Summarize the project outcomes, best performing model, and key insights]

### Recommendations
[Provide recommendations for practical deployment and future work]

---

## 8. References

1. Fei-Fei, L., Fergus, R., & Perona, P. (2007). Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories. Computer Vision and Image Understanding.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.

4. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

5. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.

---

## Appendix

### A. Code Repository Structure
```
caltech101-classification/
├── data/                    # Dataset storage
├── models/                  # Saved model checkpoints
├── notebooks/              # Jupyter notebooks
├── scripts/                # Training and evaluation scripts
├── results/                # Metrics and figures
├── src/                    # Source code
└── README.md
```

### B. Hyperparameter Details
[Detailed hyperparameter configurations for all models]

### C. Additional Visualizations
[Any additional plots, charts, or visualizations]

---

**Report Date**: [DATE]  
**Authors**: [YOUR NAME]  
**Project**: Caltech-101 Image Classification
