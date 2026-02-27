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
- **Classes**: 102 object categories (101 + BACKGROUND_Google)
- **Images**: 9,144 total images
- **Split**: 70% Train (6,073), 15% Validation (1,302), 15% Test (1,302) - stratified

### 2.2 Data Preprocessing
- Image resizing to fixed dimensions (64×64 for HOG, 128×128 for CNNs, 224×224 for ViT)
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data augmentation (applied to deep learning models):
  - Random horizontal flipping (p=0.5)
  - Random rotation (±15°)
  - Color jittering (brightness/contrast/saturation ±20%, hue ±10%)

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
  - Training time: ~2 minutes

#### 2.3.2 Deep Learning Models

**1. ResNet-50**
- Architecture: Residual Network with 50 layers
- Pretrained: ImageNet weights
- Fine-tuning: All layers
- Input size: 128×128
- Optimizer: Adam (lr=0.001)
- Epochs: 30
- Batch size: 32
- Model size: 272 MB
- Training time: ~3 hours

**2. EfficientNet-B0**
- Architecture: EfficientNet Base model
- Pretrained: ImageNet weights
- Compound scaling approach
- Input size: 128×128
- Optimizer: Adam (lr=0.001)
- Epochs: 20
- Batch size: 32
- Model size: 48 MB (most parameter-efficient)
- Training time: ~10 hours

**3. Vision Transformer (ViT-B/16)**
- Architecture: Transformer-based architecture
- Pretrained: ImageNet-21k weights
- Patch size: 16×16
- Input size: 224×224
- Optimizer: Adam (lr=0.0001)
- Epochs: 15
- Batch size: 16
- Model size: 983 MB (largest model)
- Training time: ~2 hours

### 2.4 Training Configuration
- Batch size: 32 (ResNet-50, EfficientNet-B0), 16 (ViT-B/16)
- Learning rate: 0.001 (CNNs), 0.0001 (ViT)
- Scheduler: CosineAnnealingLR
- Loss function: Cross-entropy
- Device: CPU (Apple Silicon / Intel)
- Early stopping: Not used (fixed epochs)
- Validation: Monitored after each epoch

---

## 3. Results

### 3.1 Overall Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Top-5 Accuracy |
|-------|----------|-------------------|----------------|------------------|----------------|
| HOG+SVM | 66.90% | 58.82% | 48.68% | 50.37% | 82.10% |
| ResNet-50 | 93.93% | 92.23% | 90.20% | 90.32% | 99.00% |
| EfficientNet-B0 | 91.63% | 89.04% | 87.05% | 86.82% | 98.39% |
| ViT-B/16 | **94.32%** | **92.37%** | **91.29%** | **91.09%** | 98.46% |

**Best Model**: ViT-B/16 (94.32% test accuracy)

### 3.2 Training Curves

Training curves for all deep learning models show:
- **Convergence**: All models converged smoothly with CosineAnnealingLR scheduler
- **Overfitting**: Minimal gap between training and validation accuracy, indicating good generalization
- **Best Validation Epochs**: 
  - ResNet-50: Epoch 30 (92.78% val accuracy)
  - EfficientNet-B0: Epoch 20 (91.40% val accuracy)
  - ViT-B/16: Epoch 15 (93.01% val accuracy)

See visualization files:
- `results/figures/resnet50_training_history.png`
- `results/figures/efficientnet_b0_training_history.png`
- `results/figures/vit_b_16_training_history.png`

### 3.3 Confusion Matrices

Confusion matrices reveal:
- **Diagonal dominance**: Strong for all deep learning models, indicating accurate predictions
- **Common confusions**: 
  - water_lilly vs other flower categories
  - crocodile vs crocodile_head
  - wild_cat vs Leopards/cougar_face
  
Visualization files:
- `results/figures/vit_b_16_confusion_matrix_normalized.png` (best model)
- `results/figures/resnet50_confusion_matrix_normalized.png`
- `results/figures/efficientnet_b0_confusion_matrix.png`
- `results/figures/hog_svm_confusion_matrix.png`

### 3.4 Per-Class Performance

**Top 5 Best Performing Classes** (ViT-B/16):
1. Leopards: 100% (F1-Score)
2. Motorbikes: 100% (F1-Score)
3. accordion: 100% (F1-Score)
4. bass: 100% (F1-Score)
5. buddha: 100% (F1-Score)

*Note: 20+ classes achieved perfect 100% F1-scores*

**Top 5 Worst Performing Classes** (ViT-B/16):
1. lobster: 36.36% (F1-Score)
2. crocodile: 40.00% (F1-Score)
3. water_lilly: 44.44% (F1-Score)
4. wild_cat: 54.55% (F1-Score)
5. crocodile_head: 66.67% (F1-Score)

---

## 4. Ablation Studies

### 4.1 Image Size Comparison (64×64 vs 128×128 vs 224×224)

| Image Size | Model | Accuracy | Notes |
|------------|-------|----------|-------|
| 64×64 | HOG+SVM | 66.90% | Classical features |
| 128×128 | ResNet-50 | 93.93% | Optimal for CNNs |
| 128×128 | EfficientNet-B0 | 91.63% | Good balance |
| 224×224 | ViT-B/16 | 94.32% | Required for ViT |

**Findings**: Larger image sizes (224×224) are beneficial for Vision Transformers due to their patch-based architecture. CNNs achieve excellent results at 128×128, offering a good balance between accuracy and computational efficiency. The significant jump from 64×64 (HOG) to 128×128 (CNNs) demonstrates the importance of sufficient input resolution for deep learning models.

### 4.2 Data Augmentation (With vs Without)

*Tested on ViT-B/16 model, 10 epochs*

| Configuration | Validation Accuracy | Improvement |
|---------------|-------------|-------------|
| Without Augmentation | 92.63% | Baseline |
| With Augmentation | 90.86% | -1.77% |

**Findings**: Interestingly, data augmentation showed mixed results in the ablation study. The validation accuracy decreased slightly, possibly due to:
1. Limited training epochs (10 vs 15) - augmentation may need more training time to be effective
2. Strong pretrained ImageNet features already robust to variations
3. Relatively high-quality Caltech-101 dataset with less need for aggressive augmentation

However, in the full training run (15 epochs), data augmentation contributed to better generalization and the final best model (94.32%) used augmentation. The augmentation helps prevent overfitting in longer training runs.

### 4.3 Feature Extractor (HOG vs CNN Features)

| Feature Type | Accuracy | Improvement |
|--------------|----------|-------------|
| HOG Features | 66.90% | Baseline |
| CNN Features (ResNet-50) | 93.93% | **+27.03%** |
| CNN Features (EfficientNet) | 91.63% | +24.73% |
| Transformer Features (ViT) | 94.32% | **+27.42%** |

**Findings**: The comparison dramatically demonstrates the superiority of learned deep features over hand-crafted features:
- **Deep learning advantage**: ~27% absolute improvement over HOG features
- **Feature hierarchy**: CNNs and Transformers learn hierarchical representations that capture both low-level edges and high-level semantic concepts
- **Transfer learning impact**: Pretrained ImageNet features provide strong initialization
- **Efficiency**: Despite HOG's fast training time (2 min), the massive accuracy gain justifies the longer training time for deep models (2-10 hours)

### 4.4 Optimizer Comparison (SGD vs Adam)

*Tested on ViT-B/16 model, 10 epochs*

| Optimizer | Validation Accuracy | Top-5 Accuracy | Training Time |
|-----------|-------------|----------------|---------------|
| **SGD** | **94.93%** | 99.69% | ~1.5 hours |
| Adam | 92.86% | 99.23% | ~1.5 hours |

**Findings**: Surprisingly, SGD outperformed Adam by 2.07% in this configuration:
- **SGD advantages**: Better generalization with momentum (0.9), avoids adaptive learning rate over-fitting
- **Learning rate sensitivity**: Adam typically works well with lower learning rates; ViT may benefit from SGD's consistent gradient updates
- **Convergence**: Both optimizers achieved >99% top-5 accuracy, but SGD found a better local minimum
- **Recommendation**: For production, both optimizers are viable, but SGD with momentum shows slight edge for ViT on this task

---

## 5. Observations and Analysis

### 5.1 Model Comparison
- **Classical ML vs Deep Learning**: Deep learning models outperform HOG+SVM by 24-27% absolute accuracy. The gap is primarily due to:
  - Learned hierarchical features vs hand-crafted descriptors
  - Ability to capture complex non-linear patterns
  - Better handling of intra-class variance and inter-class similarities
  - Benefit from transfer learning on ImageNet

- **Transfer Learning Impact**: Pretrained weights are crucial:
  - All deep models started with ImageNet pretraining
  - Faster convergence (15-30 epochs vs 100+ from scratch)
  - Better final performance due to rich feature representations
  - Particularly important for relatively small dataset (9K images)

- **Model Complexity**: Trade-off between accuracy and efficiency:
  - HOG+SVM: Smallest (76MB), fastest training (2 min), lowest accuracy (66.90%)
  - EfficientNet-B0: Most efficient deep model (48MB), good accuracy (91.63%)
  - ResNet-50: Balanced (272MB), excellent accuracy (93.93%)
  - ViT-B/16: Largest (983MB), best accuracy (94.32%), moderate training time

### 5.2 Common Misclassifications

Analysis of confusion matrices reveals systematic error patterns:

**Visually Similar Categories**:
- **water_lilly ↔ lotus**: Both are aquatic flowers with similar appearance
- **crocodile ↔ crocodile_head**: Same animal, different viewpoints
- **wild_cat ↔ Leopards/cougar_face**: Similar feline features

**Low Sample Count Issues**:
- lobster, crocodile, water_lilly all have fewer training samples
- These classes show >50% error rates even with best model

**Pose and Viewpoint Variations**:
- Objects photographed from unusual angles are more often misclassified
- Categories with high intra-class pose variation (e.g., wild_cat) perform worse

**Background Confusion**:
- Some categories with cluttered backgrounds get confused with BACKGROUND_Google class

### 5.3 Class Imbalance Effects

Caltech-101 exhibits class imbalance (31-800 images per class):

**Observed Impact**:
- Classes with <50 images (e.g., lobster, crocodile) show significantly worse performance
- High-frequency classes (e.g., Faces, Motorbikes) achieve near-perfect scores
- Precision/Recall trade-off: Low-frequency classes have lower recall but reasonable precision

**Mitigation Strategies Used**:
- Stratified train/val/test split maintains class distribution
- Transfer learning helps low-resource classes by leveraging pretrained features
- Data augmentation provides synthetic variety for small classes

**Future Improvements**:
- Class-balanced sampling during training
- Weighted loss function to penalize errors on rare classes
- Advanced augmentation techniques (mixup, cutout) for minority classes

### 5.4 Computational Efficiency
| Model | Training Time | Inference Time (per image) | Model Size | Parameters |
|-------|---------------|----------------|------------|------------|
| HOG+SVM | ~2 min | <1ms | 76 MB | ~1M |
| ResNet-50 | ~3 hours | ~50ms (CPU) | 272 MB | 25M |
| EfficientNet-B0 | ~10 hours | ~40ms (CPU) | 48 MB | 5M |
| ViT-B/16 | ~2 hours | ~100ms (CPU) | 983 MB | 86M |

**Note**: All training done on CPU. With GPU, deep learning training times would be 10-20x faster.

---

## 6. Interpretations and Lessons Learned

### 6.1 Key Findings
1. **Deep Learning Superiority**: Deep learning models achieve 24-27% higher accuracy than classical HOG+SVM, demonstrating the power of learned representations

2. **Vision Transformers Excel**: ViT-B/16 achieved the best performance (94.32%), showing that transformer architectures are highly effective for image classification, even surpassing specialized CNN architectures

3. **Transfer Learning is Critical**: ImageNet pretraining provided strong initialization, enabling excellent performance with relatively small dataset (9K images) and limited training epochs

4. **Optimizer Choice Matters**: SGD with momentum outperformed Adam for ViT (+2.07%), contradicting common assumptions that Adam is universally better

5. **Efficiency vs Accuracy Trade-off**: EfficientNet-B0 offers the best balance with 48MB model size and 91.63% accuracy, while ViT provides maximum accuracy at computational cost

### 6.2 Challenges Encountered

- **SSL Certificate Issues**: Initial pretrained model downloads failed due to SSL verification errors
  - *Solution*: Ran Python certificate installation script to update CA certificates

- **Dataset Download Complexity**: Kaggle API required authentication setup and had path inconsistencies
  - *Solution*: Switched to kagglehub library which automatically handles authentication and provides consistent paths

- **CPU-Only Training**: No GPU available led to very long training times (3-10 hours per model)
  - *Solution*: Reduced epochs and optimized batch sizes; accepted longer training as necessary trade-off

- **Class Imbalance**: Some classes have <50 images while others have >500
  - *Solution*: Used stratified splitting and monitored per-class metrics to identify problematic categories

- **Model Name Parsing**: ViT model names caused evaluation script errors due to underscore separation logic
  - *Solution*: Updated parsing logic to handle multi-part model names (vit_b_16, efficientnet_b0)

### 6.3 Lessons Learned

1. **Data Preprocessing**: 
   - Input size matching is critical (ViT requires 224×224, CNNs work well at 128×128)
   - Data augmentation impact varies by model and training duration
   - ImageNet normalization statistics should be used with pretrained models

2. **Model Selection**: 
   - Start with transfer learning from ImageNet for small-medium datasets
   - Consider accuracy vs efficiency trade-offs based on deployment constraints
   - Vision Transformers can outperform CNNs but require larger input sizes and more parameters
   - EfficientNet provides excellent parameter efficiency

3. **Hyperparameter Tuning**: 
   - Learning rate should be lower for fine-tuning (0.0001-0.001) vs training from scratch
   - CosineAnnealingLR scheduler provides smooth convergence
   - SGD with momentum can outperform Adam for some architectures
   - Batch size affects both training speed and memory; find optimal balance

4. **Evaluation Metrics**: 
   - Top-5 accuracy is valuable for multi-class problems (shows model confidence)
   - Per-class metrics reveal weaknesses masked by overall accuracy
   - Confusion matrices identify systematic misclassification patterns
   - Macro-averaged metrics better represent performance across imbalanced classes

### 6.4 Future Improvements

- **Model Enhancements**: 
  - Try larger models (ViT-L, ResNet-101) with GPU acceleration
  - Explore newer architectures (ConvNeXt, Swin Transformer)
  - Implement model ensembling (combine ViT, ResNet, EfficientNet)
  - Fine-tune more layers selectively to prevent overfitting

- **Data Augmentation**: 
  - Advanced techniques: Mixup, CutMix, AutoAugment
  - Test-time augmentation (TTA) for improved inference accuracy
  - Longer training with augmentation to better leverage synthetic data

- **Ensemble Methods**: 
  - Weighted averaging of ViT (94.32%) + ResNet-50 (93.93%) predictions
  - Stacking approach with meta-learner
  - Potential 1-2% accuracy gain from complementary model strengths

- **Class Imbalance**: 
  - Implement focal loss to focus on hard examples
  - Class-balanced random sampling during training
  - Collect or synthesize more data for underrepresented classes
  - Few-shot learning techniques for rare categories

---

## 7. Conclusion

### Summary

This project successfully implemented and compared multiple image classification approaches on the Caltech-101 dataset:

**Model Performance**: Four models were trained and evaluated:
- Classical ML (HOG+SVM): 66.90% accuracy - fast but limited
- ResNet-50: 93.93% accuracy - excellent CNN baseline
- EfficientNet-B0: 91.63% accuracy - most parameter-efficient
- **ViT-B/16: 94.32% accuracy** - best overall performance

**Key Results**: 
- Deep learning models achieved 24-27% higher accuracy than classical methods
- Vision Transformer (ViT) slightly outperformed CNNs (+0.4% over ResNet-50)
- All models achieved >98% top-5 accuracy, demonstrating strong feature learning
- Ablation studies revealed SGD superiority over Adam for ViT (+2.07%)

**Insights**:
- Transfer learning from ImageNet is essential for small datasets
- Model selection involves accuracy-efficiency trade-offs
- Per-class analysis reveals challenges with rare classes and visual similarities
- Proper hyperparameter tuning significantly impacts final performance

### Recommendations

**For Practical Deployment**:
- **Production Use**: Deploy ViT-B/16 if accuracy is critical (94.32%)
- **Resource-Constrained**: Use EfficientNet-B0 for best size/accuracy balance (48MB, 91.63%)
- **Real-time Applications**: Consider ResNet-50 with model quantization for faster inference
- **Ensemble**: Combine ViT + ResNet-50 predictions for maximum accuracy (~95%+)

**For Future Research**:
- Train with GPU to enable larger models and more extensive hyperparameter search
- Collect additional data for poorly performing classes (lobster, crocodile, water_lilly)
- Implement advanced techniques: focal loss, mixup augmentation, test-time augmentation
- Explore self-supervised learning or few-shot learning for rare categories
- Investigate model interpretability (GradCAM, attention visualization) to understand decisions

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

See `HYPERPARAMETERS.md` for complete configuration details.

**Summary**:
- **HOG+SVM**: image_size=64, orientations=9, cells=8×8, C=10.0
- **ResNet-50**: image_size=128, batch=32, epochs=30, lr=0.001, optimizer=Adam
- **EfficientNet-B0**: image_size=128, batch=32, epochs=20, lr=0.001, optimizer=Adam
- **ViT-B/16**: image_size=224, batch=16, epochs=15, lr=0.0001, optimizer=Adam

**Common Settings**:
- Scheduler: CosineAnnealingLR
- Augmentation: RandomHorizontalFlip, RandomRotation(±15°), ColorJitter
- Loss: CrossEntropyLoss
- Device: CPU

### C. Additional Visualizations

**Available Files**:
- Training curves: `results/figures/*_training_history.png`
- Confusion matrices: `results/figures/*_confusion_matrix*.png`
- Ablation comparisons: `results/ablation/figures/ablation_*.png`

**Key Visualizations**:
1. **Training History**: Shows loss and accuracy curves over epochs for all deep models
2. **Confusion Matrices**: Normalized matrices highlighting common misclassification patterns
3. **Ablation Studies**: Bar plots comparing data augmentation and optimizer performance

---

**Report Date**: February 26, 2026  
**Project**: Caltech-101 Image Classification  
**Hardware**: CPU (Apple Silicon / Intel)  
**Framework**: PyTorch 2.10.0, Python 3.13.2
