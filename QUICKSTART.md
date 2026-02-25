# Caltech-101 Image Classification - Quick Reference Guide

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd ~/caltech101-classification
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option 1: Automatic (requires Kaggle API)
kaggle datasets download -d imbikramsaha/caltech-101 -p data/

# Option 2: Manual
# Visit: https://www.kaggle.com/datasets/imbikramsaha/caltech-101
# Download and extract to data/ directory

# Verify download
python scripts/download_data.py
```

### 3. Quick Test (10 epochs)
```bash
./quick_start.sh
```

### 4. Full Pipeline
```bash
./run_all.sh  # This will take several hours
```

---

## 📋 Individual Commands

### Train Classical ML Models
```bash
# HOG + SVM
python scripts/train_classical.py \
    --feature_type hog \
    --classifier svm \
    --image_size 64

# HOG + Random Forest
python scripts/train_classical.py \
    --feature_type hog \
    --classifier random_forest \
    --n_estimators 100

# CNN Features + SVM
python scripts/train_classical.py \
    --feature_type cnn \
    --classifier svm \
    --cnn_model resnet50
```

### Train Deep Learning Models
```bash
# ResNet-50
python scripts/train_deep.py --model resnet50 --epochs 50

# EfficientNet-B0
python scripts/train_deep.py --model efficientnet_b0 --epochs 50

# Vision Transformer
python scripts/train_deep.py --model vit_b_16 --epochs 50

# With custom parameters
python scripts/train_deep.py \
    --model resnet50 \
    --image_size 128 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --optimizer adam \
    --augmentation
```

### Run Ablation Studies
```bash
# Image size (64x64 vs 128x128)
python scripts/ablation_studies.py --study image_size --model resnet50

# Data augmentation (with vs without)
python scripts/ablation_studies.py --study augmentation --model resnet50

# Feature extractor (HOG vs CNN)
python scripts/ablation_studies.py --study feature_extractor

# Optimizer (SGD vs Adam)
python scripts/ablation_studies.py --study optimizer --model resnet50

# Run all ablations
python scripts/ablation_studies.py --study all --model resnet50
```

### Evaluate Models
```bash
# Evaluate specific model
python scripts/evaluate.py --model_path models/resnet50_best.pth

# Evaluate all models
python scripts/evaluate.py --model_dir models --generate_report

# Evaluate only deep learning models
python scripts/evaluate.py --model_type deep --generate_report
```

---

## 📊 Analysis with Jupyter Notebooks

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```
- Dataset statistics
- Class distribution
- Sample visualizations
- Augmentation demonstrations

### Model Comparison
```bash
jupyter notebook notebooks/model_comparison.ipynb
```
- Performance metrics comparison
- Confusion matrices
- Training curves
- Ablation results analysis

---

## 📁 Project Structure

```
caltech101-classification/
├── data/                           # Dataset directory
│   └── 101_ObjectCategories/      # Caltech-101 images
│
├── models/                         # Saved model checkpoints
│   ├── resnet50_best.pth
│   ├── efficientnet_b0_best.pth
│   ├── vit_b_16_best.pth
│   └── hog_svm_classifier.pkl
│
├── results/                        # Evaluation results
│   ├── metrics/                   # JSON metrics files
│   │   ├── resnet50_test_metrics.json
│   │   └── evaluation_summary.json
│   └── figures/                   # Visualization plots
│       ├── confusion_matrices/
│       ├── training_histories/
│       └── model_comparison.png
│
├── notebooks/                      # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
│
├── scripts/                        # Training and evaluation
│   ├── download_data.py
│   ├── train_classical.py
│   ├── train_deep.py
│   ├── ablation_studies.py
│   └── evaluate.py
│
├── src/                           # Source code
│   ├── data/
│   │   ├── dataset.py            # Dataset loading
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── classical_ml.py       # HOG+SVM, etc.
│   │   └── deep_learning.py      # ResNet, EfficientNet, ViT
│   └── utils/
│       ├── evaluation.py         # Metrics computation
│       └── visualization.py      # Plotting functions
│
├── README.md                      # Main documentation
├── REPORT_TEMPLATE.md            # Report template
├── requirements.txt              # Python dependencies
├── run_all.sh                    # Complete pipeline
└── quick_start.sh                # Quick test script
```

---

## 🎯 Expected Outcomes

### Models Performance (Approximate)
| Model | Accuracy | Training Time | Inference Speed |
|-------|----------|---------------|-----------------|
| HOG+SVM | ~40-50% | Fast (~30 min) | Very Fast |
| ResNet-50 | ~70-80% | Medium (~2-3 hrs) | Fast |
| EfficientNet-B0 | ~75-85% | Medium (~2-3 hrs) | Fast |
| ViT-B/16 | ~75-85% | Slow (~3-4 hrs) | Medium |

### Ablation Study Insights
- **Image Size**: Larger images (128×128) typically improve accuracy by 5-10%
- **Data Augmentation**: Can improve accuracy by 3-7% and reduce overfitting
- **Feature Extractor**: CNN features significantly outperform HOG features
- **Optimizer**: Adam typically converges faster than SGD for these tasks

---

## 🔧 Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python scripts/train_deep.py --model resnet50 --batch_size 16

# Use smaller model
python scripts/train_deep.py --model resnet18 --batch_size 32
```

### Slow Training
```bash
# Reduce epochs for testing
python scripts/train_deep.py --model resnet50 --epochs 10

# Use CPU if GPU issues
python scripts/train_deep.py --model resnet50 --device cpu
```

### Dataset Not Found
```bash
# Check dataset location
ls data/101_ObjectCategories/

# Re-run download script
python scripts/download_data.py --data_dir data
```

---

## 📝 Report Writing

1. Run all experiments
2. Collect results from `results/metrics/`
3. Open `REPORT_TEMPLATE.md`
4. Fill in the template with your results
5. Add visualizations from `results/figures/`
6. Include insights from ablation studies
7. Write observations and conclusions

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different models
- Add new ablation studies
- Improve evaluation metrics
- Enhance visualizations

---

## 📚 Additional Resources

### Papers
- [Caltech-101 Dataset](https://authors.library.caltech.edu/7694/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the README.md
3. Examine the code comments
4. Consult the Jupyter notebooks

---

**Good luck with your image classification project! 🎉**
