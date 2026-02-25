#!/bin/bash

# Caltech-101 Image Classification - Complete Workflow Script
# This script demonstrates the full pipeline from data download to evaluation

set -e  # Exit on any error

echo "========================================="
echo "Caltech-101 Classification Pipeline"
echo "========================================="
echo ""

# Step 1: Download and prepare dataset
echo "Step 1: Download Dataset"
echo "-----------------------------------------"
python scripts/download_data.py --data_dir data
echo ""

# Step 2: Train classical ML model
echo "Step 2: Train Classical ML Model (HOG + SVM)"
echo "-----------------------------------------"
python scripts/train_classical.py \
    --data_dir data/101_ObjectCategories \
    --feature_type hog \
    --classifier svm \
    --image_size 64 \
    --batch_size 32
echo ""

# Step 3: Train deep learning models
echo "Step 3: Train Deep Learning Models"
echo "-----------------------------------------"

# ResNet-50
echo "Training ResNet-50..."
python scripts/train_deep.py \
    --model resnet50 \
    --data_dir data/101_ObjectCategories \
    --image_size 128 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --optimizer adam \
    --augmentation
echo ""

# EfficientNet-B0
echo "Training EfficientNet-B0..."
python scripts/train_deep.py \
    --model efficientnet_b0 \
    --data_dir data/101_ObjectCategories \
    --image_size 128 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --optimizer adam \
    --augmentation
echo ""

# Vision Transformer
echo "Training ViT-B/16..."
python scripts/train_deep.py \
    --model vit_b_16 \
    --data_dir data/101_ObjectCategories \
    --image_size 128 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --optimizer adam \
    --augmentation
echo ""

# Step 4: Run ablation studies
echo "Step 4: Ablation Studies"
echo "-----------------------------------------"

# Image size ablation
echo "Running image size ablation..."
python scripts/ablation_studies.py \
    --study image_size \
    --model resnet50 \
    --data_dir data/101_ObjectCategories \
    --epochs 30
echo ""

# Data augmentation ablation
echo "Running data augmentation ablation..."
python scripts/ablation_studies.py \
    --study augmentation \
    --model resnet50 \
    --data_dir data/101_ObjectCategories \
    --epochs 30
echo ""

# Feature extractor ablation
echo "Running feature extractor ablation..."
python scripts/ablation_studies.py \
    --study feature_extractor \
    --data_dir data/101_ObjectCategories
echo ""

# Optimizer ablation
echo "Running optimizer ablation..."
python scripts/ablation_studies.py \
    --study optimizer \
    --model resnet50 \
    --data_dir data/101_ObjectCategories \
    --epochs 30
echo ""

# Step 5: Comprehensive evaluation
echo "Step 5: Model Evaluation"
echo "-----------------------------------------"
python scripts/evaluate.py \
    --model_dir models \
    --data_dir data/101_ObjectCategories \
    --image_size 128 \
    --batch_size 32 \
    --generate_report
echo ""

echo "========================================="
echo "Pipeline Completed Successfully!"
echo "========================================="
echo ""
echo "Results are saved in:"
echo "  - models/          : Trained model checkpoints"
echo "  - results/metrics/ : Evaluation metrics (JSON)"
echo "  - results/figures/ : Visualizations (PNG)"
echo ""
echo "Next steps:"
echo "  1. Open notebooks/model_comparison.ipynb for detailed analysis"
echo "  2. Review results/evaluation_summary.json"
echo "  3. Fill in REPORT_TEMPLATE.md with your findings"
echo ""
