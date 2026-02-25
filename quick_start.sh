#!/bin/bash

# Quick Start Script - Train a single model quickly for testing
# This script trains ResNet-50 with reduced epochs for faster testing

echo "========================================"
echo "Quick Start: Training ResNet-50"
echo "========================================"
echo ""

# Check if data exists
if [ ! -d "data/101_ObjectCategories" ]; then
    echo "Dataset not found. Please run:"
    echo "  python scripts/download_data.py"
    exit 1
fi

# Train ResNet-50 with reduced epochs
python scripts/train_deep.py \
    --model resnet50 \
    --data_dir data/101_ObjectCategories \
    --image_size 128 \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --optimizer adam \
    --augmentation

echo ""
echo "========================================"
echo "Training completed!"
echo "Check results/ directory for outputs"
echo "========================================"
