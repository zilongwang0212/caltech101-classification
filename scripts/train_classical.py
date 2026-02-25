"""
Train classical machine learning models (HOG + SVM, HOG + Random Forest)
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.classical_ml import ClassicalMLClassifier
from src.utils.evaluation import compute_metrics, print_metrics, save_metrics
from src.utils.visualization import plot_confusion_matrix

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Train classical ML models on Caltech-101')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/101_ObjectCategories',
                        help='Path to Caltech-101 dataset')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size for feature extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Model parameters
    parser.add_argument('--feature_type', type=str, default='hog',
                        choices=['hog', 'cnn'],
                        help='Feature extraction method')
    parser.add_argument('--classifier', type=str, default='svm',
                        choices=['svm', 'linear_svm', 'random_forest'],
                        help='Classifier type')
    parser.add_argument('--cnn_model', type=str, default='resnet50',
                        help='CNN model for feature extraction (if feature_type=cnn)')
    
    # Classifier hyperparameters
    parser.add_argument('--C', type=float, default=10.0,
                        help='Regularization parameter for SVM')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees for Random Forest')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    
    print("\n" + "="*80)
    print("Training Classical ML Model on Caltech-101")
    print("="*80)
    print(f"Feature Type: {args.feature_type.upper()}")
    print(f"Classifier: {args.classifier.upper()}")
    print(f"Image Size: {args.image_size}x{args.image_size}")
    print("="*80 + "\n")
    
    # Create dataloaders (no augmentation for classical ML)
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augmentation=False,  # No augmentation for classical ML
        num_workers=4
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Initialize classifier
    print(f"\nInitializing {args.feature_type.upper()} + {args.classifier.upper()} classifier...")
    
    classifier = ClassicalMLClassifier(
        feature_type=args.feature_type,
        classifier_type=args.classifier,
        image_size=args.image_size,
        cnn_model=args.cnn_model,
        C=args.C,
        n_estimators=args.n_estimators
    )
    
    # Train classifier
    classifier.fit(train_loader)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_preds, val_labels = classifier.predict(val_loader)
    val_proba, _ = classifier.predict_proba(val_loader) if args.classifier != 'linear_svm' else (None, None)
    
    val_metrics = compute_metrics(
        y_true=val_labels,
        y_pred=val_preds,
        y_proba=val_proba,
        class_names=class_names
    )
    
    print_metrics(val_metrics, title="Validation Metrics")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_preds, test_labels = classifier.predict(test_loader)
    test_proba, _ = classifier.predict_proba(test_loader) if args.classifier != 'linear_svm' else (None, None)
    
    test_metrics = compute_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_proba=test_proba,
        class_names=class_names
    )
    
    print_metrics(test_metrics, title="Test Metrics")
    
    # Save model
    model_name = f"{args.feature_type}_{args.classifier}"
    model_path = os.path.join(args.save_dir, f"{model_name}_classifier.pkl")
    classifier.save(model_path)
    
    # Save metrics
    metrics_path = os.path.join(args.results_dir, 'metrics', f"{model_name}_test_metrics.json")
    save_metrics(test_metrics, metrics_path)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(args.results_dir, 'figures', f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        class_names=class_names,
        save_path=cm_path,
        normalize=True,
        title=f'{args.feature_type.upper()} + {args.classifier.upper()} - Confusion Matrix'
    )
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
