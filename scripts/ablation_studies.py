"""
Run ablation studies for deep learning models
Studies include:
1. Image size comparison (64x64 vs 128x128)
2. Data augmentation comparison (with vs without)
3. Feature extractor comparison (HOG vs CNN features for classical ML)
4. Optimizer comparison (SGD vs Adam for deep learning)
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.deep_learning import get_model, DeepLearningTrainer
from src.models.classical_ml import ClassicalMLClassifier
from src.utils.evaluation import compute_metrics, save_metrics
from src.utils.visualization import plot_ablation_results

import torch


def ablation_image_size(args):
    """Ablation study: Compare different image sizes."""
    print("\n" + "="*80)
    print("ABLATION STUDY: Image Size Comparison")
    print("="*80 + "\n")
    
    image_sizes = [64, 128]
    results = {}
    
    for img_size in image_sizes:
        print(f"\n--- Training with image size: {img_size}x{img_size} ---")
        
        # Create dataloaders
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir=args.data_dir,
            image_size=img_size,
            batch_size=args.batch_size,
            augmentation=True,
            num_workers=args.num_workers
        )
        
        # Initialize model
        model = get_model(
            model_name=args.model,
            num_classes=len(class_names),
            pretrained=True
        )
        
        # Train
        trainer = DeepLearningTrainer(
            model=model,
            device=args.device,
            lr=args.lr,
            optimizer='adam'
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=args.save_dir,
            model_name=f"{args.model}_size{img_size}"
        )
        
        # Evaluate
        test_preds, test_labels = trainer.predict(test_loader)
        test_proba, _ = trainer.predict_proba(test_loader)
        
        metrics = compute_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            class_names=class_names
        )
        
        results[f"{img_size}x{img_size}"] = metrics['accuracy']
        
        # Save metrics
        save_metrics(
            metrics,
            os.path.join(args.results_dir, 'metrics', f"ablation_size{img_size}_metrics.json")
        )
    
    # Plot results
    plot_ablation_results(
        results,
        study_name="Image Size Comparison",
        save_path=os.path.join(args.results_dir, 'figures', 'ablation_image_size.png')
    )
    
    return results


def ablation_augmentation(args):
    """Ablation study: Compare with and without data augmentation."""
    print("\n" + "="*80)
    print("ABLATION STUDY: Data Augmentation Comparison")
    print("="*80 + "\n")
    
    augmentation_configs = [False, True]
    config_names = ["Without Augmentation", "With Augmentation"]
    results = {}
    
    for aug, name in zip(augmentation_configs, config_names):
        print(f"\n--- Training {name} ---")
        
        # Create dataloaders
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            augmentation=aug,
            num_workers=args.num_workers
        )
        
        # Initialize model
        model = get_model(
            model_name=args.model,
            num_classes=len(class_names),
            pretrained=True
        )
        
        # Train
        trainer = DeepLearningTrainer(
            model=model,
            device=args.device,
            lr=args.lr,
            optimizer='adam'
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=args.save_dir,
            model_name=f"{args.model}_aug{aug}"
        )
        
        # Evaluate
        test_preds, test_labels = trainer.predict(test_loader)
        test_proba, _ = trainer.predict_proba(test_loader)
        
        metrics = compute_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            class_names=class_names
        )
        
        results[name] = metrics['accuracy']
        
        # Save metrics
        save_metrics(
            metrics,
            os.path.join(args.results_dir, 'metrics', f"ablation_aug{aug}_metrics.json")
        )
    
    # Plot results
    plot_ablation_results(
        results,
        study_name="Data Augmentation Comparison",
        save_path=os.path.join(args.results_dir, 'figures', 'ablation_augmentation.png')
    )
    
    return results


def ablation_feature_extractor(args):
    """Ablation study: Compare HOG vs CNN features for classical ML."""
    print("\n" + "="*80)
    print("ABLATION STUDY: Feature Extractor Comparison (HOG vs CNN)")
    print("="*80 + "\n")
    
    feature_types = ['hog', 'cnn']
    results = {}
    
    # Create dataloaders (no augmentation for classical ML)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=64,  # Smaller size for faster processing
        batch_size=args.batch_size,
        augmentation=False,
        num_workers=args.num_workers
    )
    
    for feature_type in feature_types:
        print(f"\n--- Training with {feature_type.upper()} features ---")
        
        # Initialize classifier
        classifier = ClassicalMLClassifier(
            feature_type=feature_type,
            classifier_type='svm',
            image_size=64
        )
        
        # Train
        classifier.fit(train_loader)
        
        # Evaluate
        test_preds, test_labels = classifier.predict(test_loader)
        test_proba, _ = classifier.predict_proba(test_loader)
        
        metrics = compute_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            class_names=class_names
        )
        
        results[f"{feature_type.upper()} Features"] = metrics['accuracy']
        
        # Save metrics
        save_metrics(
            metrics,
            os.path.join(args.results_dir, 'metrics', f"ablation_{feature_type}_metrics.json")
        )
    
    # Plot results
    plot_ablation_results(
        results,
        study_name="Feature Extractor Comparison",
        save_path=os.path.join(args.results_dir, 'figures', 'ablation_feature_extractor.png')
    )
    
    return results


def ablation_optimizer(args):
    """Ablation study: Compare SGD vs Adam optimizers."""
    print("\n" + "="*80)
    print("ABLATION STUDY: Optimizer Comparison (SGD vs Adam)")
    print("="*80 + "\n")
    
    optimizers = ['sgd', 'adam']
    results = {}
    
    for optimizer in optimizers:
        print(f"\n--- Training with {optimizer.upper()} optimizer ---")
        
        # Create dataloaders
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            augmentation=True,
            num_workers=args.num_workers
        )
        
        # Initialize model
        model = get_model(
            model_name=args.model,
            num_classes=len(class_names),
            pretrained=True
        )
        
        # Train
        trainer = DeepLearningTrainer(
            model=model,
            device=args.device,
            lr=args.lr,
            optimizer=optimizer
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=args.save_dir,
            model_name=f"{args.model}_{optimizer}"
        )
        
        # Evaluate
        test_preds, test_labels = trainer.predict(test_loader)
        test_proba, _ = trainer.predict_proba(test_loader)
        
        metrics = compute_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            class_names=class_names
        )
        
        results[optimizer.upper()] = metrics['accuracy']
        
        # Save metrics
        save_metrics(
            metrics,
            os.path.join(args.results_dir, 'metrics', f"ablation_{optimizer}_metrics.json")
        )
    
    # Plot results
    plot_ablation_results(
        results,
        study_name="Optimizer Comparison",
        save_path=os.path.join(args.results_dir, 'figures', 'ablation_optimizer.png')
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies on Caltech-101')
    
    # Study selection
    parser.add_argument('--study', type=str, required=True,
                        choices=['image_size', 'augmentation', 'feature_extractor', 'optimizer', 'all'],
                        help='Which ablation study to run')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/101_ObjectCategories',
                        help='Path to Caltech-101 dataset')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model architecture for deep learning studies')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (reduced for ablation)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models/ablation',
                        help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='results/ablation',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    
    # Run selected study
    all_results = {}
    
    if args.study == 'image_size' or args.study == 'all':
        results = ablation_image_size(args)
        all_results['image_size'] = results
    
    if args.study == 'augmentation' or args.study == 'all':
        results = ablation_augmentation(args)
        all_results['augmentation'] = results
    
    if args.study == 'feature_extractor' or args.study == 'all':
        results = ablation_feature_extractor(args)
        all_results['feature_extractor'] = results
    
    if args.study == 'optimizer' or args.study == 'all':
        results = ablation_optimizer(args)
        all_results['optimizer'] = results
    
    # Save all results
    summary_path = os.path.join(args.results_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*80)
    print("Ablation Studies Completed!")
    print(f"Summary saved to: {summary_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
