"""
Train deep learning models (ResNet, EfficientNet, ViT) on Caltech-101
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.models.deep_learning import get_model, DeepLearningTrainer
from src.utils.evaluation import compute_metrics, print_metrics, save_metrics
from src.utils.visualization import plot_confusion_matrix, plot_training_history

import torch


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models on Caltech-101')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/101_ObjectCategories',
                        help='Path to Caltech-101 dataset')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 
                                'efficientnet_b0', 'efficientnet_b3',
                                'vit_b_16', 'vit_b_32'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Handle augmentation flag
    if args.no_augmentation:
        args.augmentation = False
    
    # Adjust scheduler
    if args.scheduler == 'none':
        args.scheduler = None
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    
    print("\n" + "="*80)
    print("Training Deep Learning Model on Caltech-101")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Image Size: {args.image_size}x{args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augmentation=args.augmentation,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    
    # Initialize trainer
    trainer = DeepLearningTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
        model_name=args.model
    )
    
    # Plot training history
    print("\nPlotting training history...")
    history_path = os.path.join(args.results_dir, 'figures', f"{args.model}_training_history.png")
    plot_training_history(history, save_path=history_path)
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
    trainer.load_checkpoint(best_model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_preds, test_labels = trainer.predict(test_loader)
    test_proba, _ = trainer.predict_proba(test_loader)
    
    test_metrics = compute_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_proba=test_proba,
        class_names=class_names
    )
    
    print_metrics(test_metrics, title=f"{args.model.upper()} - Test Metrics")
    
    # Save metrics
    metrics_path = os.path.join(args.results_dir, 'metrics', f"{args.model}_test_metrics.json")
    save_metrics(test_metrics, metrics_path)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(args.results_dir, 'figures', f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        class_names=class_names,
        save_path=cm_path,
        normalize=True,
        title=f'{args.model.upper()} - Confusion Matrix'
    )
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Figures saved to: {args.results_dir}/figures/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
