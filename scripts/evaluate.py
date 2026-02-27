"""
Evaluate trained models and generate comprehensive reports
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
from src.utils.evaluation import (
    compute_metrics, print_metrics, save_metrics,
    compare_models, analyze_errors, get_classification_report
)
from src.utils.visualization import (
    plot_confusion_matrix, plot_per_class_accuracy, plot_model_comparison
)

import torch
import numpy as np


def evaluate_deep_learning_model(model_path: str, args):
    """Evaluate a deep learning model."""
    print(f"\nEvaluating deep learning model: {model_path}")
    
    # Extract model name from path
    model_name = Path(model_path).stem.replace('_best', '')
    
    # Create dataloaders
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augmentation=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    # For ViT and some models, keep full name; for others, extract first part
    if 'vit' in model_name or 'efficientnet' in model_name:
        model_arch = model_name
    else:
        model_arch = model_name.split('_')[0]
    
    model = get_model(
        model_name=model_arch,
        num_classes=len(class_names),
        pretrained=False
    )
    
    # Initialize trainer and load checkpoint
    trainer = DeepLearningTrainer(model=model, device=args.device)
    trainer.load_checkpoint(model_path)
    
    # Make predictions
    test_preds, test_labels = trainer.predict(test_loader)
    test_proba, _ = trainer.predict_proba(test_loader)
    
    # Compute metrics
    metrics = compute_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_proba=test_proba,
        class_names=class_names
    )
    
    # Print metrics
    print_metrics(metrics, title=f"{model_name.upper()} Evaluation")
    
    return metrics, test_preds, test_labels, class_names, model_name


def evaluate_classical_model(model_path: str, args):
    """Evaluate a classical ML model."""
    print(f"\nEvaluating classical model: {model_path}")
    
    # Extract model name from path
    model_name = Path(model_path).stem.replace('_classifier', '')
    
    # Create dataloaders
    _, _, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=64,  # Classical models typically use smaller size
        batch_size=args.batch_size,
        augmentation=False,
        num_workers=args.num_workers
    )
    
    # Load model
    classifier = ClassicalMLClassifier.load(model_path)
    
    # Make predictions
    test_preds, test_labels = classifier.predict(test_loader)
    test_proba, _ = classifier.predict_proba(test_loader)
    
    # Compute metrics
    metrics = compute_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_proba=test_proba,
        class_names=class_names
    )
    
    # Print metrics
    print_metrics(metrics, title=f"{model_name.upper()} Evaluation")
    
    return metrics, test_preds, test_labels, class_names, model_name


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on Caltech-101')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to specific model checkpoint')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models (evaluate all)')
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['deep', 'classical', 'all'],
                        help='Type of models to evaluate')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/101_ObjectCategories',
                        help='Path to Caltech-101 dataset')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate comprehensive evaluation report')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'figures'), exist_ok=True)
    
    print("\n" + "="*80)
    print("Model Evaluation on Caltech-101")
    print("="*80 + "\n")
    
    all_metrics = {}
    all_results = {}
    
    # Evaluate specific model
    if args.model_path:
        if args.model_path.endswith('.pth'):
            metrics, preds, labels, class_names, model_name = evaluate_deep_learning_model(args.model_path, args)
        elif args.model_path.endswith('.pkl'):
            metrics, preds, labels, class_names, model_name = evaluate_classical_model(args.model_path, args)
        else:
            print(f"Unsupported model format: {args.model_path}")
            return
        
        all_metrics[model_name] = metrics
        all_results[model_name] = (preds, labels, class_names)
    
    # Evaluate all models in directory
    else:
        model_dir = Path(args.model_dir)
        
        # Find all model files
        if args.model_type in ['deep', 'all']:
            deep_models = list(model_dir.glob('*_best.pth'))
            for model_path in deep_models:
                try:
                    metrics, preds, labels, class_names, model_name = evaluate_deep_learning_model(
                        str(model_path), args
                    )
                    all_metrics[model_name] = metrics
                    all_results[model_name] = (preds, labels, class_names)
                except Exception as e:
                    print(f"Error evaluating {model_path}: {e}")
        
        if args.model_type in ['classical', 'all']:
            classical_models = list(model_dir.glob('*_classifier.pkl'))
            for model_path in classical_models:
                try:
                    metrics, preds, labels, class_names, model_name = evaluate_classical_model(
                        str(model_path), args
                    )
                    all_metrics[model_name] = metrics
                    all_results[model_name] = (preds, labels, class_names)
                except Exception as e:
                    print(f"Error evaluating {model_path}: {e}")
    
    # Compare all models
    if len(all_metrics) > 1:
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        compare_models(all_metrics)
        
        # Plot comparison
        plot_model_comparison(
            all_metrics,
            save_path=os.path.join(args.results_dir, 'figures', 'model_comparison.png')
        )
    
    # Generate detailed reports for each model
    if args.generate_report:
        print("\n" + "="*80)
        print("Generating Detailed Reports")
        print("="*80)
        
        for model_name, (preds, labels, class_names) in all_results.items():
            print(f"\nGenerating report for: {model_name}")
            
            # Save metrics
            metrics_path = os.path.join(args.results_dir, 'metrics', f'{model_name}_detailed_metrics.json')
            save_metrics(all_metrics[model_name], metrics_path)
            
            # Plot confusion matrix
            cm_path = os.path.join(args.results_dir, 'figures', f'{model_name}_confusion_matrix_normalized.png')
            plot_confusion_matrix(
                y_true=labels,
                y_pred=preds,
                class_names=class_names,
                save_path=cm_path,
                normalize=True,
                title=f'{model_name.upper()} - Confusion Matrix'
            )
            
            # Plot per-class accuracy
            if 'per_class_metrics' in all_metrics[model_name]:
                pca_path = os.path.join(args.results_dir, 'figures', f'{model_name}_per_class_accuracy.png')
                plot_per_class_accuracy(
                    all_metrics[model_name]['per_class_metrics'],
                    save_path=pca_path
                )
            
            # Error analysis
            error_analysis = analyze_errors(labels, preds, class_names, top_n=10)
            error_path = os.path.join(args.results_dir, 'metrics', f'{model_name}_error_analysis.json')
            with open(error_path, 'w') as f:
                json.dump(error_analysis, f, indent=4)
            
            # Classification report
            report = get_classification_report(labels, preds, class_names)
            report_path = os.path.join(args.results_dir, 'metrics', f'{model_name}_classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
    
    # Save comparison summary
    if all_metrics:
        summary_path = os.path.join(args.results_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"\nEvaluation summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("Evaluation Completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
