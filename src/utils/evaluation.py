"""
Evaluation Metrics and Analysis Tools
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: List[str] = None,
    top_k: int = 5
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for top-k accuracy)
        class_names: List of class names
        top_k: K for top-k accuracy
    
    Returns:
        Dictionary containing all metrics
    """
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-Score (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Per-class accuracy
    conf_matrix = confusion_matrix(y_true, y_pred)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Top-k accuracy (if probabilities provided)
    top_k_acc = None
    if y_proba is not None:
        try:
            top_k_acc = top_k_accuracy_score(y_true, y_proba, k=top_k)
        except:
            pass
    
    # Build metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
    }
    
    if top_k_acc is not None:
        metrics[f'top_{top_k}_accuracy'] = float(top_k_acc)
    
    # Add per-class metrics
    if class_names:
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'accuracy': float(per_class_accuracy[i]) if i < len(per_class_accuracy) else 0.0,
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
                'support': int(support_per_class[i]) if i < len(support_per_class) else 0
            }
        metrics['per_class_metrics'] = per_class_metrics
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics") -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Metrics dictionary
        title: Title for the printout
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    
    if 'top_5_accuracy' in metrics:
        print(f"  Top-5 Accuracy:    {metrics['top_5_accuracy']:.4f}")
    
    print(f"\n  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    
    print(f"\n  Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")
    
    # Per-class summary (top 5 best and worst)
    if 'per_class_metrics' in metrics:
        per_class = metrics['per_class_metrics']
        
        # Sort by F1-score
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        print("\n📈 Top 5 Best Performing Classes (by F1-Score):")
        for i, (class_name, class_metrics) in enumerate(sorted_classes[:5], 1):
            print(f"  {i}. {class_name:30s} - F1: {class_metrics['f1_score']:.4f}, "
                  f"Acc: {class_metrics['accuracy']:.4f}, Support: {class_metrics['support']}")
        
        print("\n📉 Top 5 Worst Performing Classes (by F1-Score):")
        for i, (class_name, class_metrics) in enumerate(sorted_classes[-5:][::-1], 1):
            print(f"  {i}. {class_name:30s} - F1: {class_metrics['f1_score']:.4f}, "
                  f"Acc: {class_metrics['accuracy']:.4f}, Support: {class_metrics['support']}")
    
    print("\n" + "=" * 80 + "\n")


def save_metrics(metrics: Dict, save_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save JSON file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


def load_metrics(load_path: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        load_path: Path to JSON file
    
    Returns:
        Metrics dictionary
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def compare_models(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str] = None
) -> None:
    """
    Compare multiple models side by side.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        metric_names: List of metric names to compare (default: main metrics)
    """
    if metric_names is None:
        metric_names = [
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro',
            'top_5_accuracy'
        ]
    
    print("\n" + "=" * 100)
    print(f"{'Model Comparison':^100}")
    print("=" * 100)
    
    # Header
    header = f"{'Model':<25}"
    for metric in metric_names:
        header += f"{metric:>15}"
    print(header)
    print("-" * 100)
    
    # Rows for each model
    for model_name, metrics in metrics_dict.items():
        row = f"{model_name:<25}"
        for metric in metric_names:
            if metric in metrics:
                row += f"{metrics[metric]:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 100 + "\n")
    
    # Find best model for each metric
    print("🏆 Best Performance by Metric:")
    for metric in metric_names:
        best_model = None
        best_value = -1
        
        for model_name, metrics in metrics_dict.items():
            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                best_model = model_name
        
        if best_model:
            print(f"  {metric:25s}: {best_model:25s} ({best_value:.4f})")
    
    print()


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = 10
) -> Dict:
    """
    Analyze common misclassification patterns.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_n: Number of top errors to return
    
    Returns:
        Dictionary with error analysis
    """
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Find most common misclassifications
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i, j] > 0:
                errors.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(conf_matrix[i, j]),
                    'true_class_idx': i,
                    'predicted_class_idx': j
                })
    
    # Sort by count
    errors = sorted(errors, key=lambda x: x['count'], reverse=True)
    
    # Get top N errors
    top_errors = errors[:top_n]
    
    # Print error analysis
    print("\n" + "=" * 100)
    print(f"{'Error Analysis - Most Common Misclassifications':^100}")
    print("=" * 100)
    
    print(f"\n{'Rank':<6}{'True Class':<30}{'Predicted As':<30}{'Count':<10}")
    print("-" * 100)
    
    for i, error in enumerate(top_errors, 1):
        print(f"{i:<6}{error['true_class']:<30}{error['predicted_class']:<30}{error['count']:<10}")
    
    print("=" * 100 + "\n")
    
    return {
        'all_errors': errors,
        'top_errors': top_errors,
        'total_misclassifications': int(np.sum(y_true != y_pred))
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    return report
