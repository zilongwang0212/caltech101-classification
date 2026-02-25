"""
Visualization Tools for Model Evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (20, 18),
    normalize: bool = False,
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,  # Don't annotate all cells (too many for 101 classes)
        fmt=fmt,
        cmap='Blues',
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_per_class_accuracy(
    per_class_metrics: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    top_n: int = 20
) -> None:
    """
    Plot per-class accuracy as a bar chart.
    
    Args:
        per_class_metrics: Dictionary mapping class names to their metrics
        save_path: Path to save figure
        figsize: Figure size
        top_n: Show top N best and worst classes
    """
    # Extract class names and accuracies
    classes = list(per_class_metrics.keys())
    accuracies = [per_class_metrics[c]['accuracy'] for c in classes]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    
    # Get top N best and worst
    worst_indices = sorted_indices[:top_n]
    best_indices = sorted_indices[-top_n:]
    selected_indices = np.concatenate([worst_indices, best_indices])
    
    selected_classes = [classes[i] for i in selected_indices]
    selected_accuracies = [accuracies[i] for i in selected_indices]
    
    # Create color map (red for worst, green for best)
    colors = ['red'] * top_n + ['green'] * top_n
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(selected_classes))
    bars = ax.barh(y_pos, selected_accuracies, color=colors, alpha=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected_classes, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-Class Accuracy (Top {top_n} Best and Worst)', fontsize=14)
    ax.axvline(x=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot comparison of multiple models across different metrics.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        metric_names: List of metric names to compare
        save_path: Path to save figure
        figsize: Figure size
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Extract data
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)
    
    # Prepare data matrix
    data = np.zeros((n_models, n_metrics))
    for i, model_name in enumerate(model_names):
        for j, metric_name in enumerate(metric_names):
            if metric_name in metrics_dict[model_name]:
                data[i, j] = metrics_dict[model_name][metric_name]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, data[i], width, label=model_name, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison Across Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def plot_ablation_results(
    ablation_results: Dict[str, float],
    study_name: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Dictionary mapping experiment names to accuracy scores
        study_name: Name of the ablation study
        save_path: Path to save figure
        figsize: Figure size
    """
    experiments = list(ablation_results.keys())
    scores = list(ablation_results.values())
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(experiments)), scores, color='steelblue', alpha=0.7)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            score,
            f'{score:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Ablation Study: {study_name}', fontsize=14)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add baseline line (if "baseline" in results)
    if 'baseline' in [e.lower() for e in experiments]:
        baseline_idx = [e.lower() for e in experiments].index('baseline')
        baseline_score = scores[baseline_idx]
        ax.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_score:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation study plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_distribution: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8)
) -> None:
    """
    Plot class distribution histogram.
    
    Args:
        class_distribution: Dictionary mapping class names to sample counts
        save_path: Path to save figure
        figsize: Figure size
    """
    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(len(sorted_classes)), sorted_counts, color='coral', alpha=0.7)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14)
    ax.axhline(y=np.mean(sorted_counts), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sorted_counts):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Show only every Nth class label to avoid clutter
    step = max(1, len(sorted_classes) // 20)
    ax.set_xticks(range(0, len(sorted_classes), step))
    ax.set_xticklabels([sorted_classes[i] for i in range(0, len(sorted_classes), step)], rotation=90, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
