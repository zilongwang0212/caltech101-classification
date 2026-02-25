"""
Deep Learning Models for Image Classification
ResNet, EfficientNet, Vision Transformer (ViT), and training utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import numpy as np

import torchvision.models as models
from torchvision.models import (
    resnet18, resnet50, resnet101,
    efficientnet_b0, efficientnet_b3,
    vit_b_16, vit_b_32,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
    EfficientNet_B0_Weights, EfficientNet_B3_Weights,
    ViT_B_16_Weights, ViT_B_32_Weights
)


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Get a deep learning model for image classification.
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet50', 'efficientnet_b0', 'vit_b_16', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        PyTorch model
    """
    
    # ResNet models
    if model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet101(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    # EfficientNet models
    elif model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'efficientnet_b3':
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b3(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    # Vision Transformer models
    elif model_name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    
    elif model_name == 'vit_b_32':
        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_32(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Freeze backbone if requested
    if freeze_backbone and pretrained:
        for name, param in model.named_parameters():
            # Don't freeze the final classification layer
            if 'fc' not in name and 'classifier' not in name and 'heads' not in name:
                param.requires_grad = False
    
    return model


class DeepLearningTrainer:
    """
    Trainer for deep learning models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.001,
        optimizer: str = 'adam',
        scheduler: Optional[str] = 'cosine',
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: PyTorch model
            device: Device to train on
            lr: Learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler: Learning rate scheduler ('cosine', 'step', None)
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        
        # Initialize optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Initialize scheduler
        self.scheduler = None
        if scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50
            )
        elif scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_dir: str = 'models',
        model_name: str = 'model'
    ) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            model_name: Name for saved model
        
        Returns:
            Training history dictionary
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"\nTraining {model_name} for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                save_path = os.path.join(save_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, save_path)
                print(f"✓ Best model saved (Val Acc: {val_acc:.4f})")
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return self.history
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def predict_proba(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities.
        
        Returns:
            Tuple of (probabilities, true_labels)
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Computing probabilities"):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_probs), np.array(all_labels)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Checkpoint loaded from {checkpoint_path}")
