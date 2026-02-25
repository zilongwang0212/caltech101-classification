"""
Caltech-101 Dataset Loading and Preprocessing
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split


class Caltech101Dataset(Dataset):
    """
    Custom Dataset class for Caltech-101.
    
    Args:
        root_dir: Root directory containing Caltech-101 images
        image_size: Target image size (height, width)
        transform: Optional transform to be applied
        subset: One of 'train', 'val', 'test'
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 128,
        transform: Optional[Callable] = None,
        subset: str = 'train'
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.subset = subset
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load all image paths and corresponding labels."""
        # Get all category directories
        categories = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Filter out background and faces_easy (optional)
        categories = [c for c in categories if c.name != 'BACKGROUND_Google']
        
        self.class_names = [c.name for c in categories]
        
        for label, category in enumerate(categories):
            image_files = list(category.glob('*.jpg')) + list(category.glob('*.png'))
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} categories")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.Resize((self.image_size, self.image_size))(image)
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        return self.class_names


def get_transforms(image_size: int = 128, augmentation: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms.
    
    Args:
        image_size: Target image size
        augmentation: Whether to apply data augmentation for training
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    
    # Test/validation transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if augmentation:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Training transforms without augmentation
        train_transform = test_transform
    
    return train_transform, test_transform


def create_dataloaders(
    data_dir: str,
    image_size: int = 128,
    batch_size: int = 32,
    augmentation: bool = True,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders with stratified splitting.
    
    Args:
        data_dir: Path to Caltech-101 dataset
        image_size: Target image size
        batch_size: Batch size
        augmentation: Whether to use data augmentation
        train_split: Training split ratio (default: 0.7)
        val_split: Validation split ratio (default: 0.15)
        test_split: Test split ratio (default: 0.15)
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    
    assert train_split + val_split + test_split == 1.0, "Splits must sum to 1.0"
    
    # Get transforms
    train_transform, test_transform = get_transforms(image_size, augmentation)
    
    # Load full dataset
    full_dataset = Caltech101Dataset(
        root_dir=data_dir,
        image_size=image_size,
        transform=None  # We'll apply transforms after splitting
    )
    
    # Get labels for stratified split
    labels = np.array(full_dataset.labels)
    indices = np.arange(len(full_dataset))
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=seed
    )
    
    # Second split: val vs test
    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_split / (val_split + test_split),
        stratify=temp_labels,
        random_state=seed
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Apply transforms to each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader, full_dataset.get_class_names()


def get_dataset_statistics(data_dir: str) -> dict:
    """
    Compute dataset statistics (mean, std, class distribution).
    
    Args:
        data_dir: Path to Caltech-101 dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    dataset = Caltech101Dataset(
        root_dir=data_dir,
        image_size=128,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    )
    
    # Compute mean and std
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    # Class distribution
    labels = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(dataset.get_class_names(), counts))
    
    stats = {
        'num_images': len(dataset),
        'num_classes': len(dataset.get_class_names()),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'class_distribution': class_distribution,
        'class_names': dataset.get_class_names()
    }
    
    return stats
