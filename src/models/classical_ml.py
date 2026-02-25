"""
Classical Machine Learning Models for Image Classification
HOG + SVM and other traditional ML approaches
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import color, transform
import cv2

import torch
from torch.utils.data import DataLoader


class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2)
    ):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            HOG feature vector
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        
        # Resize image
        resized = transform.resize(gray, self.image_size)
        
        # Extract HOG features
        features = hog(
            resized,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        return features
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract HOG features from a batch of images."""
        features = []
        for img in tqdm(images, desc="Extracting HOG features"):
            features.append(self.extract(img))
        return np.array(features)


class CNNFeatureExtractor:
    """
    Extract features using a pre-trained CNN (e.g., ResNet).
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove final classification layer
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def extract_batch(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract CNN features from a dataloader.
        
        Returns:
            Tuple of (features, labels)
        """
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting {self.model_name} features"):
                images = images.to(self.device)
                features = self.model(images)
                features = features.squeeze()
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels


class ClassicalMLClassifier:
    """
    Classical ML classifier combining feature extraction and classification.
    """
    
    def __init__(
        self,
        feature_type: str = 'hog',
        classifier_type: str = 'svm',
        image_size: int = 64,
        **kwargs
    ):
        """
        Args:
            feature_type: 'hog' or 'cnn'
            classifier_type: 'svm', 'linear_svm', or 'random_forest'
            image_size: Image size for feature extraction
            **kwargs: Additional arguments for feature extractor or classifier
        """
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.image_size = image_size
        
        # Initialize feature extractor
        if feature_type == 'hog':
            self.feature_extractor = HOGFeatureExtractor(
                image_size=(image_size, image_size)
            )
        elif feature_type == 'cnn':
            self.feature_extractor = CNNFeatureExtractor(
                model_name=kwargs.get('cnn_model', 'resnet50')
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                C=kwargs.get('C', 10.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                verbose=False
            )
        elif classifier_type == 'linear_svm':
            self.classifier = LinearSVC(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 10000),
                verbose=False
            )
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                n_jobs=-1,
                verbose=False
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def fit(self, train_loader: DataLoader) -> None:
        """
        Train the classifier.
        
        Args:
            train_loader: DataLoader for training data
        """
        print(f"\nTraining {self.feature_type.upper()} + {self.classifier_type.upper()} classifier...")
        
        # Extract features
        if self.feature_type == 'hog':
            X_train = []
            y_train = []
            
            for images, labels in tqdm(train_loader, desc="Extracting features"):
                for img, label in zip(images, labels):
                    # Convert tensor to numpy
                    img_np = img.permute(1, 2, 0).numpy()
                    features = self.feature_extractor.extract(img_np)
                    X_train.append(features)
                    y_train.append(label.item())
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
        
        elif self.feature_type == 'cnn':
            X_train, y_train = self.feature_extractor.extract_batch(train_loader)
        
        print(f"Feature shape: {X_train.shape}")
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Compute training accuracy
        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"Training accuracy: {train_acc:.4f}")
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        # Extract features
        if self.feature_type == 'hog':
            X_test = []
            y_test = []
            
            for images, labels in tqdm(test_loader, desc="Extracting features"):
                for img, label in zip(images, labels):
                    img_np = img.permute(1, 2, 0).numpy()
                    features = self.feature_extractor.extract(img_np)
                    X_test.append(features)
                    y_test.append(label.item())
            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        elif self.feature_type == 'cnn':
            X_test, y_test = self.feature_extractor.extract_batch(test_loader)
        
        # Make predictions
        predictions = self.classifier.predict(X_test)
        
        return predictions, y_test
    
    def predict_proba(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities.
        
        Returns:
            Tuple of (probabilities, true_labels)
        """
        # Extract features
        if self.feature_type == 'hog':
            X_test = []
            y_test = []
            
            for images, labels in test_loader:
                for img, label in zip(images, labels):
                    img_np = img.permute(1, 2, 0).numpy()
                    features = self.feature_extractor.extract(img_np)
                    X_test.append(features)
                    y_test.append(label.item())
            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        elif self.feature_type == 'cnn':
            X_test, y_test = self.feature_extractor.extract_batch(test_loader)
        
        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(X_test)
        else:
            # For LinearSVM, use decision_function
            probabilities = self.classifier.decision_function(X_test)
        
        return probabilities, y_test
    
    def save(self, save_path: str) -> None:
        """Save the trained model."""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'feature_type': self.feature_type,
                'classifier_type': self.classifier_type,
                'image_size': self.image_size,
                'classifier': self.classifier
            }, f)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'ClassicalMLClassifier':
        """Load a trained model."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            feature_type=data['feature_type'],
            classifier_type=data['classifier_type'],
            image_size=data['image_size']
        )
        model.classifier = data['classifier']
        
        return model
