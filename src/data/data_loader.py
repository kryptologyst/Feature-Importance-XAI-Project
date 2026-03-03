"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 3,
    n_redundant: int = 1,
    n_classes: int = 2,
    task_type: str = "classification",
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load synthetic dataset for feature importance analysis.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_classes: Number of classes (for classification).
        task_type: Type of task ('classification' or 'regression').
        noise: Noise level.
        random_state: Random seed.
        
    Returns:
        Tuple of (X, y, feature_names).
    """
    np.random.seed(random_state)
    
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            random_state=random_state
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=random_state
        )
    
    # Create meaningful feature names
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    
    # Mark informative features
    for i in range(n_informative):
        feature_names[i] = f"informative_{i+1}"
    
    # Mark redundant features
    for i in range(n_informative, n_informative + n_redundant):
        feature_names[i] = f"redundant_{i-n_informative+1}"
    
    logger.info(f"Generated synthetic {task_type} dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, feature_names


def load_sklearn_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load a sklearn built-in dataset.
    
    Args:
        dataset_name: Name of the dataset ('iris', 'wine', 'breast_cancer').
        
    Returns:
        Tuple of (X, y, feature_names).
    """
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    logger.info(f"Loaded {dataset_name} dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, list(feature_names)


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, StandardScaler]:
    """Preprocess data for training and evaluation.
    
    Args:
        X: Input features.
        y: Target values.
        feature_names: List of feature names.
        test_size: Fraction of data for testing.
        random_state: Random seed.
        scale_features: Whether to scale features.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Features scaled using StandardScaler")
    
    logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def create_dataset_metadata(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    task_type: str = "classification"
) -> Dict[str, Any]:
    """Create comprehensive metadata for the dataset.
    
    Args:
        X: Input features.
        y: Target values.
        feature_names: List of feature names.
        task_type: Type of task ('classification' or 'regression').
        
    Returns:
        Dict[str, Any]: Dataset metadata.
    """
    metadata = {
        "dataset_info": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "task_type": task_type,
            "feature_names": feature_names
        },
        "features": {},
        "target": {}
    }
    
    # Feature metadata
    for i, name in enumerate(feature_names):
        feature_values = X[:, i]
        metadata["features"][name] = {
            "type": "numerical",
            "range": [float(np.min(feature_values)), float(np.max(feature_values))],
            "mean": float(np.mean(feature_values)),
            "std": float(np.std(feature_values)),
            "monotonic": False,  # Would need domain knowledge
            "sensitive": False,  # Would need domain knowledge
            "index": i
        }
    
    # Target metadata
    if task_type == "classification":
        unique_classes = np.unique(y)
        metadata["target"] = {
            "type": "classification",
            "n_classes": len(unique_classes),
            "classes": unique_classes.tolist(),
            "class_distribution": {
                int(cls): int(np.sum(y == cls)) for cls in unique_classes
            }
        }
    else:
        metadata["target"] = {
            "type": "regression",
            "range": [float(np.min(y)), float(np.max(y))],
            "mean": float(np.mean(y)),
            "std": float(np.std(y))
        }
    
    return metadata


def save_dataset_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """Save dataset metadata to file.
    
    Args:
        metadata: Dataset metadata dictionary.
        filepath: Output file path.
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset metadata saved to {filepath}")


def load_dataset_metadata(filepath: str) -> Dict[str, Any]:
    """Load dataset metadata from file.
    
    Args:
        filepath: Input file path.
        
    Returns:
        Dict[str, Any]: Dataset metadata.
    """
    import json
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Dataset metadata loaded from {filepath}")
    return metadata