"""Utility functions for the Feature Importance XAI project."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def validate_inputs(X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None) -> None:
    """Validate input data for feature importance analysis.
    
    Args:
        X: Input features array.
        y: Target values array.
        feature_names: Optional list of feature names.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    
    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match number of features")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values")


def normalize_importance_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize feature importance scores.
    
    Args:
        scores: Raw importance scores.
        method: Normalization method ('minmax', 'zscore', 'sum').
        
    Returns:
        np.ndarray: Normalized scores.
    """
    if method == "minmax":
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.ones_like(scores) / len(scores)
        return (scores - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        if std_val == 0:
            return np.ones_like(scores) / len(scores)
        return (scores - mean_val) / std_val
    
    elif method == "sum":
        sum_val = np.sum(scores)
        if sum_val == 0:
            return np.ones_like(scores) / len(scores)
        return scores / sum_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_stability_metrics(scores_list: list, method: str = "kendall") -> Dict[str, float]:
    """Compute stability metrics across multiple runs.
    
    Args:
        scores_list: List of importance score arrays from different runs.
        method: Stability metric ('kendall', 'spearman', 'rank_correlation').
        
    Returns:
        Dict[str, float]: Stability metrics.
    """
    from scipy.stats import kendalltau, spearmanr
    
    if len(scores_list) < 2:
        return {"stability": 1.0, "mean_correlation": 1.0}
    
    correlations = []
    
    for i in range(len(scores_list)):
        for j in range(i + 1, len(scores_list)):
            scores1 = scores_list[i]
            scores2 = scores_list[j]
            
            if method == "kendall":
                corr, _ = kendalltau(scores1, scores2)
            elif method == "spearman":
                corr, _ = spearmanr(scores1, scores2)
            else:
                raise ValueError(f"Unknown stability method: {method}")
            
            correlations.append(corr)
    
    return {
        "stability": np.mean(correlations),
        "mean_correlation": np.mean(correlations),
        "std_correlation": np.std(correlations),
        "min_correlation": np.min(correlations)
    }


def create_feature_metadata(feature_names: list, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Create metadata for features.
    
    Args:
        feature_names: List of feature names.
        X: Feature matrix.
        
    Returns:
        Dict[str, Dict[str, Any]]: Feature metadata.
    """
    metadata = {}
    
    for i, name in enumerate(feature_names):
        feature_values = X[:, i]
        
        metadata[name] = {
            "type": "numerical",
            "range": [float(np.min(feature_values)), float(np.max(feature_values))],
            "mean": float(np.mean(feature_values)),
            "std": float(np.std(feature_values)),
            "monotonic": False,  # Would need domain knowledge to determine
            "sensitive": False,  # Would need domain knowledge to determine
            "index": i
        }
    
    return metadata


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary.
        filepath: Output file path.
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        filepath: Input file path.
        
    Returns:
        Dict[str, Any]: Loaded results.
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filepath}")
    return results
