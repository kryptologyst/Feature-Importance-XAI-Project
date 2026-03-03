"""SHAP-based feature importance methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class SHAPImportance:
    """SHAP-based feature importance methods."""
    
    def __init__(
        self,
        method: str = "tree",
        n_samples: int = 100,
        random_state: int = 42
    ):
        """Initialize SHAP importance.
        
        Args:
            method: SHAP method ('tree', 'kernel', 'deep', 'linear').
            n_samples: Number of samples for explanation.
            random_state: Random seed.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.method = method
        self.n_samples = n_samples
        self.random_state = random_state
        self.explainer = None
    
    def _create_explainer(self, model: Any, X: np.ndarray) -> Any:
        """Create appropriate SHAP explainer based on model type and method.
        
        Args:
            model: Trained model.
            X: Background data for explainer.
            
        Returns:
            SHAP explainer.
        """
        if self.method == "tree":
            # For tree-based models
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                self.explainer = shap.TreeExplainer(model)
            else:
                logger.warning("Model may not be tree-based. Falling back to kernel explainer.")
                self.explainer = shap.KernelExplainer(model.predict, X[:100])
        
        elif self.method == "kernel":
            # Kernel SHAP (model-agnostic)
            self.explainer = shap.KernelExplainer(model.predict, X[:100])
        
        elif self.method == "deep":
            # Deep SHAP for neural networks
            if hasattr(model, 'predict_proba'):
                self.explainer = shap.DeepExplainer(model, X[:100])
            else:
                logger.warning("Model may not be suitable for Deep SHAP. Falling back to kernel explainer.")
                self.explainer = shap.KernelExplainer(model.predict, X[:100])
        
        elif self.method == "linear":
            # Linear SHAP
            self.explainer = shap.LinearExplainer(model, X)
        
        else:
            raise ValueError(f"Unknown SHAP method: {self.method}")
        
        logger.info(f"Created {self.method} SHAP explainer")
        return self.explainer
    
    def compute_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute SHAP feature importance.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values (optional).
            
        Returns:
            Dict containing SHAP importance scores and statistics.
        """
        logger.info(f"Computing SHAP importance using {self.method} method")
        
        # Create explainer
        self._create_explainer(model, X)
        
        # Sample data for explanation if needed
        if self.n_samples < len(X):
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(X), self.n_samples, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Multi-class: take mean absolute values across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)
            
            # Compute global importance (mean absolute SHAP values)
            global_importance = np.mean(shap_values, axis=0)
            
            results = {
                "importance_scores": global_importance,
                "shap_values": shap_values,
                "method": f"shap_{self.method}",
                "n_samples": len(X_sample),
                "explainer": self.explainer
            }
            
            logger.info("SHAP importance computation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            # Fallback to permutation importance
            logger.info("Falling back to permutation importance")
            from .permutation_importance import PermutationImportance
            perm_importance = PermutationImportance()
            return perm_importance.compute_importance(model, X, y)


class SAGEImportance:
    """SAGE (Shapley Additive Global Importance) method."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        random_state: int = 42
    ):
        """Initialize SAGE importance.
        
        Args:
            n_samples: Number of samples for computation.
            random_state: Random seed.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SAGE. Install with: pip install shap")
        
        self.n_samples = n_samples
        self.random_state = random_state
    
    def compute_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Compute SAGE feature importance.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            
        Returns:
            Dict containing SAGE importance scores.
        """
        logger.info("Computing SAGE importance")
        
        try:
            # Create SAGE explainer
            sage_explainer = shap.explainers.Sage(model.predict, X)
            
            # Compute SAGE values
            sage_values = sage_explainer(X)
            
            # Extract importance scores
            importance_scores = np.abs(sage_values.values).mean(axis=0)
            
            results = {
                "importance_scores": importance_scores,
                "sage_values": sage_values,
                "method": "sage",
                "n_samples": len(X)
            }
            
            logger.info("SAGE importance computation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error computing SAGE values: {e}")
            logger.info("SAGE not available, falling back to SHAP")
            
            # Fallback to SHAP
            shap_importance = SHAPImportance(method="kernel", n_samples=self.n_samples)
            return shap_importance.compute_importance(model, X, y)


class SHAPStabilityAnalysis:
    """Stability analysis for SHAP-based methods."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize SHAP stability analysis.
        
        Args:
            n_splits: Number of data splits.
            test_size: Fraction of data for testing.
            random_state: Random seed.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
    
    def compute_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        shap_method: str = "tree"
    ) -> Dict[str, Any]:
        """Compute stability of SHAP importance across splits.
        
        Args:
            X: Input features.
            y: Target values.
            model: Model to analyze.
            shap_method: SHAP method to use.
            
        Returns:
            Dict containing stability metrics.
        """
        from sklearn.model_selection import train_test_split
        from scipy.stats import kendalltau
        
        logger.info(f"Computing SHAP stability analysis with {self.n_splits} splits")
        
        importance_scores_list = []
        
        for split in tqdm(range(self.n_splits), desc="SHAP stability analysis"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state + split
            )
            
            # Train model on this split
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Compute SHAP importance
            shap_importance = SHAPImportance(method=shap_method, n_samples=100)
            importance_result = shap_importance.compute_importance(model_copy, X_test)
            scores = importance_result["importance_scores"]
            
            importance_scores_list.append(scores)
        
        # Compute stability metrics
        correlations = []
        for i in range(len(importance_scores_list)):
            for j in range(i + 1, len(importance_scores_list)):
                scores1 = importance_scores_list[i]
                scores2 = importance_scores_list[j]
                
                # Kendall tau correlation
                kendall_corr, _ = kendalltau(scores1, scores2)
                correlations.append(kendall_corr)
        
        stability_metrics = {
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "min_correlation": np.min(correlations),
            "max_correlation": np.max(correlations),
            "stability_score": np.mean(correlations),
            "importance_scores_list": importance_scores_list,
            "method": f"shap_stability_{shap_method}"
        }
        
        logger.info(f"SHAP stability analysis completed. Mean correlation: {stability_metrics['mean_correlation']:.3f}")
        
        return stability_metrics
