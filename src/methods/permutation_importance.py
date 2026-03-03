"""Feature importance methods implementation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PermutationImportance:
    """Permutation importance method for feature importance analysis."""
    
    def __init__(self, n_repeats: int = 10, random_state: int = 42):
        """Initialize permutation importance.
        
        Args:
            n_repeats: Number of permutation repeats.
            random_state: Random seed.
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def compute_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Compute permutation importance.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            scoring: Scoring metric.
            
        Returns:
            Dict containing importance scores and statistics.
        """
        logger.info(f"Computing permutation importance with {self.n_repeats} repeats")
        
        # Use sklearn's permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=self.n_repeats, random_state=self.random_state, scoring=scoring
        )
        
        results = {
            "importance_scores": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
            "importance_values": perm_importance.importances,
            "method": "permutation"
        }
        
        logger.info("Permutation importance computation completed")
        return results


class TreeBasedImportance:
    """Tree-based feature importance methods."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """Initialize tree-based importance.
        
        Args:
            model_type: Type of tree model ('random_forest', 'gradient_boosting').
            **kwargs: Additional model parameters.
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
    
    def fit_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit the tree-based model.
        
        Args:
            X: Input features.
            y: Target values.
            
        Returns:
            Trained model.
        """
        if self.model_type == "random_forest":
            if len(np.unique(y)) > 2:  # Classification
                self.model = RandomForestClassifier(**self.kwargs)
            else:  # Regression
                self.model = RandomForestRegressor(**self.kwargs)
        
        self.model.fit(X, y)
        logger.info(f"Fitted {self.model_type} model")
        return self.model
    
    def compute_importance(self, model: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """Compute tree-based feature importance.
        
        Args:
            model: Trained model (if None, uses self.model).
            
        Returns:
            Dict containing importance scores.
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available. Call fit_model first.")
        
        importance_scores = model.feature_importances_
        
        results = {
            "importance_scores": importance_scores,
            "method": f"tree_based_{self.model_type}"
        }
        
        logger.info(f"Computed {self.model_type} feature importance")
        return results


class StabilityAnalysis:
    """Stability analysis for feature importance across different splits."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize stability analysis.
        
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
        importance_method: Any,
        method_name: str = "permutation"
    ) -> Dict[str, Any]:
        """Compute stability of feature importance across splits.
        
        Args:
            X: Input features.
            y: Target values.
            importance_method: Method to compute importance.
            method_name: Name of the importance method.
            
        Returns:
            Dict containing stability metrics.
        """
        from sklearn.model_selection import train_test_split
        from scipy.stats import kendalltau, spearmanr
        
        logger.info(f"Computing stability analysis with {self.n_splits} splits")
        
        importance_scores_list = []
        
        for split in tqdm(range(self.n_splits), desc="Stability analysis"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state + split
            )
            
            # Compute importance for this split
            if hasattr(importance_method, 'compute_importance'):
                importance_result = importance_method.compute_importance(X_train, y_train)
                scores = importance_result["importance_scores"]
            else:
                # Assume it's a model with feature_importances_
                importance_method.fit(X_train, y_train)
                scores = importance_method.feature_importances_
            
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
            "stability_score": np.mean(correlations),  # Main stability metric
            "importance_scores_list": importance_scores_list,
            "method": f"stability_{method_name}"
        }
        
        logger.info(f"Stability analysis completed. Mean correlation: {stability_metrics['mean_correlation']:.3f}")
        
        return stability_metrics


class CrossValidationImportance:
    """Cross-validation based feature importance analysis."""
    
    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """Initialize cross-validation importance.
        
        Args:
            cv_folds: Number of CV folds.
            random_state: Random seed.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def compute_cv_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute feature importance using cross-validation.
        
        Args:
            X: Input features.
            y: Target values.
            model: Model to use.
            scoring: Scoring metric.
            
        Returns:
            Dict containing CV-based importance metrics.
        """
        from sklearn.model_selection import KFold
        
        logger.info(f"Computing CV importance with {self.cv_folds} folds")
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        feature_importances = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
            
            # Get validation score
            if scoring:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(scoring)
                score = scorer(model, X_val, y_val)
            else:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = accuracy_score(y_val, y_pred)
                else:
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
            
            cv_scores.append(score)
        
        results = {
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "feature_importances": feature_importances,
            "mean_feature_importance": np.mean(feature_importances, axis=0) if feature_importances else None,
            "method": "cross_validation"
        }
        
        logger.info(f"CV importance completed. Mean CV score: {results['cv_mean']:.3f}")
        
        return results