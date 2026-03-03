"""Evaluation metrics for feature importance analysis."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logger = logging.getLogger(__name__)


class FaithfulnessMetrics:
    """Metrics for evaluating faithfulness of feature importance explanations."""
    
    def __init__(self, random_state: int = 42):
        """Initialize faithfulness metrics.
        
        Args:
            random_state: Random seed.
        """
        self.random_state = random_state
    
    def deletion_auc(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        importance_scores: np.ndarray,
        n_features_to_remove: int = 5
    ) -> float:
        """Compute deletion AUC - how much performance drops as important features are removed.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            importance_scores: Feature importance scores.
            n_features_to_remove: Number of features to remove.
            
        Returns:
            Deletion AUC score.
        """
        # Get baseline performance
        baseline_score = self._get_model_score(model, X, y)
        
        # Sort features by importance (ascending - remove least important first)
        feature_order = np.argsort(importance_scores)
        
        # Remove features one by one and measure performance drop
        performance_drops = []
        
        for i in range(min(n_features_to_remove, len(feature_order))):
            # Remove i+1 least important features
            features_to_remove = feature_order[:i+1]
            X_modified = np.delete(X, features_to_remove, axis=1)
            
            # Retrain model on modified data
            model_copy = self._create_model_copy(model)
            model_copy.fit(X_modified, y)
            
            # Measure performance
            current_score = self._get_model_score(model_copy, X_modified, y)
            performance_drop = baseline_score - current_score
            performance_drops.append(performance_drop)
        
        # Compute AUC of performance drops
        auc = np.trapz(performance_drops, dx=1)
        return auc
    
    def insertion_auc(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        importance_scores: np.ndarray,
        n_features_to_add: int = 5
    ) -> float:
        """Compute insertion AUC - how much performance improves as important features are added.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            importance_scores: Feature importance scores.
            n_features_to_add: Number of features to add.
            
        Returns:
            Insertion AUC score.
        """
        # Start with no features
        baseline_score = 0.5  # Random performance baseline
        
        # Sort features by importance (descending - add most important first)
        feature_order = np.argsort(importance_scores)[::-1]
        
        # Add features one by one and measure performance improvement
        performance_improvements = []
        
        for i in range(min(n_features_to_add, len(feature_order))):
            # Add i+1 most important features
            features_to_add = feature_order[:i+1]
            X_modified = X[:, features_to_add]
            
            # Train model on subset of features
            model_copy = self._create_model_copy(model)
            model_copy.fit(X_modified, y)
            
            # Measure performance
            current_score = self._get_model_score(model_copy, X_modified, y)
            performance_improvement = current_score - baseline_score
            performance_improvements.append(performance_improvement)
        
        # Compute AUC of performance improvements
        auc = np.trapz(performance_improvements, dx=1)
        return auc
    
    def sufficiency_score(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        importance_scores: np.ndarray,
        top_k: int = 5
    ) -> float:
        """Compute sufficiency score - how well top K features explain the model's predictions.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            importance_scores: Feature importance scores.
            top_k: Number of top features to use.
            
        Returns:
            Sufficiency score.
        """
        # Get top K features
        top_features = np.argsort(importance_scores)[-top_k:]
        X_top = X[:, top_features]
        
        # Train model on top features only
        model_copy = self._create_model_copy(model)
        model_copy.fit(X_top, y)
        
        # Get predictions from both models
        y_pred_full = model.predict(X)
        y_pred_top = model_copy.predict(X_top)
        
        # Compute correlation between predictions
        correlation = np.corrcoef(y_pred_full, y_pred_top)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def necessity_score(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        importance_scores: np.ndarray,
        top_k: int = 5
    ) -> float:
        """Compute necessity score - how much performance drops when top K features are removed.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            importance_scores: Feature importance scores.
            top_k: Number of top features to remove.
            
        Returns:
            Necessity score.
        """
        # Get baseline performance
        baseline_score = self._get_model_score(model, X, y)
        
        # Get top K features
        top_features = np.argsort(importance_scores)[-top_k:]
        
        # Remove top features
        X_modified = np.delete(X, top_features, axis=1)
        
        # Retrain model without top features
        model_copy = self._create_model_copy(model)
        model_copy.fit(X_modified, y)
        
        # Measure performance drop
        modified_score = self._get_model_score(model_copy, X_modified, y)
        performance_drop = baseline_score - modified_score
        
        return performance_drop
    
    def _get_model_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Get model performance score.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            
        Returns:
            Model score.
        """
        if hasattr(model, 'predict_proba'):
            # Classification model
            y_pred = model.predict(X)
            return accuracy_score(y, y_pred)
        else:
            # Regression model
            y_pred = model.predict(X)
            return r2_score(y, y_pred)
    
    def _create_model_copy(self, model: Any) -> Any:
        """Create a copy of the model for retraining.
        
        Args:
            model: Original model.
            
        Returns:
            Model copy.
        """
        # Get model parameters
        params = model.get_params()
        
        # Create new instance
        model_class = type(model)
        return model_class(**params)


class StabilityMetrics:
    """Metrics for evaluating stability of feature importance explanations."""
    
    def __init__(self, random_state: int = 42):
        """Initialize stability metrics.
        
        Args:
            random_state: Random seed.
        """
        self.random_state = random_state
    
    def cross_validation_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        importance_method: Any,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """Compute stability across cross-validation splits.
        
        Args:
            X: Input features.
            y: Target values.
            importance_method: Method to compute importance.
            n_splits: Number of CV splits.
            test_size: Fraction of data for testing.
            
        Returns:
            Dict containing stability metrics.
        """
        from scipy.stats import kendalltau, spearmanr
        
        importance_scores_list = []
        
        for split in range(n_splits):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state + split
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
        
        return {
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "min_correlation": np.min(correlations),
            "stability_score": np.mean(correlations)
        }
    
    def bootstrap_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        importance_method: Any,
        n_bootstrap: int = 100,
        bootstrap_size: float = 0.8
    ) -> Dict[str, float]:
        """Compute stability using bootstrap sampling.
        
        Args:
            X: Input features.
            y: Target values.
            importance_method: Method to compute importance.
            n_bootstrap: Number of bootstrap samples.
            bootstrap_size: Fraction of data for bootstrap.
            
        Returns:
            Dict containing stability metrics.
        """
        from scipy.stats import kendalltau
        
        importance_scores_list = []
        n_samples = int(len(X) * bootstrap_size)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            np.random.seed(self.random_state + _)
            bootstrap_idx = np.random.choice(len(X), n_samples, replace=True)
            X_bootstrap = X[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            
            # Compute importance for this bootstrap sample
            if hasattr(importance_method, 'compute_importance'):
                importance_result = importance_method.compute_importance(X_bootstrap, y_bootstrap)
                scores = importance_result["importance_scores"]
            else:
                # Assume it's a model with feature_importances_
                importance_method.fit(X_bootstrap, y_bootstrap)
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
        
        return {
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "min_correlation": np.min(correlations),
            "stability_score": np.mean(correlations)
        }


class FidelityMetrics:
    """Metrics for evaluating fidelity of surrogate models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize fidelity metrics.
        
        Args:
            random_state: Random seed.
        """
        self.random_state = random_state
    
    def surrogate_fidelity(
        self,
        black_box_model: Any,
        surrogate_model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Compute fidelity of surrogate model to black box model.
        
        Args:
            black_box_model: Original black box model.
            surrogate_model: Surrogate model.
            X: Input features.
            y: Target values.
            
        Returns:
            Dict containing fidelity metrics.
        """
        # Get predictions from both models
        y_pred_black_box = black_box_model.predict(X)
        y_pred_surrogate = surrogate_model.predict(X)
        
        # Compute correlation
        correlation = np.corrcoef(y_pred_black_box, y_pred_surrogate)[0, 1]
        
        # Compute MSE
        mse = mean_squared_error(y_pred_black_box, y_pred_surrogate)
        
        # Compute R²
        r2 = r2_score(y_pred_black_box, y_pred_surrogate)
        
        return {
            "correlation": correlation if not np.isnan(correlation) else 0.0,
            "mse": mse,
            "r2": r2,
            "fidelity_score": abs(correlation)  # Main fidelity metric
        }
    
    def feature_importance_fidelity(
        self,
        black_box_importance: np.ndarray,
        surrogate_importance: np.ndarray
    ) -> Dict[str, float]:
        """Compute fidelity of feature importance between models.
        
        Args:
            black_box_importance: Importance scores from black box model.
            surrogate_importance: Importance scores from surrogate model.
            
        Returns:
            Dict containing fidelity metrics.
        """
        from scipy.stats import kendalltau, spearmanr
        
        # Compute correlations
        pearson_corr = np.corrcoef(black_box_importance, surrogate_importance)[0, 1]
        spearman_corr, _ = spearmanr(black_box_importance, surrogate_importance)
        kendall_corr, _ = kendalltau(black_box_importance, surrogate_importance)
        
        return {
            "pearson_correlation": pearson_corr if not np.isnan(pearson_corr) else 0.0,
            "spearman_correlation": spearman_corr,
            "kendall_correlation": kendall_corr,
            "fidelity_score": abs(kendall_corr)  # Main fidelity metric
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""
    
    def __init__(self, random_state: int = 42):
        """Initialize comprehensive evaluator.
        
        Args:
            random_state: Random seed.
        """
        self.random_state = random_state
        self.faithfulness_metrics = FaithfulnessMetrics(random_state)
        self.stability_metrics = StabilityMetrics(random_state)
        self.fidelity_metrics = FidelityMetrics(random_state)
    
    def evaluate_explanation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        importance_scores: np.ndarray,
        importance_method: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of feature importance explanation.
        
        Args:
            model: Trained model.
            X: Input features.
            y: Target values.
            importance_scores: Feature importance scores.
            importance_method: Method used to compute importance.
            
        Returns:
            Dict containing comprehensive evaluation results.
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {
            "faithfulness": {},
            "stability": {},
            "fidelity": {},
            "overall_score": 0.0
        }
        
        # Faithfulness metrics
        try:
            results["faithfulness"] = {
                "deletion_auc": self.faithfulness_metrics.deletion_auc(
                    model, X, y, importance_scores
                ),
                "insertion_auc": self.faithfulness_metrics.insertion_auc(
                    model, X, y, importance_scores
                ),
                "sufficiency_score": self.faithfulness_metrics.sufficiency_score(
                    model, X, y, importance_scores
                ),
                "necessity_score": self.faithfulness_metrics.necessity_score(
                    model, X, y, importance_scores
                )
            }
        except Exception as e:
            logger.error(f"Error computing faithfulness metrics: {e}")
            results["faithfulness"] = {"error": str(e)}
        
        # Stability metrics
        if importance_method is not None:
            try:
                results["stability"] = self.stability_metrics.cross_validation_stability(
                    X, y, importance_method
                )
            except Exception as e:
                logger.error(f"Error computing stability metrics: {e}")
                results["stability"] = {"error": str(e)}
        
        # Compute overall score
        faithfulness_score = 0.0
        if "error" not in results["faithfulness"]:
            faithfulness_metrics = results["faithfulness"]
            faithfulness_score = np.mean([
                faithfulness_metrics["deletion_auc"],
                faithfulness_metrics["insertion_auc"],
                faithfulness_metrics["sufficiency_score"],
                faithfulness_metrics["necessity_score"]
            ])
        
        stability_score = 0.0
        if "error" not in results["stability"]:
            stability_score = results["stability"]["stability_score"]
        
        results["overall_score"] = (faithfulness_score + stability_score) / 2
        
        logger.info(f"Comprehensive evaluation completed. Overall score: {results['overall_score']:.3f}")
        
        return results
