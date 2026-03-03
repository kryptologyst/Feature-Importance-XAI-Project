"""Main feature importance explainer interface."""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from methods import (
    PermutationImportance,
    TreeBasedImportance,
    StabilityAnalysis,
    CrossValidationImportance
)
from methods.shap_methods import SHAPImportance, SAGEImportance, SHAPStabilityAnalysis
from utils import validate_inputs, compute_stability_metrics, normalize_importance_scores
from data import preprocess_data

logger = logging.getLogger(__name__)


class FeatureImportanceExplainer:
    """Main interface for feature importance analysis."""
    
    def __init__(
        self,
        methods: List[str] = None,
        random_state: int = 42,
        n_samples: int = 1000,
        n_repeats: int = 10,
        n_splits: int = 5
    ):
        """Initialize feature importance explainer.
        
        Args:
            methods: List of methods to use ('permutation', 'tree', 'shap', 'sage', 'stability').
            random_state: Random seed.
            n_samples: Number of samples for SHAP/SAGE.
            n_repeats: Number of repeats for permutation importance.
            n_splits: Number of splits for stability analysis.
        """
        if methods is None:
            methods = ["permutation", "tree", "shap"]
        
        self.methods = methods
        self.random_state = random_state
        self.n_samples = n_samples
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        
        # Initialize method objects
        self._initialize_methods()
        
        logger.info(f"Initialized FeatureImportanceExplainer with methods: {methods}")
    
    def _initialize_methods(self) -> None:
        """Initialize method objects."""
        self.method_objects = {}
        
        if "permutation" in self.methods:
            self.method_objects["permutation"] = PermutationImportance(
                n_repeats=self.n_repeats, random_state=self.random_state
            )
        
        if "tree" in self.methods:
            self.method_objects["tree"] = TreeBasedImportance(
                model_type="random_forest", random_state=self.random_state
            )
        
        if "shap" in self.methods:
            try:
                self.method_objects["shap"] = SHAPImportance(
                    method="tree", n_samples=self.n_samples, random_state=self.random_state
                )
            except ImportError:
                logger.warning("SHAP not available. Skipping SHAP method.")
                self.methods.remove("shap")
        
        if "sage" in self.methods:
            try:
                self.method_objects["sage"] = SAGEImportance(
                    n_samples=self.n_samples, random_state=self.random_state
                )
            except ImportError:
                logger.warning("SAGE not available. Skipping SAGE method.")
                self.methods.remove("sage")
        
        if "stability" in self.methods:
            self.method_objects["stability"] = StabilityAnalysis(
                n_splits=self.n_splits, random_state=self.random_state
            )
        
        if "cv" in self.methods:
            self.method_objects["cv"] = CrossValidationImportance(
                cv_folds=self.n_splits, random_state=self.random_state
            )
    
    def explain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model: Optional[Any] = None,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """Compute feature importance using specified methods.
        
        Args:
            X: Input features.
            y: Target values.
            feature_names: List of feature names.
            model: Pre-trained model (optional).
            task_type: Type of task ('classification' or 'regression').
            
        Returns:
            Dict containing results from all methods.
        """
        # Validate inputs
        validate_inputs(X, y, feature_names)
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create default model if not provided
        if model is None:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            
            model.fit(X, y)
            logger.info("Created and trained default Random Forest model")
        
        # Compute explanations using all methods
        results = {
            "feature_names": feature_names,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "task_type": task_type,
            "methods": {}
        }
        
        for method_name in self.methods:
            if method_name not in self.method_objects:
                continue
            
            logger.info(f"Computing {method_name} importance")
            
            try:
                if method_name == "permutation":
                    importance_result = self.method_objects[method_name].compute_importance(model, X, y)
                
                elif method_name == "tree":
                    importance_result = self.method_objects[method_name].compute_importance(model)
                
                elif method_name == "shap":
                    importance_result = self.method_objects[method_name].compute_importance(model, X, y)
                
                elif method_name == "sage":
                    importance_result = self.method_objects[method_name].compute_importance(model, X, y)
                
                elif method_name == "stability":
                    importance_result = self.method_objects[method_name].compute_stability(
                        X, y, self.method_objects["permutation"], "permutation"
                    )
                
                elif method_name == "cv":
                    importance_result = self.method_objects[method_name].compute_cv_importance(X, y, model)
                
                else:
                    logger.warning(f"Unknown method: {method_name}")
                    continue
                
                results["methods"][method_name] = importance_result
                logger.info(f"Completed {method_name} importance computation")
                
            except Exception as e:
                logger.error(f"Error computing {method_name} importance: {e}")
                continue
        
        # Compute comparative analysis
        results["comparative_analysis"] = self._compute_comparative_analysis(results["methods"])
        
        logger.info("Feature importance analysis completed")
        return results
    
    def _compute_comparative_analysis(self, method_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comparative analysis across methods.
        
        Args:
            method_results: Results from different methods.
            
        Returns:
            Dict containing comparative analysis.
        """
        from scipy.stats import kendalltau, spearmanr
        
        comparative = {
            "method_correlations": {},
            "rank_correlations": {},
            "consensus_features": [],
            "disagreement_features": []
        }
        
        # Get methods with importance scores
        methods_with_scores = {
            name: result for name, result in method_results.items()
            if "importance_scores" in result
        }
        
        if len(methods_with_scores) < 2:
            return comparative
        
        # Compute correlations between methods
        method_names = list(methods_with_scores.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                scores1 = methods_with_scores[method1]["importance_scores"]
                scores2 = methods_with_scores[method2]["importance_scores"]
                
                # Value correlation
                pearson_corr = np.corrcoef(scores1, scores2)[0, 1]
                spearman_corr, _ = spearmanr(scores1, scores2)
                kendall_corr, _ = kendalltau(scores1, scores2)
                
                comparative["method_correlations"][f"{method1}_vs_{method2}"] = {
                    "pearson": pearson_corr,
                    "spearman": spearman_corr,
                    "kendall": kendall_corr
                }
                
                # Rank correlation
                ranks1 = np.argsort(np.argsort(scores1))
                ranks2 = np.argsort(np.argsort(scores2))
                rank_corr = np.corrcoef(ranks1, ranks2)[0, 1]
                
                comparative["rank_correlations"][f"{method1}_vs_{method2}"] = rank_corr
        
        # Find consensus and disagreement features
        if len(methods_with_scores) >= 2:
            # Get top features from each method
            top_features_per_method = {}
            for method_name, result in methods_with_scores.items():
                scores = result["importance_scores"]
                top_indices = np.argsort(scores)[-3:]  # Top 3 features
                top_features_per_method[method_name] = set(top_indices)
            
            # Find consensus features (appear in top 3 of multiple methods)
            all_features = set(range(len(scores)))
            consensus_features = all_features.copy()
            for method_features in top_features_per_method.values():
                consensus_features &= method_features
            
            # Find disagreement features (appear in top 3 of some but not all methods)
            disagreement_features = set()
            for method1_features in top_features_per_method.values():
                for method2_features in top_features_per_method.values():
                    disagreement_features |= (method1_features - method2_features)
            
            comparative["consensus_features"] = list(consensus_features)
            comparative["disagreement_features"] = list(disagreement_features)
        
        return comparative
    
    def get_top_features(
        self,
        results: Dict[str, Any],
        method: str = "permutation",
        top_k: int = 5
    ) -> List[tuple]:
        """Get top K features from specified method.
        
        Args:
            results: Results from explain method.
            method: Method to use for ranking.
            top_k: Number of top features to return.
            
        Returns:
            List of (feature_name, importance_score) tuples.
        """
        if method not in results["methods"]:
            raise ValueError(f"Method {method} not found in results")
        
        method_result = results["methods"][method]
        if "importance_scores" not in method_result:
            raise ValueError(f"Method {method} does not have importance scores")
        
        importance_scores = method_result["importance_scores"]
        feature_names = results["feature_names"]
        
        # Get top K features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        
        top_features = [
            (feature_names[i], importance_scores[i]) for i in top_indices
        ]
        
        return top_features
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save results to file.
        
        Args:
            results: Results dictionary.
            filepath: Output file path.
        """
        from ..utils import save_results as save_results_util
        save_results_util(results, filepath)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from file.
        
        Args:
            filepath: Input file path.
            
        Returns:
            Dict containing loaded results.
        """
        from ..utils import load_results as load_results_util
        return load_results_util(filepath)
