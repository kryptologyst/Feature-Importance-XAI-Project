"""Test suite for Feature Importance XAI project."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_synthetic_data, preprocess_data
from explainers import FeatureImportanceExplainer
from methods import PermutationImportance, TreeBasedImportance, StabilityAnalysis
from metrics import FaithfulnessMetrics, StabilityMetrics, ComprehensiveEvaluator
from viz import FeatureImportanceVisualizer
from utils import set_seed, validate_inputs, normalize_importance_scores


class TestDataModule:
    """Test data loading and preprocessing."""
    
    def test_load_synthetic_data_classification(self):
        """Test synthetic data loading for classification."""
        X, y, feature_names = load_synthetic_data(
            n_samples=100,
            n_features=5,
            task_type="classification",
            random_state=42
        )
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(feature_names) == 5
        assert len(np.unique(y)) == 2  # Binary classification
    
    def test_load_synthetic_data_regression(self):
        """Test synthetic data loading for regression."""
        X, y, feature_names = load_synthetic_data(
            n_samples=100,
            n_features=5,
            task_type="regression",
            random_state=42
        )
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(feature_names) == 5
        assert len(np.unique(y)) > 2  # Continuous target
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X, y, feature_names = load_synthetic_data(
            n_samples=100,
            n_features=5,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test, feature_names_out, scaler = preprocess_data(
            X, y, feature_names, test_size=0.2, random_state=42
        )
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert len(feature_names_out) == 5
        assert scaler is not None


class TestMethodsModule:
    """Test feature importance methods."""
    
    def test_permutation_importance(self):
        """Test permutation importance computation."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        perm_importance = PermutationImportance(n_repeats=5, random_state=42)
        result = perm_importance.compute_importance(model, X, y)
        
        assert "importance_scores" in result
        assert len(result["importance_scores"]) == 5
        assert result["method"] == "permutation"
    
    def test_tree_based_importance(self):
        """Test tree-based importance computation."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        tree_importance = TreeBasedImportance(model_type="random_forest", random_state=42)
        model = tree_importance.fit_model(X, y)
        result = tree_importance.compute_importance(model)
        
        assert "importance_scores" in result
        assert len(result["importance_scores"]) == 5
        assert result["method"] == "tree_based_random_forest"
    
    def test_stability_analysis(self):
        """Test stability analysis."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        perm_importance = PermutationImportance(n_repeats=3, random_state=42)
        stability = StabilityAnalysis(n_splits=3, random_state=42)
        
        result = stability.compute_stability(X, y, perm_importance, "permutation")
        
        assert "stability_score" in result
        assert "mean_correlation" in result
        assert result["method"] == "stability_permutation"


class TestExplainerModule:
    """Test main explainer interface."""
    
    def test_feature_importance_explainer_init(self):
        """Test explainer initialization."""
        explainer = FeatureImportanceExplainer(
            methods=["permutation", "tree"],
            random_state=42
        )
        
        assert explainer.methods == ["permutation", "tree"]
        assert explainer.random_state == 42
        assert "permutation" in explainer.method_objects
        assert "tree" in explainer.method_objects
    
    def test_explain_method(self):
        """Test main explain method."""
        X, y, feature_names = load_synthetic_data(
            n_samples=100,
            n_features=5,
            random_state=42
        )
        
        explainer = FeatureImportanceExplainer(
            methods=["permutation", "tree"],
            random_state=42
        )
        
        results = explainer.explain(X, y, feature_names)
        
        assert "feature_names" in results
        assert "methods" in results
        assert "permutation" in results["methods"]
        assert "tree" in results["methods"]
        assert "comparative_analysis" in results
    
    def test_get_top_features(self):
        """Test getting top features."""
        X, y, feature_names = load_synthetic_data(
            n_samples=100,
            n_features=5,
            random_state=42
        )
        
        explainer = FeatureImportanceExplainer(methods=["permutation"], random_state=42)
        results = explainer.explain(X, y, feature_names)
        
        top_features = explainer.get_top_features(results, "permutation", top_k=3)
        
        assert len(top_features) == 3
        assert all(isinstance(item, tuple) for item in top_features)
        assert all(len(item) == 2 for item in top_features)


class TestMetricsModule:
    """Test evaluation metrics."""
    
    def test_faithfulness_metrics(self):
        """Test faithfulness metrics computation."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create dummy importance scores
        importance_scores = np.random.rand(5)
        
        faithfulness = FaithfulnessMetrics(random_state=42)
        
        deletion_auc = faithfulness.deletion_auc(model, X, y, importance_scores)
        insertion_auc = faithfulness.insertion_auc(model, X, y, importance_scores)
        sufficiency = faithfulness.sufficiency_score(model, X, y, importance_scores)
        necessity = faithfulness.necessity_score(model, X, y, importance_scores)
        
        assert isinstance(deletion_auc, float)
        assert isinstance(insertion_auc, float)
        assert isinstance(sufficiency, float)
        assert isinstance(necessity, float)
    
    def test_stability_metrics(self):
        """Test stability metrics computation."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        perm_importance = PermutationImportance(n_repeats=3, random_state=42)
        stability = StabilityMetrics(random_state=42)
        
        result = stability.cross_validation_stability(X, y, perm_importance, n_splits=3)
        
        assert "stability_score" in result
        assert "mean_correlation" in result
        assert isinstance(result["stability_score"], float)
    
    def test_comprehensive_evaluator(self):
        """Test comprehensive evaluator."""
        X, y, _ = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance_scores = np.random.rand(5)
        evaluator = ComprehensiveEvaluator(random_state=42)
        
        result = evaluator.evaluate_explanation(model, X, y, importance_scores)
        
        assert "faithfulness" in result
        assert "overall_score" in result
        assert isinstance(result["overall_score"], float)


class TestVizModule:
    """Test visualization module."""
    
    def test_feature_importance_visualizer_init(self):
        """Test visualizer initialization."""
        visualizer = FeatureImportanceVisualizer(figsize=(10, 6), dpi=100)
        
        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 100
        assert len(visualizer.colors) == 10
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        visualizer = FeatureImportanceVisualizer()
        
        importance_scores = np.random.rand(5)
        feature_names = [f"feature_{i}" for i in range(5)]
        
        fig = visualizer.plot_feature_importance(
            importance_scores, feature_names, "Test Method"
        )
        
        assert fig is not None
        assert hasattr(fig, 'axes')
    
    def test_create_summary_report(self):
        """Test summary report creation."""
        visualizer = FeatureImportanceVisualizer()
        
        # Create dummy results
        results = {
            "feature_names": [f"feature_{i}" for i in range(5)],
            "methods": {
                "permutation": {
                    "importance_scores": np.random.rand(5),
                    "method": "permutation"
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer.create_summary_report(results, save_dir=temp_dir)
            
            # Check if files were created
            files = os.listdir(temp_dir)
            assert len(files) > 0


class TestUtilsModule:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy random
        np.random.seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        val2 = np.random.rand()
        
        assert val1 == val2
    
    def test_validate_inputs(self):
        """Test input validation."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]
        
        # Should not raise
        validate_inputs(X, y, feature_names)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            validate_inputs(X, y, feature_names[:3])  # Wrong length
    
    def test_normalize_importance_scores(self):
        """Test importance score normalization."""
        scores = np.array([1, 2, 3, 4, 5])
        
        # Test minmax normalization
        normalized = normalize_importance_scores(scores, method="minmax")
        assert np.min(normalized) == 0
        assert np.max(normalized) == 1
        
        # Test sum normalization
        normalized = normalize_importance_scores(scores, method="sum")
        assert np.sum(normalized) == 1.0
        
        # Test zscore normalization
        normalized = normalize_importance_scores(scores, method="zscore")
        assert np.isclose(np.mean(normalized), 0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1, atol=1e-10)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Load data
        X, y, feature_names = load_synthetic_data(
            n_samples=200,
            n_features=8,
            random_state=42
        )
        
        # Initialize explainer
        explainer = FeatureImportanceExplainer(
            methods=["permutation", "tree"],
            random_state=42
        )
        
        # Compute explanations
        results = explainer.explain(X, y, feature_names)
        
        # Evaluate explanations
        evaluator = ComprehensiveEvaluator(random_state=42)
        evaluation_results = {}
        
        for method_name, method_result in results["methods"].items():
            if "importance_scores" in method_result:
                eval_result = evaluator.evaluate_explanation(
                    None, X, y, method_result["importance_scores"]
                )
                evaluation_results[method_name] = eval_result
        
        # Create visualizations
        visualizer = FeatureImportanceVisualizer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer.create_summary_report(results, evaluation_results, save_dir=temp_dir)
            
            # Verify files were created
            files = os.listdir(temp_dir)
            assert len(files) > 0
        
        # Test top features
        top_features = explainer.get_top_features(results, "permutation", top_k=3)
        assert len(top_features) == 3
        
        # Test saving/loading results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            explainer.save_results(results, f.name)
            
            loaded_results = explainer.load_results(f.name)
            assert loaded_results["feature_names"] == results["feature_names"]
            
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
