"""Main script for running feature importance analysis."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import load_synthetic_data, load_sklearn_dataset, preprocess_data
from explainers import FeatureImportanceExplainer
from viz import FeatureImportanceVisualizer
from metrics import ComprehensiveEvaluator
from utils import set_seed, save_results

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('feature_importance.log')
        ]
    )

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Feature Importance XAI Analysis")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["synthetic", "iris", "wine", "breast_cancer"],
                       help="Dataset to use")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples for synthetic data")
    parser.add_argument("--n-features", type=int, default=10,
                       help="Number of features for synthetic data")
    parser.add_argument("--task-type", type=str, default="classification",
                       choices=["classification", "regression"],
                       help="Type of task")
    
    # Method arguments
    parser.add_argument("--methods", nargs="+", 
                       default=["permutation", "tree", "shap"],
                       choices=["permutation", "tree", "shap", "sage", "stability"],
                       help="Methods to use")
    parser.add_argument("--n-samples-shap", type=int, default=1000,
                       help="Number of samples for SHAP")
    parser.add_argument("--n-repeats", type=int, default=10,
                       help="Number of repeats for permutation importance")
    parser.add_argument("--n-splits", type=int, default=5,
                       help="Number of splits for stability analysis")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results")
    
    # Other arguments
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(args.random_state)
    
    logger.info("Starting Feature Importance XAI Analysis")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading data...")
        if args.dataset == "synthetic":
            X, y, feature_names = load_synthetic_data(
                n_samples=args.n_samples,
                n_features=args.n_features,
                task_type=args.task_type,
                random_state=args.random_state
            )
        else:
            X, y, feature_names = load_sklearn_dataset(args.dataset)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            X, y, feature_names, random_state=args.random_state
        )
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize explainer
        logger.info("Initializing explainer...")
        explainer = FeatureImportanceExplainer(
            methods=args.methods,
            random_state=args.random_state,
            n_samples=args.n_samples_shap,
            n_repeats=args.n_repeats,
            n_splits=args.n_splits
        )
        
        # Compute explanations
        logger.info("Computing feature importance...")
        results = explainer.explain(
            X_train, y_train, feature_names, task_type=args.task_type
        )
        
        # Evaluate explanations
        logger.info("Evaluating explanations...")
        evaluator = ComprehensiveEvaluator(random_state=args.random_state)
        evaluation_results = {}
        
        for method_name, method_result in results["methods"].items():
            if "importance_scores" in method_result:
                eval_result = evaluator.evaluate_explanation(
                    None, X_train, y_train, method_result["importance_scores"]
                )
                evaluation_results[method_name] = eval_result
        
        # Create visualizations
        if args.save_plots:
            logger.info("Creating visualizations...")
            visualizer = FeatureImportanceVisualizer()
            plots_dir = os.path.join(args.output_dir, "plots")
            visualizer.create_summary_report(
                results, evaluation_results, save_dir=plots_dir
            )
        
        # Save results
        if args.save_results:
            logger.info("Saving results...")
            results_file = os.path.join(args.output_dir, "results.json")
            save_results(results, results_file)
            
            eval_file = os.path.join(args.output_dir, "evaluation.json")
            save_results(evaluation_results, eval_file)
        
        # Print summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Methods applied: {list(results['methods'].keys())}")
        
        if evaluation_results:
            overall_scores = [
                eval_result.get("overall_score", 0) 
                for eval_result in evaluation_results.values()
            ]
            avg_score = sum(overall_scores) / len(overall_scores)
            logger.info(f"Average overall score: {avg_score:.3f}")
        
        # Print top features
        for method_name in results["methods"].keys():
            if "importance_scores" in results["methods"][method_name]:
                top_features = explainer.get_top_features(results, method_name, top_k=5)
                logger.info(f"Top 5 features ({method_name}): {[f[0] for f in top_features]}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
