"""Streamlit demo application for Feature Importance XAI project."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import load_synthetic_data, load_sklearn_dataset, preprocess_data
from explainers import FeatureImportanceExplainer
from viz import FeatureImportanceVisualizer
from metrics import ComprehensiveEvaluator
from utils import set_seed, get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Feature Importance XAI Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Feature Importance XAI Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This demo is for research and educational purposes only. XAI outputs may be unstable, 
    misleading, or incorrect. These explanations are not a substitute for human judgment and 
    should not be used for regulated decisions without human review.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["synthetic", "iris", "wine", "breast_cancer"],
        index=0
    )
    
    # Task type
    task_type = st.sidebar.selectbox(
        "Task Type",
        ["classification", "regression"],
        index=0
    )
    
    # Method selection
    methods = st.sidebar.multiselect(
        "Select Methods",
        ["permutation", "tree", "shap", "sage", "stability"],
        default=["permutation", "tree", "shap"]
    )
    
    # Parameters
    st.sidebar.subheader("Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    n_features = st.sidebar.slider("Number of Features", 5, 20, 10)
    n_informative = st.sidebar.slider("Informative Features", 2, n_features, min(5, n_features))
    random_state = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Load data
    with st.spinner("Loading data..."):
        if dataset_option == "synthetic":
            X, y, feature_names = load_synthetic_data(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                task_type=task_type,
                random_state=random_state
            )
        else:
            X, y, feature_names = load_sklearn_dataset(dataset_option)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
            X, y, feature_names, random_state=random_state
        )
    
    # Display dataset info
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Train Samples", len(X_train))
    with col4:
        st.metric("Test Samples", len(X_test))
    
    # Display feature names
    st.subheader("🔤 Feature Names")
    st.write(", ".join(feature_names))
    
    # Run analysis
    if st.button("🚀 Run Feature Importance Analysis", type="primary"):
        
        with st.spinner("Computing feature importance..."):
            # Set seed
            set_seed(random_state)
            
            # Initialize explainer
            explainer = FeatureImportanceExplainer(
                methods=methods,
                random_state=random_state,
                n_samples=min(500, len(X_train))  # Limit samples for demo
            )
            
            # Compute explanations
            results = explainer.explain(
                X_train, y_train, feature_names, task_type=task_type
            )
            
            # Initialize evaluator
            evaluator = ComprehensiveEvaluator(random_state=random_state)
            
            # Evaluate explanations
            evaluation_results = {}
            for method_name, method_result in results["methods"].items():
                if "importance_scores" in method_result:
                    eval_result = evaluator.evaluate_explanation(
                        explainer.method_objects.get(method_name, None),
                        X_train, y_train, method_result["importance_scores"]
                    )
                    evaluation_results[method_name] = eval_result
        
        # Display results
        st.success("Analysis completed successfully!")
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Individual Results", 
            "🔄 Comparative Analysis", 
            "📊 Evaluation Metrics",
            "📋 Summary Report",
            "💾 Download Results"
        ])
        
        with tab1:
            display_individual_results(results)
        
        with tab2:
            display_comparative_analysis(results)
        
        with tab3:
            display_evaluation_metrics(evaluation_results)
        
        with tab4:
            display_summary_report(results, evaluation_results)
        
        with tab5:
            display_download_options(results, evaluation_results)


def display_individual_results(results: Dict[str, Any]):
    """Display individual method results."""
    st.subheader("📈 Individual Method Results")
    
    for method_name, method_result in results["methods"].items():
        if "importance_scores" not in method_result:
            continue
        
        st.subheader(f"Method: {method_name.title()}")
        
        # Get top features
        importance_scores = method_result["importance_scores"]
        top_indices = np.argsort(importance_scores)[-10:][::-1]
        
        # Create DataFrame for display
        df = pd.DataFrame({
            'Feature': [results["feature_names"][i] for i in top_indices],
            'Importance': importance_scores[top_indices]
        })
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(df)), df['Importance'], color='skyblue')
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{method_name.title()} - Top 10 Features')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df['Importance'])):
            ax.text(value + 0.01 * max(df['Importance']), i, f'{value:.3f}',
                   va='center', ha='left', fontsize=9)
        
        st.pyplot(fig)


def display_comparative_analysis(results: Dict[str, Any]):
    """Display comparative analysis."""
    st.subheader("🔄 Comparative Analysis")
    
    # Method correlations
    if "comparative_analysis" in results:
        comp_analysis = results["comparative_analysis"]
        
        if "method_correlations" in comp_analysis:
            st.subheader("Method Correlations")
            
            # Create correlation matrix
            correlations = comp_analysis["method_correlations"]
            method_pairs = list(correlations.keys())
            
            corr_data = []
            for pair in method_pairs:
                corr_data.append({
                    'Method Pair': pair,
                    'Pearson': correlations[pair]['pearson'],
                    'Spearman': correlations[pair]['spearman'],
                    'Kendall': correlations[pair]['kendall']
                })
            
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True)
            
            # Plot correlation heatmap
            methods = list(results["methods"].keys())
            if len(methods) >= 2:
                corr_matrix = np.eye(len(methods))
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i != j:
                            pair_key = f"{method1}_vs_{method2}"
                            if pair_key in correlations:
                                corr_matrix[i, j] = correlations[pair_key]['kendall']
                            else:
                                pair_key = f"{method2}_vs_{method1}"
                                if pair_key in correlations:
                                    corr_matrix[i, j] = correlations[pair_key]['kendall']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           xticklabels=methods, yticklabels=methods, ax=ax)
                ax.set_title('Method Correlation Matrix (Kendall)')
                st.pyplot(fig)
        
        # Consensus features
        if "consensus_features" in comp_analysis:
            st.subheader("Consensus Features")
            consensus_features = comp_analysis["consensus_features"]
            if consensus_features:
                consensus_names = [results["feature_names"][i] for i in consensus_features]
                st.write("Features that appear in top rankings across multiple methods:")
                st.write(", ".join(consensus_names))
            else:
                st.write("No consensus features found.")
        
        # Disagreement features
        if "disagreement_features" in comp_analysis:
            st.subheader("Disagreement Features")
            disagreement_features = comp_analysis["disagreement_features"]
            if disagreement_features:
                disagreement_names = [results["feature_names"][i] for i in disagreement_features]
                st.write("Features with high disagreement between methods:")
                st.write(", ".join(disagreement_names))
            else:
                st.write("No significant disagreement between methods.")


def display_evaluation_metrics(evaluation_results: Dict[str, Any]):
    """Display evaluation metrics."""
    st.subheader("📊 Evaluation Metrics")
    
    if not evaluation_results:
        st.warning("No evaluation results available.")
        return
    
    for method_name, eval_result in evaluation_results.items():
        st.subheader(f"Method: {method_name.title()}")
        
        # Overall score
        overall_score = eval_result.get("overall_score", 0)
        st.metric("Overall Score", f"{overall_score:.3f}")
        
        # Faithfulness metrics
        if "faithfulness" in eval_result and "error" not in eval_result["faithfulness"]:
            st.subheader("Faithfulness Metrics")
            faithfulness = eval_result["faithfulness"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Deletion AUC", f"{faithfulness.get('deletion_auc', 0):.3f}")
            with col2:
                st.metric("Insertion AUC", f"{faithfulness.get('insertion_auc', 0):.3f}")
            with col3:
                st.metric("Sufficiency", f"{faithfulness.get('sufficiency_score', 0):.3f}")
            with col4:
                st.metric("Necessity", f"{faithfulness.get('necessity_score', 0):.3f}")
        
        # Stability metrics
        if "stability" in eval_result and "error" not in eval_result["stability"]:
            st.subheader("Stability Metrics")
            stability = eval_result["stability"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Correlation", f"{stability.get('mean_correlation', 0):.3f}")
            with col2:
                st.metric("Stability Score", f"{stability.get('stability_score', 0):.3f}")


def display_summary_report(results: Dict[str, Any], evaluation_results: Dict[str, Any]):
    """Display summary report."""
    st.subheader("📋 Summary Report")
    
    # Dataset summary
    st.subheader("Dataset Summary")
    st.write(f"**Total Samples:** {results['n_samples']}")
    st.write(f"**Features:** {results['n_features']}")
    st.write(f"**Task Type:** {results['task_type']}")
    
    # Methods summary
    st.subheader("Methods Applied")
    for method_name in results["methods"].keys():
        st.write(f"• {method_name.title()}")
    
    # Top features consensus
    if "comparative_analysis" in results:
        comp_analysis = results["comparative_analysis"]
        if "consensus_features" in comp_analysis:
            consensus_features = comp_analysis["consensus_features"]
            if consensus_features:
                consensus_names = [results["feature_names"][i] for i in consensus_features]
                st.subheader("Consensus Top Features")
                st.write(", ".join(consensus_names))
    
    # Evaluation summary
    if evaluation_results:
        st.subheader("Evaluation Summary")
        overall_scores = [eval_result.get("overall_score", 0) for eval_result in evaluation_results.values()]
        if overall_scores:
            avg_score = np.mean(overall_scores)
            st.metric("Average Overall Score", f"{avg_score:.3f}")


def display_download_options(results: Dict[str, Any], evaluation_results: Dict[str, Any]):
    """Display download options."""
    st.subheader("💾 Download Results")
    
    # Create downloadable files
    import json
    import io
    
    # Results JSON
    results_json = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="📄 Download Results (JSON)",
        data=results_json,
        file_name="feature_importance_results.json",
        mime="application/json"
    )
    
    # Evaluation results JSON
    if evaluation_results:
        eval_json = json.dumps(evaluation_results, indent=2, default=str)
        st.download_button(
            label="📊 Download Evaluation Results (JSON)",
            data=eval_json,
            file_name="evaluation_results.json",
            mime="application/json"
        )
    
    # CSV summary
    summary_data = []
    for method_name, method_result in results["methods"].items():
        if "importance_scores" in method_result:
            for i, (feature_name, score) in enumerate(zip(results["feature_names"], method_result["importance_scores"])):
                summary_data.append({
                    'Method': method_name,
                    'Feature': feature_name,
                    'Importance': score,
                    'Rank': i + 1
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Summary CSV",
            data=csv,
            file_name="feature_importance_summary.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
