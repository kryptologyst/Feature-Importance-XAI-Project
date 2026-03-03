"""Visualization utilities for feature importance analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class FeatureImportanceVisualizer:
    """Comprehensive visualization for feature importance analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size.
            dpi: Figure DPI.
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up color palette
        self.colors = sns.color_palette("husl", 10)
    
    def plot_feature_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str],
        method_name: str = "Feature Importance",
        top_k: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature importance as horizontal bar chart.
        
        Args:
            importance_scores: Feature importance scores.
            feature_names: List of feature names.
            method_name: Name of the method.
            top_k: Number of top features to show.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        # Get top K features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_names)), top_scores, color=self.colors[0])
        
        # Customize plot
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{method_name} - Top {top_k} Features')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score + 0.01 * max(top_scores), i, f'{score:.3f}',
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_comparative_importance(
        self,
        results: Dict[str, Any],
        methods: List[str] = None,
        top_k: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparative feature importance across methods.
        
        Args:
            results: Results from explainer.
            methods: List of methods to compare.
            top_k: Number of top features to show.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if methods is None:
            methods = list(results["methods"].keys())
        
        # Get top K features from first method
        first_method = methods[0]
        if first_method not in results["methods"]:
            raise ValueError(f"Method {first_method} not found in results")
        
        first_scores = results["methods"][first_method]["importance_scores"]
        top_indices = np.argsort(first_scores)[-top_k:][::-1]
        top_names = [results["feature_names"][i] for i in top_indices]
        
        # Create DataFrame for plotting
        plot_data = []
        for method in methods:
            if method in results["methods"] and "importance_scores" in results["methods"][method]:
                scores = results["methods"][method]["importance_scores"]
                for i, name in enumerate(top_names):
                    idx = results["feature_names"].index(name)
                    plot_data.append({
                        'Feature': name,
                        'Method': method,
                        'Importance': scores[idx]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create grouped bar plot
        sns.barplot(data=df, x='Feature', y='Importance', hue='Method', ax=ax)
        
        # Customize plot
        ax.set_title(f'Comparative Feature Importance - Top {top_k} Features')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Method')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Comparative plot saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(
        self,
        results: Dict[str, Any],
        methods: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot correlation matrix between different methods.
        
        Args:
            results: Results from explainer.
            methods: List of methods to compare.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if methods is None:
            methods = list(results["methods"].keys())
        
        # Extract importance scores
        method_scores = {}
        for method in methods:
            if method in results["methods"] and "importance_scores" in results["methods"][method]:
                method_scores[method] = results["methods"][method]["importance_scores"]
        
        if len(method_scores) < 2:
            logger.warning("Need at least 2 methods for correlation matrix")
            return None
        
        # Create correlation matrix
        df = pd.DataFrame(method_scores)
        correlation_matrix = df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        # Customize plot
        ax.set_title('Method Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        return fig
    
    def plot_stability_analysis(
        self,
        stability_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot stability analysis results.
        
        Args:
            stability_results: Results from stability analysis.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        if "importance_scores_list" not in stability_results:
            logger.warning("No stability data available")
            return None
        
        importance_scores_list = stability_results["importance_scores_list"]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Plot 1: Box plot of importance scores across runs
        importance_array = np.array(importance_scores_list)
        ax1.boxplot(importance_array.T, labels=[f'Feature {i+1}' for i in range(importance_array.shape[1])])
        ax1.set_title('Feature Importance Stability Across Runs')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Importance Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Correlation distribution
        correlations = []
        for i in range(len(importance_scores_list)):
            for j in range(i + 1, len(importance_scores_list)):
                scores1 = importance_scores_list[i]
                scores2 = importance_scores_list[j]
                corr = np.corrcoef(scores1, scores2)[0, 1]
                correlations.append(corr)
        
        ax2.hist(correlations, bins=20, alpha=0.7, color=self.colors[1])
        ax2.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        ax2.set_title('Correlation Distribution Between Runs')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Stability analysis plot saved to {save_path}")
        
        return fig
    
    def plot_evaluation_metrics(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot evaluation metrics.
        
        Args:
            evaluation_results: Results from comprehensive evaluation.
            save_path: Path to save the plot.
            
        Returns:
            matplotlib Figure object.
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        # Plot faithfulness metrics
        if "faithfulness" in evaluation_results and "error" not in evaluation_results["faithfulness"]:
            faithfulness = evaluation_results["faithfulness"]
            metrics = ["deletion_auc", "insertion_auc", "sufficiency_score", "necessity_score"]
            values = [faithfulness.get(metric, 0) for metric in metrics]
            
            bars = axes[0].bar(metrics, values, color=self.colors[:len(metrics)])
            axes[0].set_title('Faithfulness Metrics')
            axes[0].set_ylabel('Score')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot stability metrics
        if "stability" in evaluation_results and "error" not in evaluation_results["stability"]:
            stability = evaluation_results["stability"]
            metrics = ["mean_correlation", "stability_score"]
            values = [stability.get(metric, 0) for metric in metrics]
            
            bars = axes[1].bar(metrics, values, color=self.colors[2:4])
            axes[1].set_title('Stability Metrics')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot overall score
        overall_score = evaluation_results.get("overall_score", 0)
        axes[2].bar(['Overall Score'], [overall_score], color=self.colors[4])
        axes[2].set_title('Overall Evaluation Score')
        axes[2].set_ylabel('Score')
        axes[2].set_ylim(0, 1)
        
        # Add value label
        axes[2].text(0, overall_score + 0.05, f'{overall_score:.3f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot method comparison (if available)
        if "comparative_analysis" in evaluation_results:
            comp_analysis = evaluation_results["comparative_analysis"]
            if "method_correlations" in comp_analysis:
                correlations = comp_analysis["method_correlations"]
                method_pairs = list(correlations.keys())
                kendall_scores = [correlations[pair]["kendall"] for pair in method_pairs]
                
                bars = axes[3].bar(range(len(method_pairs)), kendall_scores, color=self.colors[5:])
                axes[3].set_title('Method Correlations (Kendall)')
                axes[3].set_ylabel('Correlation')
                axes[3].set_xticks(range(len(method_pairs)))
                axes[3].set_xticklabels(method_pairs, rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, kendall_scores):
                    axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Evaluation metrics plot saved to {save_path}")
        
        return fig
    
    def create_summary_report(
        self,
        results: Dict[str, Any],
        evaluation_results: Optional[Dict[str, Any]] = None,
        save_dir: str = "assets"
    ) -> None:
        """Create a comprehensive summary report with all visualizations.
        
        Args:
            results: Results from explainer.
            evaluation_results: Results from evaluation.
            save_dir: Directory to save plots.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Creating comprehensive summary report")
        
        # Plot individual method importance
        for method_name, method_result in results["methods"].items():
            if "importance_scores" in method_result:
                self.plot_feature_importance(
                    method_result["importance_scores"],
                    results["feature_names"],
                    method_name,
                    save_path=os.path.join(save_dir, f"{method_name}_importance.png")
                )
        
        # Plot comparative analysis
        self.plot_comparative_importance(
            results,
            save_path=os.path.join(save_dir, "comparative_importance.png")
        )
        
        # Plot correlation matrix
        self.plot_correlation_matrix(
            results,
            save_path=os.path.join(save_dir, "correlation_matrix.png")
        )
        
        # Plot stability analysis (if available)
        for method_name, method_result in results["methods"].items():
            if "stability_score" in method_result:
                self.plot_stability_analysis(
                    method_result,
                    save_path=os.path.join(save_dir, f"{method_name}_stability.png")
                )
        
        # Plot evaluation metrics (if available)
        if evaluation_results:
            self.plot_evaluation_metrics(
                evaluation_results,
                save_path=os.path.join(save_dir, "evaluation_metrics.png")
            )
        
        logger.info(f"Summary report created in {save_dir}")
    
    def plot_interactive_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str],
        method_name: str = "Feature Importance"
    ) -> None:
        """Create interactive plot using plotly (if available).
        
        Args:
            importance_scores: Feature importance scores.
            feature_names: List of feature names.
            method_name: Name of the method.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Create DataFrame
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            })
            
            # Sort by importance
            df = df.sort_values('Importance', ascending=True)
            
            # Create interactive bar chart
            fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                        title=f'{method_name} - Interactive View',
                        labels={'Importance': 'Importance Score', 'Feature': 'Features'})
            
            fig.update_layout(height=600, showlegend=False)
            fig.show()
            
        except ImportError:
            logger.warning("Plotly not available. Install with: pip install plotly")
            # Fallback to matplotlib
            self.plot_feature_importance(importance_scores, feature_names, method_name)
