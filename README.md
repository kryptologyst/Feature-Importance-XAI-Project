# Feature Importance XAI Project

## DISCLAIMER

**IMPORTANT: This project is for research and educational purposes only.**

XAI outputs may be unstable, misleading, or incorrect. These explanations are not a substitute for human judgment and should not be used for regulated decisions without human review. The methods implemented here are experimental and may not generalize across different datasets or model architectures.

## Overview

This project provides a comprehensive framework for feature importance visualization and analysis in machine learning models. It implements multiple state-of-the-art methods for global interpretability including permutation importance, SHAP, SAGE, and stability analysis.

## Features

- **Multiple Methods**: Permutation importance, SHAP (Tree/Kernel/Deep), SAGE, stability analysis
- **Comprehensive Evaluation**: Faithfulness, stability, and fidelity metrics
- **Interactive Demo**: Streamlit-based visualization interface
- **Production Ready**: Type hints, comprehensive testing, configuration management
- **Research Focus**: Designed for interpretability research and education

## Quick Start

1. **Installation**:
   ```bash
   pip install -e .
   ```

2. **Run Demo**:
   ```bash
   streamlit run demo/app.py
   ```

3. **Basic Usage**:
   ```python
   from src.explainers import FeatureImportanceExplainer
   from src.data import load_synthetic_data
   
   # Load data
   X, y, feature_names = load_synthetic_data()
   
   # Initialize explainer
   explainer = FeatureImportanceExplainer()
   
   # Get explanations
   explanations = explainer.explain(X, y, method='shap')
   ```

## Dataset Schema

The project supports tabular datasets with the following metadata structure:

```json
{
  "features": {
    "feature_name": {
      "type": "numerical|categorical|binary",
      "range": [min, max],
      "monotonic": true|false,
      "sensitive": true|false
    }
  },
  "target": {
    "type": "classification|regression",
    "classes": ["class1", "class2"] // for classification
  }
}
```

## Methods Implemented

### Global Feature Importance
- **Permutation Importance**: Model-agnostic feature importance
- **SHAP**: Shapley Additive Explanations (Tree, Kernel, Deep)
- **SAGE**: Shapley Additive Global Importance
- **Stability Analysis**: Cross-validation stability metrics

### Evaluation Metrics
- **Faithfulness**: Deletion/insertion AUC, sufficiency/necessity
- **Stability**: Kendall τ, Spearman ρ across seeds/splits
- **Fidelity**: Surrogate model accuracy

## Limitations

1. **Explanation Instability**: Feature importance rankings may vary across different random seeds or data splits
2. **Model Dependency**: Different models may produce different importance rankings for the same data
3. **Correlation Effects**: Highly correlated features may show misleading importance values
4. **Computational Cost**: Some methods (especially SHAP) can be computationally expensive
5. **Interpretation Complexity**: Raw importance values require domain expertise to interpret correctly

## Project Structure

```
├── src/                    # Source code
│   ├── methods/           # XAI method implementations
│   ├── explainers/        # High-level explainer interfaces
│   ├── metrics/           # Evaluation metrics
│   ├── viz/              # Visualization utilities
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model implementations
│   ├── eval/             # Evaluation framework
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── assets/               # Generated visualizations
└── demo/                 # Interactive demo
```

## Configuration

The project uses YAML configuration files for reproducible experiments:

```yaml
# configs/default.yaml
data:
  dataset: "synthetic"
  test_size: 0.2
  random_state: 42

model:
  type: "random_forest"
  n_estimators: 100

explainer:
  methods: ["permutation", "shap", "sage"]
  n_samples: 1000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{feature_importance_xai,
  title={Feature Importance XAI: A Comprehensive Framework for Interpretability},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Feature-Importance-XAI-Project}
}
```
# Feature-Importance-XAI-Project
