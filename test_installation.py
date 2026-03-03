#!/usr/bin/env python3
"""Quick test script to verify the Feature Importance XAI project installation."""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data import load_synthetic_data, preprocess_data
        print("✓ Data module imported successfully")
        
        from explainers import FeatureImportanceExplainer
        print("✓ Explainer module imported successfully")
        
        from methods import PermutationImportance, TreeBasedImportance
        print("✓ Methods module imported successfully")
        
        from metrics import ComprehensiveEvaluator
        print("✓ Metrics module imported successfully")
        
        from viz import FeatureImportanceVisualizer
        print("✓ Visualization module imported successfully")
        
        from utils import set_seed, validate_inputs
        print("✓ Utils module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Set seed
        from utils import set_seed
        set_seed(42)
        print("✓ Seed setting works")
        
        # Load data
        from data import load_synthetic_data
        X, y, feature_names = load_synthetic_data(n_samples=100, n_features=5, random_state=42)
        print(f"✓ Data loading works: {X.shape}")
        
        # Initialize explainer
        from explainers import FeatureImportanceExplainer
        explainer = FeatureImportanceExplainer(methods=["permutation", "tree"], random_state=42)
        print("✓ Explainer initialization works")
        
        # Compute explanations
        results = explainer.explain(X, y, feature_names)
        print(f"✓ Explanation computation works: {len(results['methods'])} methods")
        
        # Test visualization
        from viz import FeatureImportanceVisualizer
        visualizer = FeatureImportanceVisualizer()
        print("✓ Visualizer initialization works")
        
        # Test evaluation
        from metrics import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator(random_state=42)
        print("✓ Evaluator initialization works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_demo_import():
    """Test that the demo can be imported."""
    print("\nTesting demo import...")
    
    try:
        from demo.app import main
        print("✓ Demo app imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Demo import failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Feature Importance XAI Project - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test demo import
    demo_ok = test_demo_import()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if imports_ok:
        print("✓ All imports successful")
    else:
        print("✗ Import tests failed")
    
    if functionality_ok:
        print("✓ Basic functionality works")
    else:
        print("✗ Basic functionality tests failed")
    
    if demo_ok:
        print("✓ Demo app can be imported")
    else:
        print("✗ Demo import failed")
    
    if imports_ok and functionality_ok and demo_ok:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: streamlit run demo/app.py")
        print("2. Run the analysis script: python scripts/run_analysis.py --help")
        print("3. Check the comprehensive example notebook")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
