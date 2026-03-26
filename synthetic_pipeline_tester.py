import pandas as pd
import numpy as np
import traceback
import sys
from sklearn.datasets import make_classification, make_regression
from octolearn import AutoML

def run_test():
    print("Generating complex synthetic dataset...")
    X_np, y_np = make_classification(
        n_samples=1500, n_features=25, n_informative=10, 
        n_redundant=5, n_classes=2, weights=[0.8, 0.2], random_state=42
    )

    X = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(25)])
    
    # Introduce real-world messiness
    X.loc[np.random.choice(X.index, 100), 'feature_0'] = np.nan
    X['cat_high'] = [f'id_{i}' for i in np.random.randint(0, 500, 1500)]
    X['cat_low'] = np.random.choice(['Sector_A', 'Sector_B', 'Sector_C', np.nan], size=1500)
    X['constant_col'] = 9.99
    
    y = pd.Series(y_np, name='target')

    print("\nInitializing AutoML...")
    try:
        automl = AutoML(show_progress=True)
        automl.feature_optimization_config.enable_feature_optimization = False
        automl.fit(X, y, optuna_trials=1, optuna_timeout=30, n_models=1)
        
        print("\nChecking intermediate components...")
        assert automl.raw_profile_ is not None, "Raw profile missing"
        assert automl.clean_profile_ is not None, "Clean profile missing"
        assert automl.X_train_ is not None, "X_train_ absent"
        assert automl.best_model_ is not None, "Best model was not selected"
        
        print("Shapes -> Raw: {}, Train: {}, Final Features: {}".format(
            X.shape, automl.X_train_.shape, len(automl.clean_profile_.columns)
        ))
        
        print("\nTesting prediction endpoint...")
        preds = automl.predict(X.head(10))
        assert len(preds) == 10, "Prediction shape mismatch"
        print("Predictions generated successfully:", preds[:3])
        
        print("\nTesting report generation...")
        pdf_path = automl.generate_report()
        assert pdf_path.endswith('.pdf'), "Report generation failed"
        print(f"Report saved to: {pdf_path}")
        
        print("\nTesting pipeline code export...")
        export_path = "best_pipeline.txt"
        automl.export_pipeline_code(filepath=export_path)
        
        import os
        assert os.path.exists(export_path), "Export file missing"
        with open(export_path, "r") as f:
             contents = f.read()
             assert "import pandas as pd" in contents
             print(f"Exported pipeline preview ({len(contents)} chars) successful.")
             
        print("\nALL SYNTHETIC TESTS PASSED [OK]")

    except Exception as e:
        print("\nSYNTHETIC TEST FAILED [ERROR]")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()
