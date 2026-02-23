import os
import pandas as pd
from octolearn import AutoML, OptimizationConfig, ReportingConfig

from sklearn.datasets import make_classification
import numpy as np

# Create a more complex synthetic dataset to trigger intelligent narratives
X_np, y_np = make_classification(n_samples=1000, n_features=15, n_informative=5, n_redundant=3, random_state=42)
X = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(15)])

# Introduce missing values
missing_mask = np.random.rand(*X.shape) < 0.05
X[missing_mask] = np.nan

# Introduce a categorical feature
X['user_category'] = np.random.choice(['Premium', 'Standard', 'Basic', 'Enterprise'], size=len(X))

y = pd.Series(y_np, name='target_conversion')

# Initialize with Joint Bayesian Optimization and Baseline Score
optim_config = OptimizationConfig(
    use_optuna=True, 
    optuna_trials_per_model=15, 
    baseline_score=0.85
)

reporting_config = ReportingConfig(
    generate_report=True,
    report_title="Intelligent Enhancement Test Report"
)

automl = AutoML(
    optimization_config=optim_config,
    reporting_config=reporting_config
)

print("Starting AutoML Pipeline...")
automl.fit(X, y)

print("Generating Report...")
report_path = automl.generate_report("test_report.pdf")
print(f"Report generated successfully at: {report_path}")
