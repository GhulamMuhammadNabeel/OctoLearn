# Benchmarks

This folder contains reproducible benchmarks and example data for validating and demonstrating OctoLearn.

## Quick Validation

Use this tiny synthetic dataset for smoke tests:

- `benchmark_small.csv` - Small dataset with a 'target' column.

```bash
python - <<'PY'
import pandas as pd
from octolearn import AutoML

df = pd.read_csv('benchmarks/benchmark_small.csv')
X = df.drop('target', axis=1)
y = df['target']

automl = AutoML(train_models=False, generate_shap=False)
automl.fit(X, y)
print('Risk Score:', automl.get_risk_score())
PY
```

## Performance Benchmark

For a more comprehensive test, use the included benchmark script:

- `run_benchmarks.py` - Runs a full pipeline on the California Housing dataset.

### Execution

```bash
python benchmarks/run_benchmarks.py
```

This script will:
1. Load the California Housing dataset (20k+ rows).
2. Segment data into training and test sets.
3. Perform automated cleaning and profiling.
4. Execute hyperparameter optimization via Optuna.
5. Generate a professional PDF intelligence report.
6. Display a comparative leaderboard of all models.
