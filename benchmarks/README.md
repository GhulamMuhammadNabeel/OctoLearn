# Benchmarks

This folder contains small reproducible benchmarks and example data for
quick validation and demo usage of OctoLearn.

Files:
- `benchmark_small.csv` - tiny synthetic dataset used by `notebooks/benchmark.ipynb`.

How to run the quick benchmark:

```bash
python - <<'PY'
import pandas as pd
from octolearn import AutoML

df = pd.read_csv('benchmarks/benchmark_small.csv')
X = df.drop('target', axis=1)
y = df['target']

automl = AutoML(train_models=False, generate_shap=False)
automl.fit(X, y)
print('Profile:', automl.report())
print('Risk:', automl.get_risk_score())
PY
```

For notebook usage, open `notebooks/benchmark.ipynb`.
