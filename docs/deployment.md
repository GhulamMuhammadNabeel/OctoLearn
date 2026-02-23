# Deployment Guide

Learn how to deploy your OctoLearn models to production and keep your documentation updated using GitHub Pages.

---

## 🚀 Deploying OctoLearn Models

Moving from training to production with OctoLearn follows a two-step process: **Export** and **Inference**.

### 1. Exporting the Champion Pipeline
After calling `fit()`, OctoLearn identifies the best model. You can extract this model along with all necessary preprocessing steps as a unified scikit-learn pipeline.

```python
from octolearn import AutoML
import joblib

# 1. Fit the orchestrator
automl = AutoML()
automl.fit(X, y)

# 2. Extract the unified pipeline (Preprocessing + Best Model)
pipeline = automl.get_pipeline()

# 3. Save for production
joblib.dump(pipeline, "octolearn_pipeline.pkl")
```

### 2. Side-car Inference
In your production environment, you only need the `.pkl` file and standard libraries (`scikit-learn`, `pandas`). You **do not** strictly need OctoLearn installed for inference, as the exported pipeline is a standard scikit-learn object.

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load("octolearn_pipeline.pkl")

# Predict on new data
new_data = pd.read_csv("live_feed.csv")
predictions = pipeline.predict(new_data)
```

---

## 📖 Deploying Documentation to GitHub Pages

OctoLearn uses **MkDocs** to generate its documentation website. You can automate the deployment process using GitHub Actions.

### Automation with GitHub Actions
Create a file at `.github/workflows/docs.yml` in your repository with the following content:

```yaml
name: Deploy Documentation
on:
  push:
    branches:
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.x
      - run: pip install mkdocs-material mkdocstrings python-handler
      - run: mkdocs gh-deploy --force
```

### Manual Deployment
If you prefer to deploy manually from your local machine:

1. Install the documentation dependencies:
   ```bash
   pip install mkdocs-material mkdocstrings[python]
   ```
2. Run the deployment command:
   ```bash
   mkdocs gh-deploy
   ```

The site will be hosted at `https://<your-username>.github.io/OctoLearn/`.

---

## 🛡️ Production Best Practices

- **Versioning**: Always tag your models with a version number or timestamp.
- **Monitoring**: Use OctoLearn's `RiskScorer` on incoming live data periodically to detect feature drift.
- **Retraining**: Schedule retraining cycles when your primary metrics start to degrade below the baseline established during the initial `fit()`.
