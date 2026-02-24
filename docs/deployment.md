# Deployment Guide

Learn how to deploy your OctoLearn models to production and manage your live documentation. 

> [!NOTE]
> While GitHub Pages is a standard option, this documentation is currently hosted on **Vercel** ([octolearn.vercel.app](https://octolearn.vercel.app)) for superior reliability and to bypass GitHub Actions billing limitations.

---

## Deploying OctoLearn Models

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

## Deploying Documentation to GitHub Pages

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

> [!WARNING]
> If your GitHub Actions are disabled (e.g., due to billing issues), follow the **Alternative Hosting** instructions below.

---

## Alternative Hosting Solutions

If you encounter issues with GitHub Actions, you can use these free alternatives that don't rely on GitHub's internal billing.

### 1. GitHub Pages (Manual/Branch Method)
Instead of using GitHub Actions to deploy, you can configure GitHub to serve your documentation directly from a branch.

1.  **Push the branch**: Run `mkdocs gh-deploy` from your local terminal. This creates and pushes the `gh-pages` branch.
2.  **Change Source**: In your GitHub Repository, go to **Settings** > **Pages**.
3.  **Select Branch**: Under "Build and deployment", change the Source to **Deploy from a branch** and select `gh-pages` and `/ (root)`.

### 2. Netlify (Recommended)
Netlify provides superior performance and bypasses GitHub billing locks.

1.  Create a Free account at [Netlify.com](https://www.netlify.com).
2.  Select **Import from GitHub** and authorize the OctoLearn repository.
3.  **Branch to deploy**: Select `master` (or your main branch).
4.  Netlify will automatically detect the settings from the `netlify.toml` file we've included in the repository.
5.  **Build Command**: `mkdocs build`
6.  **Publish Directory**: `site`

### 3. Vercel
Vercel is another industrial-strength free alternative.

1.  Connect your GitHub account to [Vercel](https://vercel.com).
2.  Import OctoLearn.
3.  **Branch to deploy**: Select `master` (NOT `gh-pages`).
4.  Set the Framework Preset to **Other** (if not detected).
5.  **Build Command**: `mkdocs build`
6.  **Output Directory**: `site`
7.  **Dependencies**: Vercel will automatically install dependencies from the `requirements.txt` file in the root.

> [!TIP]
> **Why not deploy the `gh-pages` branch?**
> The `gh-pages` branch only contains the final HTML files. If you deploy it, you don't need a "Build Command". However, it's better to deploy the `master` branch so that your site updates automatically whenever you push code changes.

---

## Production Best Practices

- **Versioning**: Always tag your models with a version number or timestamp.
- **Monitoring**: Use OctoLearn's `RiskScorer` on incoming live data periodically to detect feature drift.
- **Retraining**: Schedule retraining cycles when your primary metrics start to degrade below the baseline established during the initial `fit()`.
