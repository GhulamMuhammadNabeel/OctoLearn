# Experiments Module

> `octolearn/experiments/`

## Purpose

The experiments module contains all **intelligence and reporting** components: PDF report generation, visualization, risk scoring, outlier detection, feature importance, and recommendation engines. These components are invoked during the final phases of the pipeline to produce the OctoLearn Intelligence Report.

## Files

### `report_generator.py`

#### `ReportGenerator`
Produces the professional PDF intelligence report using ReportLab.

**Report sections (in order):**
1. Cover page (title, timestamp, dataset shape, task type)
2. Data Risk Assessment (0–100 score + factor breakdown)
3. Dataset Health (missing values, duplicates, feature types)
4. Feature Distribution Plots
5. Correlation Heatmap + Narrative
6. Feature Importance Plot
7. Outlier Narratives (extreme values mapped to targets)
8. Data Journey (before/after cleaning distributions)
9. Feature Optimization Results (when enabled)
10. Model Arena Leaderboard
11. Champion Model Deep-Dive (confusion matrix, ROC, PR curves)
12. SHAP Global Importance
13. Preprocessing Suggestions
14. Recommendations

**Design rationale:** Uses ReportLab instead of HTML/CSS for pixel-perfect control over layout, consistent rendering across platforms, and no browser dependency. Custom ShantellSans font files are embedded from `octolearn/fonts/`.

### `plot_generator.py`

#### `PlotGenerator`
Creates matplotlib/seaborn visualizations saved as temporary PNG files.

**Key methods:**
| Method | Description |
|:---|:---|
| `generate_smart_visuals(limit)` | Distribution plots for top features |
| `generate_correlation_heatmap(corr_top_n)` | Full heatmap or Top-N bar chart |
| `generate_feature_importance_plot(importances)` | Horizontal bar chart |
| `generate_performance_plots(y_true, y_pred, y_proba)` | ROC, PR, residual plots |
| `generate_shap_plot()` | SHAP summary plot |

**Adaptive behavior:** When >10 features exist, the heatmap automatically switches to a Top-N correlation bar chart to prevent visual noise.

### `risk_scorer.py`

#### `RiskScorer`
Computes a 0–100 data quality risk score based on:
- Missing data prevalence and patterns
- Class imbalance severity
- Leakage suspects
- Feature skewness
- High cardinality

Returns: `(score: int, category: str, factors: dict)`

Categories: `0-20` Low Risk, `21-40` Moderate, `41-60` Elevated, `61-80` High, `81-100` Critical

### `outlier_detector.py`

#### `OutlierDetector`
Multi-method outlier detection combining three approaches:
1. **IQR** — Interquartile range with 1.5× multiplier
2. **Z-Score** — Standard deviations from mean (threshold: 3.0)
3. **Isolation Forest** — Unsupervised anomaly detection

Returns per-feature outlier counts, bounds, indices, and narrative recommendations.

### `baseline_importance.py`

#### `BaselineImportance`
Computes feature importance before model training using permutation importance with a simple model. This provides early insight into which features are likely predictive.

### `recommendation_engine.py`

#### `RecommendationEngine`
Generates plain-English ML recommendations categorized by priority (`high`, `medium`, `informational`).

**Example outputs:**
- "Remove highly correlated feature `account_status` (r=0.99 with target)"
- "Apply SMOTE to handle class imbalance (ratio: 0.12)"
- "Consider log transformation for `salary` (skewness: 3.2)"

### `preprocessing_suggester.py`

#### `PreprocessingSuggester`
Generates structured preprocessing advice organized into 6 categories:
- Missing value strategy
- Categorical encoding
- Scaling strategy
- Feature engineering
- Column actions
- Risk mitigation

## Data Flow

```
AutoML (post-training)
    │
    ▼
_generate_report_components()
    ├── PlotGenerator → distribution, heatmap, importance, performance PNGs
    ├── BaselineImportance → feature scores
    ├── RiskScorer → score, category, factors
    ├── RecommendationEngine → categorized text recommendations
    └── OutlierDetector → outlier narratives
    │
    ▼
ReportGenerator.generate_report()
    │
    ▼
PDF Intelligence Report (saved to disk)
```

## Dependencies

- `reportlab` — PDF generation
- `matplotlib`, `seaborn` — visualizations
- `shap` — SHAP importance analysis
- `sklearn.ensemble.IsolationForest` — outlier detection
