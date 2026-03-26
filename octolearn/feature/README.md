# Feature Module

> `octolearn/feature/`

## Purpose

The feature module provides feature engineering capabilities: date extraction, skew correction, and pairwise interaction analysis. It operates during **Phase 5** of the pipeline (after cleaning, before modeling).

## Files

### `generator.py`

#### `FeatureGenerator`
Scikit-learn compatible transformer (`BaseEstimator`, `TransformerMixin`) for automated feature engineering.

**Transformations (applied in order):**

1. **Date Feature Extraction** — Detects datetime columns (via `pd.to_datetime` heuristic) and extracts: year, month, day, weekday, is_weekend. Optionally drops the original date column.

2. **Skewed Feature Correction** — Identifies numeric features with absolute skewness above a threshold (default: 1.0) and applies log transform:
   - For positive values: `log1p(x)`
   - For non-positive values: `log(x + |min| + 1)` (shift to positive domain)
   - Creates `log_{col}` columns; optionally drops originals.

3. **Interaction Features** — Uses `FeatureInteractionAnalyzer` to discover top-N statistically significant interactions and generates them as new columns.

**Usage:**
```python
gen = FeatureGenerator(profile=dataset_profile)
gen.fit(X_train, y_train)   # Learn which features are skewed, which are dates, etc.
X_enhanced = gen.transform(X_train)
```

**Design note:** The generator is separate from the `_FeaturePoolBuilder` in the optimization module. The generator is used for basic feature engineering (Phase 5), while the pool builder is used for the more aggressive synthetic feature search (Phase 5.5).

### `interaction_analyzer.py`

#### `FeatureInteractionAnalyzer`
Analyzes three types of feature interactions and ranks them by correlation with the target.

**Interaction types:**

| Type | Formula | Naming Convention |
|:---|:---|:---|
| Polynomial | A² | `A^2` |
| Pairwise | A × B | `A_x_B` |
| Ratio | A / B | `A_/_B` |

**Key methods:**
| Method | Description |
|:---|:---|
| `analyze()` | Run all interaction analyses, return ranked results |
| `create_interaction_features(names)` | Generate specified interaction columns |

**How analysis works:**
For each pair of numeric features, the analyzer:
1. Computes the interaction (product, ratio, or polynomial)
2. Calculates Pearson correlation with the target
3. Filters by `min_corr_interaction` threshold
4. Returns top-K sorted by absolute correlation

**Key design decisions:**
- Division-by-zero is handled by replacing zeros with NaN before division
- Infinite values from division are replaced with NaN then dropped for correlation
- The analyzer stores a reference to the original data, so `create_interaction_features()` can regenerate features for the transform step

## Data Flow

```
Cleaned X, y, DatasetProfile
    │
    ▼
FeatureGenerator.fit()
    ├── _find_date_columns()     → self.date_cols_
    ├── _find_skewed_features()  → self.skewed_feats_
    └── _find_interactions()     → self.interaction_names_, self.analyzer_
    │
    ▼
FeatureGenerator.transform()
    ├── Date extraction (year, month, day, ...)
    ├── Log transforms for skewed features
    └── Interaction features from analyzer
    │
    ▼
Enhanced X with new columns
```

## Dependencies

- `pandas`, `numpy` — data operations
- `sklearn.preprocessing.PolynomialFeatures` — polynomial generation
- `../config.py` — `FEATURE_GENERATION_CONFIG`, `INTERACTION_CONFIG`
