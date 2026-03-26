# Preprocessing Module

> `octolearn/preprocessing/`

## Purpose

The preprocessing module transforms raw, messy data into clean, numeric, model-ready features. It is the **third phase** of the pipeline (after profiling and train/test splitting). It enforces the **leakage prevention rule**: statistics are learned only from training data via `fit()`, then applied consistently to test/new data via `transform()`.

## Files

### `auto_cleaner.py`

#### `AutoCleaner`
The core cleaning engine. Implements `fit()` / `transform()` / `fit_transform()` to ensure train-test consistency.

**Cleaning pipeline (executed in order):**

1. **ID Column Removal** — Drops columns identified as identifiers by the profiler (high unique ratio, naming patterns like `*_id`, `*_uuid`).
2. **Constant Column Removal** — Drops zero-variance columns that add no predictive value.
3. **Missing Value Imputation** — Strategy per type:
   - Numeric: `mean` (default), `median`, or `constant`
   - Categorical: `mode` (most frequent) or `constant`
   - Users can override per-column via `imputer_strategy={'age': 'median'}`
4. **Categorical Encoding** — Smart selection based on cardinality:
   - Low cardinality (≤10 unique): One-Hot Encoding
   - High cardinality (>10 unique): Ordinal Encoding
   - Users can override via `encoder_strategy`
5. **Rare Category Grouping** — Categories appearing in <1% of rows are collapsed into an `_OTHER` category to prevent feature explosion.
6. **Feature Scaling** — Applied after encoding:
   - `standard` (default): Zero mean, unit variance
   - `robust`: Median-based (resistant to outliers)
   - `minmax`: Scale to [0, 1]
   - `None`: Skip scaling

**Key design decisions:**
- All statistics (mean, mode, category mappings, scaler params) are stored during `fit()` and reused in `transform()`. This prevents data leakage when cleaning test data.
- The cleaner returns a `cleaning_log_` dict documenting every operation performed — enabling the "Data Journey" section in PDF reports.
- Target encoding is handled separately: string labels are converted via `LabelEncoder` before entering the cleaner.

### `sampler.py`

#### `AutoSampler`
Handles class imbalance for classification tasks via resampling strategies.

**Strategies:**
| Strategy | Method | Best For |
|:---|:---|:---|
| `auto` | Auto-selects based on imbalance ratio and dataset size | Default |
| `smote` | Synthetic Minority Over-sampling | Small-medium datasets |
| `adasyn` | Adaptive Synthetic Sampling | Boundary-focused |
| `undersample` | Random majority undersampling | Large datasets (>100k) |
| `combine` | SMOTE + Tomek links | Noisy boundaries |
| `none` | Skip sampling | Balanced data |

**Auto-selection logic:**
- Imbalance ratio < 1.5 → `none` (not imbalanced enough)
- Minority class < 6 samples → `undersample` (SMOTE needs k=5 neighbors)
- Dataset > 100k rows → `undersample` (SMOTE too slow)
- Imbalance ratio > 10 → `combine` (severe imbalance)
- Default → `smote`

**Important:** Sampling is applied **only to training data**, never to test data, to prevent evaluation leakage.

### `pipeline_builder.py`

#### `PipelineBuilder`
Converts the fitted `AutoCleaner` configuration into a standalone scikit-learn `Pipeline` object for deployment.

**Why a separate builder?**
The `AutoCleaner` uses custom logic (rare category grouping, dynamic encoding selection) that doesn't map 1:1 to scikit-learn transformers. The `PipelineBuilder` bridges this gap by constructing a `ColumnTransformer` with the exact steps used during training, wrapped in a standard `Pipeline` that can be deployed without OctoLearn.

```python
builder = PipelineBuilder(automl.cleaner_, automl.clean_profile_)
pipeline = builder.build()  # Returns sklearn Pipeline
```

## Data Flow

```
X_train, y_train (raw)
    │
    ▼
AutoCleaner.fit(X_train, y_train)
    ├── Learn imputation stats
    ├── Learn encoding mappings
    └── Learn scaler params
    │
    ▼
AutoCleaner.transform(X_train) → X_train_clean
AutoCleaner.transform(X_test)  → X_test_clean  (same stats applied)
    │
    ▼
AutoSampler.fit_resample(X_train_clean, y_train)  (classification only)
    │
    ▼
Clean, balanced, numeric data ready for modeling
```

## Dependencies

- `pandas`, `numpy`, `scipy` — data operations
- `sklearn.preprocessing` — `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`
- `imblearn` (optional) — SMOTE, ADASYN, RandomUnderSampler, SMOTETomek
