# Profiling Module

> `octolearn/profiling/`

## Purpose

The profiling module provides automated dataset analysis and health assessment. It is the **first phase** of the AutoML pipeline — executed before any preprocessing or training begins. Its output, the `DatasetProfile` dataclass, drives every subsequent decision in the pipeline.

## Files

### `data_profiler.py`

Contains two main components:

#### `DataProfiler`
The stateless analysis engine that produces a `DatasetProfile` from raw `(X, y)` data.

**Key method:**
```python
profiler = DataProfiler()
profile = profiler.profile(X: pd.DataFrame, y: pd.Series) -> DatasetProfile
```

**What it does (in order):**
1. **Task Type Inference** — Determines `classification` vs `regression` by analyzing the target variable's dtype and unique count.
2. **Feature Type Detection** — Classifies each column as `numeric`, `categorical`, `id`, `date`, or `text` using layered heuristics (unique ratio, dtype, naming patterns).
3. **Statistical Profiling** — Computes mean, median, std, skewness, kurtosis for numeric features; value counts and mode for categoricals.
4. **Quality Analysis** — Calculates missing ratios, identifies constant columns, low-variance features, and high-cardinality categoricals.
5. **Leakage Detection** — Flags features with >0.95 absolute correlation with the target as potential data leakage suspects.
6. **Imbalance Ratio** — Computes `min_class / max_class` for classification targets.

**Design rationale:**
- Profiling is run **twice** during the pipeline: once on raw data (Phase 1) and once on cleaned data (Phase 4). This enables the "Data Journey" comparison in reports.
- Type inference uses a multi-step heuristic rather than just `dtype` because many datasets have numeric IDs stored as integers or dates stored as strings.

#### `DatasetProfile`
A `@dataclass` that stores all profiling results. It is the "brain" of OctoLearn — nearly every downstream module reads from it.

**Key attributes:**
| Attribute | Type | Description |
|:---|:---|:---|
| `shape` | `tuple` | `(n_rows, n_cols)` |
| `task_type` | `str` | `'classification'` or `'regression'` |
| `feature_types` | `dict` | `{col: 'numeric'/'categorical'/'id'/...}` |
| `numeric_columns` | `list` | All detected numeric columns |
| `categorical_columns` | `list` | All detected categorical columns |
| `id_like_columns` | `list` | Auto-detected identifier columns |
| `constant_columns` | `list` | Zero-variance columns |
| `leakage_suspects` | `list` | Features with >0.95 target correlation |
| `missing_ratio` | `dict` | `{col: float}` missing value proportions |
| `imbalance_ratio` | `float` | Class ratio for classification |
| `data_quality_score` | `float` | 0–100 health score |

## Data Flow

```
Raw X, y
    │
    ▼
DataProfiler.profile()
    │
    ├── Task type inference
    ├── Feature type detection
    ├── Statistical profiling
    ├── Quality analysis
    └── Leakage detection
    │
    ▼
DatasetProfile (used by AutoCleaner, ModelTrainer, ReportGenerator, etc.)
```

## Dependencies

- `pandas`, `numpy` — data manipulation
- `scipy.stats` — skewness/kurtosis calculation
- `../config.py` — thresholds for type detection (e.g., `ID_UNIQUE_RATIO`, `CARDINALITY_THRESHOLD`)
