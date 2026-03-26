# Utils Module

> `octolearn/utils/`

## Purpose

The utils module provides cross-cutting concerns: logging, error handling decorators, input validation, and general-purpose utility functions used throughout the library.

## Files

### `helpers.py`

#### Custom Exceptions

All OctoLearn-specific exceptions inherit from `OctolearnError`:

| Exception | Typical Trigger |
|:---|:---|
| `OctolearnError` | Base exception |
| `ProfilingError` | Profiling phase failure |
| `PreprocessingError` | Cleaning/encoding failure |
| `FeatureEngineeringError` | Feature generation failure |
| `ModelTrainingError` | Training phase failure |
| `OptimizationError` | Optuna optimization failure |
| `EvaluationError` | Metrics computation failure |
| `ReportGenerationError` | PDF report failure |
| `RiskScoringError` | Risk scoring failure |

#### Logging

```python
from octolearn.utils.helpers import setup_logger
logger = setup_logger(__name__)
```

`setup_logger()` creates a logger with:
- Console handler (stdout) with configurable format from `LOGGING_CONFIG`
- Optional file handler (if `LOGGING_CONFIG['file']` is set)
- Level from `LOGGING_CONFIG['level']` (default: `INFO`)

Every module in OctoLearn creates its own logger via `setup_logger(__name__)`.

#### Decorators

**`@log_execution(logger_obj)`** — Logs start time and duration of the decorated function. Used on every major pipeline method (`fit()`, `transform()`, `evaluate()`, etc.).

**`@handle_exceptions(raise_error, logger_obj)`** — Catches exceptions, logs the traceback, and optionally re-raises as `OctolearnError`. Returns `None` on suppressed errors.

**`@retry_with_backoff(max_retries, backoff)`** — Retries failed operations with exponential or linear backoff. Useful for flaky I/O operations.

#### Validation

```python
validate_dataframe(df, name="df")   # Raises TypeError/ValueError if invalid
validate_series(series, name="y")   # Raises TypeError/ValueError if invalid
```

These are used at the entry point of `AutoML.fit()` to fail fast with clear error messages.

#### Utility Functions

| Function | Description |
|:---|:---|
| `flatten_dict(d, sep='_')` | Flatten nested dicts into single-level `{'a_b_c': val}` |
| `get_memory_usage(obj)` | Human-readable memory footprint (KB/MB/GB) |
| `dict_to_table(data, max_rows)` | Convert dict-of-lists to formatted ASCII table |

## Design Rationale

- **Centralized logging** ensures consistent formatting across all modules.
- **Decorator-based error handling** keeps business logic clean — errors are caught and logged without cluttering the core pipeline code.
- **Validation at boundaries** follows the "fail fast" principle — invalid inputs are caught at `fit()` entry, not deep inside the pipeline.

## Dependencies

- `logging`, `sys`, `time`, `traceback` — Python stdlib
- `pandas`, `numpy` — type checking for validation
- `../config.py` — `LOGGING_CONFIG` for logger configuration
