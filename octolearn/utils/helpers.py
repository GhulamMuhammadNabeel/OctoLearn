"""
Helper Utilities Module - Modern & Complete
"""

import logging
import sys
import time
import traceback
from typing import Any, List, Dict, Optional, Callable
from functools import wraps

import pandas as pd
import numpy as np
from ..config import LOGGING_CONFIG

# ====================================================================
# CUSTOM EXCEPTIONS
# ====================================================================

class OctolearnError(Exception):
    """Base exception for Octolearn errors."""
    pass

class ProfilingError(OctolearnError): pass
class RiskScoringError(OctolearnError): pass
class PreprocessingError(OctolearnError): pass
class FeatureEngineeringError(OctolearnError): pass
class ModelTrainingError(OctolearnError): pass
class OptimizationError(OctolearnError): pass
class EvaluationError(OctolearnError): pass
class ReportGenerationError(OctolearnError): pass

# ====================================================================
# LOGGING SETUP
# ====================================================================

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOGGING_CONFIG.get('level', logging.INFO))

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional File Handler
        log_file = LOGGING_CONFIG.get('file')
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except IOError:
                logger.warning(f"Cannot write log file: {log_file}")

    return logger

# ====================================================================
# DECORATORS
# ====================================================================

def handle_exceptions(raise_error: bool = False, logger_obj: Optional[logging.Logger] = None):
    """Decorator to handle exceptions gracefully."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"Error in {func.__name__}: {e}"
                tb = traceback.format_exc()
                if logger_obj:
                    logger_obj.error(f"{msg}\n{tb}")
                if raise_error:
                    raise OctolearnError(msg) from e
                return None
        return wrapper
    return decorator

def log_execution(logger_obj: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            if logger_obj:
                logger_obj.info(f"Starting {func.__name__}...")
            result = func(*args, **kwargs)
            end = time.time()
            if logger_obj:
                logger_obj.info(f"{func.__name__} completed in {end - start:.2f}s")
            return result
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, backoff: str = 'exponential'):
    """Retry function execution with backoff on failure."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = (2 ** attempt) if backoff == 'exponential' else (attempt + 1)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator

# ====================================================================
# VALIDATION UTILITIES
# ====================================================================

def validate_dataframe(df: Any, name: str = "df") -> bool:
    """Validate that input is a non-empty pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df)}")
    if df.empty:
        raise ValueError(f"{name} is empty")
    return True

def validate_series(series: Any, name: str = "series") -> bool:
    """Validate that input is a non-empty pandas Series or DataFrame."""
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pandas Series/DataFrame, got {type(series)}")
    if len(series) == 0:
        raise ValueError(f"{name} is empty")
    return True

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_memory_usage(obj: Any) -> str:
    """Return memory usage of an object in human-readable format."""
    import sys
    bytes_used = sys.getsizeof(obj)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.1f} TB"

def dict_to_table(data: Dict[str, List[Any]], max_rows: int = 10) -> str:
    """Convert dictionary of lists to a formatted table string."""
    for k in data:
        if isinstance(data[k], list):
            data[k] = data[k][:max_rows]

    col_widths = {col: max(len(str(col)), max(len(str(v)) for v in data[col])) for col in data}
    header = " | ".join(f"{col.ljust(col_widths[col])}" for col in data)
    lines = [header, "-" * len(header)]
    num_rows = len(next(iter(data.values())))
    for i in range(num_rows):
        row = " | ".join(f"{str(data[col][i]).ljust(col_widths[col])}" for col in data)
        lines.append(row)
    return "\n".join(lines)
