# Helper utilities module

import logging
import sys
from typing import Any, List, Dict, Optional
from functools import wraps
import traceback
from ..config import LOGGING_CONFIG

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class OctoLearnError(Exception):
    """Base exception for OctoLearn"""
    pass

class ProfilingError(OctoLearnError):
    """Raised when dataset profiling fails"""
    pass

class RiskScoringError(OctoLearnError):
    """Raised when risk scoring fails"""
    pass

class PreprocessingError(OctoLearnError):
    """Raised when preprocessing fails"""
    pass

class FeatureEngineeringError(OctoLearnError):
    """Raised when feature engineering fails"""
    pass

class ModelTrainingError(OctoLearnError):
    """Raised when model training fails"""
    pass

class OptimizationError(OctoLearnError):
    """Raised when hyperparameter optimization fails"""
    pass

class EvaluationError(OctoLearnError):
    """Raised when evaluation fails"""
    pass

class ReportGenerationError(OctoLearnError):
    """Raised when report generation fails"""
    pass

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger for OctoLearn modules.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
        
    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOGGING_CONFIG['level']))
        
        # Formatter
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    return logger

# ============================================================================
# DECORATORS FOR ERROR HANDLING & LOGGING
# ============================================================================

def handle_exceptions(raise_error: bool = False, logger_obj: Optional[logging.Logger] = None):
    """
    Decorator to handle exceptions gracefully.
    
    Parameters
    ----------
    raise_error : bool
        Whether to raise exception or return None
    logger_obj : logging.Logger, optional
        Logger object for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                error_traceback = traceback.format_exc()
                
                if logger_obj:
                    logger_obj.error(f"{error_msg}\n{error_traceback}")
                
                if raise_error:
                    raise OctoLearnError(error_msg) from e
                return None
        return wrapper
    return decorator

def log_execution(logger_obj: Optional[logging.Logger] = None):
    """
    Decorator to log function execution.
    
    Parameters
    ----------
    logger_obj : logging.Logger, optional
        Logger object for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if logger_obj:
                logger_obj.info(f"Starting {func_name}...")
            
            result = func(*args, **kwargs)
            
            if logger_obj:
                logger_obj.info(f"Completed {func_name}.")
            
            return result
        return wrapper
    return decorator

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_dataframe(X: Any, name: str = "X") -> bool:
    """
    Validate that input is a pandas DataFrame.
    
    Parameters
    ----------
    X : Any
        Input to validate
    name : str
        Name for error messages
        
    Returns
    -------
    bool
        True if valid
        
    Raises
    ------
    ValueError
        If not a DataFrame
    """
    import pandas as pd
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(X)}")
    
    if X.empty:
        raise ValueError(f"{name} is empty")
    
    return True

def validate_series(y: Any, name: str = "y") -> bool:
    """
    Validate that input is a pandas Series.
    
    Parameters
    ----------
    y : Any
        Input to validate
    name : str
        Name for error messages
        
    Returns
    -------
    bool
        True if valid
        
    Raises
    ------
    ValueError
        If not a Series
    """
    import pandas as pd
    
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise ValueError(f"{name} must be a pandas Series/DataFrame, got {type(y)}")
    
    if len(y) == 0:
        raise ValueError(f"{name} is empty")
    
    return True

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten nested dictionary.
    
    Parameters
    ----------
    d : dict
        Dictionary to flatten
    parent_key : str
        Parent key for recursion
    sep : str
        Separator for keys
        
    Returns
    -------
    dict
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def get_memory_usage(obj: Any) -> str:
    """
    Get memory usage of an object.
    
    Parameters
    ----------
    obj : Any
        Object to measure
        
    Returns
    -------
    str
        Memory usage string (e.g., "50.2 MB")
    """
    import sys
    
    bytes_used = sys.getsizeof(obj)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024
    
    return f"{bytes_used:.1f} TB"

def retry_with_backoff(max_retries: int = 3, backoff: str = 'exponential'):
    """
    Decorator to retry function with backoff.
    
    Parameters
    ----------
    max_retries : int
        Maximum number of retries
    backoff : str
        'linear' or 'exponential' backoff
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    # Calculate wait time
                    if backoff == 'exponential':
                        wait_time = 2 ** attempt
                    else:
                        wait_time = attempt + 1
                    
                    print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        return wrapper
    return decorator

def dict_to_table(data: Dict[str, List], max_rows: int = 10) -> str:
    """
    Convert dictionary to formatted table string.
    
    Parameters
    ----------
    data : dict
        Dictionary with columns as keys and lists as values
    max_rows : int
        Maximum rows to show
        
    Returns
    -------
    str
        Formatted table
    """
    # Limit rows
    for key in data:
        if isinstance(data[key], list):
            data[key] = data[key][:max_rows]
    
    # Find column widths
    col_widths = {}
    for col, values in data.items():
        col_widths[col] = max(len(str(col)), max(len(str(v)) for v in values))
    
    # Build table
    result = []
    
    # Header
    header = " | ".join(f"{col.ljust(col_widths[col])}" for col in data.keys())
    result.append(header)
    result.append("-" * len(header))
    
    # Rows
    num_rows = len(next(iter(data.values())))
    for i in range(num_rows):
        row = " | ".join(
            f"{str(data[col][i]).ljust(col_widths[col])}"
            for col in data.keys()
        )
        result.append(row)
    
    return "\n".join(result)
