

"""
Octolearn - Automated Machine Learning Pipeline with Intelligent Dataset Profiling

This package provides a complete AutoML solution with the following phases:
    - Phase 1: Dataset profiling and analysis
    - Phase 2: Exploratory data analysis and risk assessment
    - Phase 3: Feature engineering and automatic cleaning
    - Phase 4: Multi-model training with Optuna HPO and registry

Attributes:
    __version__ (str): The version of the Octolearn package.
    __author__ (str): The author of the Octolearn package.
    __license__ (str): The license of the Octolearn package.
    __all__ (list): List of public objects of this module.

Example:
    >>> from octolearn import AutoML
    >>> automl = AutoML()
    >>> automl.fit(X, y)
    >>> automl.generate_report()

Note:
    See HOW_TO_USE.md and README.md for full usage details and advanced options.
"""

__version__ = "0.5.2"
__author__ = "Ghulam Muhammad Nabeel"
__license__ = "MIT"


from .core import AutoML
from .profiling.data_profiler import DataProfiler
from .models.registry import ModelRegistry

__all__ = [
    "AutoML",
    "DataProfiler",
    "ModelRegistry"
]