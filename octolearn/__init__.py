"""
OctoLearn - Automated Machine Learning Pipeline with Intelligent Dataset Profiling

Complete AutoML with Phase 1-4:
- Phase 1: Dataset profiling and analysis
- Phase 2: Exploratory data analysis and risk assessment
- Phase 3: Feature engineering and automatic cleaning
- Phase 4: Multi-model training with Optuna HPO and registry
"""

__version__ = "0.4.0"
__author__ = "Ghulam Muhammad Nabeel"
__license__ = "MIT"

from .core import AutoML

__all__ = ["AutoML", "__version__"]
