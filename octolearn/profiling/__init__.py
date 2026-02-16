

"""
Profiling submodule for Octolearn.

This module provides dataset profiling utilities including DataProfiler and DatasetProfile.

Attributes:
	DataProfiler (class): Main class for dataset profiling.
	DatasetProfile (class): Data structure for storing profile results.
	__all__ (list): List of public objects of this submodule.

Example:
	>>> from octolearn.profiling import DataProfiler
	>>> profiler = DataProfiler()
	>>> profile = profiler.profile(X, y)
"""

from .data_profiler import DataProfiler, DatasetProfile

__all__ = ["DataProfiler", "DatasetProfile"]
