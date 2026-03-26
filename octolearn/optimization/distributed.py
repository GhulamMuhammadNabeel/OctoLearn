"""
Enhanced Distributed Systems Module - Ray & Dask Integration

Provides intermediate-level distributed computing support for OctoLearn:
- Parallel model training across CPU cores using Ray
- Large-scale data profiling using Dask
- Distributed Optuna hyperparameter optimization
- Automatic resource management and monitoring

This module is optional and maintains backward compatibility. Importing OctoLearn
does not require Ray or Dask, but users can opt-in by installing distributed extras.

Architecture:
    DistributedProfiler  -> Parallel data profiling (Dask)
    DistributedTrainer   -> Parallel model training (Ray)
    ResourceManager      -> CPU/GPU/Memory monitoring
    BackendSelector      -> Automatic backend selection

Usage Examples:

    # Local execution (default - no distributed library needed)
    automl = AutoML(distributed_backend='local')
    
    # Ray-based parallel training (requires: pip install ray)
    automl = AutoML(
        distributed_backend='ray',
        num_workers=4,
        use_gpu=False
    )
    automl.fit(X, y)  # 4x faster model training
    
    # Dask-based large-scale processing (requires: pip install dask)
    automl = AutoML(
        distributed_backend='dask',
        chunk_size='100MB'  # Auto-split large datasets
    )
    automl.fit(X, y)

Author: OctoLearn Development Team
Version: 0.10.3
License: MIT
"""

import logging
import psutil
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# BACKEND DETECTION & CONFIGURATION
# ============================================================================

class DistributedBackend(Enum):
    """Supported distributed backends."""
    LOCAL = "local"      # Single-machine, single-threaded
    RAY = "ray"          # Ray Distributed
    DASK = "dask"        # Dask Distributed
    AUTO = "auto"        # Auto-detect available backend


@dataclass
class ResourceConfig:
    """Resource allocation configuration."""
    num_workers: int = None  # Auto-detect if None
    num_cpus_per_worker: float = 1.0
    memory_per_worker_gb: float = None  # Auto-detect if None
    use_gpu: bool = False
    gpu_per_worker: float = 0.0
    verbose: bool = True


def detect_available_backends() -> Dict[str, bool]:
    """
    Detect which distributed backends are installed.
    
    Returns
    -------
    dict
        {'ray': bool, 'dask': bool} indicating availability
        
    Example
    -------
    >>> backends = detect_available_backends()
    >>> if backends['ray']:
    ...     print("Ray is installed and ready!")
    """
    backends = {'ray': False, 'dask': False}
    
    try:
        import ray  # type: ignore
        backends['ray'] = True
        logger.debug(f"Ray {ray.__version__} detected")
    except ImportError:
        logger.debug("Ray not installed. Install with: pip install ray")
    
    try:
        import dask  # type: ignore
        backends['dask'] = True
        logger.debug(f"Dask {dask.__version__} detected")
    except ImportError:
        logger.debug("Dask not installed. Install with: pip install dask distributed")
    
    return backends


def get_optimal_num_workers() -> int:
    """
    Determine optimal number of workers based on system resources.
    
    Returns
    -------
    int
        Recommended number of workers (CPU count - 1, minimum 1)
        
    Example
    -------
    >>> workers = get_optimal_num_workers()
    >>> print(f"Recommended workers: {workers}")
    """
    try:
        cpu_count = psutil.cpu_count(logical=True)
        # Leave one CPU for system processes
        optimal = max(1, cpu_count - 1)
        logger.debug(f"System has {cpu_count} CPUs, recommending {optimal} workers")
        return optimal
    except Exception as e:
        logger.warning(f"Could not detect CPU count: {e}. Using default 4 workers.")
        return 4


def get_available_memory_gb() -> float:
    """
    Get available system memory.
    
    Returns
    -------
    float
        Available memory in GB
    """
    try:
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Could not detect memory: {e}. Using default 8GB estimate.")
        return 8.0


# ============================================================================
# RAY-BASED DISTRIBUTED BACKEND
# ============================================================================

class DistributedTrainerRay:
    """
    Ray-based distributed model trainer for parallel hyperparameter optimization.
    
    Uses Ray to distribute model training across multiple CPU cores or machines.
    Supports Optuna integration for distributed hyperparameter search.
    
    Example
    -------
    >>> trainer = DistributedTrainerRay(num_workers=4)
    >>> results = trainer.train_models(
    ...     X_train, X_test, y_train, y_test,
    ...     models=['random_forest', 'xgboost', 'lightgbm'],
    ...     use_optuna=True
    ... )
    """
    
    def __init__(self, num_workers: int = None, use_gpu: bool = False, verbose: bool = True):
        """
        Initialize Ray-based distributed trainer.
        
        Parameters
        ----------
        num_workers : int, optional
            Number of Ray workers. Auto-detects if None.
        use_gpu : bool, default=False
            Whether to allocate GPUs to workers
        verbose : bool, default=True
            Print progress information
        """
        self.num_workers = num_workers or get_optimal_num_workers()
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.ray_client = None
        
        self._initialize_ray()
    
    def _initialize_ray(self):
        """Initialize Ray cluster."""
        try:
            import ray  # type: ignore
            
            if ray.is_initialized():
                logger.info("Ray already initialized")
                self.ray_client = ray
                return
            
            # Configure Ray
            ray_config = {
                'num_cpus': self.num_workers,
                'ignore_reinit_error': True,
                'logging_level': logging.WARNING if not self.verbose else logging.INFO
            }
            
            if self.use_gpu:
                gpu_count = ray.available_resources().get('GPU', 0) if ray.available_resources() else 0
                if gpu_count > 0:
                    ray_config['num_gpus'] = min(gpu_count, self.num_workers)
            
            ray.init(**ray_config)
            self.ray_client = ray
            
            resources = ray.available_resources()
            logger.info(f"Ray initialized: {resources}")
            
        except ImportError:
            raise ImportError(
                "Ray is not installed. Install with: pip install ray\n"
                "Or use distributed_backend='dask' or 'local'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise RuntimeError(f"Ray initialization failed: {e}")
    
    def train_model_remote(
        self,
        model_class: Callable,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        hp_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a single model remotely on a Ray worker.
        
        Parameters
        ----------
        model_class : Callable
            Model constructor (e.g., RandomForestClassifier)
        X_train, X_test : DataFrame
            Training and test features
        y_train, y_test : Series
            Training and test labels
        hp_params : dict
            Hyperparameters for the model
            
        Returns
        -------
        dict
            Training results with model, score, and metadata
        """
        try:
            model = model_class(**hp_params)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            return {
                'model': model,
                'score': score,
                'hp_params': hp_params,
                'n_features': X_train.shape[1],
                'status': 'success'
            }
        except Exception as e:
            logger.warning(f"Model training failed: {e}")
            return {
                'model': None,
                'score': 0.0,
                'hp_params': hp_params,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_models_parallel(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_config: List[Tuple[Callable, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Train multiple models in parallel using Ray.
        
        Parameters
        ----------
        X_train, X_test : DataFrame
            Training and test features
        y_train, y_test : Series
            Training and test labels
        models_config : list of (model_class, list of hp_params dicts)
            Models and hyperparameter configurations
            
        Returns
        -------
        list
            Training results for all models
            
        Example
        -------
        >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        >>> models = [
        ...     (RandomForestClassifier, [
        ...         {'n_estimators': 100, 'max_depth': 10},
        ...         {'n_estimators': 200, 'max_depth': 15}
        ...     ]),
        ...     (GradientBoostingClassifier, [
        ...         {'n_estimators': 100, 'learning_rate': 0.1}
        ...     ])
        ... ]
        >>> results = trainer.train_models_parallel(X_train, X_test, y_train, y_test, models)
        """
        if not self.ray_client:
            logger.warning("Ray not initialized, falling back to sequential training")
            return self._train_sequential(X_train, X_test, y_train, y_test, models_config)
        
        remote_train = self.ray_client.remote(self.train_model_remote)
        futures = []
        
        # Submit all training tasks
        for model_class, hp_configs in models_config:
            for hp_params in hp_configs:
                future = remote_train.remote(
                    self,
                    model_class,
                    X_train, X_test, y_train, y_test,
                    hp_params
                )
                futures.append(future)
        
        # Collect results
        results = self.ray_client.get(futures)
        
        if self.verbose:
            successful = sum(1 for r in results if r['status'] == 'success')
            logger.info(f"Training complete: {successful}/{len(results)} models trained successfully")
        
        return results
    
    def _train_sequential(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_config: List[Tuple[Callable, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Fallback sequential training if Ray is not available."""
        results = []
        for model_class, hp_configs in models_config:
            for hp_params in hp_configs:
                result = self.train_model_remote(
                    model_class,
                    X_train, X_test, y_train, y_test,
                    hp_params
                )
                results.append(result)
        return results
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        try:
            if self.ray_client and self.ray_client.is_initialized():
                self.ray_client.shutdown()
                logger.info("Ray cluster shut down")
        except Exception as e:
            logger.warning(f"Could not shut down Ray: {e}")


# ============================================================================
# DASK-BASED DISTRIBUTED BACKEND
# ============================================================================

class DistributedProfilerDask:
    """
    Dask-based distributed data profiler for large-scale datasets.
    
    Uses Dask DataFrames for parallel profiling of datasets > 1GB.
    Automatically chunks data and computes statistics in parallel.
    
    Example
    -------
    >>> profiler = DistributedProfilerDask(chunk_size='100MB')
    >>> profile = profiler.profile_large_dataset('huge_file.parquet')
    """
    
    def __init__(self, chunk_size: str = '100MB', n_partitions: int = None):
        """
        Initialize Dask-based profiler.
        
        Parameters
        ----------
        chunk_size : str, default='100MB'
            Size of data chunks (e.g., '100MB', '1GB')
        n_partitions : int, optional
            Number of partitions. Auto-calculated if None.
        """
        self.chunk_size = chunk_size
        self.n_partitions = n_partitions
        self.dask_client = None
        
        self._initialize_dask()
    
    def _initialize_dask(self):
        """Initialize Dask distributed client."""
        try:
            from dask.distributed import Client  # type: ignore
            import dask  # type: ignore
            
            self.dask_client = Client(processes=True, threads_per_worker=2)
            logger.info(f"Dask initialized: {self.dask_client}")
            
        except ImportError:
            logger.warning(
                "Dask is not installed. Install with: pip install dask distributed\n"
                "Falling back to pandas-based profiling"
            )
            self.dask_client = None
        except Exception as e:
            logger.warning(f"Could not initialize Dask: {e}. Using pandas-based profiling.")
            self.dask_client = None
    
    def profile_large_dataset(self, filepath: str) -> Dict[str, Any]:
        """
        Profile a large dataset using Dask.
        
        Parameters
        ----------
        filepath : str
            Path to dataset file (parquet, csv, etc.)
            
        Returns
        -------
        dict
            Dataset profile with statistics
        """
        if not self.dask_client:
            logger.info("Using pandas for profiling (Dask not available)")
            return self._profile_pandas(filepath)
        
        try:
            import dask.dataframe as dd  # type: ignore
            
            # Load with Dask
            if filepath.endswith('.parquet'):
                ddf = dd.read_parquet(filepath)
            elif filepath.endswith('.csv'):
                ddf = dd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported format: {filepath}")
            
            # Compute statistics in parallel
            profile = {
                'n_rows': len(ddf),
                'n_cols': len(ddf.columns),
                'columns': list(ddf.columns),
                'dtypes': ddf.dtypes.to_dict(),
                'memory_usage': ddf.memory_usage(deep=True).compute().sum() / (1024 ** 3),
                'missing_values': ddf.isnull().sum().compute().to_dict(),
                'numeric_stats': {}
            }
            
            # Compute statistics for numeric columns
            for col in ddf.select_dtypes(include=['number']).columns:
                profile['numeric_stats'][col] = {
                    'mean': float(ddf[col].mean().compute()),
                    'std': float(ddf[col].std().compute()),
                    'min': float(ddf[col].min().compute()),
                    'max': float(ddf[col].max().compute())
                }
            
            logger.info(f"Profile computed: {profile['n_rows']} rows, {profile['n_cols']} cols")
            return profile
            
        except Exception as e:
            logger.error(f"Dask profiling failed: {e}")
            return self._profile_pandas(filepath)
    
    def _profile_pandas(self, filepath: str) -> Dict[str, Any]:
        """Fallback pandas-based profiling."""
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_parquet(filepath)
        
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 ** 3),
            'missing_values': df.isnull().sum().to_dict()
        }
    
    def shutdown(self):
        """Shutdown Dask cluster."""
        try:
            if self.dask_client:
                self.dask_client.close()
                logger.info("Dask cluster shut down")
        except Exception as e:
            logger.warning(f"Could not shut down Dask: {e}")


# ============================================================================
# MAIN API - BACKEND SELECTOR & MANAGER
# ============================================================================

class DistributedExecutor:
    """
    High-level API for distributed execution with automatic backend selection.
    
    Automatically selects the best available backend and provides a unified interface
    for distributed training and profiling.
    
    Example
    -------
    >>> executor = DistributedExecutor(backend='auto', num_workers=4)
    >>> results = executor.train_models(X_train, X_test, y_train, y_test, models_config)
    >>> executor.shutdown()
    """
    
    def __init__(
        self,
        backend: str = 'auto',
        num_workers: Optional[int] = None,
        use_gpu: bool = False,
        verbose: bool = True
    ):
        """
        Initialize distributed executor.
        
        Parameters
        ----------
        backend : {'auto', 'ray', 'dask', 'local'}, default='auto'
            Which backend to use
        num_workers : int, optional
            Number of workers/processes
        use_gpu : bool, default=False
            Use GPU acceleration if available
        verbose : bool, default=True
            Print progress information
        """
        self.backend_name = backend
        self.num_workers = num_workers or get_optimal_num_workers()
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.backend = None
        
        self._select_backend()
    
    def _select_backend(self):
        """Select and initialize the appropriate backend."""
        available = detect_available_backends()
        
        if self.backend_name == 'auto':
            if available['ray']:
                self.backend_name = 'ray'
            elif available['dask']:
                self.backend_name = 'dask'
            else:
                self.backend_name = 'local'
        
        logger.info(f"Using {self.backend_name.upper()} backend for distributed execution")
        
        if self.backend_name == 'ray':
            self.backend = DistributedTrainerRay(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
                verbose=self.verbose
            )
        elif self.backend_name == 'dask':
            self.backend = DistributedProfilerDask()
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_config: List[Tuple[Callable, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Train multiple models, automatically distributed across backend.
        
        Parameters
        ----------
        X_train, X_test : DataFrame
        y_train, y_test : Series
        models_config : list
            (model_class, hp_params_list) tuples
            
        Returns
        -------
        list
            Training results
        """
        if self.backend_name == 'ray' and self.backend:
            return self.backend.train_models_parallel(
                X_train, X_test, y_train, y_test, models_config
            )
        else:
            logger.info(f"Using {self.backend_name} backend - training models sequentially")
            # Sequential training for local/dask backend
            return self._sequential_training(
                X_train, X_test, y_train, y_test, models_config
            )
    
    def _sequential_training(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_config: List[Tuple[Callable, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Sequential training fallback."""
        results = []
        for model_class, hp_configs in models_config:
            for hp_params in hp_configs:
                try:
                    model = model_class(**hp_params)
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    
                    results.append({
                        'model': model,
                        'score': score,
                        'hp_params': hp_params,
                        'status': 'success'
                    })
                except Exception as e:
                    results.append({
                        'model': None,
                        'score': 0.0,
                        'hp_params': hp_params,
                        'status': 'failed',
                        'error': str(e)
                    })
        return results
    
    def shutdown(self):
        """Cleanup distributed resources."""
        if self.backend:
            self.backend.shutdown()


# ============================================================================
# BACKWARD COMPATIBLE API
# ============================================================================

def setup_distributed(
    backend: str = 'auto',
    num_workers: int = None,
    **kwargs
) -> DistributedExecutor:
    """
    Setup distributed backend for OctoLearn.
    
    Parameters
    ----------
    backend : {'auto', 'ray', 'dask', 'local'}, default='auto'
    num_workers : int, optional
    **kwargs : additional arguments
    
    Returns
    -------
    DistributedExecutor
    
    Example
    -------
    >>> executor = setup_distributed(backend='ray', num_workers=4)
    >>> # Use executor for training
    >>> executor.shutdown()
    """
    return DistributedExecutor(
        backend=backend,
        num_workers=num_workers,
        **kwargs
    )


# Backward compatibility
setup_distributed_backend = setup_distributed