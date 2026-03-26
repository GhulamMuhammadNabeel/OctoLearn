import time
import os
import psutil

class TimeEstimator:
    """
    Heuristic-based ETA Estimator for OctoLearn AutoML pipelines.
    
    Uses dataset dimensions, optimization configs, and hardware specs
    to project roughly how long the pipeline will take.
    """
    
    def __init__(self, data_config, profiling_config, modeling_config, optimization_config, parallel_config):
        self.data_config = data_config
        self.profiling_config = profiling_config
        self.modeling_config = modeling_config
        self.opt_config = optimization_config
        self.parallel_config = parallel_config

    def estimate(self, X_shape):
        """
        Estimate completion time in seconds.
        
        Parameters:
        -----------
        X_shape : tuple
            (n_samples, n_features) of the input data.
            
        Returns:
        --------
        estimated_seconds : int
            Projected time in seconds.
        eta_string : str
            Human readable ETA string.
        """
        n_samples, n_features = X_shape
        
        # Consider sampling if enabled
        if not self.data_config.use_full_data and n_samples > self.data_config.sample_size:
            n_samples = self.data_config.sample_size
            
        # Hardware capacity scaling
        cores = self.parallel_config.n_jobs
        if cores == -1 or cores is None:
            cores = os.cpu_count() or 2
        
        # Base processing power heuristic (ops per second roughly normalized)
        # Assumes standard mixed-type data overhead
        base_ops = n_samples * (n_features ** 1.2) 
        
        # 1. Profiling + Cleaning Time
        prep_time = (base_ops / 50000) / (cores * 0.5) 
        
        # 2. Optimization + Modeling Time
        modeling_time = 0
        if self.modeling_config.train_models:
            n_models = self.modeling_config.n_models if self.modeling_config.models_to_train is None else len(self.modeling_config.models_to_train)
            
            if self.opt_config.use_optuna:
                trials = self.opt_config.optuna_trials_per_model
                cv_folds = 3 # assumed
                # Each trial fits cv_folds times
                total_fits = n_models * trials * cv_folds
                # tree model heuristic
                fit_time = (base_ops / 30000)
                modeling_time = total_fits * fit_time
                
                # Parallel boost
                opt_cores = self.opt_config.optuna_parallel_jobs
                if opt_cores == -1: opt_cores = cores
                modeling_time = modeling_time / max(1, opt_cores * 0.8)
            else:
                modeling_time = (n_models * (base_ops / 20000)) / cores
                
        # 3. Report Generation Time
        report_time = 5 + (n_features * 0.5) # mostly plotting distribution overhead
        
        total_seconds = int(prep_time + modeling_time + report_time)
        
        # Formulate string
        if total_seconds < 60:
            eta_string = f"~{total_seconds} seconds"
        else:
            mins = total_seconds // 60
            secs = total_seconds % 60
            eta_string = f"~{mins} min {secs} sec"
            
        return total_seconds, eta_string
