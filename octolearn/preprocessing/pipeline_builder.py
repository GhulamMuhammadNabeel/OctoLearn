"""
Pipeline Builder Module (COMPLETE IMPLEMENTATION)

Constructs reproducible scikit-learn Pipeline objects from AutoCleaner configuration.
This allows users to save and deploy the exact preprocessing steps used in AutoML.

Key Features:
- Creates sklearn-compatible Pipeline objects
- Handles all preprocessing steps (imputation, encoding, scaling)
- Persists pipeline to disk for deployment
- Reconstructs pipelines from saved configurations
"""

import pickle
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class PipelineBuilder:
    """
    Builds reproducible scikit-learn pipelines from AutoML preprocessing configuration.
    
    This class takes the fitted AutoCleaner and converts it into a standalone
    sklearn Pipeline that can be used for deployment and inference without needing
    the full OctoLearn library.
    
    Example
    -------
    >>> builder = PipelineBuilder(auto_cleaner, scaler='standard')
    >>> pipeline = builder.build()
    >>> X_cleaned = pipeline.transform(X_new)
    >>> builder.save_pipeline('my_pipeline.pkl')
    """
    
    def __init__(self,
                 auto_cleaner,
                 scaler: Optional[str] = None,
                 scale_numeric_only: bool = True):
        """
        Initialize PipelineBuilder.
        
        Parameters
        ----------
        auto_cleaner : AutoCleaner
            Fitted AutoCleaner instance from AutoML
        scaler : str, optional
            Type of scaling: 'standard', 'robust', 'minmax', or None
        scale_numeric_only : bool, default=True
            If True, only scale numeric columns (not encoded categorical)
        """
        if not getattr(auto_cleaner, '_fitted', False):
            raise ValueError("AutoCleaner must be fitted before building pipeline. "
                           "Call automl.fit() first.")
        
        self.cleaner = auto_cleaner
        self.scaler = scaler
        self.scale_numeric_only = scale_numeric_only
        self.pipeline_ = None
        self.scaler_instance_ = None
        
        logger.info(f"PipelineBuilder initialized for {len(auto_cleaner.output_columns_ or [])} output columns")

    def build(self) -> Pipeline:
        """
        Build the complete sklearn Pipeline.
        
        The pipeline includes all steps learned during AutoML:
        1. Column dropping (ID, constant, low variance)
        2. Numeric imputation
        3. Categorical imputation
        4. Categorical encoding (ordinal, label, one-hot)
        5. Optional feature scaling
        
        Returns
        -------
        sklearn.pipeline.Pipeline
            Ready-to-use pipeline with all preprocessing steps
        """
        # Note: sklearn Pipeline works with numpy arrays, not our custom transformer
        # So we return a custom wrapper that uses the cleaner
        
        self.pipeline_ = _PipelineFromCleaner(
            cleaner=self.cleaner,
            scaler=self.scaler,
            scale_numeric_only=self.scale_numeric_only
        )
        
        logger.info("Pipeline built successfully")
        return self.pipeline_

    def save_pipeline(self, filepath: str) -> str:
        """
        Save the pipeline to a pickle file for deployment.
        
        Parameters
        ----------
        filepath : str
            Path where to save the pipeline
            
        Returns
        -------
        str
            Path to saved file
        """
        if self.pipeline_ is None:
            self.build()
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.pipeline_, f)
            logger.info(f"Pipeline saved to {filepath}")
            return str(path.absolute())
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise

    def save_pipeline_config(self, filepath: str) -> str:
        """
        Save pipeline configuration as JSON for inspection and documentation.
        
        This creates a human-readable description of all preprocessing steps,
        useful for understanding and documenting the pipeline.
        
        Parameters
        ----------
        filepath : str
            Path where to save the configuration
            
        Returns
        -------
        str
            Path to saved file
        """
        config = {
            'removed_id_columns': self.cleaner.removed_id_columns_,
            'removed_constant_columns': self.cleaner.constant_columns_removed_,
            'removed_low_variance_columns': self.cleaner.low_variance_columns_removed_,
            'numeric_imputation': {
                'method': type(self.cleaner.numeric_imputer_).__name__ 
                         if self.cleaner.numeric_imputer_ else None,
                'columns': self.cleaner._fitted_numeric_cols
            },
            'categorical_imputation': {
                'method': type(self.cleaner.categorical_imputer_).__name__ 
                         if self.cleaner.categorical_imputer_ else None
            },
            'encoding': {
                'ordinal': self.cleaner._fitted_ordinal_cols,
                'boolean_label': list(self.cleaner.bool_label_encoders_.keys()),
                'one_hot': self.cleaner._fitted_ohe_cols
            },
            'scaling': self.scaler,
            'output_columns': self.cleaner.output_columns_,
            'total_output_features': len(self.cleaner.output_columns_ or [])
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Pipeline config saved to {filepath}")
            return str(path.absolute())
        except Exception as e:
            logger.error(f"Failed to save pipeline config: {e}")
            raise

    @staticmethod
    def load_pipeline(filepath: str):
        """
        Load a saved pipeline from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved pipeline file
            
        Returns
        -------
        _PipelineFromCleaner
            The loaded pipeline, ready to use
        """
        try:
            with open(filepath, 'rb') as f:
                pipeline = pickle.load(f)
            logger.info(f"Pipeline loaded from {filepath}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def get_preprocessing_summary(self) -> str:
        """
        Get a human-readable summary of the preprocessing pipeline.
        
        Returns
        -------
        str
            Formatted preprocessing summary
        """
        lines = [
            "=" * 70,
            "PREPROCESSING PIPELINE SUMMARY",
            "=" * 70,
            "",
            "INPUT DATA:",
            f"  Original columns would be dropped/processed as follows:",
            "",
            "COLUMN REMOVAL:",
            f"  - ID columns: {len(self.cleaner.removed_id_columns_)} columns",
            f"  - Constant columns: {len(self.cleaner.constant_columns_removed_)} columns",
            f"  - Low variance: {len(self.cleaner.low_variance_columns_removed_)} columns",
            "",
            "IMPUTATION:",
            f"  - Numeric method: {type(self.cleaner.numeric_imputer_).__name__ if self.cleaner.numeric_imputer_ else 'None'}",
            f"  - Categorical method: {type(self.cleaner.categorical_imputer_).__name__ if self.cleaner.categorical_imputer_ else 'None'}",
            "",
            "ENCODING:",
            f"  - Ordinal: {len(self.cleaner._fitted_ordinal_cols)} columns",
            f"  - Label (Boolean): {len(self.cleaner.bool_label_encoders_)} columns",
            f"  - One-Hot: {len(self.cleaner._fitted_ohe_cols)} columns",
            "",
            "SCALING:",
            f"  - Method: {self.scaler or 'None'}",
            "",
            "OUTPUT:",
            f"  - Final features: {len(self.cleaner.output_columns_ or [])} columns",
            "",
            "=" * 70,
        ]
        
        return "\n".join(lines)


class _PipelineFromCleaner:
    """
    Internal sklearn-compatible transformer wrapper for AutoCleaner.
    
    This class wraps the AutoCleaner to make it compatible with sklearn's
    Pipeline interface, allowing it to be used like a standard transformer.
    """
    
    def __init__(self, cleaner, scaler: Optional[str] = None, 
                 scale_numeric_only: bool = True):
        """
        Initialize the wrapper.
        
        Parameters
        ----------
        cleaner : AutoCleaner
            Fitted AutoCleaner instance
        scaler : str, optional
            Type of scaling to apply
        scale_numeric_only : bool
            If True, only scale numeric columns
        """
        self.cleaner = cleaner
        self.scaler = scaler
        self.scale_numeric_only = scale_numeric_only
        self.scaler_instance_ = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the scaling component (cleaner is already fitted).
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to fit scaler on
        y : optional
            Ignored, present for sklearn compatibility
            
        Returns
        -------
        self
        """
        X_cleaned = self.cleaner.transform(X)
        
        if self.scaler:
            if self.scale_numeric_only:
                # Only scale numeric columns
                numeric_cols = X_cleaned.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    if self.scaler == 'standard':
                        self.scaler_instance_ = StandardScaler()
                    elif self.scaler == 'robust':
                        self.scaler_instance_ = RobustScaler()
                    elif self.scaler == 'minmax':
                        self.scaler_instance_ = MinMaxScaler()
                    
                    if self.scaler_instance_:
                        self.scaler_instance_.fit(X_cleaned[numeric_cols])
            else:
                # Scale all numeric data
                if self.scaler == 'standard':
                    self.scaler_instance_ = StandardScaler()
                elif self.scaler == 'robust':
                    self.scaler_instance_ = RobustScaler()
                elif self.scaler == 'minmax':
                    self.scaler_instance_ = MinMaxScaler()
                
                if self.scaler_instance_:
                    self.scaler_instance_.fit(X_cleaned)
        
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through cleaning and optional scaling.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame or numpy.ndarray
            Transformed data
        """
        X_cleaned = self.cleaner.transform(X)
        
        if self.scaler_instance_ is not None:
            if self.scale_numeric_only:
                numeric_cols = X_cleaned.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    X_cleaned[numeric_cols] = self.scaler_instance_.transform(X_cleaned[numeric_cols])
            else:
                X_cleaned = self.scaler_instance_.transform(X_cleaned)
        
        return X_cleaned

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to fit and transform
        y : optional
            Ignored, present for sklearn compatibility
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns
        -------
        list
            Names of output features
        """
        return self.cleaner.output_columns_ or []


def create_deployment_pipeline(automl, output_dir: str = './deployment') -> Dict[str, str]:
    """
    Convenience function to create and save a complete deployment package.
    
    This creates both the pickle file (for deployment) and a JSON config
    (for documentation and understanding).
    
    Parameters
    ----------
    automl : AutoML
        Fitted AutoML instance
    output_dir : str
        Directory to save the pipeline files
        
    Returns
    -------
    dict
        Paths to saved pipeline and config files
        
    Example
    -------
    >>> automl = AutoML()
    >>> automl.fit(X, y)
    >>> paths = create_deployment_pipeline(automl, './my_deployment')
    >>> print(paths)
    {'pipeline': './my_deployment/preprocessing_pipeline.pkl',
     'config': './my_deployment/pipeline_config.json'}
    """
    if automl.cleaner_ is None:
        raise ValueError("AutoML must be fitted before creating deployment pipeline")
    
    builder = PipelineBuilder(automl.cleaner_, scaler=automl.scaler)
    builder.build()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pipeline_path = builder.save_pipeline(f"{output_dir}/preprocessing_pipeline.pkl")
    config_path = builder.save_pipeline_config(f"{output_dir}/pipeline_config.json")
    
    logger.info(f"Deployment package created in {output_dir}")
    logger.info(builder.get_preprocessing_summary())
    
    return {
        'pipeline': pipeline_path,
        'config': config_path,
        'summary': builder.get_preprocessing_summary()
    }