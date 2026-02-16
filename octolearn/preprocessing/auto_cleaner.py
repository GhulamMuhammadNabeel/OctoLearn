"""
Automatic Data Cleaning Module

Applies automated cleaning actions: remove duplicates, impute missing values, remove ID columns, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')

from ..config import AUTO_CLEAN_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class AutoCleaner:
    """
    Automatically cleans dataset based on profile and configuration.

    Applies actions such as duplicate removal, missing value imputation, ID column removal, encoding, and scaling.

    Attributes:
        X (pd.DataFrame): Working feature dataframe.
        y (pd.Series): Working target variable.
        profile (DatasetProfile): Dataset profile object.
        original_shape (tuple): Original shape of X.
        cleaning_log (dict): Log of all cleaning actions performed.
        imputer_strategy (dict): User-specified imputation strategy.
        encoder_strategy (dict): User-specified encoding strategy.
        scaler (str): User-specified scaler.
        id_columns (list): User-specified ID columns.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, profile,
                 imputer_strategy: dict = None,
                 encoder_strategy: dict = None,
                 scaler: str = None,
                 id_columns: list = None):
        """
        Initialize AutoCleaner with data, profile, and user overrides.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target variable.
            profile (DatasetProfile): Dataset profile object.
            imputer_strategy (dict, optional): User-specified imputation strategy.
            encoder_strategy (dict, optional): User-specified encoding strategy.
            scaler (str, optional): User-specified scaler.
            id_columns (list, optional): User-specified ID columns.
        """
        self.X = X.copy()
        self.y = y.copy()
        self.profile = profile
        self.original_shape = X.shape
        self.cleaning_log = {}
        # User param overrides
        self.imputer_strategy = imputer_strategy or {}
        self.encoder_strategy = encoder_strategy or {}
        self.scaler = scaler
        self.id_columns = id_columns
    
    @log_execution(logger_obj=logger)
    def clean(self, return_log: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Apply all enabled cleaning actions to the dataset.

        Args:
            return_log (bool): Whether to return the cleaning log.

        Returns:
            Tuple[pd.DataFrame, pd.Series, dict]: Cleaned features, target, and cleaning log.
        """
        logger.info(f"Starting data cleaning. Original shape: {self.original_shape}")
        
        # Apply cleaning actions
        if AUTO_CLEAN_CONFIG['actions']['remove_duplicates']:
            self.X, self.y = self._remove_duplicates()

        # Use user-specified ID columns if provided, else profile
        if AUTO_CLEAN_CONFIG['actions']['remove_id_columns']:
            if self.id_columns is not None:
                id_cols = self.id_columns
                if id_cols:
                    self.X = self.X.drop(columns=id_cols, errors='ignore')
                    self.cleaning_log['id_columns_removed'] = id_cols
                    logger.info(f"Removed user-specified ID columns: {id_cols}")
                else:
                    self.cleaning_log['id_columns_removed'] = []
            else:
                self.X = self._remove_id_columns()

        if AUTO_CLEAN_CONFIG['actions']['remove_constants']:
            self.X = self._remove_constant_columns()

        if AUTO_CLEAN_CONFIG['actions']['remove_low_variance']:
            self.X = self._remove_low_variance_columns()

        if AUTO_CLEAN_CONFIG['actions']['impute_missing']:
            self.X = self._impute_missing_values()

        # Encoding step (NEW): handle ordinal, bool, and OHE
        self.X = self._encode_categoricals()
        def _encode_categoricals(self) -> pd.DataFrame:
                """
                Encode categorical columns using user strategy or defaults.
                - Ordinal columns: ordinal encoding (user-specified)
                - Bool columns: label encoding (0/1)
                - All others: OHE
                """
                import pandas as pd
                from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
                X = self.X
                encoder_strategy = self.encoder_strategy or {}
                ordinal_cols = encoder_strategy.get('ordinal_cols', [])
                bool_cols = encoder_strategy.get('bool_cols', [])
                # If not specified, try to auto-detect bool columns (2 unique values, e.g. sex)
                auto_bool_cols = [col for col in X.select_dtypes(include=['object', 'category', 'bool']).columns
                                  if col not in ordinal_cols and X[col].nunique(dropna=True) == 2]
                bool_cols = list(set(bool_cols) | set(auto_bool_cols))
                # Ordinal encoding
                if ordinal_cols:
                    oe = OrdinalEncoder()
                    X[ordinal_cols] = oe.fit_transform(X[ordinal_cols])
                # Label encoding for bool
                for col in bool_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                # OHE for remaining categoricals
                cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                cat_cols = [col for col in cat_cols if col not in ordinal_cols and col not in bool_cols]
                if cat_cols:
                    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                return X
        
        # Align X and y
        common_idx = self.X.index.intersection(self.y.index)
        self.X = self.X.loc[common_idx]
        self.y = self.y.loc[common_idx]
        
        logger.info(f"Cleaning complete. Final shape: {self.X.shape}")
        logger.info(f"Removed {self.original_shape[0] - self.X.shape[0]} rows and {self.original_shape[1] - self.X.shape[1]} columns")
        
        if return_log:
            return self.X, self.y, self.cleaning_log
        return self.X, self.y, {}
    
    def _remove_duplicates(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove duplicate rows.
        
        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        initial_rows = len(self.X)
        
        # Identify duplicates
        dup_mask = self.X.duplicated()
        dup_count = dup_mask.sum()
        
        if dup_count > 0:
            # Keep first occurrence
            self.X = self.X[~dup_mask]
            self.y = self.y[~dup_mask]
            
            self.cleaning_log['duplicates_removed'] = int(dup_count)
            logger.info(f"Removed {dup_count} duplicate rows")
        else:
            self.cleaning_log['duplicates_removed'] = 0
        
        return self.X, self.y
    
    def _remove_id_columns(self) -> pd.DataFrame:
        """
        Remove ID-like columns.
        
        Returns
        -------
        pd.DataFrame
            Data without ID columns
        """
        id_cols = self.profile.id_like_columns
        
        if id_cols:
            self.X = self.X.drop(columns=id_cols, errors='ignore')
            self.cleaning_log['id_columns_removed'] = id_cols
            logger.info(f"Removed ID columns: {id_cols}")
        else:
            self.cleaning_log['id_columns_removed'] = []
        
        return self.X
    
    def _remove_constant_columns(self) -> pd.DataFrame:
        """
        Remove constant columns (only 1 unique value).
        
        Returns
        -------
        pd.DataFrame
            Data without constant columns
        """
        constant_cols = self.profile.constant_columns
        
        if constant_cols:
            self.X = self.X.drop(columns=constant_cols, errors='ignore')
            self.cleaning_log['constant_columns_removed'] = constant_cols
            logger.info(f"Removed constant columns: {constant_cols}")
        else:
            self.cleaning_log['constant_columns_removed'] = []
        
        return self.X
    
    def _remove_low_variance_columns(self) -> pd.DataFrame:
        """
        Remove low variance columns.
        
        Returns
        -------
        pd.DataFrame
            Data without low variance columns
        """
        low_var_cols = self.profile.low_variance_columns
        
        if low_var_cols:
            self.X = self.X.drop(columns=low_var_cols, errors='ignore')
            self.cleaning_log['low_variance_columns_removed'] = low_var_cols
            logger.info(f"Removed low variance columns: {low_var_cols}")
        else:
            self.cleaning_log['low_variance_columns_removed'] = []
        
        return self.X
    
    def _impute_missing_values(self) -> pd.DataFrame:
        """
        Impute missing values in numeric and categorical features.
        
        Returns
        -------
        pd.DataFrame
            Data with imputed missing values
        """
        imputation_log = {}
        # Use user param if provided, else config
        missing_threshold = self.imputer_strategy.get('missing_threshold', AUTO_CLEAN_CONFIG['imputation']['missing_threshold'])
        
        # Check for columns with too much missing data
        cols_to_drop = []
        for col in self.X.columns:
            missing_pct = self.X[col].isnull().sum() / len(self.X)
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
                imputation_log[col] = f"Dropped (>{missing_threshold*100}% missing)"
        
        if cols_to_drop:
            self.X = self.X.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with too much missing data: {cols_to_drop}")
        
        # Impute numeric features
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_missing = self.X[numeric_cols].isnull().sum()
            if numeric_missing.sum() > 0:
                method = self.imputer_strategy.get('numeric', AUTO_CLEAN_CONFIG['imputation']['numeric'])
                if method == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                    self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])
                    imputation_log['numeric_method'] = 'mean'
                elif method == 'median':
                    imputer = SimpleImputer(strategy='median')
                    self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])
                    imputation_log['numeric_method'] = 'median'
                elif method == 'knn' and len(self.X) > 5:
                    try:
                        imputer = KNNImputer(n_neighbors=5)
                        self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])
                        imputation_log['numeric_method'] = 'knn'
                    except:
                        imputer = SimpleImputer(strategy='mean')
                        self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])
                        imputation_log['numeric_method'] = 'mean (fallback)'
                logger.info(f"Imputed {numeric_missing.sum()} missing numeric values using {method}")
        
        # Impute categorical features
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            cat_missing = self.X[categorical_cols].isnull().sum()
            if cat_missing.sum() > 0:
                method = self.imputer_strategy.get('categorical', AUTO_CLEAN_CONFIG['imputation']['categorical'])
                if method == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                    self.X[categorical_cols] = imputer.fit_transform(self.X[categorical_cols])
                    imputation_log['categorical_method'] = 'mode'
                elif method == 'constant':
                    imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
                    self.X[categorical_cols] = imputer.fit_transform(self.X[categorical_cols])
                    imputation_log['categorical_method'] = 'constant (MISSING)'
                logger.info(f"Imputed {cat_missing.sum()} missing categorical values using {method}")
        
        self.cleaning_log['imputation'] = imputation_log
        return self.X
    
    def get_cleaning_report(self) -> str:
        """
        Get a formatted cleaning report.
        
        Returns
        -------
        str
            Formatted cleaning report
        """
        report = [
            "=" * 60,
            "DATA CLEANING REPORT",
            "=" * 60,
            f"\nOriginal shape: {self.original_shape}",
            f"Final shape: {self.X.shape}",
            f"\nRows removed: {self.original_shape[0] - self.X.shape[0]}",
            f"Columns removed: {self.original_shape[1] - self.X.shape[1]}",
            "\nDETAILED CLEANING LOG:",
            "-" * 60,
        ]
        
        for action, details in self.cleaning_log.items():
            if isinstance(details, list) and details:
                report.append(f"\n{action.upper()}:")
                for item in details:
                    report.append(f"  - {item}")
            elif isinstance(details, dict):
                report.append(f"\n{action.upper()}:")
                for key, val in details.items():
                    report.append(f"  - {key}: {val}")
            elif isinstance(details, int) and details > 0:
                report.append(f"\n{action.upper()}: {details}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
