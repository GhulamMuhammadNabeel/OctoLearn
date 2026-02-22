"""
Data Profiler Module

Provides automated dataset profiling to infer feature types, summary statistics,
and data quality issues using robust heuristics for real-world messy datasets.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


@dataclass
class DatasetProfile:
    """
    Metadata container for dataset analysis results.

    Stores comprehensive information about a dataset's structure, quality,
    and statistical distributions across all features.

    Attributes
    ----------
    shape : tuple of int
        The dimensions of the dataset (n_rows, n_cols).
    columns : list of str
        The original column names.
    feature_types : dict of str: str
        The inferred semantic type for each column (e.g., 'numeric', 'categorical').
    stats : dict of str: dict
        Summary statistics (mean, std, min, max, etc.) for each feature.
    missing_ratio : dict of str: float
        The proportion of missing values per column.
    unique_counts : dict of str: int
        The number of unique values per column.
    task_type : str, optional
        The inferred machine learning task type (e.g., 'classification', 'regression').
    target_col : str, optional
        The name of the target column, if specified.
    id_like_columns : list of str
        Columns identified as potential ID or key columns.
    constant_columns : list of str
        Columns with only one unique value.
    low_variance_columns : list of str
        Numeric columns with very low variance.
    numeric_columns : list of str
        Columns identified as numeric.
    categorical_columns : list of str
        Columns identified as categorical.
    date_columns : list of str
        Columns identified as datetime.
    text_columns : list of str
        Columns identified as free text.
    leakage_suspects : list of str
        Columns potentially exhibiting data leakage with the target.
    high_cardinality_cols : list of str
        Categorical columns with a high number of unique values.
    imbalance_ratio : float, optional
        The ratio of the smallest class count to the largest class count for classification targets.
    duplicate_rows : int
        The total count of identical rows in the dataset.
    data_quality_score : float
        A health score from 0 to 100 representing overall data cleanliness.
    """

    shape: Tuple[int, int]
    columns: List[str]
    feature_types: Dict[str, str]
    stats: Dict[str, Dict[str, Any]]
    missing_ratio: Dict[str, float]
    unique_counts: Dict[str, int]

    task_type: str = "unknown"
    target_col: Optional[str] = None

    id_like_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    low_variance_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)

    # 🔧 NEW FIELDS
    leakage_suspects: List[str] = field(default_factory=list)
    high_cardinality_cols: List[str] = field(default_factory=list)
    imbalance_ratio: float = None
    duplicate_rows: int = 0
    data_quality_score: float = 0.0  # New field from smart pipeline specs

    @property
    def n_rows(self) -> int:
        """Number of rows in the dataset."""
        return self.shape[0]

    @property
    def n_columns(self) -> int:
        """Number of columns in the dataset."""
        return self.shape[1]


class DataProfiler:
    """
    Performs automated profiling of a pandas DataFrame.
    """

    def __init__(self):
        self.DOMAIN_NUMERIC = [
            "age", "income", "salary", "revenue", "budget", "price",
            "cost", "sales", "year", "day", "month", "width", "length",
            "height", "weight", "score", "percent", "ratio", "count",
            "balance", "amount", "profit", "target"
        ]

        self.DOMAIN_CATEGORICAL = [
            "gender", "sex", "category", "status", "type", "mode",
            "zip", "postal", "country", "city", "state", "region",
            "group", "class", "rank", "grade", "is_", "has_"
        ]

        self.DOMAIN_ID = ["id", "key", "index", "sku", "customer_no"]

    # --------------------------------------------------------
    # MAIN PROFILING METHOD
    # --------------------------------------------------------
    def profile(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        user_id_cols: Optional[List[str]] = None
    ) -> DatasetProfile:
        """
        Analyze a DataFrame and generate a comprehensive `DatasetProfile`.

        This method coordinates the entire profiling process, from feature type
        inference to data quality scoring and correlation analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset to profile.
        target : pd.Series, optional
            The target variable, used to infer task type and calculate correlations.
        user_id_cols : list of str, optional
            A list of column names to be explicitly marked as identifiers.

        Returns
        -------
        profile : DatasetProfile
            A populated profile object containing all analysis results.
        """

        feature_types = {}
        numeric_cols, categorical_cols, date_cols, text_cols, id_cols = [], [], [], [], []
        constant_cols, low_variance_cols = [], []

        stats, missing_ratio, unique_counts = {}, {}, {}

        for col in df.columns:
            series = df[col]
            if user_id_cols and col in user_id_cols:
                col_type = "id"
            else:
                col_type = self._detect_column_type(series, col)

            feature_types[col] = col_type

            # Safe numeric conversion for stats
            if col_type == "numeric" and not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors='coerce')

            if col_type == "numeric":
                numeric_cols.append(col)
            elif col_type == "categorical":
                categorical_cols.append(col)
            elif col_type == "date":
                date_cols.append(col)
            elif col_type == "text":
                text_cols.append(col)
            elif col_type == "id":
                id_cols.append(col)

            n_unique = series.nunique(dropna=True)
            unique_counts[col] = n_unique
            missing_ratio[col] = series.isna().mean()

            if n_unique <= 1:
                constant_cols.append(col)

            if col_type == "numeric" and n_unique > 1:
                # Use the potentially converted series
                if series.var() < 1e-5:
                    low_variance_cols.append(col)

        # ----------------------------------------------------
        # High Cardinality columns (categorical only)
        # ----------------------------------------------------
        high_cardinality_cols = [
            col for col in categorical_cols
            if unique_counts.get(col, 0) / len(df) > 0.5 and col not in id_cols
        ]

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------
        stats = self._calculate_statistics(df, feature_types)
        
        # ----------------------------------------------------
        # TASK TYPE
        # ----------------------------------------------------
        task_type = "unknown"
        if target is not None:
             task_type = self._detect_task_type(target)

        # ----------------------------------------------------
        # DUPLICATE ROWS
        # ----------------------------------------------------
        duplicate_rows = int(df.duplicated().sum())

        # ----------------------------------------------------
        # LEAKAGE SUSPECTS
        # ----------------------------------------------------
        leakage_suspects = []
        target_name = target.name if target is not None else None

        if target is not None:
            if pd.api.types.is_numeric_dtype(target):
                for col in numeric_cols:
                    if target_name and col == target_name:
                        continue
                    try:
                        # Safe conversion for correlation
                        series = df[col]
                        if not pd.api.types.is_numeric_dtype(series):
                             series = pd.to_numeric(series, errors='coerce')
                        
                        corr = series.corr(target)
                        if corr is not None and abs(corr) > 0.95:
                            leakage_suspects.append(col)
                    except Exception:
                        pass
            # Name-based heuristic
            for col in df.columns:
                if target_name and target_name.lower() in col.lower() and col != target_name:
                    # Double check equality to be safe
                    if col == target_name: continue
                    leakage_suspects.append(col)
            leakage_suspects = list(set(leakage_suspects))

        # ----------------------------------------------------
        # CLASS IMBALANCE
        # ----------------------------------------------------
        imbalance_ratio = None
        imbalance_ratio = None
        if target is not None:
            # Use the passed target series directly
            if target.nunique() > 1:
                counts = target.value_counts()
                imbalance_ratio = counts.min() / counts.max()
            else:
                imbalance_ratio = 1.0
            
            # Ensure target is in unique_counts for RecommendationEngine
            if target.name:
                unique_counts[target.name] = target.nunique()

        return DatasetProfile(
            shape=df.shape,
            columns=list(df.columns),
            feature_types=feature_types,
            stats=stats,
            missing_ratio=missing_ratio,
            unique_counts=unique_counts,
            task_type=task_type,
            target_col=target_name,
            id_like_columns=id_cols,
            constant_columns=constant_cols,
            low_variance_columns=low_variance_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            date_columns=date_cols,
            text_columns=text_cols,
            duplicate_rows=duplicate_rows,
            imbalance_ratio=imbalance_ratio,
            leakage_suspects=leakage_suspects,
            high_cardinality_cols=high_cardinality_cols,
            data_quality_score=self._calculate_data_quality_score(
                df, missing_ratio, duplicate_rows, leakage_suspects
            )
        )

    def _calculate_data_quality_score(
        self, 
        df: pd.DataFrame, 
        missing_ratio: Dict[str, float], 
        duplicate_rows: int,
        leakage_suspects: List[str]
    ) -> float:
        """
        Compute a 0-100 Data Quality Score based on completeness, uniqueness, and leakage.
        
        Formula:
        - Start: 100
        - Completeness: -1 point for every 1% of missing data (avg across cols)
        - Uniqueness: -1 point for every 1% of duplicate rows
        - Leakage: -20 points if any leakage suspect is found (critical)
        - Constant Cols: -5 points if > 10% of columns are constant
        """
        score = 100.0
        n_rows = len(df)
        if n_rows == 0:
            return 0.0

        # Completeness penalty
        avg_missing = np.mean(list(missing_ratio.values())) if missing_ratio else 0.0
        score -= (avg_missing * 100)

        # Uniqueness penalty
        dup_rate = duplicate_rows / n_rows
        score -= (dup_rate * 100)

        # Leakage penalty (Binary heavy penalty)
        if leakage_suspects:
            score -= 20.0

        return max(0.0, round(score, 1))

    # --------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------
    def _detect_column_type(self, series: pd.Series, col_name: str) -> str:
        """
        Infer the semantic type of a single column using layered heuristics.

        The detection order is:
            1. Domain keyword overrides
            2. Datetime and boolean checks
            3. Numeric dtype analysis
            4. Object/string analysis with numeric convertibility
            5. Cardinality and distribution-based heuristics

        Args:
            series (pd.Series): Column data to analyze.
            col_name (str): Name of the column.

        Returns:
            str: Inferred column type ('numeric', 'categorical', 'date', 'text', or 'id').
        """

        col_lower = col_name.lower()
        n_unique = series.nunique(dropna=True)
        n_rows = len(series)
        unique_ratio = n_unique / n_rows if n_rows else 0

        if any(x in col_lower for x in self.DOMAIN_ID) and unique_ratio > 0.9:
            return "id"

        if any(x in col_lower for x in self.DOMAIN_NUMERIC):
            if self._can_convert_to_numeric(series):
                return "numeric"

        if any(x in col_lower for x in self.DOMAIN_CATEGORICAL):
            return "categorical"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"

        if pd.api.types.is_bool_dtype(series):
            return "categorical"

        if pd.api.types.is_numeric_dtype(series):
            return self._analyze_numeric_column(series)

        return self._analyze_object_column(series)

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        """
        Check whether a series can be fully converted to numeric values
        without introducing additional missing values.

        Args:
            series (pd.Series): Input data.

        Returns:
            bool: True if all non-null values are numeric-convertible, otherwise False.
        """
        converted = pd.to_numeric(series, errors="coerce")
        return converted.notna().sum() == series.notna().sum()

    def _analyze_object_column(self, series: pd.Series) -> str:
        """
        Analyze an object or string column to determine whether it represents
        categorical data, numeric data stored as strings, identifiers, or free text.

        Args:
            series (pd.Series): Column data with object or string dtype.

        Returns:
            str: Inferred column type ('numeric', 'categorical', 'text', or 'id').
        """
        n_unique = series.nunique(dropna=True)
        n_rows = len(series)
        unique_ratio = n_unique / n_rows if n_rows else 0
        avg_len = series.dropna().astype(str).str.len().mean()

        if unique_ratio > 0.9:
            return "text" if avg_len > 20 else "id"

        if self._can_convert_to_numeric(series):
            converted = pd.to_numeric(series, errors="coerce")
            return self._analyze_numeric_column(converted)

        return "categorical"

    def _analyze_numeric_column(self, series: pd.Series) -> str:
        """
        Determine whether a numeric column should be treated as a true numeric
        feature or as a categorical feature based on distribution and structure.

        Args:
            series (pd.Series): Numeric column data.

        Returns:
            str: 'numeric' or 'categorical'.
        """

        series = series.dropna()
        n_unique = series.nunique()
        unique_ratio = n_unique / len(series) if len(series) else 0

        if pd.api.types.is_float_dtype(series.dtype):
            if (series % 1 != 0).any():
                return "numeric"

        if n_unique <= 2:
            return "categorical"

        value_counts = series.value_counts(normalize=True)
        top_freq = value_counts.iloc[0]

        if n_unique < 20 and top_freq > 0.3:
            return "categorical"

        unique_vals = np.sort(series.unique())
        if len(unique_vals) > 1:
            diffs = np.diff(unique_vals)
            if np.all(diffs == 1):
                return "numeric"
            data_range = unique_vals.max() - unique_vals.min()
            if data_range < n_unique * 2:
                return "numeric"

        if unique_ratio < 0.05:
            return "categorical"

        return "numeric"

    def _detect_task_type(self, target: pd.Series) -> str:
        """
        Infer whether the prediction task is classification or regression
        based on the target variable characteristics.

        Heuristics:
        1. Non-numeric -> Classification
        2. Float with decimals -> Regression
        3. 2 or fewer unique values -> Classification (Binary)
        4. High cardinality (>50 unique & >5% ratio) -> Regression
        5. Low cardinality integers (3-50 unique):
           - If consecutive (0,1,2... or 10,11,12) -> Classification (Label Encoded)
           - If gaps (10, 20, 50) -> Regression
        """
        # 1. Non-numeric
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        target_valid = target.dropna()
        n_unique = target_valid.nunique()
        n_total = len(target_valid)
        
        # 2. Float with decimals
        if pd.api.types.is_float_dtype(target.dtype):
            if (target_valid % 1 != 0).any():
                return "regression"

        # 3. Binary
        if n_unique <= 2:
            return "classification"

        # 4. High Cardinality (Likely continuous quantity)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        if n_unique > 50 and unique_ratio > 0.05:
            return "regression"

        # 5. Low Cardinality Integers (The ambiguous zone)
        # Check for order/consecutiveness
        unique_vals = np.sort(target_valid.unique())
        
        # Check if values are consecutive integers (e.g. 0,1,2,3 or 1,2,3,4)
        # This strongly suggests Label Encoding -> Classification
        if len(unique_vals) > 1:
            diffs = np.diff(unique_vals)
            is_consecutive = np.all(diffs == 1)
            
            if is_consecutive:
                return "classification"
            else:
                # Gaps exist (e.g. 10, 20, 50). Likely regression (discrete counts, ratings with gaps, prices)
                # But could be sparse class labels. 
                # If unique counts are very low (<10) and not consecutive, it's ambiguous.
                # User preference: "if it aint have any order... call it regression"
                return "regression"
                
        return "classification"

    def _calculate_statistics(self, df: pd.DataFrame,
                              feature_types: Dict[str, str]) -> Dict[str, Dict]:
        """
        Compute summary statistics for each column based on its inferred feature type.

        Args:
            df (pd.DataFrame): Input dataset.
            feature_types (Dict[str, str]): Mapping of column names to feature types.

        Returns:
            Dict[str, Dict]: Dictionary containing statistics for each column.
        """

        stats = {}
        for col, dtype in feature_types.items():
            series = df[col]
            
            # Convert if necessary (redundant logic but ensures safety for stats)
            if dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors='coerce')
                
            if dtype == "numeric":
                desc = series.describe()
                stats[col] = {
                    "mean": desc.get("mean"),
                    "std": desc.get("std"),
                    "min": desc.get("min"),
                    "max": desc.get("max"),
                    "median": series.median(),
                    "skew": series.skew(),
                    "kurtosis": series.kurtosis()
                }
            elif dtype == "categorical":
                stats[col] = {
                    "top_counts": series.value_counts().head(5).to_dict(),
                    "mode": series.mode().iloc[0] if not series.mode().empty else None
                }
            else:
                stats[col] = {}
        return stats
