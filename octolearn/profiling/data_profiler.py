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
    Container for storing dataset profiling results.
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

        feature_types = {}
        numeric_cols, categorical_cols, date_cols, text_cols, id_cols = [], [], [], [], []
        constant_cols, low_variance_cols = [], []

        stats, missing_ratio, unique_counts = {}, {}, {}

        for col in df.columns:
            if user_id_cols and col in user_id_cols:
                col_type = "id"
            else:
                col_type = self._detect_column_type(df[col], col)

            feature_types[col] = col_type

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

            n_unique = df[col].nunique(dropna=True)
            unique_counts[col] = n_unique
            missing_ratio[col] = df[col].isna().mean()

            if n_unique <= 1:
                constant_cols.append(col)

            if col_type == "numeric" and n_unique > 1:
                if df[col].var() < 1e-5:
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
                    try:
                        corr = df[col].corr(target)
                        if corr is not None and abs(corr) > 0.95:
                            leakage_suspects.append(col)
                    except Exception:
                        pass
            # Name-based heuristic
            for col in df.columns:
                if target_name and target_name.lower() in col.lower() and col != target_name:
                    leakage_suspects.append(col)
            leakage_suspects = list(set(leakage_suspects))

        # ----------------------------------------------------
        # CLASS IMBALANCE
        # ----------------------------------------------------
        imbalance_ratio = None
        if target is not None and target_name in df.columns:
            target_series = df[target_name]
            if target_series.nunique() > 1:
                counts = target_series.value_counts()
                imbalance_ratio = counts.min() / counts.max()
            else:
                imbalance_ratio = 1.0

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
            high_cardinality_cols=high_cardinality_cols
        )

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

        Args:
            target (pd.Series): Target column.

        Returns:
            str: 'classification' or 'regression'.
        """

        if pd.api.types.is_numeric_dtype(target):
            n_unique = target.nunique()
            if n_unique < 20 or (n_unique / len(target) < 0.05):
                return "classification"
            if pd.api.types.is_float_dtype(target.dtype):
                if (target.dropna() % 1 != 0).any():
                    return "regression"
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
