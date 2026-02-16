import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_object_dtype, is_string_dtype


@dataclass
class DatasetProfile:
    """
    Data structure for storing the results of dataset profiling.

    Attributes:
        dataset_hash (str): Unique hash of the dataset.
        n_rows (int): Number of rows in the dataset.
        n_columns (int): Number of columns in the dataset.
        numeric_features (List[str]): List of numeric feature names.
        categorical_features (List[str]): List of categorical feature names.
        datetime_features (List[str]): List of datetime feature names.
        missing_report (Dict[str, float]): Missing value percentage per column.
        imbalance_ratio (Optional[float]): Class imbalance ratio (if classification).
        skewed_columns (List[str]): List of skewed columns.
        constant_columns (List[str]): List of constant columns.
        low_variance_columns (List[str]): List of low variance columns.
        id_like_columns (List[str]): List of columns detected as IDs.
        high_cardinality_cols (List[str]): List of high cardinality columns.
        duplicate_rows (int): Number of duplicate rows.
        leakage_suspects (List[str]): Columns suspected of data leakage.
        task_type (str): 'classification' or 'regression'.
    """
    dataset_hash: str
    n_rows: int
    n_columns: int
    numeric_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    missing_report: Dict[str, float]
    imbalance_ratio: Optional[float]
    skewed_columns: List[str]
    constant_columns: List[str]
    low_variance_columns: List[str]
    id_like_columns: List[str]
    high_cardinality_cols: List[str]
    duplicate_rows: int
    leakage_suspects: List[str]
    task_type: str


class DataProfiler:
    """
    Main class for profiling datasets in Octolearn.

    Provides methods for feature type inference, task detection, and summary statistics.
    """

    # -----------------------------------
    # Utility Functions
    # -----------------------------------

    def _generate_hash(self, X: pd.DataFrame) -> str:
        """
        Generate a unique hash for the dataset based on the first 1000 rows.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            str: 12-character MD5 hash string.
        """
        raw = pd.util.hash_pandas_object(X.head(1000), index=True).values
        return hashlib.md5(raw).hexdigest()[:12]

    def _smart_sample(self, X, y, max_rows=100_000):
        """
        Sample the dataset if it exceeds max_rows for efficient profiling.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target variable.
            max_rows (int): Maximum number of rows to keep.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Sampled X and y.
        """
        if len(X) > max_rows:
            idx = X.sample(max_rows, random_state=42).index
            return X.loc[idx], y.loc[idx]
        return X, y

    def detect_task(self, y: pd.Series) -> str:
        """
        Detect the machine learning task type based on the target variable.

        Args:
            y (pd.Series): Target variable.

        Returns:
            str: 'classification' or 'regression'.
        """
        if y.dtype == "object" or y.nunique() < 20:
            return "classification"
        return "regression"

    # -----------------------------------
    # Smart Feature Type Inference
    # -----------------------------------

    def _infer_feature_types(self, X: pd.DataFrame):
        """
        Infer feature types (numeric, categorical, datetime, ID-like) for all columns.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: Lists of feature names by type.
        """

        numeric_features = []
        categorical_features = []
        datetime_features = []
        id_like_columns = []

        for col in X.columns:

            series = X[col]
            unique_count = series.nunique(dropna=True)
            total_count = len(series)
            unique_ratio = unique_count / total_count if total_count > 0 else 0

            # ---- 1. Datetime detection (string/object only) ----
            if is_object_dtype(series) or is_string_dtype(series):
                try:
                    converted = pd.to_datetime(series, errors="raise")
                    datetime_features.append(col)
                    X[col] = converted
                    continue
                except Exception:
                    pass

            # ---- 2. ID-like detection ----
            if unique_count == total_count and unique_count > 0:
                id_like_columns.append(col)
                continue

            # ---- 3. Numeric detection ----
            if is_numeric_dtype(series):

                # Binary numeric → categorical
                if unique_count == 2:
                    categorical_features.append(col)

                # Low cardinality numeric → categorical
                elif unique_count <= 10:
                    categorical_features.append(col)

                # Very small unique ratio → categorical
                elif unique_ratio < 0.01 and unique_count < 50:
                    categorical_features.append(col)

                else:
                    numeric_features.append(col)

            # ---- 4. Categorical detection ----
            elif is_object_dtype(series) or is_categorical_dtype(series) or is_string_dtype(series):
                categorical_features.append(col)

            # ---- 5. Fallback ----
            else:
                numeric_features.append(col)

        return numeric_features, categorical_features, datetime_features, id_like_columns
    # -----------------------------------
    # Main Profiling Function
    # -----------------------------------

    def profile(self, X: pd.DataFrame, y: pd.Series) -> DatasetProfile:

        X = X.copy()
        X_sample, y_sample = self._smart_sample(X, y)

        numeric_features, categorical_features, datetime_features, id_like_columns = self._infer_feature_types(X)

        # Missing %
        missing_report = (X.isnull().mean() * 100).round(2).to_dict()

        # Task detection
        task_type = self.detect_task(y)

        imbalance_ratio = None
        if task_type == "classification":
            class_counts = y.value_counts(normalize=True)
            imbalance_ratio = round(class_counts.max(), 3)

        # Skewness
        skewed_columns = []
        for col in numeric_features:
            try:
                if abs(X_sample[col].skew()) > 1:
                    skewed_columns.append(col)
            except Exception:
                continue

        # Constant columns
        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]

        # Low variance numeric
        low_variance_columns = []
        for col in numeric_features:
            try:
                if X_sample[col].var() < 1e-5:
                    low_variance_columns.append(col)
            except Exception:
                continue

        # Duplicates
        duplicate_rows = X.duplicated().sum()

        # High cardinality categorical
        high_cardinality_cols = [
            col for col in categorical_features
            if X[col].nunique() > 0.3 * len(X)
        ]

        # Leakage detection (regression only)
        leakage_suspects = []
        if task_type == "regression" and numeric_features:
            try:
                corr = X_sample[numeric_features].corrwith(y_sample).abs()
                leakage_suspects = corr[corr > 0.95].index.tolist()
            except Exception:
                leakage_suspects = []

        dataset_hash = self._generate_hash(X)

        return DatasetProfile(
            dataset_hash=dataset_hash,
            n_rows=X.shape[0],
            n_columns=X.shape[1],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            datetime_features=datetime_features,
            missing_report=missing_report,
            imbalance_ratio=imbalance_ratio,
            skewed_columns=skewed_columns,
            constant_columns=constant_columns,
            low_variance_columns=low_variance_columns,
            id_like_columns=id_like_columns,
            high_cardinality_cols=high_cardinality_cols,
            duplicate_rows=duplicate_rows,
            leakage_suspects=leakage_suspects,
            task_type=task_type
        )
