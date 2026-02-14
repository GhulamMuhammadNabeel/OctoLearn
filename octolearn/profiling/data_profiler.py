import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DatasetProfile:
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

    def _generate_hash(self, X: pd.DataFrame) -> str:
        raw = pd.util.hash_pandas_object(X.head(1000), index=True).values
        return hashlib.md5(raw).hexdigest()[:12]

    def _smart_sample(self, X, y, max_rows=100_000):
        if len(X) > max_rows:
            idx = X.sample(max_rows, random_state=42).index
            return X.loc[idx], y.loc[idx]
        return X, y

    def detect_task(self, y: pd.Series) -> str:
        if y.dtype == "object" or y.nunique() < 20:
            return "classification"
        return "regression"

    def profile(self, X: pd.DataFrame, y: pd.Series) -> DatasetProfile:

        X_sample, y_sample = self._smart_sample(X, y)

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_features = X.select_dtypes(include=["datetime"]).columns.tolist()

        missing_report = (X.isnull().mean() * 100).round(2).to_dict()

        task_type = self.detect_task(y)

        imbalance_ratio = None
        if task_type == "classification":
            class_counts = y.value_counts(normalize=True)
            imbalance_ratio = round(class_counts.max(), 3)

        skewed_columns = [
            col for col in numeric_features
            if abs(X_sample[col].skew()) > 1
        ]

        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]

        low_variance_columns = [
            col for col in numeric_features
            if X_sample[col].var() < 1e-5
        ]

        id_like_columns = [
            col for col in X.columns
            if X[col].nunique() == len(X)
        ]

        duplicate_rows = X.duplicated().sum()

        high_cardinality_cols = [
            col for col in categorical_features
            if X[col].nunique() > 0.3 * len(X)
        ]

        leakage_suspects = []
        if task_type == "regression" and numeric_features:
            corr = X_sample[numeric_features].corrwith(y_sample).abs()
            leakage_suspects = corr[corr > 0.95].index.tolist()

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
