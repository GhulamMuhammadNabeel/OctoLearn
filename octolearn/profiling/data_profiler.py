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
    duplicate_rows: int
    high_cardinality_cols: List[str]
    task_type: str


class DataProfiler:

    def _generate_hash(self, X: pd.DataFrame) -> str:
        raw = pd.util.hash_pandas_object(X, index=True).values
        return hashlib.md5(raw).hexdigest()[:12]

    def detect_task(self, y: pd.Series) -> str:
        if y.dtype == "object" or y.nunique() < 20:
            return "classification"
        return "regression"

    def profile(self, X: pd.DataFrame, y: pd.Series) -> DatasetProfile:

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_features = X.select_dtypes(include=["datetime"]).columns.tolist()

        missing_report = (X.isnull().mean() * 100).round(2).to_dict()

        task_type = self.detect_task(y)

        imbalance_ratio = None
        if task_type == "classification":
            class_counts = y.value_counts(normalize=True)
            imbalance_ratio = round(class_counts.max(), 3)

        skewed_columns = []
        for col in numeric_features:
            if abs(X[col].skew()) > 1:
                skewed_columns.append(col)

        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]

        duplicate_rows = X.duplicated().sum()

        high_cardinality_cols = [
            col for col in categorical_features
            if X[col].nunique() > 0.3 * len(X)
        ]

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
            duplicate_rows=duplicate_rows,
            high_cardinality_cols=high_cardinality_cols,
            task_type=task_type
        )
