"""
Automatic Data Cleaning Module

Applies automated cleaning actions: remove duplicates, impute missing values, remove ID columns, encoding,
and provides fit / transform methods so we can avoid leakage (fit only on training data).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

from ..config import AUTO_CLEAN_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class AutoCleaner:
    """
    Automatically cleans dataset based on profile and configuration.

    New API:
      - fit(X_train, y_train) -> fits imputers/encoders and applies cleaning to training set
      - transform(X) -> transforms new data using fitted transformers
      - fit_transform(X_train, y_train) -> convenience

    The class also keeps internal state:
      - numeric_imputer_, categorical_imputer_
      - ordinal_encoder_, bool_label_encoders_ (dict)
      - ohe_, output_columns_, dropped_columns_
    """

    def __init__(self,
                 profile=None,
                 imputer_strategy: dict = None,
                 encoder_strategy: dict = None,
                 scaler: Optional[str] = None,
                 id_columns: Optional[List[str]] = None):
        self.profile = profile
        self.imputer_strategy = imputer_strategy or {}
        self.encoder_strategy = encoder_strategy or {}
        self.scaler = scaler
        self.id_columns = id_columns or []

        # fitted objects
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.ordinal_encoder_ = None
        self.bool_label_encoders_ = {}  # per-col label encoders
        self.ohe_ = None

        # metadata
        self.dropped_columns_ = []
        self.removed_id_columns_ = []
        self.low_variance_columns_removed_ = []
        self.constant_columns_removed_ = []
        self.duplicates_removed_ = 0
        self.output_columns_ = None  # final column list after transform

    # -------------------------
    # Fit / Transform API
    # -------------------------
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Fit cleaning pipeline on training data and return cleaned X_train, y_train, and cleaning log.
        """
        # 1. Fit the cleaner (calculates stats, learns encoders on deduped data)
        self.fit(X, y)

        # 2. FIX: Explicit duplicate removal before transformation
        X_fit = X.copy()
        y_fit = y.copy()
        if self.duplicates_removed_ > 0 and AUTO_CLEAN_CONFIG['actions']['remove_duplicates']:
            dup_mask = X_fit.duplicated()
            X_fit = X_fit[~dup_mask]
            y_fit = y_fit[~dup_mask]

        # 3. Transform the (now deduped) X
        X_clean = self.transform(X_fit)

        # 4. Align y (safe fallback)
        if len(y_fit) != len(X_clean):
            y_fit = y_fit.loc[X_clean.index]

        log = self._build_log()
        return X_clean, y_fit, log

    @log_execution(logger_obj=logger)
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit imputers and encoders on training data and apply deterministic dropping decisions.
        Mutates internal state but does not return dataset (use fit_transform).
        """
        X = X.copy()
        y = y.copy()

        # 1. Remove duplicates from training only
        if AUTO_CLEAN_CONFIG['actions']['remove_duplicates']:
            dup_mask = X.duplicated()
            dup_count = dup_mask.sum()
            self.duplicates_removed_ = int(dup_count)
            if dup_count > 0:
                X = X[~dup_mask]
                y = y[~dup_mask]
                logger.info(f"[Cleaner.fit] Removed {dup_count} duplicate rows from training data")
        else:
            self.duplicates_removed_ = 0

        # 2. Remove ID columns (user-specified preferred, fallback to profile)
        id_cols = self.id_columns or (self.profile.id_like_columns if self.profile is not None else [])
        id_cols = [c for c in id_cols if c in X.columns]
        if id_cols:
            X = X.drop(columns=id_cols, errors='ignore')
            self.removed_id_columns_ = id_cols
            logger.info(f"[Cleaner.fit] Removed ID columns: {id_cols}")

        # 3. Remove constant columns
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols, errors='ignore')
            self.constant_columns_removed_ = constant_cols
            logger.info(f"[Cleaner.fit] Removed constant columns: {constant_cols}")

        # 4. Remove low variance columns
        low_var_cols = []
        if self.profile and getattr(self.profile, 'low_variance_columns', None):
            low_var_cols = [c for c in self.profile.low_variance_columns if c in X.columns]
        else:
            numeric_cols_temp = X.select_dtypes(include=[np.number]).columns.tolist()
            for c in numeric_cols_temp:
                if X[c].var() < 1e-5:
                    low_var_cols.append(c)
        if low_var_cols:
            X = X.drop(columns=low_var_cols, errors='ignore')
            self.low_variance_columns_removed_ = low_var_cols
            logger.info(f"[Cleaner.fit] Removed low variance columns: {low_var_cols}")

        # 5. Imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_missing = X[numeric_cols].isnull().sum().sum() if numeric_cols else 0
        numeric_method = self.imputer_strategy.get('numeric', AUTO_CLEAN_CONFIG['imputation']['numeric'])
        if numeric_cols and numeric_missing > 0:
            if numeric_method == 'mean':
                self.numeric_imputer_ = SimpleImputer(strategy='mean')
            elif numeric_method == 'median':
                self.numeric_imputer_ = SimpleImputer(strategy='median')
            elif numeric_method == 'knn' and len(X) > 5:
                try:
                    self.numeric_imputer_ = KNNImputer(n_neighbors=5)
                except Exception:
                    self.numeric_imputer_ = SimpleImputer(strategy='mean')
            else:
                self.numeric_imputer_ = SimpleImputer(strategy='mean')
            self.numeric_imputer_.fit(X[numeric_cols])

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_missing = X[cat_cols].isnull().sum().sum() if cat_cols else 0
        cat_method = self.imputer_strategy.get('categorical', AUTO_CLEAN_CONFIG['imputation']['categorical'])
        if cat_cols and cat_missing > 0:
            if cat_method == 'mode':
                self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
            else:
                self.categorical_imputer_ = SimpleImputer(strategy='constant', fill_value='MISSING')
            self.categorical_imputer_.fit(X[cat_cols])

        # 6. Encoding
        encoder_strategy = self.encoder_strategy or {}
        ordinal_cols = encoder_strategy.get('ordinal_cols', [])
        ordinal_cols = [c for c in ordinal_cols if c in X.columns]
        if ordinal_cols:
            try:
                self.ordinal_encoder_ = OrdinalEncoder()
                self.ordinal_encoder_.fit(X[ordinal_cols])
            except Exception as e:
                logger.warning(f"[Cleaner.fit] OrdinalEncoder fit failed: {e}")
                self.ordinal_encoder_ = None

        auto_bool_cols = [col for col in cat_cols if col not in ordinal_cols and X[col].nunique(dropna=True) == 2]
        bool_cols = list(set(encoder_strategy.get('bool_cols', []) + auto_bool_cols))
        bool_cols = [c for c in bool_cols if c in X.columns]
        for col in bool_cols:
            try:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.bool_label_encoders_[col] = le
            except Exception as e:
                logger.warning(f"[Cleaner.fit] LabelEncoder failed for {col}: {e}")

        to_ohe = [c for c in cat_cols if c not in ordinal_cols and c not in bool_cols]
        if to_ohe:
            try:
                try:
                    self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse=False)
                except TypeError:
                    self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.ohe_.fit(X[to_ohe])
            except Exception as e:
                logger.warning(f"[Cleaner.fit] OneHotEncoder fit failed: {e}")
                self.ohe_ = None

        # Build expected output columns
        out_cols = []
        numeric_cols_after = [c for c in numeric_cols if c in X.columns]
        out_cols.extend(numeric_cols_after)
        if ordinal_cols and self.ordinal_encoder_ is not None:
            out_cols.extend(ordinal_cols)
        out_cols.extend([c for c in bool_cols if c in X.columns])
        if self.ohe_ is not None and to_ohe:
            try:
                ohe_names = list(self.ohe_.get_feature_names_out(to_ohe))
            except Exception:
                ohe_names = []
                for c in to_ohe:
                    uniques = self.ohe_.categories_[to_ohe.index(c)]
                    for u in uniques:
                        ohe_names.append(f"{c}_{u}")
            out_cols.extend(ohe_names)

        self.output_columns_ = out_cols
        self.dropped_columns_ = [c for c in self.profile.columns if c not in X.columns] if self.profile else []

        self._fitted = True
        self._fitted_numeric_cols = numeric_cols_after
        self._fitted_ordinal_cols = ordinal_cols
        self._fitted_bool_cols = bool_cols
        self._fitted_ohe_cols = to_ohe
        self._train_index_after_duplicate_removal = X.index.copy()

        return self

    @log_execution(logger_obj=logger)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted cleaning pipeline to new data (train/test/new).
        Returns a DataFrame with columns = self.output_columns_.
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError("AutoCleaner must be fitted before calling transform().")

        X = X.copy()
        if self.removed_id_columns_:
            X = X.drop(columns=[c for c in self.removed_id_columns_ if c in X.columns], errors='ignore')
        if self.constant_columns_removed_:
            X = X.drop(columns=[c for c in self.constant_columns_removed_ if c in X.columns], errors='ignore')
        if self.low_variance_columns_removed_:
            X = X.drop(columns=[c for c in self.low_variance_columns_removed_ if c in X.columns], errors='ignore')

        if self._fitted_numeric_cols and self.numeric_imputer_ is not None:
            numeric_present = [c for c in self._fitted_numeric_cols if c in X.columns]
            if numeric_present:
                X[numeric_present] = self.numeric_imputer_.transform(X[numeric_present])

        if getattr(self, 'categorical_imputer_', None) is not None:
            cat_present = [c for c in X.select_dtypes(include=['object', 'category']).columns.tolist() if c in X.columns]
            if cat_present:
                X[cat_present] = self.categorical_imputer_.transform(X[cat_present])

        if self._fitted_ordinal_cols and self.ordinal_encoder_ is not None:
            ordinal_present = [c for c in self._fitted_ordinal_cols if c in X.columns]
            if ordinal_present:
                X[ordinal_present] = self.ordinal_encoder_.transform(X[ordinal_present])

        for col, le in self.bool_label_encoders_.items():
            if col in X.columns:
                try:
                    X[col] = le.transform(X[col].astype(str))
                except Exception:
                    classes = list(le.classes_)
                    mapping = {v: i for i, v in enumerate(classes)}
                    X[col] = X[col].astype(str).map(mapping).fillna(0).astype(int)

        ohe_cols = getattr(self, "_fitted_ohe_cols", []) or []
        ohe_df = pd.DataFrame(index=X.index)
        if self.ohe_ is not None and ohe_cols:
            to_encode = [c for c in ohe_cols if c in X.columns]
            if to_encode:
                try:
                    arr = self.ohe_.transform(X[to_encode])
                    try:
                        feature_names = list(self.ohe_.get_feature_names_out(to_encode))
                    except Exception:
                        feature_names = []
                        for i, c in enumerate(to_encode):
                            cats = self.ohe_.categories_[i]
                            for u in cats:
                                feature_names.append(f"{c}_{u}")
                    ohe_df = pd.DataFrame(arr, index=X.index, columns=feature_names)
                except Exception as e:
                    logger.warning(f"[Cleaner.transform] OHE transform failed: {e}")
                    ohe_df = pd.DataFrame(index=X.index)

        out = pd.DataFrame(index=X.index)
        for col in (self._fitted_numeric_cols or []) + (self._fitted_ordinal_cols or []) + (self._fitted_bool_cols or []):
            if col in X.columns:
                out[col] = X[col]
            else:
                out[col] = 0

        if not ohe_df.empty:
            for c in ohe_df.columns:
                out[c] = ohe_df[c]

        if self.output_columns_:
            for c in self.output_columns_:
                if c not in out.columns:
                    out[c] = 0
            out = out[self.output_columns_]

        out = out.apply(pd.to_numeric, errors='ignore')
        return out

    def _build_log(self) -> Dict:
        log = {
            'duplicates_removed': int(self.duplicates_removed_),
            'id_columns_removed': self.removed_id_columns_,
            'constant_columns_removed': self.constant_columns_removed_,
            'low_variance_columns_removed': self.low_variance_columns_removed_,
            'numeric_imputer': type(self.numeric_imputer_).__name__ if self.numeric_imputer_ else None,
            'categorical_imputer': type(self.categorical_imputer_).__name__ if self.categorical_imputer_ else None,
            'ordinal_encoder': type(self.ordinal_encoder_).__name__ if self.ordinal_encoder_ else None,
            'onehot_encoder': type(self.ohe_).__name__ if self.ohe_ else None,
            'output_columns': self.output_columns_
        }
        return log

    def get_cleaning_report(self) -> str:
        report = [
            "=" * 60,
            "DATA CLEANING REPORT (AutoCleaner fitted)",
            "=" * 60,
            f"\nDuplicates removed (train): {self.duplicates_removed_}",
            f"ID columns removed: {self.removed_id_columns_}",
            f"\nConstant columns removed: {self.constant_columns_removed_}",
            f"Low variance columns removed: {self.low_variance_columns_removed_}",
            f"\nNumeric imputer: {type(self.numeric_imputer_).__name__ if self.numeric_imputer_ else 'None'}",
            f"Categorical imputer: {type(self.categorical_imputer_).__name__ if self.categorical_imputer_ else 'None'}",
            f"Ordinal encoder: {type(self.ordinal_encoder_).__name__ if self.ordinal_encoder_ else 'None'}",
            f"OneHot encoder: {type(self.ohe_).__name__ if self.ohe_ else 'None'}",
            f"\nFinal columns after transform: {self.output_columns_}",
            "\n" + "=" * 60,
        ]
        return "\n".join(report)
