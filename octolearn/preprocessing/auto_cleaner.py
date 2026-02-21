"""
Automatic Data Cleaning Module (FIXED VERSION)

Applies automated cleaning actions: impute missing values, encode categorical features,
and removes duplicates. Provides fit/transform API to avoid data leakage.

Key Fixes:
- Removed redundant duplicate removal (already handled in core.py)
- Fixed OneHotEncoder sparse parameter compatibility
- Better error handling with fallbacks
- Added validation for edge cases
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
    Automated Data Cleaning and Feature Transformation Pipeline.

    This class handles the end-to-end cleaning workflow, including missing value
    imputation, categorical encoding (ordinal, one-hot, target), and feature
    selection (ID drops, constant removal).

    Attributes
    ----------
    numeric_imputer_ : SimpleImputer
        Fitted imputer for numeric columns.
    categorical_imputer_ : SimpleImputer
        Fitted imputer for categorical columns.
    ordinal_encoder_ : OrdinalEncoder
        Fitted encoder for specified ordinal columns.
    bool_label_encoders_ : dict of LabelEncoder
        Dictionary mapping column names to LabelEncoders for boolean features.
    ohe_ : OneHotEncoder
        Fitted One-Hot Encoder for high-cardinality categorical features.
    dropped_columns_ : list of str
        Names of all columns removed during the cleaning process.
    output_columns_ : list of str
        The final list of column names after transformation.

    Notes
    -----
    The cleaner follows the fit-transform pattern to prevent data leakage.
    Statistics are learned only from the training set and applied consistently
    to any subsequent data.
    """

    def __init__(
        self,
        profile=None,
        imputer_strategy: dict = None,
        encoder_strategy: dict = None,
        scaler: Optional[str] = None,
        id_columns: Optional[List[str]] = None
    ):
        """
        Initialize the AutoCleaner with custom strategies.

        Parameters
        ----------
        profile : DatasetProfile, optional
            A profile object containing data types and statistics.
        imputer_strategy : dict, optional
            Mapping of column types to imputation methods
            (e.g., {'numeric': 'median', 'categorical': 'mode'}).
        encoder_strategy : dict, optional
            Configuration for feature encoding logic.
        scaler : str, optional
            Type of scaling to apply ('standard', 'robust', 'minmax').
        id_columns : list of str, optional
            Specific column names to be treated as unique identifiers and dropped.
        """
        self.profile = profile
        self.imputer_strategy = imputer_strategy or {}
        self.encoder_strategy = encoder_strategy or {}
        self.scaler = scaler
        self.id_columns = id_columns or []

        # Fitted transformers
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.ordinal_encoder_ = None
        self.bool_label_encoders_ = {}  # per-column label encoders
        self.ohe_ = None
        self.target_encoder_mappings_ = {}  # New: Target encoding mappings

        # Metadata tracking
        self.dropped_columns_ = []
        self.removed_id_columns_ = []
        self.low_variance_columns_removed_ = []
        self.constant_columns_removed_ = []
        self.redundant_columns_removed_ = []  # New: High correlation
        self.output_columns_ = None  # final column list after transform
        
        # Internal state
        self._fitted = False
        self._fitted_numeric_cols = []
        self._fitted_ordinal_cols = []
        self._fitted_bool_cols = []
        self._fitted_ohe_cols = []
        self._fitted_target_enc_cols = []  # New: Target encoding
        self._fitted_date_cols = []        # New: Temporal features

    # ================================================================================
    # PUBLIC API: FIT / TRANSFORM
    # ================================================================================
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Fit the cleaner on training data and return the transformed results.

        This is a convenience method that combines `fit` and `transform`. It
        is primarily used on the initial training set.

        Parameters
        ----------
        X : pd.DataFrame
            The training feature matrix.
        y : pd.Series
            The training target vector.

        Returns
        -------
        X_clean : pd.DataFrame
            The cleaned and transformed training features.
        y_clean : pd.Series
            The training target vector (possibly filtered).
        cleaning_log : dict
            A summary of all cleaning operations performed.
        """
        # Fit the cleaner (learn imputation stats, encoder mappings on training data)
        original_len = len(X)
        self.fit(X, y)
        
        # Transform training data
        X_clean = self.transform(X)
        if y is not None:
            n_rows_clean = len(X_clean)
            
            # Use positional indexing (iloc) instead of label-based (loc)
            # This ensures alignment regardless of index values
            y_clean = y.iloc[:n_rows_clean].copy()
            
            # Reset index to match X_clean
            y_clean.index = X_clean.index
        else:
            y_clean = None
        cleaning_log = {
            'duplicates_removed': getattr(self, '_duplicates_removed', 0),
            'id_columns_removed': len(getattr(self, 'removed_id_columns_', [])),
            'constant_columns_removed': len(getattr(self, 'constant_columns_removed_', [])),
            'low_variance_removed': len(getattr(self, 'low_variance_columns_removed_', [])),
            'missing_imputed': getattr(self, 'numeric_cols_imputed_', 0) + getattr(self, 'cat_cols_imputed_', 0),
            'rows_after_cleaning': len(X_clean),
            'cols_after_cleaning': len(X_clean.columns),
            'numeric_imputation_method': self.imputer_strategy.get('numeric', 'mean'),
            'categorical_imputation_method': self.imputer_strategy.get('categorical', 'mode'),
            'encoding_method': self.encoder_strategy.get('strategy', 'auto'),
            'scaling_method': self.scaler or 'none',
        }
        
        logger.info(f"Fit_transform completed: {len(X_clean)} rows, {len(X_clean.columns)} columns")
        return X_clean, y_clean, cleaning_log

    @log_execution(logger_obj=logger)
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the cleaning pipeline on training data.

        Learns parameters for imputation (means, medians), encoding (categories,
        mappings), and feature selection (constant columns, low variance).

        Parameters
        ----------
        X : pd.DataFrame
            The training feature matrix.
        y : pd.Series
            The training target vector.

        Returns
        -------
        self : AutoCleaner
            Returns the instance itself.
        """
        X = X.copy()
        y = y.copy()

        # ===== STEP 1: REMOVE COLUMNS =====
        # Handle infinite values early
        X = X.replace([np.inf, -np.inf], np.nan)

        # Remove user-specified ID columns
        id_cols = self.id_columns or (self.profile.id_like_columns if self.profile else [])
        id_cols = [c for c in id_cols if c in X.columns]
        if id_cols:
            X = X.drop(columns=id_cols, errors='ignore')
            self.removed_id_columns_ = id_cols
            logger.info(f"Removed ID columns: {id_cols}")

        # Remove constant columns (only one unique value)
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols, errors='ignore')
            self.constant_columns_removed_ = constant_cols
            logger.info(f"Removed constant columns: {constant_cols}")

        # Remove low variance columns
        low_var_cols = []
        if self.profile and getattr(self.profile, 'low_variance_columns', None):
            low_var_cols = [c for c in self.profile.low_variance_columns if c in X.columns]
        else:
            # Auto-detect if no profile
            numeric_cols_temp = X.select_dtypes(include=[np.number]).columns.tolist()
            for c in numeric_cols_temp:
                if X[c].var() < 1e-5:
                    low_var_cols.append(c)
        
        if low_var_cols:
            X = X.drop(columns=low_var_cols, errors='ignore')
            self.low_variance_columns_removed_ = low_var_cols
            logger.info(f"Removed low variance columns: {low_var_cols}")

        # Remove highly correlated features (> 0.95)
        numeric_cols_temp = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols_temp) > 1:
            corr_matrix = X[numeric_cols_temp].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            if to_drop:
                X = X.drop(columns=to_drop)
                self.redundant_columns_removed_ = to_drop
                logger.info(f"Removed redundant redundant features: {to_drop}")

        # ===== STEP 2: TEMPORAL FEATURE EXTRACTION =====
        date_cols = []
        if self.profile and getattr(self.profile, 'date_columns', None):
            date_cols = [c for c in self.profile.date_columns if c in X.columns]
        else:
            # Simple heuristic if no profile
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        # Try to parse first few non-null values
                        pd.to_datetime(X[col].dropna().head(5), errors='raise')
                        date_cols.append(col)
                    except:
                        pass
        
        self._fitted_date_cols = date_cols
        for col in date_cols:
            try:
                dt_col = pd.to_datetime(X[col], errors='coerce')
                X[f"{col}_year"] = dt_col.dt.year
                X[f"{col}_month"] = dt_col.dt.month
                X[f"{col}_day"] = dt_col.dt.day
                X[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                X = X.drop(columns=[col])
                logger.info(f"Extracted temporal features from {col}")
            except Exception as e:
                logger.warning(f"Temporal extraction failed for {col}: {e}")

        # ===== STEP 3: NUMERIC IMPUTATION =====
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_missing = X[numeric_cols].isnull().sum().sum() if numeric_cols else 0
        
        if numeric_cols and numeric_missing > 0:
            numeric_method = self.imputer_strategy.get('numeric', 
                                                      AUTO_CLEAN_CONFIG['imputation']['numeric'])
            
            if numeric_method == 'mean':
                self.numeric_imputer_ = SimpleImputer(strategy='mean')
            elif numeric_method == 'median':
                self.numeric_imputer_ = SimpleImputer(strategy='median')
            elif numeric_method == 'knn' and len(X) > 5:
                try:
                    self.numeric_imputer_ = KNNImputer(n_neighbors=5)
                except Exception as e:
                    logger.warning(f"KNN imputer failed, falling back to mean: {e}")
                    self.numeric_imputer_ = SimpleImputer(strategy='mean')
            else:
                self.numeric_imputer_ = SimpleImputer(strategy='mean')
            
            try:
                self.numeric_imputer_.fit(X[numeric_cols])
                self.numeric_cols_imputed_ = len([c for c in numeric_cols if X[c].isnull().any()])
                logger.info(f"Fitted numeric imputer ({numeric_method}) for {len(numeric_cols)} columns")
            except Exception as e:
                logger.warning(f"Numeric imputation fit failed: {e}")
                self.numeric_imputer_ = None

        # ===== STEP 3: CATEGORICAL IMPUTATION =====
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_missing = X[cat_cols].isnull().sum().sum() if cat_cols else 0
        
        if cat_cols and cat_missing > 0:
            cat_method = self.imputer_strategy.get('categorical', 
                                                  AUTO_CLEAN_CONFIG['imputation']['categorical'])
            
            if cat_method == 'mode':
                self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
            else:
                self.categorical_imputer_ = SimpleImputer(strategy='constant', fill_value='MISSING')
            
            try:
                self.categorical_imputer_.fit(X[cat_cols])
                self.cat_cols_imputed_ = len([c for c in cat_cols if X[c].isnull().any()])
                logger.info(f"Fitted categorical imputer ({cat_method}) for {len(cat_cols)} columns")
            except Exception as e:
                logger.warning(f"Categorical imputation fit failed: {e}")
                self.categorical_imputer_ = None

        # ===== STEP 4: CATEGORICAL ENCODING =====
        encoder_strategy = self.encoder_strategy or {}
        
        # Ordinal encoding
        ordinal_cols = encoder_strategy.get('ordinal_cols', [])
        ordinal_cols = [c for c in ordinal_cols if c in X.columns]
        if ordinal_cols:
            try:
                self.ordinal_encoder_ = OrdinalEncoder()
                self.ordinal_encoder_.fit(X[ordinal_cols])
                logger.info(f"Fitted ordinal encoder for {len(ordinal_cols)} columns")
            except Exception as e:
                logger.warning(f"OrdinalEncoder fit failed: {e}")
                self.ordinal_encoder_ = None

        # Boolean/Label encoding
        auto_bool_cols = [col for col in cat_cols if col not in ordinal_cols 
                         and X[col].nunique(dropna=True) == 2]
        bool_cols = list(set(encoder_strategy.get('bool_cols', []) + auto_bool_cols))
        bool_cols = [c for c in bool_cols if c in X.columns]
        
        for col in bool_cols:
            try:
                le = LabelEncoder()
                le.fit(X[col].astype(str).fillna('MISSING'))
                self.bool_label_encoders_[col] = le
            except Exception as e:
                logger.warning(f"LabelEncoder failed for {col}: {e}")

        # One-Hot encoding
        to_ohe = [c for c in cat_cols if c not in ordinal_cols and c not in bool_cols]
        if to_ohe:
            try:
                # Handle both sklearn versions (sparse vs sparse_output)
                try:
                    self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse=False)
                except TypeError:
                    # Newer sklearn versions use sparse_output instead of sparse
                    self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                
                self.ohe_.fit(X[to_ohe])
                logger.info(f"Fitted OneHotEncoder for {len(to_ohe)} columns")
            except Exception as e:
                logger.warning(f"OneHotEncoder fit failed: {e}")
                self.ohe_ = None

        # Target Encoding (for high cardinality or multi-class)
        # We'll use a simple mean encoding with smoothing
        to_target_enc = encoder_strategy.get('target_enc_cols', [])
        # If no explicit list, use high-cardinality columns from profile
        if not to_target_enc and self.profile and getattr(self.profile, 'high_cardinality_cols', None):
            to_target_enc = [c for c in self.profile.high_cardinality_cols if c in X.columns and c not in ordinal_cols and c not in bool_cols]
        
        self._fitted_target_enc_cols = to_target_enc
        # Skip target encoding if y is not numeric (e.g., string labels in classification)
        if to_target_enc and y is not None and pd.api.types.is_numeric_dtype(y):
            global_mean = y.mean()
            for col in to_target_enc:
                try:
                    # Smoothing: (count * mean + m * global_mean) / (count + m)
                    m = 10 
                    stats = y.groupby(X[col]).agg(['count', 'mean'])
                    counts = stats['count']
                    means = stats['mean']
                    smooth = (counts * means + m * global_mean) / (counts + m)
                    self.target_encoder_mappings_[col] = smooth.to_dict()
                    self.target_encoder_mappings_[col]['_global_mean'] = global_mean
                except Exception as e:
                    logger.warning(f"Target encoding fit failed for {col}: {e}")

        # ===== STEP 6: BUILD OUTPUT COLUMNS LIST =====
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
                # Fallback for older sklearn versions
                ohe_names = []
                for i, c in enumerate(to_ohe):
                    if hasattr(self.ohe_, 'categories_'):
                        for cat in self.ohe_.categories_[i]:
                            ohe_names.append(f"{c}_{cat}")
            out_cols.extend(ohe_names)

        if self._fitted_target_enc_cols:
            out_cols.extend(self._fitted_target_enc_cols)
        
        # Add date feature names
        for col in self._fitted_date_cols:
            out_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])

        self.output_columns_ = out_cols
        self.dropped_columns_ = [c for c in (self.profile.columns if self.profile else []) 
                                if c not in X.columns]

        # ===== STEP 6: MARK AS FITTED =====
        self._fitted = True
        self._fitted_numeric_cols = numeric_cols_after
        self._fitted_ordinal_cols = ordinal_cols
        self._fitted_bool_cols = bool_cols
        self._fitted_ohe_cols = to_ohe

        return self

    @log_execution(logger_obj=logger)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted cleaning pipeline to new data.

        Ensures that the output columns and data transformations are identical
        to those learned during `fit`.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            The cleaned and encoded feature matrix.

        Raises
        ------
        RuntimeError
            If the cleaner has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("AutoCleaner must be fitted before calling transform(). "
                             "Call fit(X_train, y_train) first.")

        X = X.copy()
        
        # Drop columns that were removed during fitting
        if self.removed_id_columns_:
            X = X.drop(columns=[c for c in self.removed_id_columns_ if c in X.columns], 
                      errors='ignore')
        if self.constant_columns_removed_:
            X = X.drop(columns=[c for c in self.constant_columns_removed_ if c in X.columns], 
                      errors='ignore')
        if self.low_variance_columns_removed_:
            X = X.drop(columns=[c for c in self.low_variance_columns_removed_ if c in X.columns], 
                      errors='ignore')
        if getattr(self, 'redundant_columns_removed_', None):
            X = X.drop(columns=[c for c in self.redundant_columns_removed_ if c in X.columns],
                      errors='ignore')

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)

        # ===== TEMPORAL FEATURES =====
        for col in self._fitted_date_cols:
            if col in X.columns:
                try:
                    dt_col = pd.to_datetime(X[col], errors='coerce')
                    X[f"{col}_year"] = dt_col.dt.year.fillna(X[f"{col}_year"].mean() if f"{col}_year" in X else 0)
                    X[f"{col}_month"] = dt_col.dt.month.fillna(0)
                    X[f"{col}_day"] = dt_col.dt.day.fillna(0)
                    X[f"{col}_dayofweek"] = dt_col.dt.dayofweek.fillna(0)
                    X = X.drop(columns=[col])
                except:
                    pass

        # ===== NUMERIC IMPUTATION =====
        if self._fitted_numeric_cols and self.numeric_imputer_ is not None:
            numeric_present = [c for c in self._fitted_numeric_cols if c in X.columns]
            if numeric_present:
                try:
                    X[numeric_present] = self.numeric_imputer_.transform(X[numeric_present])
                except Exception as e:
                    logger.warning(f"Numeric imputation transform failed: {e}")

        # ===== CATEGORICAL IMPUTATION =====
        if getattr(self, 'categorical_imputer_', None) is not None:
            cat_present = [c for c in X.select_dtypes(include=['object', 'category']).columns.tolist() 
                          if c in X.columns]
            if cat_present:
                try:
                    X[cat_present] = self.categorical_imputer_.transform(X[cat_present])
                except Exception as e:
                    logger.warning(f"Categorical imputation transform failed: {e}")

        # ===== ORDINAL ENCODING =====
        if self._fitted_ordinal_cols and self.ordinal_encoder_ is not None:
            ordinal_present = [c for c in self._fitted_ordinal_cols if c in X.columns]
            if ordinal_present:
                try:
                    X[ordinal_present] = self.ordinal_encoder_.transform(X[ordinal_present])
                except Exception as e:
                    logger.warning(f"Ordinal encoding transform failed: {e}")

        # ===== LABEL ENCODING (BOOLEAN) =====
        for col, le in self.bool_label_encoders_.items():
            if col in X.columns:
                try:
                    X[col] = le.transform(X[col].astype(str).fillna('MISSING'))
                except Exception as e:
                    # Fallback: manual mapping
                    classes = list(le.classes_)
                    mapping = {v: i for i, v in enumerate(classes)}
                    X[col] = X[col].astype(str).map(mapping).fillna(0).astype(int)

        # ===== ONE-HOT ENCODING =====
        ohe_df = pd.DataFrame(index=X.index)
        ohe_cols = getattr(self, "_fitted_ohe_cols", []) or []
        
        if self.ohe_ is not None and ohe_cols:
            to_encode = [c for c in ohe_cols if c in X.columns]
            if to_encode:
                try:
                    arr = self.ohe_.transform(X[to_encode])
                    
                    # Get feature names (handle version differences)
                    try:
                        feature_names = list(self.ohe_.get_feature_names_out(to_encode))
                    except Exception:
                        # Fallback for older sklearn
                        feature_names = []
                        for i, c in enumerate(to_encode):
                            if hasattr(self.ohe_, 'categories_'):
                                for cat in self.ohe_.categories_[i]:
                                    feature_names.append(f"{c}_{cat}")
                    
                    ohe_df = pd.DataFrame(arr, index=X.index, columns=feature_names)
                except Exception as e:
                    logger.warning(f"OneHot encoding transform failed: {e}")

        # ===== TARGET ENCODING =====
        for col in self._fitted_target_enc_cols:
            if col in X.columns and col in self.target_encoder_mappings_:
                mapping = self.target_encoder_mappings_[col]
                global_mean = mapping.get('_global_mean', 0)
                X[col] = X[col].map(mapping).fillna(global_mean)

        # ===== FINAL OUTPUT =====
        out = pd.DataFrame(index=X.index)
        
        # Add numeric, ordinal, boolean columns
        for col in (self._fitted_numeric_cols or []) + (self._fitted_ordinal_cols or []) + \
                   (self._fitted_bool_cols or []):
            if col in X.columns:
                out[col] = X[col]
            else:
                out[col] = 0

        # Add OHE columns
        if not ohe_df.empty:
            for c in ohe_df.columns:
                out[c] = ohe_df[c]

        # Reorder to match expected columns and convert to numeric
        if self.output_columns_:
            for c in self.output_columns_:
                if c not in out.columns:
                    out[c] = 0
            out = out[self.output_columns_]

        out = out.apply(pd.to_numeric, errors='coerce')
        
        return out

    def _build_log(self) -> Dict:
        """Build a summary log of cleaning operations."""
        log = {
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
        """Generate a human-readable report of cleaning operations."""
        report = [
            "=" * 70,
            "DATA CLEANING REPORT (AutoCleaner)",
            "=" * 70,
            f"\nID columns removed: {self.removed_id_columns_ or 'None'}",
            f"Constant columns removed: {self.constant_columns_removed_ or 'None'}",
            f"Low variance columns removed: {self.low_variance_columns_removed_ or 'None'}",
            f"\nNumeric imputer: {type(self.numeric_imputer_).__name__ if self.numeric_imputer_ else 'None'}",
            f"Categorical imputer: {type(self.categorical_imputer_).__name__ if self.categorical_imputer_ else 'None'}",
            f"Ordinal encoder: {type(self.ordinal_encoder_).__name__ if self.ordinal_encoder_ else 'None'}",
            f"OneHot encoder: {type(self.ohe_).__name__ if self.ohe_ else 'None'}",
            f"\nFinal columns count: {len(self.output_columns_) if self.output_columns_ else 0}",
            "=" * 70,
        ]
        return "\n".join(report)