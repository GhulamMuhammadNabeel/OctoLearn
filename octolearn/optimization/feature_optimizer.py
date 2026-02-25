"""
Optuna-Driven Feature Optimization Engine

Jointly optimizes feature selection, feature generation, and model+hyperparameter
selection using Bayesian optimization (Optuna). Instead of a static feature set,
this module iteratively discovers the best combination of:

    1. Which original features to include
    2. Which synthetic features to generate (interactions, ratios, polynomials, logs)
    3. Which model + hyperparameters to use with that feature set

The result is the optimal (features, model, hyperparameters) triple that maximizes
cross-validated performance — solving overfitting/underfitting automatically.

Architecture
------------
    FeatureOptimizer
        ├── _build_feature_pool()     → creates candidate feature pool
        ├── _optimize()               → runs Optuna study
        │     └── _objective(trial)   → single trial evaluation
        └── get_result()              → returns FeatureOptimizationResult

Example
-------
    >>> optimizer = FeatureOptimizer(X_train, y_train, X_test, y_test, profile)
    >>> result = optimizer.optimize()
    >>> print(result.best_score, result.best_model_name)
    >>> X_train_opt = result.transform(X_train)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ..config import (
    FEATURE_OPTIMIZATION_CONFIG,
    MODEL_TRAINING_CONFIG,
    OPTUNA_CONFIG,
)
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class FeatureOptimizationResult:
    """
    Immutable result container for the feature optimization process.

    Attributes
    ----------
    best_features : list of str
        Names of the selected features in the optimal feature set.
    best_model_name : str
        Name of the optimal model algorithm (e.g. 'xgboost').
    best_params : dict
        Optimal hyperparameters for the best model.
    best_score : float
        Cross-validated score achieved by the optimal configuration.
    baseline_score : float
        Score achieved with all original features and default settings.
    feature_pool_size : int
        Total number of candidate features considered.
    n_original_features : int
        Number of original (non-synthetic) features.
    n_synthetic_features : int
        Number of synthetic features in the optimal set.
    optimization_history : list of dict
        Per-trial results for reporting/visualization.
    feature_generation_recipe : dict
        Serializable recipe describing how to recreate synthetic features.
    elapsed_seconds : float
        Total optimization wall-clock time.
    """

    best_features: List[str] = field(default_factory=list)
    best_model_name: str = ''
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    baseline_score: float = 0.0
    feature_pool_size: int = 0
    n_original_features: int = 0
    n_synthetic_features: int = 0
    optimization_history: List[Dict] = field(default_factory=list)
    feature_generation_recipe: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ============================================================================
# FEATURE POOL BUILDER
# ============================================================================

class _FeaturePoolBuilder:
    """
    Builds a pool of candidate features from the cleaned training data.

    The pool consists of:
        - All original numeric features
        - Top-K interaction features (A * B, sorted by |corr| with target)
        - Top-K ratio features (A / B, sorted by |corr| with target)
        - Polynomial features (A², sorted by |corr| with target)
        - Log-transformed skewed features
    """

    def __init__(self, config: Dict):
        self.config = config
        self.original_columns_: List[str] = []
        self.synthetic_columns_: List[str] = []
        self.recipe_: Dict[str, Any] = {
            'interactions': [],
            'ratios': [],
            'polynomials': [],
            'log_transforms': [],
        }

    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Build the full candidate feature pool.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned training features (already numeric after AutoCleaner).
        y : pd.Series
            Target variable.

        Returns
        -------
        pd.DataFrame
            Feature pool with original + synthetic columns.
        """
        self.original_columns_ = list(X.columns)
        pool = X.copy()

        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) < 2:
            logger.info("Not enough numeric features for synthetic generation.")
            return pool

        max_synthetic = self.config.get('max_synthetic_features', 30)
        generated = 0

        # --- Interaction features (A * B) ---
        if self.config.get('generate_interactions', True):
            top_k = self.config.get('top_k_interactions', 15)
            pool, n = self._add_interactions(pool, numeric_cols, y, top_k, max_synthetic - generated)
            generated += n

        # --- Ratio features (A / B) ---
        if self.config.get('generate_ratios', True) and generated < max_synthetic:
            top_k = self.config.get('top_k_ratios', 10)
            pool, n = self._add_ratios(pool, numeric_cols, y, top_k, max_synthetic - generated)
            generated += n

        # --- Polynomial features (A²) ---
        if self.config.get('generate_polynomials', True) and generated < max_synthetic:
            pool, n = self._add_polynomials(pool, numeric_cols, y, max_synthetic - generated)
            generated += n

        # --- Log transforms ---
        if self.config.get('generate_log_transforms', True) and generated < max_synthetic:
            pool, n = self._add_log_transforms(pool, numeric_cols, max_synthetic - generated)
            generated += n

        self.synthetic_columns_ = [c for c in pool.columns if c not in self.original_columns_]
        logger.info(
            f"Feature pool built: {len(self.original_columns_)} original + "
            f"{len(self.synthetic_columns_)} synthetic = {len(pool.columns)} total"
        )
        return pool

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same synthetic feature generation to new data (e.g. test set).

        Uses the stored recipe to recreate only the features that were generated
        during `build()`.
        """
        result = X.copy()

        # Interactions
        for feat1, feat2, name in self.recipe_['interactions']:
            if feat1 in result.columns and feat2 in result.columns:
                result[name] = result[feat1] * result[feat2]

        # Ratios
        for feat1, feat2, name in self.recipe_['ratios']:
            if feat1 in result.columns and feat2 in result.columns:
                denom = result[feat2].replace(0, np.nan)
                ratio = result[feat1] / denom
                result[name] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Polynomials
        for feat, name in self.recipe_['polynomials']:
            if feat in result.columns:
                result[name] = result[feat] ** 2

        # Log transforms
        for feat, name, offset in self.recipe_['log_transforms']:
            if feat in result.columns:
                if offset > 0:
                    result[name] = np.log(result[feat] + offset)
                else:
                    result[name] = np.log1p(result[feat])

        return result

    # ---- Private generation methods ----

    def _add_interactions(
        self, pool: pd.DataFrame, numeric_cols: List[str],
        y: pd.Series, top_k: int, budget: int
    ) -> Tuple[pd.DataFrame, int]:
        """Generate top-K pairwise interaction features ranked by |corr| with target."""
        candidates = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                try:
                    interaction = pool[c1] * pool[c2]
                    interaction = interaction.replace([np.inf, -np.inf], np.nan)
                    valid = interaction.dropna()
                    if len(valid) < 10:
                        continue
                    corr = abs(valid.corr(y.loc[valid.index]))
                    if pd.notna(corr):
                        name = f"int_{c1}_x_{c2}"
                        candidates.append((c1, c2, name, corr))
                except Exception:
                    continue

        # Sort by correlation desc, take top_k limited by budget
        candidates.sort(key=lambda x: x[3], reverse=True)
        n_added = 0
        for c1, c2, name, _ in candidates[:min(top_k, budget)]:
            interaction = pool[c1] * pool[c2]
            pool[name] = interaction.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.recipe_['interactions'].append((c1, c2, name))
            n_added += 1

        return pool, n_added

    def _add_ratios(
        self, pool: pd.DataFrame, numeric_cols: List[str],
        y: pd.Series, top_k: int, budget: int
    ) -> Tuple[pd.DataFrame, int]:
        """Generate top-K ratio features ranked by |corr| with target."""
        candidates = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                try:
                    denom = pool[c2].replace(0, np.nan)
                    ratio = pool[c1] / denom
                    ratio = ratio.replace([np.inf, -np.inf], np.nan)
                    valid = ratio.dropna()
                    if len(valid) < 10:
                        continue
                    corr = abs(valid.corr(y.loc[valid.index]))
                    if pd.notna(corr):
                        name = f"rat_{c1}_div_{c2}"
                        candidates.append((c1, c2, name, corr))
                except Exception:
                    continue

        candidates.sort(key=lambda x: x[3], reverse=True)
        n_added = 0
        for c1, c2, name, _ in candidates[:min(top_k, budget)]:
            denom = pool[c2].replace(0, np.nan)
            ratio = pool[c1] / denom
            pool[name] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.recipe_['ratios'].append((c1, c2, name))
            n_added += 1

        return pool, n_added

    def _add_polynomials(
        self, pool: pd.DataFrame, numeric_cols: List[str],
        y: pd.Series, budget: int
    ) -> Tuple[pd.DataFrame, int]:
        """Generate squared features ranked by |corr| with target."""
        candidates = []
        for col in numeric_cols:
            try:
                squared = pool[col] ** 2
                squared = squared.replace([np.inf, -np.inf], np.nan)
                valid = squared.dropna()
                if len(valid) < 10:
                    continue
                corr = abs(valid.corr(y.loc[valid.index]))
                if pd.notna(corr):
                    name = f"poly_{col}_sq"
                    candidates.append((col, name, corr))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[2], reverse=True)
        n_added = 0
        for col, name, _ in candidates[:budget]:
            pool[name] = (pool[col] ** 2).replace([np.inf, -np.inf], np.nan).fillna(0)
            self.recipe_['polynomials'].append((col, name))
            n_added += 1

        return pool, n_added

    def _add_log_transforms(
        self, pool: pd.DataFrame, numeric_cols: List[str], budget: int
    ) -> Tuple[pd.DataFrame, int]:
        """Generate log-transformed features for skewed columns."""
        n_added = 0
        for col in numeric_cols:
            if n_added >= budget:
                break
            try:
                skew = abs(pool[col].skew())
                if skew < 1.0:
                    continue

                min_val = pool[col].min()
                name = f"log_{col}"
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    pool[name] = np.log(pool[col] + offset)
                    self.recipe_['log_transforms'].append((col, name, offset))
                else:
                    pool[name] = np.log1p(pool[col])
                    self.recipe_['log_transforms'].append((col, name, 0))
                pool[name] = pool[name].replace([np.inf, -np.inf], np.nan).fillna(0)
                n_added += 1
            except Exception:
                continue

        return pool, n_added


# ============================================================================
# FEATURE OPTIMIZER (MAIN CLASS)
# ============================================================================

class FeatureOptimizer:
    """
    Optuna-driven joint optimizer for feature selection, feature generation,
    and model+hyperparameter selection.

    This optimizer replaces the traditional "fixed features → tune model" pipeline
    with a unified search over the full (features × model × hyperparameters) space.

    Parameters
    ----------
    X_train : pd.DataFrame
        Cleaned training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Cleaned test features.
    y_test : pd.Series
        Test target.
    profile : DatasetProfile
        Dataset profile with task_type and metadata.
    config : dict, optional
        Override for FEATURE_OPTIMIZATION_CONFIG.
    models_list : list of str, optional
        Explicit list of model names to include. If None, uses task-type default.
    evaluation_metric : str, optional
        Metric to optimize (e.g. 'accuracy', 'r2').

    Example
    -------
    >>> optimizer = FeatureOptimizer(X_train, y_train, X_test, y_test, profile)
    >>> result = optimizer.optimize()
    >>> print(f"Best: {result.best_model_name} with {len(result.best_features)} features → {result.best_score:.4f}")
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        profile,
        config: Optional[Dict] = None,
        models_list: Optional[List[str]] = None,
        evaluation_metric: Optional[str] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.profile = profile
        self.task_type = getattr(profile, 'task_type', 'classification')
        self.config = config or FEATURE_OPTIMIZATION_CONFIG

        # Model list
        if models_list:
            self.models = models_list
        elif self.task_type == 'classification':
            self.models = MODEL_TRAINING_CONFIG.get('classification_models', [])
        else:
            self.models = MODEL_TRAINING_CONFIG.get('regression_models', [])

        # Metric
        if evaluation_metric:
            self.eval_metric = evaluation_metric
        else:
            self.eval_metric = 'accuracy' if self.task_type == 'classification' else 'r2'

        self.direction = 'maximize' if self.eval_metric not in ['rmse', 'mse', 'mae', 'mape'] else 'minimize'

        # State
        self.pool_builder_: Optional[_FeaturePoolBuilder] = None
        self.feature_pool_train_: Optional[pd.DataFrame] = None
        self.result_: Optional[FeatureOptimizationResult] = None

    def optimize(self) -> FeatureOptimizationResult:
        """
        Run the full feature optimization process.

        Returns
        -------
        FeatureOptimizationResult
            Complete result with best features, model, params, and history.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("FEATURE OPTIMIZATION ENGINE — Starting")
        logger.info("=" * 60)

        # Step 1: Preprocess data for modeling
        X_train_proc = self._preprocess(self.X_train)
        X_test_proc = self._preprocess(self.X_test)

        # Step 2: Build feature pool from training data
        self.pool_builder_ = _FeaturePoolBuilder(self.config)
        self.feature_pool_train_ = self.pool_builder_.build(X_train_proc, self.y_train)
        all_features = list(self.feature_pool_train_.columns)

        # Step 3: Compute baseline score (all original features, default model)
        baseline_score = self._compute_baseline(X_train_proc)
        logger.info(f"Baseline score (all original features): {baseline_score:.4f}")

        # Step 4: Run Optuna study
        n_trials = self.config.get('n_trials', 40)
        timeout = self.config.get('timeout', 300)
        cv_folds = self.config.get('cv_folds', 3)
        min_features = self.config.get('min_features', 3)

        study = optuna.create_study(
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5),
            direction=self.direction,
        )

        history = []

        def objective(trial):
            """Joint optimization objective: features + model + hyperparameters."""
            try:
                # --- Feature selection ---
                # Use importance-based selection: choose a subset size then pick features
                n_features_to_select = trial.suggest_int(
                    'n_features', min_features, len(all_features)
                )

                # For each feature, decide include/exclude with a bias toward original features
                selected_features = []
                original_cols = self.pool_builder_.original_columns_
                synthetic_cols = self.pool_builder_.synthetic_columns_

                # Always include a random subset of originals
                for feat in original_cols:
                    if len(selected_features) >= n_features_to_select:
                        break
                    include = trial.suggest_categorical(f'feat_{feat}', [True, False])
                    if include:
                        selected_features.append(feat)

                # Add synthetic features if budget allows
                for feat in synthetic_cols:
                    if len(selected_features) >= n_features_to_select:
                        break
                    include = trial.suggest_categorical(f'feat_{feat}', [True, False])
                    if include:
                        selected_features.append(feat)

                # Ensure minimum features
                if len(selected_features) < min_features:
                    # Fall back to top-k originals by variance
                    remaining = [f for f in original_cols if f not in selected_features]
                    for f in remaining[:min_features - len(selected_features)]:
                        selected_features.append(f)

                if len(selected_features) == 0:
                    raise optuna.exceptions.TrialPruned()

                X_trial = self.feature_pool_train_[selected_features].copy()

                # Handle any remaining NaN/inf
                X_trial = X_trial.replace([np.inf, -np.inf], np.nan)
                X_trial = X_trial.fillna(X_trial.mean(numeric_only=True))
                X_trial = X_trial.fillna(0)

                # --- Model selection ---
                model_name = trial.suggest_categorical('model_type', self.models)

                # --- Hyperparameter selection ---
                hp_config = OPTUNA_CONFIG.get('hyperparameters', {}).get(model_name, {})
                params = self._suggest_params(trial, model_name, hp_config)

                # --- Build model ---
                model = self._build_model(model_name, params)

                # --- Cross-validate ---
                scoring = self.eval_metric
                if scoring == 'rmse':
                    scoring = 'neg_root_mean_squared_error'
                elif scoring == 'mse':
                    scoring = 'neg_mean_squared_error'
                elif scoring == 'mae':
                    scoring = 'neg_mean_absolute_error'
                elif scoring == 'f1':
                    scoring = 'f1_weighted'

                cv_scores = cross_val_score(
                    model, X_trial, self.y_train,
                    cv=min(cv_folds, len(self.y_train)),
                    scoring=scoring,
                    n_jobs=1,
                )
                score = float(np.mean(cv_scores))

                # Track history
                history.append({
                    'trial': trial.number,
                    'n_features': len(selected_features),
                    'model': model_name,
                    'score': round(score, 6),
                    'features': selected_features,
                    'params': params,
                })

                return score

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.debug(f"Trial {trial.number} failed: {e}")
                raise optuna.exceptions.TrialPruned()

        # Run optimization
        logger.info(f"Running Optuna study: {n_trials} trials, {timeout}s timeout")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        # Step 5: Extract best result
        elapsed = time.time() - start_time
        best_trial = study.best_trial

        # Recover selected features from best trial
        best_features = []
        for feat in all_features:
            key = f'feat_{feat}'
            if key in best_trial.params and best_trial.params[key]:
                best_features.append(feat)

        # Ensure minimum features
        if len(best_features) < min_features:
            best_features = list(self.pool_builder_.original_columns_[:min_features])

        best_model_name = best_trial.params.get('model_type', self.models[0])

        # Extract hyperparams (remove feature toggles and meta-params)
        best_params = {}
        skip_prefixes = ('feat_', 'n_features', 'model_type')
        for k, v in best_trial.params.items():
            if not any(k.startswith(p) for p in skip_prefixes):
                best_params[k] = v

        n_original_in_best = len([f for f in best_features if f in self.pool_builder_.original_columns_])
        n_synthetic_in_best = len(best_features) - n_original_in_best

        self.result_ = FeatureOptimizationResult(
            best_features=best_features,
            best_model_name=best_model_name,
            best_params=best_params,
            best_score=study.best_value,
            baseline_score=baseline_score,
            feature_pool_size=len(all_features),
            n_original_features=n_original_in_best,
            n_synthetic_features=n_synthetic_in_best,
            optimization_history=history,
            feature_generation_recipe={
                'interactions': self.pool_builder_.recipe_['interactions'],
                'ratios': self.pool_builder_.recipe_['ratios'],
                'polynomials': self.pool_builder_.recipe_['polynomials'],
                'log_transforms': self.pool_builder_.recipe_['log_transforms'],
            },
            elapsed_seconds=round(elapsed, 2),
        )

        improvement = self.result_.best_score - baseline_score
        direction_word = "improvement" if (
            (self.direction == 'maximize' and improvement > 0) or
            (self.direction == 'minimize' and improvement < 0)
        ) else "no improvement"

        logger.info("=" * 60)
        logger.info(f"FEATURE OPTIMIZATION COMPLETE ({elapsed:.1f}s)")
        logger.info(f"  Baseline: {baseline_score:.4f}")
        logger.info(f"  Best:     {self.result_.best_score:.4f} ({direction_word})")
        logger.info(f"  Model:    {best_model_name}")
        logger.info(f"  Features: {len(best_features)} ({n_original_in_best} original + {n_synthetic_in_best} synthetic)")
        logger.info(f"  Trials:   {len(study.trials)}")
        logger.info("=" * 60)

        return self.result_

    def get_optimized_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and test data with only the optimized feature set applied.

        Returns
        -------
        X_train_opt, X_test_opt : tuple of pd.DataFrame
            Feature-optimized train and test sets.
        """
        if self.result_ is None:
            raise RuntimeError("Must call optimize() first.")

        # Preprocess
        X_train_proc = self._preprocess(self.X_train)
        X_test_proc = self._preprocess(self.X_test)

        # Apply synthetic feature generation to both
        X_train_full = self.pool_builder_.build(X_train_proc, self.y_train)
        X_test_full = self.pool_builder_.transform(X_test_proc)

        # Select only the best features (that exist in both)
        best_features = [f for f in self.result_.best_features if f in X_train_full.columns and f in X_test_full.columns]

        X_train_opt = X_train_full[best_features].copy()
        X_test_opt = X_test_full[best_features].copy()

        # Clean any remaining issues
        for df in [X_train_opt, X_test_opt]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df.fillna(0, inplace=True)

        return X_train_opt, X_test_opt

    # ---- Private helpers ----

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple preprocessing: fill NaN + encode categoricals."""
        X_proc = X.copy()
        X_proc = X_proc.fillna(X_proc.mean(numeric_only=True))
        for col in X_proc.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))
        X_proc = X_proc.fillna(0)
        return X_proc

    def _compute_baseline(self, X_train_proc: pd.DataFrame) -> float:
        """Compute baseline CV score with all original features + default model."""
        try:
            if self.task_type == 'classification':
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(random_state=42, n_estimators=100)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42, n_estimators=100)

            scoring = self.eval_metric
            if scoring == 'rmse':
                scoring = 'neg_root_mean_squared_error'
            elif scoring == 'mse':
                scoring = 'neg_mean_squared_error'
            elif scoring == 'mae':
                scoring = 'neg_mean_absolute_error'
            elif scoring == 'f1':
                scoring = 'f1_weighted'

            cv_folds = self.config.get('cv_folds', 3)
            scores = cross_val_score(
                model, X_train_proc, self.y_train,
                cv=min(cv_folds, len(self.y_train)),
                scoring=scoring,
                n_jobs=1,
            )
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Baseline computation failed: {e}")
            return 0.0

    def _suggest_params(self, trial, model_name: str, hp_config: Dict) -> Dict:
        """Map hyperparameter config to Optuna trial suggestions."""
        params = {}
        for hp_name, hp_range in hp_config.items():
            try:
                if isinstance(hp_range, list) and len(hp_range) == 2:
                    if isinstance(hp_range[0], int) and isinstance(hp_range[1], int):
                        params[hp_name] = trial.suggest_int(hp_name, hp_range[0], hp_range[1])
                    elif isinstance(hp_range[0], (int, float)) and isinstance(hp_range[1], (int, float)):
                        params[hp_name] = trial.suggest_float(hp_name, float(hp_range[0]), float(hp_range[1]))
                    else:
                        params[hp_name] = trial.suggest_categorical(hp_name, hp_range)
                elif isinstance(hp_range, list):
                    params[hp_name] = trial.suggest_categorical(hp_name, hp_range)
            except Exception:
                pass
        return params

    def _build_model(self, model_name: str, params: Dict):
        """Build a model instance with the given parameters."""
        # Remove threading params
        params = params.copy()
        params.pop('n_jobs', None)
        params.pop('nthread', None)

        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR

        try:
            import xgboost as xgb
        except ImportError:
            xgb = None

        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None

        model_map_clf = {
            'logistic_regression': lambda p: LogisticRegression(max_iter=1000, random_state=42, n_jobs=1, **p),
            'random_forest': lambda p: RandomForestClassifier(random_state=42, n_jobs=1, **p),
            'gradient_boosting': lambda p: GradientBoostingClassifier(random_state=42, **p),
            'xgboost': lambda p: xgb.XGBClassifier(random_state=42, verbosity=0, n_jobs=1, nthread=1, **p) if xgb else None,
            'lightgbm': lambda p: lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, **p) if lgb else None,
            'svm': lambda p: SVC(probability=True, random_state=42, **p),
        }

        model_map_reg = {
            'linear_regression': lambda p: LinearRegression(**p),
            'random_forest': lambda p: RandomForestRegressor(random_state=42, n_jobs=1, **p),
            'gradient_boosting': lambda p: GradientBoostingRegressor(random_state=42, **p),
            'xgboost': lambda p: xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1, nthread=1, **p) if xgb else None,
            'lightgbm': lambda p: lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1, **p) if lgb else None,
            'svr': lambda p: SVR(**p),
        }

        model_map = model_map_clf if self.task_type == 'classification' else model_map_reg

        builder = model_map.get(model_name)
        if builder is None:
            raise ValueError(f"Unknown model: {model_name}")

        model = builder(params)
        if model is None:
            raise ImportError(f"Required package not installed for {model_name}")

        return model
