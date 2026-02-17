"""
Octolearn Core Module: Main AutoML Orchestrator
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Phase 1 modules
from .profiling.data_profiler import DataProfiler
from .experiments.report_generator import ReportGenerator
from .experiments.plot_generator import PlotGenerator
from .experiments.recommendation_engine import RecommendationEngine
from .experiments.risk_scorer import RiskScorer
from .experiments.preprocessing_suggester import PreprocessingSuggester
from .experiments.baseline_importance import BaselineImportance

# Phase 2 modules
from .experiments.outlier_detector import OutlierDetector

# Phase 3 modules
from .feature.interaction_analyzer import FeatureInteractionAnalyzer
from .preprocessing.auto_cleaner import AutoCleaner

# Phase 4 modules
from .models.model_trainer import ModelTrainer
from .models.registry import ModelRegistry
from .evaluation.metrics import ModelEvaluator

# Utilities
from .utils.helpers import setup_logger, validate_dataframe, validate_series
from .config import PARALLEL_CONFIG
from sklearn.model_selection import train_test_split

logger = setup_logger(__name__)


class AutoML:
    """
    Complete AutoML pipeline.
    """

    def __init__(
        self,
        use_full_data: bool = False,
        sample_size: int = 500,
        parallel_workers: int = 7,
        show_progress: bool = True,
        generate_shap: bool = True,
        calculate_feature_importance: bool = True,
        generate_recommendations: bool = True,
        detect_outliers: bool = True,
        analyze_interactions: bool = True,
        auto_clean: bool = True,
        imputer_strategy: dict = None,
        encoder_strategy: dict = None,
        scaler: str = None,
        id_columns: list = None,
        train_models: bool = True,
        use_optuna: bool = True,
        use_registry: bool = True,
        parallel_processing: bool = True,
        n_models: int = 5,
        evaluation_metric: str = None,
        test_size: float = 0.2,
        random_state: int = 42,
        visuals_limit: int = 10,
        report_detail: str = 'detailed' # 'brief' or 'detailed'
    ):
        # Phase 1
        self.profiler = DataProfiler()
        self.profile_ = None
        self.X_ = None
        self.y_ = None
        self.X_original_ = None
        self.y_original_ = None

        # training/test split artifacts (after cleaning)
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None

        # store cleaner
        self.cleaner_ = None

        # Configuration
        self.use_full_data = use_full_data
        self.sample_size = sample_size
        self.parallel_workers = parallel_workers
        self.show_progress = show_progress
        self.generate_shap = generate_shap
        self.calculate_feature_importance = calculate_feature_importance
        self.generate_recommendations = generate_recommendations
        self.parallel_processing = parallel_processing

        # Visuals
        self.visuals_limit = visuals_limit
        self.report_detail = report_detail

        # Phase 3
        self.detect_outliers = detect_outliers
        self.analyze_interactions = analyze_interactions
        self.auto_clean = auto_clean
        self.outlier_results_ = None
        self.interaction_results_ = None
        self.cleaning_log_ = None
        self.preprocessing_suggester_ = None
        self.preprocessing_suggestions_ = None

        # User param overrides
        self.imputer_strategy = imputer_strategy or {}
        self.encoder_strategy = encoder_strategy or {}
        self.scaler = scaler
        self.id_columns = id_columns

        # Phase 4
        self.train_models = train_models
        self.use_optuna = use_optuna
        self.use_registry = use_registry
        self.n_models = n_models
        self.trained_models_ = None
        self.best_model_ = None
        self.model_benchmarks_ = None
        self.registry_ = None
        self.evaluation_metric = evaluation_metric

        # split params
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series,
            imputer_strategy: dict = None,
            encoder_strategy: dict = None,
            scaler: str = None,
            id_columns: list = None):

        validate_dataframe(X, "X")
        validate_series(y, "y")

        # Alignment check
        if not X.index.equals(y.index):
            logger.warning("X and y indices do not match. Attempting to align them...")
            common_idx = X.index.intersection(y.index)
            if len(common_idx) == 0:
                raise ValueError("X and y have no common indices. Cannot align data.")
            X = X.loc[common_idx]
            y = y.loc[common_idx]

        if self.show_progress:
            logger.info("=" * 40)
            logger.info("Octolearn AutoML Pipeline Started")
            logger.info("=" * 40)

        self.X_original_ = X.copy()
        self.y_original_ = y.copy()

        if imputer_strategy is not None: self.imputer_strategy = imputer_strategy
        if encoder_strategy is not None: self.encoder_strategy = encoder_strategy
        if scaler is not None: self.scaler = scaler
        if id_columns is not None: self.id_columns = id_columns

        if not self.use_full_data and X.shape[0] > self.sample_size:
            if self.show_progress:
                logger.info(f"Sampling {self.sample_size} rows from {X.shape[0]} total rows...")
            X_sampled = X.sample(n=self.sample_size, random_state=self.random_state)
            y_sampled = y.loc[X_sampled.index]
        else:
            X_sampled = X.copy()
            y_sampled = y.copy()

        # Keep sample for profiling and suggestions (lightweight)
        self.X_ = X_sampled
        self.y_ = y_sampled

        # PHASE 1
        if self.show_progress: logger.info("PHASE 1: Dataset Profiling...")
        self.profile_ = self.profiler.profile(self.X_, self.y_, user_id_cols=self.id_columns)

        if self.show_progress:
            logger.info(f"Dataset profiled: {self.X_.shape[0]} rows, {self.X_.shape[1]} columns")

        # PHASE 2
        if self.show_progress: logger.info("PHASE 2: Exploratory Data Analysis...")

        if self.detect_outliers:
            try:
                if self.show_progress: logger.info("Detecting outliers...")
                outlier_detector = OutlierDetector(self.X_, self.profile_)
                self.outlier_results_ = outlier_detector.detect()
                if self.show_progress: logger.info("Outlier detection complete")
            except Exception as e:
                logger.warning(f"Outlier detection failed: {str(e)}")

        if self.analyze_interactions:
            try:
                if self.show_progress: logger.info("Analyzing feature interactions...")
                interaction_analyzer = FeatureInteractionAnalyzer(self.X_, self.y_, self.profile_)
                self.interaction_results_ = interaction_analyzer.analyze()
                if self.show_progress: logger.info("Interaction analysis complete")
            except Exception as e:
                logger.warning(f"Interaction analysis failed: {str(e)}")

        # PHASE 2.5
        if self.show_progress: logger.info("Generating preprocessing suggestions...")
        self.preprocessing_suggester_ = PreprocessingSuggester(self.profile_, self.X_)
        self.preprocessing_suggestions_ = self.preprocessing_suggester_.generate_suggestions()

        if self.show_progress:
            logger.info("Preprocessing Suggestions:")
            for key, suggestions in self.preprocessing_suggestions_.items():
                logger.info(f"  {key}: {suggestions}")

        # PHASE 3 - IMPORTANT: split BEFORE cleaning to avoid leakage
        if self.auto_clean:
            try:
                if self.show_progress: logger.info("PHASE 3: Automatic Data Cleaning (train/test split BEFORE cleaning)...")

                # create train/test split on the sampled data (not re-sampling original)
                stratify = None
                if self.profile_.task_type == 'classification':
                    # ensure stratify works only when >1 class
                    if self.y_.nunique() > 1:
                        stratify = self.y_

                X_train, X_test, y_train, y_test = train_test_split(
                    self.X_, self.y_, test_size=self.test_size,
                    random_state=self.random_state, stratify=stratify
                )

                # Fit cleaner only on TRAIN to avoid leakage
                cleaner = AutoCleaner(
                    profile=self.profile_,
                    imputer_strategy=self.imputer_strategy,
                    encoder_strategy=self.encoder_strategy,
                    scaler=self.scaler,
                    id_columns=self.id_columns
                )

                # Fit-transform on train, transform on test
                X_train_clean, y_train_clean, log_train = cleaner.fit_transform(X_train, y_train)
                X_test_clean = cleaner.transform(X_test)

                # store cleaner and logs
                self.cleaner_ = cleaner
                self.cleaning_log_ = {"train": log_train}

                # store cleaned train/test sets and full cleaned dataset (concatenate)
                self.X_train_ = X_train_clean
                self.X_test_ = X_test_clean
                self.y_train_ = y_train_clean
                self.y_test_ = y_test

                # Concatenate to keep downstream logic using self.X_ & self.y_ as cleaned full sample
                self.X_ = pd.concat([self.X_train_, self.X_test_], axis=0).sort_index()
                self.y_ = pd.concat([self.y_train_, self.y_test_], axis=0).sort_index()

                if self.show_progress:
                    logger.info("Data cleaning complete (trained on train only).")
                    logger.info(f"   Train shape: {self.X_train_.shape}, Test shape: {self.X_test_.shape}")
            except Exception as e:
                logger.warning(f"Auto cleaning failed: {str(e)}")

        # Re-profile cleaned sample if cleaning was applied
        if self.auto_clean and self.cleaning_log_:
            self.profile_ = self.profiler.profile(self.X_, self.y_, user_id_cols=self.id_columns)

        if self.show_progress: logger.info("Phase 1-3 Complete: Ready for reporting and modeling")

        # PHASE 4
        if self.train_models:
            self.train_auto_models()

        return self

    def train_auto_models(self, evaluation_metric: str = None) -> Dict:
        if self.profile_ is None: raise ValueError("Run fit() before training models.")
        if self.show_progress: logger.info("PHASE 4: Model Training & Optimization...")

        try:
            metric = evaluation_metric or self.evaluation_metric
            if metric is None:
                metric = 'f1' if self.profile_.task_type == 'classification' else 'rmse'

            # --- FIX: Pass pre-split data via Constructor to avoid immediate error ---
            if self.X_train_ is not None and self.X_test_ is not None:
                trainer = ModelTrainer(
                    X=None, y=None,  # X/y handled via splits
                    profile=self.profile_,
                    task_type=self.profile_.task_type,
                    evaluation_metric=metric,
                    X_train=self.X_train_,  # <--- Passed HERE
                    X_test=self.X_test_,    # <--- Passed HERE
                    y_train=self.y_train_,  # <--- Passed HERE
                    y_test=self.y_test_     # <--- Passed HERE
                )
            else:
                # fallback if no split exists (e.g. auto_clean=False)
                trainer = ModelTrainer(
                    self.X_, self.y_, self.profile_,
                    task_type=self.profile_.task_type,
                    evaluation_metric=metric
                )

            results = trainer.train_all_models()

            self.trained_models_ = trainer.trained_models
            self.best_model_ = trainer.best_model
            self.model_benchmarks_ = getattr(trainer, 'model_benchmarks', [])

            if self.use_registry:
                self.registry_ = ModelRegistry()
                for model_name, model in trainer.trained_models.items():
                    score = trainer.model_scores.get(model_name, 0)
                    params = trainer.best_hp_params.get(model_name, {})
                    self.registry_.register_model(
                        name=model_name, model=model, task_type=self.profile_.task_type,
                        metrics={'score': score}, parameters=params
                    )
                if self.show_progress: logger.info(f"{len(trainer.trained_models)} models registered in registry")

            if self.show_progress:
                logger.info("Model training complete")
                logger.info(f"   Best model: {results.get('best_model')}")
                logger.info(f"   Best score: {results.get('best_score'):.4f}")

            return results
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {'error': str(e)}

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Apply saved cleaning pipeline (fit on train) then predict using best_model_.
        """
        if self.best_model_ is None:
            raise ValueError("No trained model available. Run fit() first.")

        if self.cleaner_ is None:
            # If cleaning not used, try a minimal transform (noop)
            logger.warning("No cleaner saved — assuming raw features are ready for model.")
            X_clean = X_new.copy()
        else:
            X_clean = self.cleaner_.transform(X_new)

        return self.best_model_.predict(X_clean)


    def generate_report(self) -> str:
        if self.profile_ is None: raise ValueError("Run fit() before generating report.")
        if self.show_progress: logger.info("Generating Comprehensive Report...")

        results = {}
        plotter = PlotGenerator(self.X_, self.y_, self.profile_)

        try:
            # Use smart visuals with user-defined limit
            results["dist_paths"] = plotter.generate_smart_visuals(limit=self.visuals_limit)
        except Exception as e:
            logger.warning(f"Distribution plots failed: {e}")

        try:
            results["heatmap_path"] = plotter.generate_correlation_heatmap()
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")

        if self.generate_shap:
            try:
                results["shap_path"] = plotter.generate_shap_plot()
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

        if self.calculate_feature_importance:
            try:
                results["feature_importance"] = BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()
            except Exception as e:
                logger.warning(f"Feature importance failed: {e}")

        try:
            scorer = RiskScorer(self.profile_, self.X_)
            results["risk"] = scorer.calculate_risk_score()
        except Exception as e:
            logger.warning(f"Risk scoring failed: {e}")

        if self.generate_recommendations:
            try:
                recommender = RecommendationEngine(self.profile_)
                results["recommendations"] = recommender.generate()
            except Exception as e:
                logger.warning(f"Recommendations failed: {e}")

        risk_res = results.get("risk")
        risk_score, risk_category, risk_factors = (None, None, None)
        if risk_res and isinstance(risk_res, tuple) and len(risk_res) == 3:
            risk_score, risk_category, risk_factors = risk_res

        # Pass detail setting and logo path
        logo_path = os.path.join(os.path.dirname(__file__), 'images', 'logo.png')
        if not os.path.exists(logo_path):
             logo_path = None

        generator = ReportGenerator(
            self.profile_,
            results.get("dist_paths"),
            results.get("heatmap_path"),
            results.get("recommendations"),
            risk_score=risk_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            preprocessing_suggestions=self.preprocessing_suggestions_,
            feature_importance=results.get("feature_importance"),
            shap_path=results.get("shap_path"),
            model_benchmarks=self.model_benchmarks_,
            best_model_name=self.best_model_.__class__.__name__ if self.best_model_ else None,
            detail_level=self.report_detail,
            logo_path=logo_path,
            cleaning_log=self.cleaning_log_
        )

        if self.show_progress: logger.info("Composing PDF...")
        pdf_file = generator.generate()
        if self.show_progress: logger.info(f"Report saved: {pdf_file}")
        return pdf_file

    def get_risk_score(self) -> Dict:
        if self.profile_ is None: raise ValueError("Run fit() first.")
        scorer = RiskScorer(self.profile_, self.X_)
        score, category, factors = scorer.calculate_risk_score()
        return {"score": score, "category": category, "factors": factors}

    def get_preprocessing_suggestions(self) -> Dict:
        if self.profile_ is None: raise ValueError("Run fit() first.")
        if self.preprocessing_suggestions_: return self.preprocessing_suggestions_
        suggester = PreprocessingSuggester(self.profile_, self.X_)
        return suggester.generate_suggestions()

    def get_feature_importance(self) -> Dict:
        if self.profile_ is None: raise ValueError("Run fit() first.")
        return BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()

    def get_outlier_analysis(self) -> Dict: return self.outlier_results_ or {}
    def get_interaction_analysis(self) -> Dict: return self.interaction_results_ or {}
    def get_cleaning_log(self) -> Dict: return self.cleaning_log_ or {}
    def get_trained_models(self) -> Dict: return self.trained_models_ or {}
    def get_best_model(self): return self.best_model_
    def get_profile(self): return self.profile_