"""
Octolearn Core Module: Main AutoML Orchestrator

Phase 1-4 Complete Pipeline:
- Dataset profiling and analysis
- Outlier detection
- Feature interaction analysis
- Automatic data cleaning
- Multiple model training with Optuna HPO
- Model registry and versioning
- Comprehensive evaluation and reporting
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

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

logger = setup_logger(__name__)


class AutoML:
    """
    Complete AutoML pipeline with all Phase 1-4 features.

    Phases:
        - Phase 1: Dataset profiling (COMPLETE ✅)
        - Phase 2: EDA & visualization (COMPLETE ✅)
        - Phase 3: Feature engineering & auto-cleaning (COMPLETE ✅)
        - Phase 4: Model training & optimization (COMPLETE ✅)

    Attributes:
        profiler (DataProfiler): Dataset profiling object.
        profile_ (DatasetProfile): Profiled dataset object.
        X_, y_ (pd.DataFrame, pd.Series): Cleaned features and target.
        X_original_, y_original_ (pd.DataFrame, pd.Series): Original data.
        outlier_results_ (dict): Outlier analysis results.
        interaction_results_ (dict): Feature interaction results.
        cleaning_log_ (dict): Data cleaning log.
        trained_models_ (dict): All trained model objects.
        best_model_ (object): Best trained model.
        model_benchmarks_ (list): Model benchmark results.
        registry_ (ModelRegistry): Model registry object.
        evaluation_metric (str): Metric for best model selection.
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
        # Phase 3 parameters
        detect_outliers: bool = True,
        analyze_interactions: bool = True,
        auto_clean: bool = True,
        # User param overrides for preprocessing
        imputer_strategy: dict = None,  # e.g. {"numeric": "mean", "categorical": "mode"}
        encoder_strategy: dict = None,  # e.g. {"ordinal_cols": ["col1"], "bool_cols": ["col2"], "default": "ohe"}
        scaler: str = None,             # e.g. "standard", "minmax", None
        id_columns: list = None,        # User-specified ID columns to drop
        # Phase 4 parameters
        train_models: bool = True,
        use_optuna: bool = True,
        use_registry: bool = True,
        parallel_processing: bool = True,
        n_models: int = 5,
        evaluation_metric: str = None   # User can specify e.g. "f1", "rmse", etc.
    ):
        """
        Initialize the complete AutoML pipeline with all configuration options.

        Args:
            use_full_data (bool): Use full dataset or sample (default False).
            sample_size (int): Rows to sample if use_full_data=False (default 500).
            parallel_workers (int): Parallel threads for report generation (default 7).
            show_progress (bool): Print progress messages (default True).
            generate_shap (bool): Generate SHAP plots (default True).
            calculate_feature_importance (bool): Calculate feature importance (default True).
            generate_recommendations (bool): Generate strategic recommendations (default True).
            detect_outliers (bool): Detect outliers (Phase 3, default True).
            analyze_interactions (bool): Analyze feature interactions (Phase 3, default True).
            auto_clean (bool): Automatically clean data (Phase 3, default True).
            imputer_strategy (dict): Imputation strategy for missing values.
            encoder_strategy (dict): Encoding strategy for categorical features.
            scaler (str): Scaling method for numeric features.
            id_columns (list): Columns to treat as IDs and exclude from modeling.
            train_models (bool): Train multiple models (Phase 4, default True).
            use_optuna (bool): Use Optuna for HPO (Phase 4, default True).
            use_registry (bool): Use model registry (Phase 4, default True).
            parallel_processing (bool): Enable parallel processing (default True).
            n_models (int): Number of models to train (default 5).
            evaluation_metric (str): Metric for best model selection (e.g., 'f1', 'rmse').
        """
        # Phase 1
        self.profiler = DataProfiler()
        self.profile_ = None
        self.X_ = None
        self.y_ = None
        self.X_original_ = None
        self.y_original_ = None
        
        # Configuration
        self.use_full_data = use_full_data
        self.sample_size = sample_size
        self.parallel_workers = parallel_workers
        self.show_progress = show_progress
        self.generate_shap = generate_shap
        self.calculate_feature_importance = calculate_feature_importance
        self.generate_recommendations = generate_recommendations
        self.parallel_processing = parallel_processing
        
        # Phase 3
        self.detect_outliers = detect_outliers
        self.analyze_interactions = analyze_interactions
        self.auto_clean = auto_clean
        self.outlier_results_ = None
        self.interaction_results_ = None
        self.cleaning_log_ = None
        # User param overrides for preprocessing
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
        self.registry_ = None if use_registry else None
        self.evaluation_metric = evaluation_metric

    def fit(self, X: pd.DataFrame, y: pd.Series,
            imputer_strategy: dict = None,
            encoder_strategy: dict = None,
            scaler: str = None,
            id_columns: list = None):
        """
        Fit the complete AutoML pipeline (profiling, EDA, cleaning, modeling).

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target variable.
            imputer_strategy (dict, optional): Override imputation strategy.
            encoder_strategy (dict, optional): Override encoding strategy.
            scaler (str, optional): Override scaling method.
            id_columns (list, optional): Override ID columns.

        Returns:
            AutoML: Self instance for chaining.

        Side Effects:
            Sets all internal attributes (profile_, X_, y_, etc.)
        """
        # Validate inputs
        validate_dataframe(X, "X")
        validate_series(y, "y")
        
        if self.show_progress:
            print("=" * 60)
            print("Octolearn AutoML Pipeline")
            print("=" * 60)
        
        # Store original data
        self.X_original_ = X.copy()
        self.y_original_ = y.copy()
        # Allow user to override params at fit time
        if imputer_strategy is not None:
            self.imputer_strategy = imputer_strategy
        if encoder_strategy is not None:
            self.encoder_strategy = encoder_strategy
        if scaler is not None:
            self.scaler = scaler
        if id_columns is not None:
            self.id_columns = id_columns
        
        # Sample if necessary
        if not self.use_full_data and X.shape[0] > self.sample_size:
            if self.show_progress:
                print(f"\n📊 Sampling {self.sample_size} rows from {X.shape[0]} total rows...")
            X_sampled = X.sample(n=self.sample_size, random_state=42)
            y_sampled = y.loc[X_sampled.index]
        else:
            X_sampled = X
            y_sampled = y
        
        self.X_ = X_sampled
        self.y_ = y_sampled
        
        # ============================================================
        # PHASE 1: DATASET PROFILING
        # ============================================================
        if self.show_progress:
            logger.info("\n📈 PHASE 1: Dataset Profiling...")
        
        self.profile_ = self.profiler.profile(self.X_, self.y_)
        
        if self.show_progress:
            logger.info(f"✅ Dataset profiled: {self.X_.shape[0]} rows, {self.X_.shape[1]} columns")
        
        # ============================================================
        # PHASE 2: EXPLORATORY DATA ANALYSIS
        # ============================================================
        if self.show_progress:
            logger.info("\n📊 PHASE 2: Exploratory Data Analysis...")

        # Outlier detection (Phase 2/3)
        if self.detect_outliers:
            try:
                if self.show_progress:
                    logger.info("🔍 Detecting outliers...")
                outlier_detector = OutlierDetector(self.X_, self.profile_)
                self.outlier_results_ = outlier_detector.detect()
                if self.show_progress:
                    logger.info(f"✅ Outlier detection complete")
            except Exception as e:
                logger.warning(f"⚠️ Outlier detection failed: {str(e)}")

        # Feature interaction analysis (Phase 2/3)
        if self.analyze_interactions:
            try:
                if self.show_progress:
                    logger.info("🔗 Analyzing feature interactions...")
                interaction_analyzer = FeatureInteractionAnalyzer(self.X_, self.y_, self.profile_)
                self.interaction_results_ = interaction_analyzer.analyze()
                if self.show_progress:
                    logger.info(f"✅ Interaction analysis complete")
            except Exception as e:
                logger.warning(f"⚠️ Interaction analysis failed: {str(e)}")

        # ============================================================
        # PHASE 2.5: PREPROCESSING SUGGESTIONS (NEW)
        # ============================================================
        if self.show_progress:
            logger.info("\n💡 Generating preprocessing suggestions (imputer, encoder, scaler, etc.)...")
        self.preprocessing_suggester_ = PreprocessingSuggester(self.profile_, self.X_)
        self.preprocessing_suggestions_ = self.preprocessing_suggester_.generate_suggestions()
        if self.show_progress:
            logger.info("Preprocessing Suggestions:")
            for key, suggestions in self.preprocessing_suggestions_.items():
                logger.info(f"  {key}: {suggestions}")

        # ============================================================
        # PHASE 3: AUTOMATIC DATA CLEANING & PREPROCESSING
        # ============================================================
        if self.auto_clean:
            try:
                if self.show_progress:
                    logger.info("\n🧹 PHASE 3: Automatic Data Cleaning...")
                cleaner = AutoCleaner(
                    self.X_,
                    self.y_,
                    self.profile_,
                    imputer_strategy=self.imputer_strategy,
                    encoder_strategy=self.encoder_strategy,
                    scaler=self.scaler,
                    id_columns=self.id_columns
                )
                self.X_, self.y_, self.cleaning_log_ = cleaner.clean()
                if self.show_progress:
                    logger.info(f"✅ Data cleaning complete")
                    if self.cleaning_log_:
                        logger.info(f"   Rows after cleaning: {self.X_.shape[0]}")
            except Exception as e:
                logger.warning(f"⚠️ Auto cleaning failed: {str(e)}")

        # Re-profile after cleaning
        if self.auto_clean and self.cleaning_log_:
            self.profile_ = self.profiler.profile(self.X_, self.y_)

        if self.show_progress:
            logger.info("\n✅ Phase 1-3 Complete: Ready for reporting and modeling")

        # ============================================================
        # PHASE 4: MODEL TRAINING (AUTOMATIC if enabled)
        # ============================================================
        if self.train_models:
            self.train_auto_models()

        return self

    def generate_report(self) -> str:
        """
        Generate a comprehensive PDF report with all analyses and visualizations.

        Returns:
            str: Path to the generated PDF file.

        Raises:
            ValueError: If fit() has not been called.
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before generating report.")

        if self.show_progress:
            logger.info("\n📄 Generating Comprehensive Report...")

        results = {}

        # ============================================================
        # PHASE 1: REPORTING TASKS
        # ============================================================
        
        def gen_distributions():
            if self.show_progress:
                logger.info("  📊 Generating feature distributions...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            paths = plotter.generate_distributions()
            return paths

        def gen_heatmap():
            if self.show_progress:
                logger.info("  🔥 Generating correlation heatmap...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            path = plotter.generate_correlation_heatmap()
            return path

        def gen_shap_plot():
            if not self.generate_shap:
                return None
            if self.show_progress:
                logger.info("  🎯 Generating SHAP explanations...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            path = plotter.generate_shap_plot()
            return path

        def calc_feature_importance():
            if not self.calculate_feature_importance:
                return None
            if self.show_progress:
                logger.info("  ⭐ Calculating feature importance...")
            importance = BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()
            return importance

        def calc_risk():
            if self.show_progress:
                logger.info("  ⚠️ Assessing data quality risk...")
            scorer = RiskScorer(self.profile_, self.X_)
            score, category, factors = scorer.calculate_risk_score()
            return score, category, factors

        def gen_preprocessing():
            if self.show_progress:
                logger.info("  🔧 Generating preprocessing strategy...")
            suggester = PreprocessingSuggester(self.profile_, self.X_)
            suggestions = suggester.generate_suggestions()
            return suggestions

        def gen_recommendations():
            if not self.generate_recommendations:
                return None
            if self.show_progress:
                logger.info("  💡 Generating recommendations...")
            recommender = RecommendationEngine(self.profile_)
            recs = recommender.generate()
            return recs

        tasks = {
            "dist_paths": gen_distributions,
            "heatmap_path": gen_heatmap,
            "shap_path": gen_shap_plot,
            "feature_importance": calc_feature_importance,
            "risk": calc_risk,
            "preprocessing_suggestions": gen_preprocessing,
            "recommendations": gen_recommendations,
        }

        # Run tasks in parallel
        n_workers = self.parallel_workers if self.parallel_processing else 1
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_name = {executor.submit(func): name for name, func in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning(f"Task {name} failed: {e}")
                    results[name] = None

        # Generate PDF
        generator = ReportGenerator(
            self.profile_,
            results.get("dist_paths"),
            results.get("heatmap_path"),
            results.get("recommendations"),
            risk_score=results.get("risk")[0] if results.get("risk") else None,
            risk_category=results.get("risk")[1] if results.get("risk") else None,
            risk_factors=results.get("risk")[2] if results.get("risk") else None,
            preprocessing_suggestions=results.get("preprocessing_suggestions"),
            feature_importance=results.get("feature_importance"),
            shap_path=results.get("shap_path")
        )

        if self.show_progress:
            logger.info("  📝 Composing PDF...")
        pdf_file = generator.generate()
        
        if self.show_progress:
            logger.info(f"✅ Report saved: {pdf_file}")

        return pdf_file

    def train_auto_models(self, evaluation_metric: str = None) -> Dict:
        """
        Train multiple models with Optuna hyperparameter optimization and select the best model.

        Args:
            evaluation_metric (str, optional): Metric to use for best model selection (overrides class default).

        Returns:
            dict: Training results and model comparison, including all benchmarks.

        Raises:
            ValueError: If fit() has not been called.
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before training models.")

        if self.show_progress:
            logger.info("\n🤖 PHASE 4: Model Training & Optimization...")

        try:
            metric = evaluation_metric or self.evaluation_metric
            trainer = ModelTrainer(self.X_, self.y_, self.profile_, evaluation_metric=metric)
            results = trainer.train_all_models()
            self.trained_models_ = trainer.trained_models
            self.best_model_ = trainer.best_model
            self.model_benchmarks_ = getattr(trainer, 'model_benchmarks', None)
            # Register models if enabled
            if self.use_registry:
                self.registry_ = ModelRegistry()
                for model_name, model in trainer.trained_models.items():
                    score = trainer.model_scores.get(model_name, 0)
                    params = trainer.best_hp_params.get(model_name, {})
                    self.registry_.register_model(
                        name=model_name,
                        model=model,
                        task_type=self.profile_.task_type,
                        metrics={'score': score},
                        parameters=params
                    )
                if self.show_progress:
                    logger.info(f"✅ {len(trainer.trained_models)} models registered in registry")
            if self.show_progress:
                logger.info(f"✅ Model training complete")
                logger.info(f"   Best model: {results.get('best_model')}")
                logger.info(f"   Best score: {results.get('best_score'):.4f}")
            return results
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {'error': str(e)}

    def evaluate_best_model(self) -> Dict:
        """
        Evaluate the best trained model on a holdout set using all relevant metrics.

        Returns:
            dict: Evaluation results (metrics, confusion matrix, etc.)

        Raises:
            ValueError: If no model has been trained.
        """
        if self.best_model_ is None:
            logger.error("No trained model available. Run train_auto_models() first.")
            return {'error': 'No trained model'}

        try:
            from sklearn.model_selection import train_test_split
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_, self.y_,
                test_size=0.2,
                random_state=42,
                stratify=self.y_ if self.profile_.task_type == 'classification' else None
            )
            
            evaluator = ModelEvaluator(
                self.best_model_,
                X_test,
                y_test,
                self.profile_.task_type
            )
            
            return evaluator.evaluate()

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {'error': str(e)}

    # ========================================================================
    # API METHODS (PHASE 1)
    # ========================================================================

    def get_risk_score(self) -> Dict:
        """
        Get dataset risk score (0-100) and contributing factors.

        Returns:
            dict: {"score": int, "category": str, "factors": dict}
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting risk score.")
        scorer = RiskScorer(self.profile_, self.X_)
        score, category, factors = scorer.calculate_risk_score()
        return {"score": score, "category": category, "factors": factors}

    def get_preprocessing_suggestions(self) -> Dict:
        """
        Get recommended preprocessing strategy (imputer, encoder, scaler, etc.).

        Returns:
            dict: Preprocessing suggestions by category.
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting preprocessing suggestions.")
        suggester = PreprocessingSuggester(self.profile_, self.X_)
        return suggester.generate_suggestions()

    def get_feature_importance(self) -> Dict:
        """
        Get feature importance ranking for all features.

        Returns:
            dict: {feature: importance_score}
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting feature importance.")
        importance = BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()
        return importance

    def get_outlier_analysis(self) -> Dict:
        """
        Get outlier detection results from the pipeline.

        Returns:
            dict: Outlier analysis results.
        """
        return self.outlier_results_ or {}

    def get_interaction_analysis(self) -> Dict:
        """
        Get feature interaction analysis results from the pipeline.

        Returns:
            dict: Feature interaction results.
        """
        return self.interaction_results_ or {}

    def get_cleaning_log(self) -> Dict:
        """
        Get the data cleaning log (all cleaning steps performed).

        Returns:
            dict: Cleaning log.
        """
        return self.cleaning_log_ or {}

    def get_trained_models(self) -> Dict:
        """
        Get all trained model objects from the pipeline.

        Returns:
            dict: {model_name: model_object}
        """
        return self.trained_models_ or {}

    def get_best_model(self):
        """
        Get the best trained model object from the pipeline.

        Returns:
            object: Best model.
        """
        return self.best_model_

    def report(self):
        """
        Return the dataset profile object (all metrics, types, etc.).

        Returns:
            DatasetProfile: Profiled dataset object.
        """
        return self.profile_
