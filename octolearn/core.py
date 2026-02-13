from .profiling.data_profiler import DataProfiler
from .experiments.report_generator import ReportGenerator
from .experiments.plot_generator import PlotGenerator
from .experiments.recommendation_engine import RecommendationEngine
from .experiments.risk_scorer import RiskScorer
from .experiments.preprocessing_suggester import PreprocessingSuggester
from .experiments.baseline_importance import BaselineImportance


class AutoML:

    def __init__(self):
        self.profiler = DataProfiler()
        self.profile_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.profile_ = self.profiler.profile(X, y)
        return self

    def generate_report(self):
        """Generate comprehensive PDF intelligence report with all analyses"""
        if self.profile_ is None:
            raise ValueError("Run fit() before generating report.")

        # Generate visual plots
        plotter = PlotGenerator(self.X_, self.y_, self.profile_)
        dist_paths = plotter.generate_distributions()
        heatmap_path = plotter.generate_correlation_heatmap()
        shap_path = plotter.generate_shap_plot()

        # Generate recommendations
        recommender = RecommendationEngine(self.profile_)
        recommendations = recommender.generate()

        # Calculate risk score
        risk_scorer = RiskScorer(self.profile_, self.X_)
        risk_score, risk_category, risk_factors = risk_scorer.calculate_risk_score()

        # Generate preprocessing suggestions
        preprocessor = PreprocessingSuggester(self.profile_, self.X_)
        preprocessing_suggestions = preprocessor.generate_suggestions()

        # Calculate feature importance
        importance_calculator = BaselineImportance(self.X_, self.y_, self.profile_)
        feature_importance = importance_calculator.calculate_importance()

        # Generate report
        generator = ReportGenerator(
            self.profile_,
            dist_paths,
            heatmap_path,
            recommendations,
            risk_score=risk_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            preprocessing_suggestions=preprocessing_suggestions,
            feature_importance=feature_importance,
            shap_path=shap_path
        )

        return generator.generate()

    def get_risk_score(self):
        """Get dataset risk score without generating full report"""
        if self.profile_ is None:
            raise ValueError("Run fit() before getting risk score.")
        
        risk_scorer = RiskScorer(self.profile_, self.X_)
        score, category, factors = risk_scorer.calculate_risk_score()
        return {"score": score, "category": category, "factors": factors}

    def get_preprocessing_suggestions(self):
        """Get preprocessing suggestions without generating full report"""
        if self.profile_ is None:
            raise ValueError("Run fit() before getting preprocessing suggestions.")
        
        preprocessor = PreprocessingSuggester(self.profile_, self.X_)
        return preprocessor.generate_suggestions()

    def get_feature_importance(self):
        """Get feature importance without generating full report"""
        if self.profile_ is None:
            raise ValueError("Run fit() before getting feature importance.")
        
        importance_calculator = BaselineImportance(self.X_, self.y_, self.profile_)
        return importance_calculator.calculate_importance()

    def report(self):
        """Return the dataset profile dataclass"""
        return self.profile_
