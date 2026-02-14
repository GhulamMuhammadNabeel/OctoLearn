from concurrent.futures import ThreadPoolExecutor, as_completed
from .profiling.data_profiler import DataProfiler
from .experiments.report_generator import ReportGenerator
from .experiments.plot_generator import PlotGenerator
from .experiments.recommendation_engine import RecommendationEngine
from .experiments.risk_scorer import RiskScorer
from .experiments.preprocessing_suggester import PreprocessingSuggester
from .experiments.baseline_importance import BaselineImportance


class AutoML:
    """
    AutoML pipeline for data profiling, visualization, risk scoring, preprocessing
    suggestions, feature importance, and strategic recommendations.
    """

    def __init__(
        self,
        use_full_data: bool = False,
        sample_size: int = 500,
        parallel_workers: int = 7,
        show_progress: bool = True,
        generate_shap: bool = True,
        calculate_feature_importance: bool = True,
        generate_recommendations: bool = True
    ):
        """
        Initialize AutoML with configurable parameters.

        Parameters
        ----------
        use_full_data : bool
            Whether to use full dataset or sample (default False)
        sample_size : int
            Number of rows to sample if use_full_data=False (default 500)
        parallel_workers : int
            Number of parallel threads for report generation (default 7)
        show_progress : bool
            Whether to print progress messages during report generation
        generate_shap : bool
            Generate SHAP plots (default True)
        calculate_feature_importance : bool
            Calculate feature importance (default True)
        generate_recommendations : bool
            Generate strategic recommendations (default True)
        """
        self.profiler = DataProfiler()
        self.profile_ = None
        self.X_ = None
        self.y_ = None

        self.use_full_data = use_full_data
        self.sample_size = sample_size
        self.parallel_workers = parallel_workers
        self.show_progress = show_progress
        self.generate_shap = generate_shap
        self.calculate_feature_importance = calculate_feature_importance
        self.generate_recommendations = generate_recommendations

    def fit(self, X, y):
        """
        Profile the dataset and optionally sample rows for faster processing.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series or pd.DataFrame
            Target variable

        Returns
        -------
        self : AutoML
        """
        if not self.use_full_data and X.shape[0] > self.sample_size:
            if self.show_progress:
                print(f"Sampling {self.sample_size} rows from dataset...")
            X_sampled = X.sample(n=self.sample_size, random_state=42)
            y_sampled = y.loc[X_sampled.index]
        else:
            X_sampled = X
            y_sampled = y

        self.X_ = X_sampled
        self.y_ = y_sampled

        if self.show_progress:
            print("Profiling dataset...")
        self.profile_ = self.profiler.profile(self.X_, self.y_)
        if self.show_progress:
            print("Dataset profiling completed.")
        return self

    def generate_report(self):
        """
        Generate the full PDF intelligence report with all analyses.
        Uses parallel processing to speed up tasks.

        Returns
        -------
        pdf_file : str
            Path to the generated PDF file
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before generating report.")

        results = {}

        # -----------------------------
        # Define tasks to run in parallel
        # -----------------------------
        def gen_distributions():
            """Generate histograms for numeric and categorical features."""
            if self.show_progress:
                print("Generating feature distributions...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            paths = plotter.generate_distributions()
            if self.show_progress:
                print("Feature distributions done.")
            return paths

        def gen_heatmap():
            """Generate correlation heatmap and top correlation CSV."""
            if self.show_progress:
                print("Generating correlation heatmap...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            path = plotter.generate_correlation_heatmap()
            if self.show_progress:
                print("Correlation heatmap done.")
            return path

        def gen_shap_plot():
            """Generate SHAP summary plot if enabled."""
            if not self.generate_shap:
                return None
            if self.show_progress:
                print("Generating SHAP plot...")
            plotter = PlotGenerator(self.X_, self.y_, self.profile_)
            path = plotter.generate_shap_plot()
            if self.show_progress:
                print("SHAP plot done.")
            return path

        def calc_feature_importance():
            """Calculate top feature importance if enabled."""
            if not self.calculate_feature_importance:
                return None
            if self.show_progress:
                print("Calculating feature importance...")
            importance = BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()
            if self.show_progress:
                print("Feature importance done.")
            return importance

        def calc_risk():
            """Calculate dataset risk score and category."""
            if self.show_progress:
                print("Calculating risk score...")
            scorer = RiskScorer(self.profile_, self.X_)
            score, category, factors = scorer.calculate_risk_score()
            if self.show_progress:
                print(f"Risk score done: {score} ({category})")
            return score, category, factors

        def gen_preprocessing():
            """Generate preprocessing suggestions."""
            if self.show_progress:
                print("Generating preprocessing suggestions...")
            suggester = PreprocessingSuggester(self.profile_, self.X_)
            suggestions = suggester.generate_suggestions()
            if self.show_progress:
                print("Preprocessing suggestions done.")
            return suggestions

        def gen_recommendations():
            """Generate dataset/model recommendations if enabled."""
            if not self.generate_recommendations:
                return None
            if self.show_progress:
                print("Generating strategic recommendations...")
            recommender = RecommendationEngine(self.profile_)
            recs = recommender.generate()
            if self.show_progress:
                print("Recommendations done.")
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

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_name = {executor.submit(func): name for name, func in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    if self.show_progress:
                        print(f"Task {name} failed: {e}")
                    results[name] = None

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
            print("Generating PDF report...")
        pdf_file = generator.generate()
        if self.show_progress:
            print(f"Report saved: {pdf_file}")
        return pdf_file

    def get_risk_score(self):
        """
        Get dataset risk score without generating full report.

        Returns
        -------
        dict : {"score": float, "category": str, "factors": dict}
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting risk score.")
        scorer = RiskScorer(self.profile_, self.X_)
        score, category, factors = scorer.calculate_risk_score()
        return {"score": score, "category": category, "factors": factors}

    def get_preprocessing_suggestions(self):
        """
        Get preprocessing suggestions without generating full report.

        Returns
        -------
        dict : section -> list of suggestions
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting preprocessing suggestions.")
        suggester = PreprocessingSuggester(self.profile_, self.X_)
        return suggester.generate_suggestions()

    def get_feature_importance(self):
        """
        Get feature importance without generating full report.

        Returns
        -------
        dict : feature -> importance score
        """
        if self.profile_ is None:
            raise ValueError("Run fit() before getting feature importance.")
        importance = BaselineImportance(self.X_, self.y_, self.profile_).calculate_importance()
        return importance

    def report(self):
        """
        Return the dataset profile dataclass.

        Returns
        -------
        profile : DataProfile object
        """
        return self.profile_
