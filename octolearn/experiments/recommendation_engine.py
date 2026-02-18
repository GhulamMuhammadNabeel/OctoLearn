"""
Recommendation Engine Module (COMPLETE IMPLEMENTATION)

Synthesizes profiling data, preprocessing suggestions, and model performance
into actionable, prioritized recommendations for data scientists.

This module generates intelligent insights that guide model selection, feature
engineering, and further analysis based on the data characteristics detected
during the AutoML pipeline.
"""

from typing import Dict, List, Tuple
import pandas as pd
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class RecommendationEngine:
    """
    Generates actionable, prioritized recommendations based on dataset profile
    and model results.

    The engine analyzes various data quality metrics and patterns to produce
    recommendations organized by priority level (critical → informational).
    It uses the **raw** profile for sample-size checks (so row counts reflect
    the original dataset before deduplication) and the **clean** profile for
    feature quality checks.

    Parameters
    ----------
    profile : DatasetProfile
        Profile of the **cleaned** dataset (post-deduplication, post-encoding).
        Used for feature quality, cardinality, and missing-value checks.
    raw_profile : DatasetProfile, optional
        Profile of the **raw** dataset (before any cleaning). When provided,
        sample-size recommendations reference the original row count rather
        than the post-cleaning count.

    Examples
    --------
    >>> engine = RecommendationEngine(clean_profile, raw_profile=raw_profile)
    >>> recommendations = engine.generate()
    >>> for priority, recs in recommendations.items():
    ...     print(f"{priority}: {recs}")
    """

    def __init__(self, profile, raw_profile=None):
        """
        Initialize the RecommendationEngine.

        Parameters
        ----------
        profile : DatasetProfile
            Cleaned dataset profile used for feature-level checks.
        raw_profile : DatasetProfile, optional
            Raw dataset profile used for original sample-size checks.
            If None, falls back to ``profile`` for all checks.
        """
        self.profile = profile
        self.raw_profile = raw_profile  # Used for original row-count checks
        self.recommendations = {}

        logger.info("RecommendationEngine initialized")

    def generate(self) -> Dict[str, List[str]]:
        """
        Generate recommendations based on profile characteristics.

        Runs all internal check methods and returns a dict of prioritized
        recommendations. Empty priority buckets are excluded from the result.

        Returns
        -------
        dict
            Recommendations organized by priority level:

            - ``'critical'``: Must address before production
            - ``'high'``: Should address for better performance
            - ``'medium'``: Consider for improvements
            - ``'low'``: Optional optimizations
            - ``'informational'``: Interesting observations

        Examples
        --------
        >>> recs = engine.generate()
        >>> print(recs.get('critical', []))
        """
        self.recommendations = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'informational': []
        }

        # Generate recommendations from various analyses
        self._check_target_quality()
        self._check_feature_quality()
        self._check_class_imbalance()
        self._check_feature_count()
        self._check_cardinality()
        self._check_duplicates()
        self._check_missing_values()
        self._check_feature_engineering()
        self._check_model_considerations()

        # Filter out empty categories
        final_recs = {k: v for k, v in self.recommendations.items() if v}

        logger.info(f"Generated {sum(len(v) for v in final_recs.values())} recommendations")

        return final_recs

    def _check_target_quality(self):
        """Check target variable quality and provide recommendations."""
        if self.profile.task_type == 'unknown':
            self.recommendations['critical'].append(
                "Task type could not be determined automatically. "
                "Verify if this is classification or regression and specify explicitly."
            )

        if self.profile.task_type == 'classification':
            unique_classes = len(self.profile.stats.get('target', {}).get('unique', []))

            if unique_classes < 2:
                self.recommendations['critical'].append(
                    "Classification target has only one class. "
                    "This is not a valid classification problem. "
                    "Consider treating as regression or checking your target variable."
                )
            elif unique_classes > 20:
                self.recommendations['high'].append(
                    f"Target has {unique_classes} classes (multi-class). "
                    "Consider hierarchical classification or dimensionality reduction of target."
                )

    def _check_feature_quality(self):
        """Check overall feature quality."""
        total_features = len(self.profile.columns)
        numeric_features = len(self.profile.numeric_columns or [])
        categorical_features = len(self.profile.categorical_columns or [])

        # Check for text columns that might need NLP preprocessing
        if self.profile.text_columns:
            self.recommendations['high'].append(
                f"Detected {len(self.profile.text_columns)} text columns: {self.profile.text_columns}. "
                "Consider using text embedding models (TF-IDF, Word2Vec, or transformers) "
                "for improved performance."
            )

        # Check for date columns
        if self.profile.date_columns:
            self.recommendations['medium'].append(
                f"Detected {len(self.profile.date_columns)} date columns: {self.profile.date_columns}. "
                "Consider extracting temporal features (day of week, month, year, etc.) "
                "for better temporal pattern capture."
            )

        # Check feature type balance
        if numeric_features == 0 and categorical_features > 0:
            self.recommendations['high'].append(
                "Dataset contains only categorical features. "
                "Consider encoding strategies that preserve ordinal relationships or using "
                "tree-based models which handle categorical features natively."
            )

        if categorical_features == 0 and numeric_features > 0:
            self.recommendations['medium'].append(
                "Dataset contains only numeric features. "
                "Consider polynomial features or interaction terms to capture non-linear patterns."
            )

    def _check_class_imbalance(self):
        """Check for class imbalance in classification tasks."""
        if self.profile.task_type == 'classification':
            imbalance = self.profile.imbalance_ratio

            if imbalance is not None and imbalance > 0.8:
                self.recommendations['high'].append(
                    f"Detected significant class imbalance (ratio: {imbalance:.2%}). "
                    "Use stratified cross-validation and consider: "
                    "SMOTE for oversampling, class weights, or stratified sampling "
                    "to prevent model bias toward majority class."
                )
            elif imbalance is not None and imbalance > 0.7:
                self.recommendations['medium'].append(
                    f"Detected moderate class imbalance (ratio: {imbalance:.2%}). "
                    "Consider using stratified k-fold cross-validation and metric balancing."
                )

    def _check_feature_count(self):
        """
        Check feature-to-sample ratio using the original (raw) row count.

        Uses ``raw_profile`` if available so that deduplication does not
        artificially reduce the reported sample size in recommendations.
        """
        # Use raw profile for row count so deduplication doesn't skew the message
        ref_profile = self.raw_profile if self.raw_profile is not None else self.profile
        n_samples = ref_profile.shape[0]
        n_features = self.profile.shape[1]  # Feature count from clean profile

        origin_note = " (original dataset)" if self.raw_profile is not None else ""

        if n_samples < 100:
            self.recommendations['high'].append(
                f"Very small dataset ({n_samples:,} samples{origin_note}). "
                "Risk of overfitting is high. Use cross-validation aggressively, "
                "consider regularization, and validate on external holdout set."
            )
        elif n_samples < 500:
            self.recommendations['medium'].append(
                f"Relatively small dataset ({n_samples:,} samples{origin_note}). "
                "Use k-fold cross-validation (k≥5) and regularization techniques."
            )

        feature_ratio = n_features / max(n_samples, 1)

        if feature_ratio > 0.1:
            self.recommendations['high'].append(
                f"High feature-to-sample ratio ({feature_ratio:.2%}). "
                "Risk of overfitting. Consider feature selection, dimensionality reduction "
                "or regularized models (Ridge, Lasso, Elastic Net)."
            )
        elif feature_ratio > 0.05:
            self.recommendations['medium'].append(
                f"Moderately high feature-to-sample ratio ({feature_ratio:.2%}). "
                "Monitor model complexity and consider feature selection."
            )


    def _check_cardinality(self):
        """Check for high cardinality categorical features."""
        if self.profile.high_cardinality_cols:
            self.recommendations['medium'].append(
                f"Detected {len(self.profile.high_cardinality_cols)} high-cardinality columns "
                f"with >30% unique values: {self.profile.high_cardinality_cols[:5]}. "
                "One-hot encoding may create too many features. "
                "Consider: target encoding, frequency encoding, or keeping as categorical in tree models."
            )

    def _check_duplicates(self):
        """Check for duplicate rows."""
        dup_ratio = self.profile.duplicate_rows / max(self.profile.shape[0], 1)
        
        if dup_ratio > 0.05:
            self.recommendations['high'].append(
                f"Detected {self.profile.duplicate_rows} duplicate rows ({dup_ratio:.1%} of data). "
                "Remove duplicates before training to ensure valid evaluation. "
                "Investigate why duplicates exist."
            )
        elif self.profile.duplicate_rows > 0:
            self.recommendations['medium'].append(
                f"Detected {self.profile.duplicate_rows} duplicate rows. "
                "Remove duplicates before training to get accurate metrics."
            )

    def _check_missing_values(self):
        """Check for missing values and provide imputation recommendations."""
        missing_ratio_dict = self.profile.missing_ratio or {}
        cols_with_missing = {col: ratio for col, ratio in missing_ratio_dict.items() if ratio > 0}
        
        if not cols_with_missing:
            self.recommendations['informational'].append(
                "No missing values detected. Data quality is excellent."
            )
            return
        
        high_missing = {col: ratio for col, ratio in cols_with_missing.items() if ratio > 0.3}
        moderate_missing = {col: ratio for col, ratio in cols_with_missing.items() 
                           if 0.1 <= ratio <= 0.3}
        low_missing = {col: ratio for col, ratio in cols_with_missing.items() 
                       if 0 < ratio < 0.1}
        
        if high_missing:
            cols_list = ', '.join(list(high_missing.keys())[:3])
            self.recommendations['high'].append(
                f"Detected high missing values (>30%) in {len(high_missing)} columns: {cols_list}. "
                "Consider dropping these columns or using advanced imputation "
                "(KNN, iterative imputation, or domain knowledge)."
            )
        
        if moderate_missing:
            cols_list = ', '.join(list(moderate_missing.keys())[:3])
            self.recommendations['medium'].append(
                f"Detected moderate missing values (10-30%) in {len(moderate_missing)} columns: {cols_list}. "
                "Use appropriate imputation: median for numeric, mode for categorical."
            )
        
        if low_missing:
            self.recommendations['low'].append(
                f"Detected low missing values (<10%) in {len(low_missing)} columns. "
                "Simple imputation strategies should work well."
            )

    def _check_feature_engineering(self):
        """Provide feature engineering recommendations."""
        numeric_cols = self.profile.numeric_columns or []
        categorical_cols = self.profile.categorical_columns or []
        
        if len(numeric_cols) >= 3:
            self.recommendations['medium'].append(
                f"With {len(numeric_cols)} numeric features, consider: "
                "polynomial features, interactions between important features, "
                "and feature scaling/normalization for distance-based models."
            )
        
        if len(categorical_cols) >= 2:
            self.recommendations['low'].append(
                f"Consider creating interaction features from your {len(categorical_cols)} categorical features. "
                "Combinations like (gender, age_group) might be predictive."
            )
        
        # Check for low variance features
        if self.profile.low_variance_columns:
            self.recommendations['medium'].append(
                f"Detected {len(self.profile.low_variance_columns)} low-variance features. "
                "These provide little discriminative power. Remove them to improve model efficiency "
                "and reduce noise."
            )

    def _check_model_considerations(self):
        """Provide model selection recommendations."""
        task = self.profile.task_type
        n_features = self.profile.shape[1]
        n_samples = self.profile.shape[0]
        
        if task == 'classification':
            self.recommendations['informational'].append(
                "Classification detected. Start with: "
                "Logistic Regression (baseline), Random Forest (robust), or XGBoost (best performance). "
                "Use F1-score for imbalanced data, accuracy for balanced."
            )
        elif task == 'regression':
            self.recommendations['informational'].append(
                "Regression detected. Start with: "
                "Linear Regression (baseline), Random Forest Regressor (robust), or XGBoost (best performance). "
                "Use RMSE or MAE as evaluation metric."
            )
        
        # Model complexity recommendations
        if n_features > 50 and n_samples < 1000:
            self.recommendations['high'].append(
                f"High-dimensional data ({n_features} features, {n_samples} samples). "
                "Use regularized models (Ridge/Lasso), feature selection, or dimensionality reduction "
                "(PCA, feature importance) to avoid overfitting."
            )
        
        if n_samples > 100000:
            self.recommendations['medium'].append(
                f"Large dataset ({n_samples} samples). "
                "Consider: SGDClassifier/Regressor for linear models, "
                "mini-batch training for neural networks, and sampling for hyperparameter tuning."
            )

    def _get_recommendations_summary(self) -> str:
        """Get a summary string of recommendations."""
        lines = ["RECOMMENDATIONS SUMMARY", "=" * 70]
        
        for priority, recs in self.recommendations.items():
            if recs:
                lines.append(f"\n{priority.upper()} PRIORITY:")
                for i, rec in enumerate(recs, 1):
                    lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation shows recommendations summary."""
        return self._get_recommendations_summary()