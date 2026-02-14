"""
Advanced Preprocessing Suggestions Engine
Context-aware, task-aware, intelligence-driven
"""


class PreprocessingSuggester:

    def __init__(self, profile, X):
        self.profile = profile
        self.X = X

    def generate_suggestions(self) -> dict:

        return {
            "missing_value_strategy": self._suggest_missing_handling(),
            "categorical_encoding": self._suggest_categorical_encoding(),
            "scaling_strategy": self._suggest_scaling(),
            "feature_engineering": self._suggest_feature_engineering(),
            "column_actions": self._suggest_column_actions(),
            "risk_mitigation": self._suggest_risk_controls()
        }

    # --------------------------------------------------------
    # Missing Handling (Task-aware)
    # --------------------------------------------------------

    def _suggest_missing_handling(self):

        if not self.profile.missing_report:
            return ["No missing values detected."]

        max_missing = max(self.profile.missing_report.values())

        if max_missing == 0:
            return ["No missing values detected."]

        suggestions = []

        for col, pct in self.profile.missing_report.items():

            if pct == 0:
                continue

            if pct > 50:
                suggestions.append(
                    f"'{col}': Extremely high missing rate ({pct}%). Consider dropping."
                )

            elif pct > 20:
                if self.profile.task_type == "regression":
                    suggestions.append(
                        f"'{col}': Use IterativeImputer or model-based imputation ({pct}%)."
                    )
                else:
                    suggestions.append(
                        f"'{col}': Use KNN or class-conditional imputation ({pct}%)."
                    )

            elif pct <= 5:
                suggestions.append(
                    f"'{col}': Mean/Median imputation sufficient ({pct}%)."
                )

        return suggestions or ["No significant missing issues detected."]

    # --------------------------------------------------------
    # Categorical Encoding (Cardinality-aware)
    # --------------------------------------------------------

    def _suggest_categorical_encoding(self):

        if not self.profile.categorical_features:
            return ["No categorical features."]

        suggestions = []

        for col in self.profile.categorical_features:

            n_unique = self.X[col].nunique()

            if n_unique <= 5:
                suggestions.append(
                    f"'{col}': One-Hot Encoding recommended ({n_unique} categories)."
                )

            elif n_unique <= 20:
                suggestions.append(
                    f"'{col}': Ordinal or Target Encoding suitable ({n_unique} categories)."
                )

            else:
                suggestions.append(
                    f"'{col}': High cardinality. Prefer Target/Frequency Encoding."
                )

        return suggestions

    # --------------------------------------------------------
    # Scaling (Model-aware logic)
    # --------------------------------------------------------

    def _suggest_scaling(self):

        if not self.profile.numeric_features:
            return ["No numeric features to scale."]

        suggestions = []

        if self.profile.skewed_columns:
            suggestions.append(
                f"Apply log transformation to skewed features: {self.profile.skewed_columns}"
            )

        suggestions.append(
            "Scaling required for linear models, SVM, KNN, and neural networks."
        )

        suggestions.append(
            "Scaling NOT required for tree-based models (RandomForest, XGBoost)."
        )

        return suggestions

    # --------------------------------------------------------
    # Feature Engineering (Intelligent triggers)
    # --------------------------------------------------------

    def _suggest_feature_engineering(self):

        suggestions = []

        if len(self.profile.numeric_features) >= 2:
            suggestions.append(
                "Explore feature interactions (ratios, products, differences)."
            )

        if self.profile.datetime_features:
            suggestions.append(
                "Extract temporal features (month, weekday, quarter, trend indicators)."
            )

        if self.profile.n_rows > 50000:
            suggestions.append(
                "Consider dimensionality reduction (PCA) if training speed becomes issue."
            )

        return suggestions or ["No immediate feature engineering requirements."]

    # --------------------------------------------------------
    # Column Actions (Now includes ID + variance)
    # --------------------------------------------------------

    def _suggest_column_actions(self):

        suggestions = []

        if self.profile.id_like_columns:
            suggestions.append(
                f"Remove identifier columns: {self.profile.id_like_columns}"
            )

        if self.profile.constant_columns:
            suggestions.append(
                f"Remove constant columns: {self.profile.constant_columns}"
            )

        if self.profile.low_variance_columns:
            suggestions.append(
                f"Remove near-zero variance columns: {self.profile.low_variance_columns}"
            )

        if self.profile.high_cardinality_cols:
            suggestions.append(
                f"Apply encoding strategy for high-cardinality columns: {self.profile.high_cardinality_cols}"
            )

        return suggestions or ["No column-level structural issues detected."]

    # --------------------------------------------------------
    # Risk Mitigation (NEW SECTION)
    # --------------------------------------------------------

    def _suggest_risk_controls(self):

        suggestions = []

        if self.profile.leakage_suspects:
            suggestions.append(
                f"Investigate potential target leakage in: {self.profile.leakage_suspects}"
            )

        if self.profile.imbalance_ratio and self.profile.imbalance_ratio > 0.85:
            suggestions.append(
                "Apply class balancing (SMOTE, class_weight, focal loss)."
            )

        if not suggestions:
            suggestions.append("No critical risk controls required.")

        return suggestions
