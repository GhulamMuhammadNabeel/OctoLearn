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
    # Missing Values
    # --------------------------------------------------------
    def _suggest_missing_handling(self):
        if not self.profile.missing_ratio:
            return ["No missing values detected."]

        suggestions = []
        for col, ratio in self.profile.missing_ratio.items():
            if ratio > 0.5:
                suggestions.append(f"Drop '{col}' (>50% missing).")
            elif ratio > 0:
                dtype = self.profile.feature_types.get(col, 'unknown')
                if dtype == 'numeric':
                    suggestions.append(f"Impute '{col}' with mean/median.")
                else:
                    suggestions.append(f"Impute '{col}' with mode.")

        return suggestions or ["No missing values to handle."]

    # --------------------------------------------------------
    # Categorical Encoding
    # --------------------------------------------------------
    def _suggest_categorical_encoding(self):
        suggestions = []
        for col in self.profile.categorical_columns:
            n_unique = self.profile.unique_counts.get(col, 0)
            if n_unique <= 2:
                suggestions.append(f"Label Encode '{col}' (Binary/Low Cardinality).")
            elif n_unique < 10:
                suggestions.append(f"One-Hot Encode '{col}'.")
            else:
                suggestions.append(f"Target/Frequency Encode '{col}' (High Cardinality).")
        return suggestions or ["No categorical encoding needed."]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    def _suggest_scaling(self):
        if not self.profile.numeric_columns:
            return ["No numeric features to scale."]

        suggestions = []

        # Identify skewed numeric columns dynamically
        skewed_columns = [
            col for col, stat in self.profile.stats.items()
            if "skew" in stat and abs(stat["skew"]) > 1
        ]

        if skewed_columns:
            suggestions.append(
                f"Apply log or Box-Cox transformation to skewed features: {skewed_columns}"
            )

        suggestions.append(
            "Scaling required for linear models, SVM, KNN, and neural networks."
        )
        suggestions.append(
            "Scaling NOT required for tree-based models (RandomForest, XGBoost)."
        )

        return suggestions

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------
    def _suggest_feature_engineering(self):
        suggestions = []

        if len(self.profile.numeric_columns) >= 2:
            suggestions.append("Explore feature interactions (ratios, products, differences).")

        if self.profile.date_columns:
            suggestions.append("Extract temporal features (month, weekday, quarter, trend indicators).")

        if self.profile.shape[0] > 50000:
            suggestions.append("Consider dimensionality reduction (PCA) if training speed becomes an issue.")

        return suggestions or ["No immediate feature engineering requirements."]

    # --------------------------------------------------------
    # Column Actions
    # --------------------------------------------------------
    def _suggest_column_actions(self):
        suggestions = []

        if self.profile.id_like_columns:
            suggestions.append(f"Remove identifier columns: {self.profile.id_like_columns}")

        if self.profile.constant_columns:
            suggestions.append(f"Remove constant columns: {self.profile.constant_columns}")

        if self.profile.low_variance_columns:
            suggestions.append(f"Remove near-zero variance columns: {self.profile.low_variance_columns}")

        if self.profile.high_cardinality_cols:
            suggestions.append(
                f"Apply encoding strategy for high-cardinality columns: {self.profile.high_cardinality_cols}"
            )

        return suggestions or ["No column-level structural issues detected."]

    # --------------------------------------------------------
    # Risk Mitigation
    # --------------------------------------------------------
    def _suggest_risk_controls(self):
        suggestions = []

        if self.profile.leakage_suspects:
            suggestions.append(
                f"Investigate potential target leakage in: {self.profile.leakage_suspects}"
            )

        if getattr(self.profile, "imbalance_ratio", None) is not None and self.profile.imbalance_ratio < 0.85:
            suggestions.append("Apply class balancing (SMOTE, class_weight, focal loss).")

        if getattr(self.profile, "duplicate_rows", 0) > 0:
            suggestions.append(f"Handle {self.profile.duplicate_rows} duplicate rows.")

        return suggestions or ["No critical risk controls required."]
