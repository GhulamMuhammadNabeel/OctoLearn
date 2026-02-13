"""
Preprocessing suggestions engine - generates automated preprocessing recommendations
"""


class PreprocessingSuggester:

    def __init__(self, profile, X):
        self.profile = profile
        self.X = X

    def generate_suggestions(self) -> dict:
        """Generate preprocessing recommendations"""
        suggestions = {
            "missing_value_strategy": self._suggest_missing_handling(),
            "categorical_encoding": self._suggest_categorical_encoding(),
            "scaling_strategy": self._suggest_scaling(),
            "feature_engineering": self._suggest_feature_engineering(),
            "column_actions": self._suggest_column_actions()
        }
        return suggestions

    def _suggest_missing_handling(self) -> list:
        """Suggest missing value handling strategies"""
        suggestions = []
        
        if not self.profile.missing_report:
            return ["No missing values detected."]

        max_missing = max(self.profile.missing_report.values())
        
        if max_missing == 0:
            return ["No missing values detected."]
        
        if max_missing > 50:
            suggestions.append("Columns with >50% missing: Consider removal or careful imputation.")
        
        for col, pct in self.profile.missing_report.items():
            if 0 < pct <= 5:
                suggestions.append(f"'{col}': Use mean/median imputation (missing: {pct}%)")
            elif 5 < pct <= 20:
                suggestions.append(f"'{col}': Use KNN imputation or iterative imputation (missing: {pct}%)")
            elif pct > 20:
                suggestions.append(f"'{col}': High missing rate. Consider feature removal or specialized method (missing: {pct}%)")
        
        return suggestions

    def _suggest_categorical_encoding(self) -> list:
        """Suggest categorical encoding strategies"""
        suggestions = []
        
        if not self.profile.categorical_features:
            return ["No categorical features."]
        
        for col in self.profile.categorical_features:
            n_unique = self.X[col].nunique()
            
            if n_unique <= 5:
                suggestions.append(f"'{col}': Use One-Hot Encoding ({n_unique} categories)")
            elif n_unique <= 20:
                suggestions.append(f"'{col}': Use Ordinal Encoding or Target Encoding ({n_unique} categories)")
            else:
                suggestions.append(f"'{col}': Use Target Encoding or Frequency Encoding ({n_unique} categories)")
        
        return suggestions

    def _suggest_scaling(self) -> list:
        """Suggest feature scaling strategies"""
        suggestions = []
        
        if not self.profile.numeric_features:
            return ["No numeric features to scale."]
        
        if len(self.profile.skewed_columns) > 0:
            suggestions.append("Apply log transformation to skewed features before scaling.")
        
        suggestions.append("Use StandardScaler or RobustScaler for tree-based models.")
        suggestions.append("Use MinMaxScaler for neural networks or distance-based models.")
        
        return suggestions

    def _suggest_feature_engineering(self) -> list:
        """Suggest feature engineering opportunities"""
        suggestions = []
        
        if len(self.profile.numeric_features) >= 2:
            suggestions.append("Consider polynomial features for non-linear relationships.")
            suggestions.append("Analyze feature interactions (e.g., product, ratio of top features).")
        
        if len(self.profile.datetime_features) > 0:
            suggestions.append("Extract temporal features (month, day_of_week, season, etc.) from datetime columns.")
        
        if self.profile.n_rows > 1000:
            suggestions.append("Consider domain-specific feature combinations based on problem context.")
        
        return suggestions

    def _suggest_column_actions(self) -> list:
        """Suggest actions for specific columns"""
        suggestions = []
        
        if self.profile.constant_columns:
            suggestions.append(f"Remove constant columns: {self.profile.constant_columns}")
        
        if self.profile.high_cardinality_cols:
            suggestions.append(f"Handle high-cardinality features: {self.profile.high_cardinality_cols}")
        
        return suggestions
