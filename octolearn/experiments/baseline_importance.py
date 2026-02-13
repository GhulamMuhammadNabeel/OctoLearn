"""
Baseline feature importance calculator - trains quick model for feature insights
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class BaselineImportance:

    def __init__(self, X, y, profile):
        self.X = X
        self.y = y
        self.profile = profile

    def calculate_importance(self) -> dict:
        """
        Train quick model and extract feature importance.
        Returns: {feature: importance_score}
        """
        try:
            X_processed = self._preprocess_features()
            
            if self.profile.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            model.fit(X_processed, self.y)
            
            importance_dict = {}
            for col, imp in zip(self.X.columns, model.feature_importances_):
                importance_dict[col] = round(float(imp), 4)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return sorted_importance
            
        except Exception as e:
            return {"error": str(e)}

    def _preprocess_features(self):
        """Simple preprocessing for model training"""
        X_copy = self.X.copy()
        
        # Handle missing values
        X_copy = X_copy.fillna(X_copy.mean(numeric_only=True))
        
        # Encode categorical features
        for col in self.profile.categorical_features:
            le = LabelEncoder()
            X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        
        return X_copy
