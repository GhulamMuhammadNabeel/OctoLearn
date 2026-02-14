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

    def _smart_sample(self, max_rows=100_000):
        if len(self.X) > max_rows:
            idx = self.X.sample(max_rows, random_state=42).index
            return self.X.loc[idx], self.y.loc[idx]
        return self.X, self.y

    def calculate_importance(self) -> dict:

        try:
            X_sample, y_sample = self._smart_sample()
            X_processed = self._preprocess(X_sample)

            if self.profile.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=40,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=40,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )

            model.fit(X_processed, y_sample)

            importance_dict = {
                col: round(float(imp), 4)
                for col, imp in zip(X_sample.columns, model.feature_importances_)
            }

            return dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))

        except Exception as e:
            return {"error": str(e)}

    def _preprocess(self, X):
        X_copy = X.copy()
        X_copy = X_copy.fillna(X_copy.mean(numeric_only=True))

        for col in self.profile.categorical_features:
            le = LabelEncoder()
            X_copy[col] = le.fit_transform(X_copy[col].astype(str))

        return X_copy
