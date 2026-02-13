import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PlotGenerator:

    def __init__(self, X, y, profile):
        self.X = X
        self.y = y
        self.profile = profile
        self.plot_dir = f"_octolearn_plots_{profile.dataset_hash}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def generate_distributions(self):
        paths = []

        for col in self.profile.numeric_features[:5]:  # limit for MVP
            plt.figure(figsize=(6, 4))
            sns.histplot(self.X[col], kde=True)
            path = os.path.join(self.plot_dir, f"{col}_dist.png")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(path, dpi=100)
            plt.close()
            paths.append(path)

        return paths

    def generate_correlation_heatmap(self):
        if len(self.profile.numeric_features) < 2:
            return None

        plt.figure(figsize=(8, 6))
        corr = self.X[self.profile.numeric_features].corr()
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
        path = os.path.join(self.plot_dir, "correlation_heatmap.png")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()

        return path

    def generate_shap_plot(self):
        """Generate SHAP summary plot for model interpretation"""
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # Preprocess data
            X_processed = self.X.copy()
            X_processed = X_processed.fillna(X_processed.mean(numeric_only=True))
            
            # Encode categoricals
            for col in self.profile.categorical_features:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Train quick model
            if self.profile.task_type == "classification":
                model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            
            model.fit(X_processed, self.y)
            
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)
            
            # Handle multi-class classification
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
            path = os.path.join(self.plot_dir, "shap_importance.png")
            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return path
            
        except Exception as e:
            return None
