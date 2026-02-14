import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings('ignore')


class PlotGenerator:
    """
    Generates dataset visualizations: distributions, correlations, SHAP plots.
    """

    def __init__(self, X, y, profile):
        """
        Initialize PlotGenerator.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        profile : DataProfile
            Dataset profile object
        """
        self.X = X
        self.y = y
        self.profile = profile
        self.plot_dir = f"_octolearn_plots_{profile.dataset_hash}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def _smart_sample(self, max_rows=50_000):
        """
        Return a sampled subset of X for faster plotting.

        Parameters
        ----------
        max_rows : int
            Maximum rows to sample

        Returns
        -------
        pd.DataFrame
        """
        if len(self.X) > max_rows:
            return self.X.sample(max_rows, random_state=42)
        return self.X

    def generate_distributions(self):
        """
        Generate distribution plots for numeric and categorical features.

        Returns
        -------
        list : paths to saved plot images
        """
        paths = []
        X_sample = self._smart_sample()

        # Numeric features
        for col in self.profile.numeric_features[:5]:
            plt.figure(figsize=(6, 4))
            plt.hist(X_sample[col].dropna(), bins=30, color='#FF4500', alpha=0.7)
            plt.title(f"Distribution of {col}", color='#FF0000')
            path = os.path.join(self.plot_dir, f"{col}_dist.png")
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            paths.append(path)

        # Categorical features
        for col in self.profile.categorical_features[:5]:
            plt.figure(figsize=(6, 4))
            counts = X_sample[col].value_counts()
            sns.barplot(x=counts.index, y=counts.values, palette='magma')
            plt.title(f"Categorical Distribution: {col}", color='#FF0000')
            plt.xticks(rotation=45)
            path = os.path.join(self.plot_dir, f"{col}_cat.png")
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            paths.append(path)

        return paths

    def generate_correlation_heatmap(self):
        """
        Generate correlation table CSV for numeric features.

        Returns
        -------
        str : path to CSV file (currently no image)
        """
        if len(self.profile.numeric_features) < 2:
            return None

        X_sample = self._smart_sample()
        corr = X_sample[self.profile.numeric_features].corr().abs()
        corr_unstacked = corr.unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1]
        top_corr = corr_unstacked.sort_values(ascending=False).head(15)

        path = os.path.join(self.plot_dir, "top_correlations.csv")
        top_corr.to_csv(path)
        return path

    def generate_shap_plot(self, model=None):
        """
        Generate SHAP summary plot using RandomForest if model not provided.

        Parameters
        ----------
        model : sklearn model, optional

        Returns
        -------
        str : path to SHAP plot image
        """
        X_sample = self._smart_sample(max_rows=10_000)
        y_sample = self.y.loc[X_sample.index]

        if model is None:
            if self.profile.target_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_sample, y_sample)

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        path = os.path.join(self.plot_dir, "shap_summary.png")
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()

        return path
