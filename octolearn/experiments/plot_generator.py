"""
Plot Generator Module

Generates intelligent, beautiful visualization dashboards or simple plots.
Now supports 'simple' vs 'dashboard' modes via user config.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from typing import List, Optional

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class PlotGenerator:
    """
    Generates visualizations for the AutoML report.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, profile, mode: str = 'simple'):
        self.X = X
        self.y = y
        self.profile = profile
        self.mode = mode
        self.output_dir = ".octolearn/plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set professional dark theme
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", palette="deep", rc={
            "axes.facecolor": "#1B1B1B",
            "figure.facecolor": "#0D0D15",
            "grid.color": "#333333",
            "text.color": "#E0E0E0",
            "xtick.color": "#E0E0E0",
            "ytick.color": "#E0E0E0",
            "axes.labelcolor": "#00F0FF",
            "axes.titlecolor": "#00F0FF"
        })

    def generate_smart_visuals(self, limit: int = 10) -> List[str]:
        """Generate distribution plots for top numeric features"""
        paths = []
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns for visualization")
            return []
        
        ranked_cols = self._rank_features(numeric_cols)
        for col in ranked_cols[:limit]:
            try:
                path = self._plot_feature_distribution(col)
                if path:
                    paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to plot {col}: {e}")
        
        return paths
    def _rank_features(self, cols: List[str]) -> List[str]:
        """
        Ranks columns based on importance.
        """
        if self.y is not None:
            # Temporary dataframe for correlation
            temp_df = self.X[cols].copy()
            y_enc = self.y
            if not pd.api.types.is_numeric_dtype(self.y):
                y_enc = pd.factorize(self.y)[0]
            
            temp_df['__target__'] = y_enc
            
            try:
                corrs = temp_df.corr()['__target__'].drop('__target__').abs()
                return corrs.sort_values(ascending=False).index.tolist()
            except Exception:
                pass 

        # Fallback: Rank by Variance
        temp_df = self.X[cols].copy()
        try:
            normalized = (temp_df - temp_df.mean()) / temp_df.std()
            variances = normalized.var().sort_values(ascending=False)
            return variances.index.tolist()
        except Exception:
            return cols[:10]

    def _plot_feature_simple(self, col: str) -> Optional[str]:
        """
        Generates a single, high-quality distribution plot.
        """
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.X, x=col, kde=True, color='teal')
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col)
        
        filename = f"viz_simple_{col}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        return path

    def _plot_feature_dashboard(self, col: str) -> Optional[str]:
        """
        Creates a 3-panel dashboard for a single feature.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Feature Analysis: {col}", fontsize=16, fontweight='bold')
        
        is_class = self.profile.task_type == 'classification' if self.profile else False
        
        # Panel 1: Dist
        sns.histplot(data=self.X, x=col, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("Distribution")
        
        # Panel 2: Box
        sns.boxplot(data=self.X, x=col, ax=axes[1], color='lightgreen')
        axes[1].set_title("Outliers")
        
        # Panel 3: Relation
        if self.y is not None:
            if is_class:
                top_classes = self.y.value_counts().head(5).index
                mask = self.y.isin(top_classes)
                sns.violinplot(x=self.y[mask], y=self.X.loc[mask, col], ax=axes[2], palette="pastel")
                axes[2].set_title(f"By Target Class")
            else:
                sns.regplot(x=self.X[col], y=self.y, ax=axes[2], 
                           scatter_kws={'alpha':0.5, 's':20}, line_kws={'color':'red'})
                axes[2].set_title(f"vs Target")
        else:
            axes[2].axis('off')

        plt.tight_layout()
        
        filename = f"viz_dashboard_{col}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        return path

    def _plot_feature_distribution(self, col: str):
        """Create histogram with mean/median lines"""
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            self.X[col].hist(bins=30, ax=ax, alpha=0.7, color='#2E86C1')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            
            mean_val = self.X[col].mean()
            median_val = self.X[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            filename = os.path.join(self.output_dir, f'dist_{col.replace(" ", "_")}.png')
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            logger.warning(f"Failed to generate plot: {e}")
            return None

    def generate_correlation_heatmap(self):
        """Create correlation matrix heatmap"""
        try:
            numeric_df = self.X.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontweight='bold')
            
            filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            return filename
        except Exception as e:
            logger.warning(f"Failed to generate heatmap: {e}")
            return None

    def generate_shap_plot(self, model=None) -> Optional[str]:
        """
        Generate SHAP summary plot for feature importance.
        """
        if model is None or self.X is None:
            return None

        try:
            # Create object-only explainer if needed, or TreeExplainer if tree-based
            # For simplicity using basic TreeExplainer as most models in OctoLearn are tree-based
            # or LinearExplainer for linear models.
            
            # Simple heuristic for explainer type
            model_type = type(model).__name__.lower()
            
            if 'linear' in model_type or 'regression' in model_type and 'tree' not in model_type and 'forest' not in model_type and 'boost' not in model_type:
                 explainer = shap.LinearExplainer(model, self.X)
                 shap_values = explainer.shap_values(self.X)
            else:
                # Tree based defaults
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X)

            # Handle binary classification case (shap returns list of arrays)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 6))
            # Use a dark background compatible style
            plt.style.use('dark_background')
            
            shap.summary_plot(shap_values, self.X, show=False, color_bar=True, cmap='cool')
            
            filename = os.path.join(self.output_dir, 'shap_summary.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='#0D0D15') # Match report BG
            plt.close()
            
            # Reset style
            plt.style.use('default')
            sns.set_theme(style="whitegrid", palette="muted")
            
            return filename
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {e}")
            return None
