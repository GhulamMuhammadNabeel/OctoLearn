"""
Plot Generator Module

Generates intelligent, beautiful visualization dashboards including distributions,
KDEs, Box Plots, and Target Relationships.
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

    def __init__(self, X: pd.DataFrame, y: pd.Series, profile):
        self.X = X
        self.y = y
        self.profile = profile
        self.output_dir = ".octolearn/plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set professional theme
        sns.set_theme(style="whitegrid", palette="muted")

    def generate_smart_visuals(self, limit: int = 10) -> List[str]:
        """
        Generates comprehensive 'Dashboard' style visuals for top features.
        
        Logic:
        1. Ranks features by 'interest' (Correlation with target or Variance).
        2. Generates a 3-panel plot for each top feature:
           - Panel 1: Distribution (Hist + KDE)
           - Panel 2: Box Plot (Outlier Focus)
           - Panel 3: Relationship with Target (Scatter or Violin)
        """
        paths = []
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return []

        # 1. Rank features
        ranked_cols = self._rank_features(numeric_cols)
        
        # 2. Limit features
        cols_to_plot = ranked_cols[:limit]
        logger.info(f"Generating smart visuals for top {len(cols_to_plot)} features: {cols_to_plot}")

        # 3. Generate Dashboard Plots
        for col in cols_to_plot:
            try:
                path = self._plot_feature_dashboard(col)
                if path:
                    paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to plot dashboard for {col}: {e}")
        
        return paths

    def _rank_features(self, cols: List[str]) -> List[str]:
        """
        Ranks columns based on importance.
        - If Target is available & numeric: Rank by absolute correlation.
        - Otherwise: Rank by Variance.
        """
        if self.y is not None:
            # Temporary dataframe for correlation
            temp_df = self.X[cols].copy()
            
            # If target is categorical, encode it temporarily for ranking
            y_enc = self.y
            if not pd.api.types.is_numeric_dtype(self.y):
                y_enc = pd.factorize(self.y)[0]
            
            temp_df['__target__'] = y_enc
            
            try:
                corrs = temp_df.corr()['__target__'].drop('__target__').abs()
                return corrs.sort_values(ascending=False).index.tolist()
            except Exception:
                pass # Fallback to variance

        # Fallback: Rank by Variance (normalized)
        # Normalize first to compare apples to apples
        temp_df = self.X[cols].copy()
        try:
            normalized = (temp_df - temp_df.mean()) / temp_df.std()
            variances = normalized.var().sort_values(ascending=False)
            return variances.index.tolist()
        except Exception:
            return cols[:10] # Desperate fallback

    def _plot_feature_dashboard(self, col: str) -> Optional[str]:
        """
        Creates a 3-panel dashboard for a single feature.
        """
        # Figure setup: Wide layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Feature Analysis: {col}", fontsize=16, fontweight='bold')
        
        # Determine target type for visuals
        is_class = self.profile.task_type == 'classification' if self.profile else False
        
        # --- PANEL 1: Distribution (Hist + KDE) ---
        sns.histplot(data=self.X, x=col, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("Distribution & KDE")
        axes[0].set_xlabel(col)
        
        # --- PANEL 2: Box Plot (Outliers) ---
        sns.boxplot(data=self.X, x=col, ax=axes[1], color='lightgreen')
        axes[1].set_title("Box Plot (Outlier Detection)")
        axes[1].set_xlabel(col)
        
        # --- PANEL 3: Relation with Target ---
        if self.y is not None:
            if is_class:
                # Classification: Violin Plot or Box Plot split by class
                # We limit classes to top 5 to prevent mess
                top_classes = self.y.value_counts().head(5).index
                mask = self.y.isin(top_classes)
                sns.violinplot(x=self.y[mask], y=self.X.loc[mask, col], ax=axes[2], palette="pastel")
                axes[2].set_title(f"{col} by Target Class")
            else:
                # Regression: Scatter Plot with Trendline
                sns.regplot(x=self.X[col], y=self.y, ax=axes[2], 
                           scatter_kws={'alpha':0.5, 's':20}, line_kws={'color':'red'})
                axes[2].set_title(f"{col} vs Target")
                axes[2].set_ylabel("Target")
        else:
            # Unsupervised: QQ Plot or just empty
            axes[2].text(0.5, 0.5, "No Target Variable", ha='center', va='center')
            axes[2].axis('off')

        plt.tight_layout()
        
        filename = f"viz_dashboard_{col}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return path

    def generate_correlation_heatmap(self) -> Optional[str]:
        """
        Generates a smart correlation heatmap.
        If cols > 15, it selects the top 15 features correlated with the target.
        """
        numeric_df = self.X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None

        plt.figure(figsize=(10, 8))
        
        # SMART LOGIC: If too many columns, filter them
        if numeric_df.shape[1] > 15:
            logger.info("High dimensionality detected. Generating 'Smart Heatmap' for top 15 features.")
            
            cols_to_plot = []
            
            # Scenario A: Target is numeric/convertible -> Select features correlated with target
            if self.y is not None and pd.api.types.is_numeric_dtype(self.y):
                # Create temp frame
                temp_df = numeric_df.copy()
                temp_df['__target__'] = self.y
                
                # Get correlations with target
                corr_with_target = temp_df.corr()['__target__'].abs().sort_values(ascending=False)
                
                # Top 15 features + target (total 16 max)
                # Drop target from list itself to avoid duplicate, but keep in plot columns
                top_features = corr_with_target.index.tolist()
                if '__target__' in top_features:
                    top_features.remove('__target__')
                
                cols_to_plot = top_features[:15]
                numeric_df = numeric_df[cols_to_plot]
                
                # Add target back for the plot to show relationship
                numeric_df['Target'] = self.y
                
            # Scenario B: No valid target -> Select features with highest variance
            else:
                variances = numeric_df.var().sort_values(ascending=False)
                cols_to_plot = variances.head(15).index.tolist()
                numeric_df = numeric_df[cols_to_plot]

        # Standard Heatmap
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix (Top Features)")
        
        path = os.path.join(self.output_dir, "heatmap.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path

    def generate_shap_plot(self, model=None) -> Optional[str]:
        """
        Generates SHAP summary plot if a tree-based model is used.
        """
        # Placeholder for now as SHAP requires trained model passed explicitly
        return None