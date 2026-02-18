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
        """Generate professional dashboard-style visualizations"""
        paths = []
        
        # identifying feature types
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Rank features by importance/variance
        ranked_cols = self._rank_features(numeric_cols + cat_cols)
        
        count = 0
        for col in ranked_cols:
            if count >= limit:
                break
                
            try:
                if col in numeric_cols:
                    path = self._plot_numeric_dashboard(col)
                else:
                    path = self._plot_categorical_dashboard(col)
                    
                if path:
                    paths.append(path)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to plot {col}: {e}")
        
        return paths

    def _rank_features(self, cols: List[str]) -> List[str]:
        """Ranks columns based on importance (correlation with target or variance)."""
        if not cols:
            return []
            
        if self.y is not None:
            # Temporary dataframe for correlation
            temp_df = self.X[cols].copy()
            y_enc = self.y
            
            # Encode target if needed
            if not pd.api.types.is_numeric_dtype(self.y):
                if self.profile and self.profile.task_type == 'classification':
                    y_enc = pd.factorize(self.y)[0]
                else:
                     # For regression with non-numeric target? Unlikely, but fallback
                    pass

            # Handle categorical features for correlation (simple factorize)
            for c in temp_df.columns:
                if not pd.api.types.is_numeric_dtype(temp_df[c]):
                    temp_df[c] = pd.factorize(temp_df[c])[0]
            
            # Add target
            temp_df['__target__'] = y_enc
            
            try:
                corrs = temp_df.corr()['__target__'].drop('__target__').abs()
                return corrs.sort_values(ascending=False).index.tolist()
            except Exception:
                pass 

        # Fallback: Rank by Variance (for numeric) or just list (for cat)
        return cols

    def _plot_numeric_dashboard(self, col: str) -> Optional[str]:
        """Creates a 3-panel dashboard for a numeric feature."""
        # Setup Figure
        fig = plt.figure(figsize=(10, 3.5))
        gs = fig.add_gridspec(1, 3)
        fig.suptitle(f"Feature Analysis: {col}", fontsize=12, fontweight='bold', color='#00F0FF')
        
        # Panel 1: Distribution with Mean/Median
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=self.X, x=col, kde=True, ax=ax1, color='#00F0FF', fill=True, alpha=0.3)
        mean_val = self.X[col].mean()
        median_val = self.X[col].median()
        ax1.axvline(mean_val, color='#FF0055', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='#00FFAA', linestyle=':', label=f'Median: {median_val:.2f}')
        ax1.legend(facecolor='#1B1B1B', edgecolor='white')
        ax1.set_title("Distribution", color='#E0E0E0')
        ax1.set_xlabel(col, color='#E0E0E0')
        
        # Panel 2: Boxplot for Outliers
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=self.X, x=col, ax=ax2, color='#BD00FF', flierprops={'markerfacecolor':'white'})
        ax2.set_title("Outliers & Range", color='#E0E0E0')
        ax2.set_xlabel(col, color='#E0E0E0')
        
        # Panel 3: Relationship with Target
        ax3 = fig.add_subplot(gs[0, 2])
        if self.y is not None:
            if self.profile and self.profile.task_type == 'classification':
                # Violin plot by class
                # Limit classes to top 5 to avoid clutter
                top_classes = self.y.value_counts().head(5).index
                mask = self.y.isin(top_classes)
                y_plot = self.y[mask]
                X_plot = self.X.loc[mask, col]
                sns.violinplot(x=y_plot, y=X_plot, ax=ax3, palette="viridis")
                ax3.set_title("Distribution by Target Class", color='#E0E0E0')
            else:
                # Scatter/Reg plot
                sns.regplot(x=self.X[col], y=self.y, ax=ax3, 
                           scatter_kws={'alpha':0.5, 'color':'#00F0FF'}, 
                           line_kws={'color':'#FF0055'})
                ax3.set_title("Relationship vs Target", color='#E0E0E0')
        else:
            ax3.text(0.5, 0.5, "No Target Available", ha='center', va='center', color='gray')
            ax3.axis('off')

        plt.tight_layout()
        filename = f"viz_dash_num_{col.replace(' ', '_')}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=120, facecolor='#0D0D15')
        plt.close()
        return path

    def _plot_categorical_dashboard(self, col: str) -> Optional[str]:
        """Creates a 2-panel dashboard for a categorical feature."""
        # Skip high cardinality (e.g. IDs)
        if self.X[col].nunique() > 20:
            return None
            
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 2)
        fig.suptitle(f"Feature Analysis: {col}", fontsize=16, fontweight='bold', color='#00F0FF')
        
        # Panel 1: Count Plot
        ax1 = fig.add_subplot(gs[0, 0])
        # Get value counts
        counts = self.X[col].value_counts().head(10)
        sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette="viridis")
        ax1.set_title(f"Top Categories (Max 10)", color='#E0E0E0')
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Target Impact (if classification)
        ax2 = fig.add_subplot(gs[0, 1])
        if self.y is not None and self.profile and self.profile.task_type == 'classification':
            # Stacked bar chart (crosstab)
            ct = pd.crosstab(self.X[col], self.y, normalize='index')
            # Limit to top 10 categories
            if len(ct) > 10:
                ct = ct.loc[counts.index]
                
            ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
            ax2.set_title("Target Distribution by Category", color='#E0E0E0')
            ax2.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
        elif self.y is not None:
             # Regression: Boxplot
             top_cats = counts.index
             mask = self.X[col].isin(top_cats)
             sns.boxplot(x=self.X.loc[mask, col], y=self.y.loc[mask], ax=ax2, palette="viridis")
             ax2.set_title("Target Distribution by Category", color='#E0E0E0')
             ax2.tick_params(axis='x', rotation=45)
        else:
             ax2.text(0.5, 0.5, "No Target Available", ha='center', va='center', color='gray')
             ax2.axis('off')

        plt.tight_layout()
        filename = f"viz_dash_cat_{col.replace(' ', '_')}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=120, facecolor='#0D0D15')
        plt.close()
        return path

    def generate_correlation_heatmap(self):
        """Create correlation matrix heatmap"""
        try:
            numeric_df = self.X.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Feature Correlation Matrix', fontweight='bold', color='#00F0FF', fontsize=14)
            
            filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='#0D0D15')
            plt.close()
            return filename
        except Exception as e:
            logger.warning(f"Failed to generate heatmap: {e}")
            return None
    
    # SHAP Generation remains the same but ensure plot paths are correct
    def generate_shap_plot(self, model=None) -> Optional[str]:
        """Generate SHAP summary plot"""
        if model is None or self.X is None:
            return None
        try:
            # Type detection for explainer
            model_type = type(model).__name__.lower()
            if 'linear' in model_type or 'regression' in model_type and 'tree' not in model_type and 'forest' not in model_type and 'boost' not in model_type:
                 explainer = shap.LinearExplainer(model, self.X)
                 shap_values = explainer.shap_values(self.X)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 6))
            plt.style.use('dark_background')
            shap.summary_plot(shap_values, self.X, show=False, color_bar=True, cmap='cool')
            
            filename = os.path.join(self.output_dir, 'shap_summary.png')
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='#0D0D15')
            plt.close()
            
            # Reset style
            plt.style.use('default')
            sns.set_theme(style="whitegrid", palette="muted")
            
            return filename
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {e}")
            return None
