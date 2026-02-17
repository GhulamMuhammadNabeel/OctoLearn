"""
Plot Generator Module

Generates professional, classic-themed visualizations for the report.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shap
from typing import List, Optional
from ..utils.helpers import setup_logger
from sklearn.preprocessing import LabelEncoder

logger = setup_logger(__name__)

class PlotGenerator:
    """
    Generates high-quality plots for dataset distributions, correlations, and SHAP explanations.
    Uses a professional 'Whitegrid' theme with a classic blue/grey palette.
    """

    def __init__(self, df: pd.DataFrame, target: pd.Series, profile, model=None):
        self.df = df
        self.target = target
        self.profile = profile
        self.best_model = model  # <- store the trained model for SHAP
        self.output_dir = f"_octolearn_plots_{id(self)}"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # --- VISUAL THEME SETUP ---
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('ggplot')

        # Professional Palette
        self.palette = sns.color_palette("Blues_r")
        self.categorical_palette = sns.color_palette("viridis")
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white'
        })

    def generate_distributions(self, max_cols: int = 6) -> List[str]:
        paths = []
        
        numeric_cols = self.profile.numeric_columns[:max_cols]
        for col in numeric_cols:
            try:
                plt.figure(figsize=(8, 4))
                sns.histplot(self.df[col], kde=True, color='#2E86C1', edgecolor='white')
                plt.title(f'Distribution of {col}', fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Count')
                
                path = os.path.join(self.output_dir, f"{col}_dist.png")
                plt.tight_layout()
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to plot distribution for {col}: {e}")

        categorical_cols = self.profile.categorical_columns[:max_cols]
        for col in categorical_cols:
            try:
                plt.figure(figsize=(8, 4))
                top_n = self.df[col].value_counts().head(10)
                sns.barplot(x=top_n.index, y=top_n.values, palette="viridis")
                plt.title(f'Top Categories: {col}', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')
                
                path = os.path.join(self.output_dir, f"{col}_dist.png")
                plt.tight_layout()
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to plot category dist for {col}: {e}")
                
        return paths

    def generate_correlation_heatmap(self) -> Optional[str]:
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None
                
            plt.figure(figsize=(10, 8))
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(
                corr, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap="RdBu_r", 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5}
            )
            
            plt.title('Feature Correlation Matrix', fontweight='bold', pad=20)
            
            path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate heatmap: {e}")
            return None

    def generate_shap_plot(self) -> Optional[str]:
        """
        Generate SHAP summary plot using the trained model from AutoML.
        """
        if self.best_model is None:
            logger.warning("No trained model provided for SHAP. Skipping plot.")
            return None
        
        try:
            X_temp = self.df.select_dtypes(include=[np.number]).fillna(0)
            if X_temp.shape[1] == 0: return None
            
            y_temp = self.target
            if y_temp.dtype == 'object':
                y_temp = LabelEncoder().fit_transform(y_temp.astype(str))
            
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X_temp)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For classification: positive class
                
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_temp, show=False, plot_type="dot")
            plt.title("SHAP Feature Impact", fontweight='bold')
            
            path = os.path.join(self.output_dir, "shap_summary.png")
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {e}")
            return None
