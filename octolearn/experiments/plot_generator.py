"""
Plot Generator Module
=====================

Generates intelligent, beautiful visualization dashboards or simple plots.
Supports 'simple' vs 'dashboard' modes via user config.

For correlation analysis, automatically adapts to the number of features:
- ≤15 numeric features: full annotated heatmap
- >15 numeric features: ranked bar chart of top-N correlated pairs

This module is used internally by :class:`~octolearn.core.AutoML` and
:class:`~octolearn.experiments.report_generator.ReportGenerator`.

Examples
--------
>>> from octolearn.experiments.plot_generator import PlotGenerator
>>> plotter = PlotGenerator(X_clean, y, profile, mode='simple')
>>> paths = plotter.generate_smart_visuals(limit=5)
>>> heatmap_path, corr_summary = plotter.generate_correlation_heatmap(corr_top_n=15)
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class PlotGenerator:
    """
    Generates visualizations for the OctoLearn AutoML report.

    Supports two plot modes:

    - ``'simple'``: Single-panel plots, fast and lightweight.
    - ``'dashboard'``: Multi-panel dashboards with distribution, outlier,
      and target-relationship panels.

    For correlation analysis, the class automatically selects the best
    visualization strategy based on the number of numeric features:

    - **≤15 features**: Full annotated heatmap (readable, detailed).
    - **>15 features**: Ranked bar chart of the top-N most correlated pairs
      (avoids the unreadable "wall of numbers" problem).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (cleaned, post-preprocessing).
    y : pd.Series
        Target variable. Used for target-relationship plots.
    profile : DatasetProfile
        Dataset profile from :class:`~octolearn.profiling.DataProfiler`.
        Used to determine task type and column metadata.
    mode : str, optional
        Plot mode. One of ``'simple'`` or ``'dashboard'``. Default ``'simple'``.

    Examples
    --------
    >>> plotter = PlotGenerator(X_clean, y_train, clean_profile, mode='simple')
    >>> visual_paths = plotter.generate_smart_visuals(limit=8)
    >>> heatmap_path, corr_summary = plotter.generate_correlation_heatmap(corr_top_n=15)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, profile, mode: str = 'simple', theme: str = 'light'):
        """
        Initialize PlotGenerator.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned feature matrix.
        y : pd.Series
            Target variable.
        profile : DatasetProfile
            Dataset profile containing column type metadata.
        mode : str, optional
            ``'simple'`` for single-panel plots, ``'dashboard'`` for
            multi-panel dashboards. Default ``'simple'``.
        theme : str, optional
            ``'light'`` for professional white-background plots (best for PDFs),
            ``'dark'`` for dark-mode screens. Default ``'light'``.
        """
        self.X = X
        self.y = y
        self.profile = profile
        self.mode = mode
        self.theme = theme
        self.output_dir = ".octolearn/plots"
        os.makedirs(self.output_dir, exist_ok=True)

        # Set professional theme
        if self.theme == 'dark':
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
            self.colors = {
                'text': '#E0E0E0',
                'title': '#00F0FF',
                'bg': '#0D0D15',
                'grid': '#333333',
                'accent1': '#00F0FF',
                'accent2': '#FF0055',
                'accent3': '#00FFAA',
                'accent4': '#BD00FF',
            }
        else:
            # Professional Light Theme (Publication Quality)
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_theme(style="whitegrid", palette="deep", rc={
                "axes.facecolor": "#F8F9FA",
                "figure.facecolor": "#FFFFFF",
                "grid.color": "#E9ECEF",
                "text.color": "#212529",
                "axes.labelcolor": "#495057",
                "axes.titlecolor": "#1A3A5C",  # Navy Blue
                "font.family": "sans-serif",
            })
            self.colors = {
                'text': '#212529',
                'title': '#1A3A5C',
                'bg': '#FFFFFF',
                'grid': '#E9ECEF',
                'accent1': '#1A3A5C',  # Navy
                'accent2': '#E74C3C',  # Red
                'accent3': '#27AE60',  # Green
                'accent4': '#8E44AD',  # Purple
            }

    def generate_smart_visuals(self, limit: int = 10) -> List[str]:
        """
        Generate professional visualizations for the top-ranked features.

        Features are ranked by their absolute correlation with the target
        (if available) or by variance. The top ``limit`` features are plotted.

        Parameters
        ----------
        limit : int, optional
            Maximum number of feature plots to generate. Default 10.

        Returns
        -------
        list of str
            Absolute file paths to the generated PNG images.
            Empty list if no plots could be generated.

        Examples
        --------
        >>> paths = plotter.generate_smart_visuals(limit=5)
        >>> print(paths[0])
        '.octolearn/plots/viz_dash_num_age.png'
        """
        if self.X is None or self.X.empty:
            logger.warning("PlotGenerator: X is empty, skipping smart visuals.")
            return []

        paths = []

        # Identify feature types
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
                logger.warning(f"Failed to plot '{col}': {e}")

        return paths

    def _rank_features(self, cols: List[str]) -> List[str]:
        """
        Rank columns by predictive relevance.

        Uses absolute correlation with the target variable when available.
        Falls back to variance ranking for numeric columns, or returns
        the original order for categorical columns.

        Parameters
        ----------
        cols : list of str
            Column names to rank.

        Returns
        -------
        list of str
            Column names sorted from most to least relevant.
        """
        if not cols:
            return []

        if self.y is not None:
            temp_df = self.X[cols].copy()
            y_enc = self.y

            # Encode target if categorical
            if not pd.api.types.is_numeric_dtype(self.y):
                if self.profile and self.profile.task_type == 'classification':
                    y_enc = pd.factorize(self.y)[0]

            # Factorize categorical features for correlation
            for c in temp_df.columns:
                if not pd.api.types.is_numeric_dtype(temp_df[c]):
                    temp_df[c] = pd.factorize(temp_df[c])[0]

            temp_df['__target__'] = y_enc

            try:
                corrs = temp_df.corr()['__target__'].drop('__target__').abs()
                return corrs.sort_values(ascending=False).index.tolist()
            except Exception as e:
                logger.warning(f"Feature ranking by correlation failed: {e}")

        # Fallback: original order
        return cols

    def _plot_numeric_dashboard(self, col: str) -> Optional[str]:
        """
        Create a multi-panel dashboard for a single numeric feature.

        Generates three panels:

        1. **Distribution** — histogram with KDE, mean and median lines.
        2. **Outliers & Range** — box plot highlighting outlier points.
        3. **Target Relationship** — violin plot (classification) or
           scatter/regression plot (regression).

        Parameters
        ----------
        col : str
            Name of the numeric column to visualize.

        Returns
        -------
        str or None
            Absolute path to the saved PNG, or ``None`` on failure.
        """
        if col not in self.X.columns:
            return None

        fig = plt.figure(figsize=(10, 3.5))
        gs = fig.add_gridspec(1, 3)
        fig.suptitle(f"Feature Analysis: {col}", fontsize=12, fontweight='bold', color=self.colors['title'])

        # Panel 1: Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            sns.histplot(data=self.X, x=col, kde=True, ax=ax1, color=self.colors['accent1'], fill=True, alpha=0.3)
            mean_val = self.X[col].mean()
            median_val = self.X[col].median()
            ax1.axvline(mean_val, color=self.colors['accent2'], linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color=self.colors['accent3'], linestyle=':', label=f'Median: {median_val:.2f}')
            ax1.legend(facecolor=self.colors['bg'], edgecolor=self.colors['grid'], fontsize=7, labelcolor=self.colors['text'])
        except Exception:
            ax1.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
        ax1.set_title("Distribution", color=self.colors['text'])
        ax1.set_xlabel(col, color=self.colors['text'])

        # Panel 2: Boxplot
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            sns.boxplot(data=self.X, x=col, ax=ax2, color=self.colors['accent4'],
                        flierprops={'markerfacecolor': self.colors['text']})
        except Exception:
            ax2.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
        ax2.set_title("Outliers & Range", color=self.colors['text'])
        ax2.set_xlabel(col, color=self.colors['text'])

        # Panel 3: Target relationship
        ax3 = fig.add_subplot(gs[0, 2])
        try:
            if self.y is not None:
                if self.profile and self.profile.task_type == 'classification':
                    top_classes = self.y.value_counts().head(5).index
                    mask = self.y.isin(top_classes)
                    sns.violinplot(x=self.y[mask], y=self.X.loc[mask, col],
                                   ax=ax3, palette="viridis")
                    ax3.set_title("Distribution by Target Class", color='#E0E0E0')
                else:
                    sns.regplot(x=self.X[col], y=self.y, ax=ax3,
                                scatter_kws={'alpha': 0.5, 'color': self.colors['accent1']},
                                line_kws={'color': self.colors['accent2']})
                    ax3.set_title("Relationship vs Target", color=self.colors['text'])
            else:
                ax3.text(0.5, 0.5, "No Target Available", ha='center', va='center', color='gray')
                ax3.axis('off')
        except Exception as e:
            logger.warning(f"Target panel failed for '{col}': {e}")
            ax3.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
            ax3.axis('off')

        plt.tight_layout()
        filename = f"viz_dash_num_{col.replace(' ', '_')}.png"
        path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(path, bbox_inches='tight', dpi=120, facecolor=self.colors['bg'])
        except Exception as e:
            logger.warning(f"Could not save plot for '{col}': {e}")
            path = None
        plt.close()
        return path

    def _plot_categorical_dashboard(self, col: str) -> Optional[str]:
        """
        Create a multi-panel dashboard for a single categorical feature.

        Skips columns with more than 20 unique values (likely IDs).
        Generates two panels:

        1. **Count Plot** — bar chart of top-10 category frequencies.
        2. **Target Impact** — stacked bar (classification) or box plot
           (regression) showing target distribution per category.

        Parameters
        ----------
        col : str
            Name of the categorical column to visualize.

        Returns
        -------
        str or None
            Absolute path to the saved PNG, or ``None`` if skipped/failed.
        """
        if col not in self.X.columns:
            return None

        # Skip high-cardinality columns (likely IDs)
        if self.X[col].nunique() > 20:
            return None

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 2)
        fig.suptitle(f"Feature Analysis: {col}", fontsize=16, fontweight='bold', color=self.colors['title'])

        # Panel 1: Count plot
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            counts = self.X[col].value_counts().head(10)
            sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette="viridis")
            ax1.set_title("Top Categories (Max 10)", color='#E0E0E0')
            ax1.tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.warning(f"Count panel failed for '{col}': {e}")
            ax1.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')

        # Panel 2: Target impact
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            if self.y is not None and self.profile and self.profile.task_type == 'classification':
                ct = pd.crosstab(self.X[col], self.y, normalize='index')
                if len(ct) > 10:
                    ct = ct.loc[counts.index]
                ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
                ax2.set_title("Target Distribution by Category", color='#E0E0E0')
                ax2.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.tick_params(axis='x', rotation=45)
            elif self.y is not None:
                top_cats = self.X[col].value_counts().head(10).index
                mask = self.X[col].isin(top_cats)
                sns.boxplot(x=self.X.loc[mask, col], y=self.y.loc[mask],
                            ax=ax2, palette="viridis")
                ax2.set_title("Target Distribution by Category", color='#E0E0E0')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, "No Target Available", ha='center', va='center', color='gray')
                ax2.axis('off')
        except Exception as e:
            logger.warning(f"Target panel failed for '{col}': {e}")
            ax2.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
            ax2.axis('off')

        plt.tight_layout()
        filename = f"viz_dash_cat_{col.replace(' ', '_')}.png"
        path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(path, bbox_inches='tight', dpi=120, facecolor=self.colors['bg'])
        except Exception as e:
            logger.warning(f"Could not save plot for '{col}': {e}")
            path = None
        plt.close()
        return path

    def generate_correlation_heatmap(
        self,
        corr_top_n: int = 15,
    ) -> Tuple[Optional[str], Dict]:
        """
        Generate a correlation visualization that adapts to the number of features.

        **Strategy:**

        - **≤15 numeric features**: Full annotated heatmap — every cell is
          labeled, easy to read.
        - **>15 numeric features**: Ranked horizontal bar chart of the
          ``corr_top_n`` most correlated feature pairs. This avoids the
          unreadable "wall of numbers" problem with wide datasets.

        In both cases, a ``corr_summary`` dict is returned containing the
        top and bottom correlated pairs for use in narrative paragraphs.

        Parameters
        ----------
        corr_top_n : int, optional
            Number of top correlated pairs to show in the bar chart (used
            only when features > 15). Default 15.

        Returns
        -------
        path : str or None
            Absolute path to the saved PNG, or ``None`` on failure.
        corr_summary : dict
            Dictionary with keys:

            - ``'top_pairs'``: list of (feat_a, feat_b, corr) tuples,
              highest absolute correlations first.
            - ``'bottom_pairs'``: list of (feat_a, feat_b, corr) tuples,
              lowest absolute correlations first.
            - ``'n_features'``: total number of numeric features.
            - ``'strategy'``: ``'heatmap'`` or ``'bar_chart'``.

        Examples
        --------
        >>> path, summary = plotter.generate_correlation_heatmap(corr_top_n=10)
        >>> print(summary['top_pairs'][:3])
        [('age', 'fare', 0.87), ('pclass', 'fare', -0.73), ...]
        """
        corr_summary = {
            'top_pairs': [],
            'bottom_pairs': [],
            'n_features': 0,
            'strategy': 'none',
        }

        try:
            numeric_df = self.X.select_dtypes(include=[np.number])
            n_features = numeric_df.shape[1]
            corr_summary['n_features'] = n_features

            if n_features < 2:
                logger.info("Fewer than 2 numeric features — skipping correlation plot.")
                return None, corr_summary

            corr_matrix = numeric_df.corr()

            # Build a flat list of unique pairs (upper triangle, excluding diagonal)
            pairs = []
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

            # Sort by absolute value descending
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
            corr_summary['top_pairs'] = pairs_sorted[:5]
            corr_summary['bottom_pairs'] = pairs_sorted[-5:]

            if n_features <= 15:
                # ── Full annotated heatmap ────────────────────────────────
                corr_summary['strategy'] = 'heatmap'
                fig, ax = plt.subplots(figsize=(max(8, n_features * 0.7),
                                                max(6, n_features * 0.6)))
                sns.heatmap(
                    corr_matrix, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'},
                    linewidths=0.5, linecolor=self.colors['grid'],
                )
                ax.set_title(
                    f'Feature Correlation Matrix ({n_features} features)',
                    fontweight='bold', color=self.colors['title'], fontsize=14,
                )
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', rotation=0, labelsize=9)

            else:
                # ── Ranked bar chart of top-N pairs ───────────────────────
                corr_summary['strategy'] = 'bar_chart'
                top_pairs = pairs_sorted[:corr_top_n]

                labels = [f"{a} ↔ {b}" for a, b, _ in top_pairs]
                values = [c for _, _, c in top_pairs]
                colors_bar = [self.colors['accent2'] if v < 0 else self.colors['accent1'] for v in values]

                fig_height = max(5, len(top_pairs) * 0.45)
                fig, ax = plt.subplots(figsize=(10, fig_height))

                bars = ax.barh(range(len(values)), values, color=colors_bar, alpha=0.85)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.invert_yaxis()  # Highest at top
                ax.axvline(0, color=self.colors['grid'], linewidth=0.8, linestyle='--')
                ax.set_xlabel('Pearson Correlation Coefficient', color=self.colors['text'])
                ax.set_title(
                    f'Top {len(top_pairs)} Correlated Feature Pairs\n'
                    f'({n_features} total numeric features — full heatmap would be unreadable)',
                    fontweight='bold', color=self.colors['title'], fontsize=12,
                )

                # Legend
                pos_patch = mpatches.Patch(color=self.colors['accent1'], label='Positive correlation')
                neg_patch = mpatches.Patch(color=self.colors['accent2'], label='Negative correlation')
                ax.legend(handles=[pos_patch, neg_patch], loc='lower right',
                          facecolor=self.colors['bg'], edgecolor=self.colors['grid'], fontsize=8, labelcolor=self.colors['text'])

                # Value labels on bars
                for bar, val in zip(bars, values):
                    x_pos = val + 0.01 if val >= 0 else val - 0.01
                    ha = 'left' if val >= 0 else 'right'
                    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                            f'{val:.2f}', va='center', ha=ha,
                            color=self.colors['text'], fontsize=8)

            plt.tight_layout()
            filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close()
            return filename, corr_summary

        except Exception as e:
            logger.warning(f"Failed to generate correlation visualization: {e}")
            plt.close('all')
            return None, corr_summary

    def generate_shap_plot(self, model=None) -> Optional[str]:
        """
        Generate a SHAP summary plot for model explainability.

        Uses :class:`shap.TreeExplainer` for tree-based models and
        :class:`shap.LinearExplainer` for linear models. The plot shows
        feature impact on model output, sorted by mean absolute SHAP value.

        Parameters
        ----------
        model : sklearn estimator, optional
            Trained model to explain. If ``None``, returns ``None``.

        Returns
        -------
        str or None
            Absolute path to the saved SHAP summary PNG, or ``None`` on failure.

        Notes
        -----
        SHAP computation can be slow for large datasets (>10,000 rows).
        Consider passing a sample of ``X`` to the plotter for faster results.

        Examples
        --------
        >>> shap_path = plotter.generate_shap_plot(model=automl.best_model_)
        """
        if model is None or self.X is None:
            return None
        try:
            import shap

            model_type = type(model).__name__.lower()
            is_linear = (
                'linear' in model_type
                or ('regression' in model_type
                    and 'tree' not in model_type
                    and 'forest' not in model_type
                    and 'boost' not in model_type)
            )

            if is_linear:
                explainer = shap.LinearExplainer(model, self.X)
                shap_values = explainer.shap_values(self.X)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 6))
            if self.theme == 'dark':
                plt.style.use('dark_background')
            else:
                plt.style.use('seaborn-v0_8-whitegrid')
            
            shap.summary_plot(shap_values, self.X, show=False, color_bar=True, cmap='cool')

            filename = os.path.join(self.output_dir, 'shap_summary.png')
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close()

            # Reset style
            plt.style.use('default')
            sns.set_theme(style="whitegrid", palette="muted")

            return filename
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {e}")
            plt.close('all')
            return None
