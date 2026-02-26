"""
OctoLearn Intelligence Report Generator
========================================

Generates professional, narrative-driven PDF reports that tell the story
of your data — from raw input to trained models and actionable insights.

Features:
- Professional light theme with readable dark text
- Auto-discovered OctoLearn logo on cover + page headers
- Storytelling narrative flow across 8+ sections
- Color-coded Data Health Dashboard with risk gauge
- Before/After data transformation comparison
- Model Arena with populated benchmark tables
- Priority-grouped recommendations (Critical/High/Medium/Low)
- Page headers and footers with branding

Author: OctoLearn Development Team
Version: 0.10.0 (Redesigned)
License: MIT
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import warnings

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics import renderPDF

# Safe import with fallback
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    from ..utils.helpers import setup_logger
except ImportError:
    import logging
    def setup_logger(name):
        return logging.getLogger(name)

logger = setup_logger(__name__)


# ============================================================================
# COLOR PALETTE — Professional Light Theme
# ============================================================================

class ReportColors:
    """Premium light-theme color palette for modern PDF output."""

    # === Primary brand colors ===
    PRIMARY = colors.HexColor('#0F172A')        # Slate 900 (Deep modern navy)
    PRIMARY_LIGHT = colors.HexColor('#334155')   # Slate 700
    ACCENT = colors.HexColor('#3B82F6')          # Blue 500 (Vibrant accent for insights)
    ACCENT_MUTED = colors.HexColor('#DBEAFE')    # Blue 100 

    # === Backgrounds ===
    WHITE = colors.HexColor('#FFFFFF')
    PAGE_BG = colors.HexColor('#F8FAFC')         # Slate 50 (Very soft grey-blue)
    CARD_BG = colors.HexColor('#FFFFFF')         # White cards
    HEADER_BG = colors.HexColor('#0F172A')       # Slate 900 headers
    ROW_ALT = colors.HexColor('#F1F5F9')         # Slate 100 for alternating rows

    # === Text colors ===
    TEXT_PRIMARY = colors.HexColor('#1E293B')    # Slate 800 for high-contrast body
    TEXT_SECONDARY = colors.HexColor('#475569')  # Slate 600 for secondary text
    TEXT_ON_DARK = colors.HexColor('#F8FAFC')     # Slate 50 on dark headers
    TEXT_CAPTION = colors.HexColor('#64748B')    # Slate 500 for captions

    # === Status/Semantic colors ===
    SUCCESS = colors.HexColor('#10B981')         # Emerald 500
    WARNING = colors.HexColor('#F59E0B')         # Amber 500
    DANGER = colors.HexColor('#EF4444')          # Red 500
    INFO = colors.HexColor('#0EA5E9')            # Sky 500

    # === Risk gauge ===
    RISK_LOW = colors.HexColor('#10B981')
    RISK_MODERATE = colors.HexColor('#F59E0B')
    RISK_HIGH = colors.HexColor('#EF4444')

    # === Decorative ===
    DIVIDER = colors.HexColor('#E2E8F0')         # Slate 200 light divider
    BORDER = colors.HexColor('#CBD5E1')          # Slate 300 card borders

    @staticmethod
    def get_risk_color(score: int) -> colors.Color:
        if score <= 30:
            return ReportColors.RISK_LOW
        elif score <= 60:
            return ReportColors.RISK_MODERATE
        else:
            return ReportColors.RISK_HIGH

    @staticmethod
    def get_risk_label(score: int) -> str:
        if score <= 30:
            return "Healthy"
        elif score <= 60:
            return "Needs Attention"
        else:
            return "Critical"


# ============================================================================
# LOGO DISCOVERY
# ============================================================================

def _find_logo() -> Optional[str]:
    """Auto-discover the OctoLearn logo from common locations."""
    candidates = [
        Path(__file__).parent.parent / 'images' / 'logo.png',
        Path(__file__).parent.parent / 'images' / 'logo.jpg',
        Path('octolearn') / 'images' / 'logo.png',
        Path.cwd() / 'octolearn' / 'images' / 'logo.png',
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


# ============================================================================
# CUSTOM PAGE TEMPLATE — Headers & Footers
# ============================================================================




# ============================================================================
# REPORT GENERATOR — MAIN CLASS
# ============================================================================

class ReportGenerator:
    """
    Professional PDF Report Generator with intelligent storytelling.

    Produces narrative-driven reports that guide users through:
    1. Cover Page — branding and dataset overview
    2. Data Story — narrative introduction
    3. Data Health Dashboard — risk score and quality metrics
    4. Before & After — data transformation journey
    5. Feature Intelligence — importance rankings
    6. Model Arena — benchmark comparison
    7. Recommendations — priority-grouped actions
    8. Visual Insights — charts and correlation maps
    """

    def __init__(
        self,
        raw_profile: Any,
        clean_profile: Any,
        mode: str = 'detailed',
        dist_plots: Optional[List[str]] = None,
        heatmap_plot: Optional[str] = None,
        recommendations: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        risk_score: Optional[int] = None,
        risk_category: Optional[str] = None,
        risk_factors: Optional[Dict[str, Any]] = None,
        preprocessing_suggestions: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        shap_path: Optional[str] = None,
        model_benchmarks: Optional[List[Dict[str, Any]]] = None,
        best_model_name: Optional[str] = None,
        logo_path: Optional[str] = None,
        cleaning_log: Optional[Dict] = None,
        outlier_results: Optional[Dict] = None,
        interaction_results: Optional[Dict] = None,
        title: str = "OctoLearn Intelligence Report",
        author: str = "OctoLearn AutoML",
        company: str = "Data Science Team",
        # Raw and cleaned DataFrames for before/after distribution plots
        raw_X: Optional[Any] = None,
        clean_X: Optional[Any] = None,
        corr_summary: Optional[Dict[str, Any]] = None,
        # Performance Visuals
        confusion_matrix_plot: Optional[str] = None,
        roc_curve_plot: Optional[str] = None,
        residual_plot: Optional[str] = None,
        feature_importance_plot: Optional[str] = None,
    ):
        if mode not in ['brief', 'detailed']:
            raise ValueError(f"Mode must be 'brief' or 'detailed', got '{mode}'")

        self.raw_profile = raw_profile
        self.clean_profile = clean_profile
        self.mode = mode
        self.dist_plots = dist_plots or []
        self.heatmap_plot = heatmap_plot
        self.corr_summary = corr_summary or {}  # {top_pairs, bottom_pairs, n_features, strategy}
        self.recommendations = recommendations or []
        self.risk_score = risk_score if risk_score is not None else 50
        self.risk_category = risk_category or "Moderate"
        self.risk_factors = risk_factors or {}
        self.preprocessing_suggestions = preprocessing_suggestions or {}
        self.feature_importance = feature_importance or {}
        self.shap_path = shap_path
        self.model_benchmarks = model_benchmarks or []
        self.best_model_name = best_model_name or "N/A"
        self.logo_path = logo_path or _find_logo()
        self.cleaning_log = cleaning_log or {}
        self.outlier_results = outlier_results or {}
        self.interaction_results = interaction_results or {}
        self.title = title
        self.author = author
        self.company = company
        self.raw_X = raw_X
        self.clean_X = clean_X
        # Visuals
        self.confusion_matrix_plot = confusion_matrix_plot
        self.roc_curve_plot = roc_curve_plot
        self.residual_plot = residual_plot
        self.feature_importance_plot = feature_importance_plot
        
        # Temp files created during report generation (cleaned up after)
        self._temp_files: List[str] = []

        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._create_styles()

        logger.info(f"Report initialized in {mode.upper()} mode")

    # ------------------------------------------------------------------
    # Font Registration
    # ------------------------------------------------------------------
    def _register_fonts(self):
        """Load ShantellSans fonts with fallback to Helvetica."""
        self.font_regular = 'Helvetica'
        self.font_bold = 'Helvetica-Bold'
        self.font_title = 'Helvetica-Bold'
        self.font_italic = 'Helvetica-Oblique'

        possible_dirs = [
            Path(__file__).parent.parent / 'fonts',
            Path('octolearn') / 'fonts',
            Path('fonts'),
            Path.cwd() / 'octolearn' / 'fonts',
        ]

        font_files = {
            'ShantellSans-Regular': 'ShantellSans-Regular.ttf',
            'ShantellSans-Bold': 'ShantellSans-Bold.ttf',
            'ShantellSans-ExtraBold': 'ShantellSans-ExtraBold.ttf',
            'ShantellSans-Italic': 'ShantellSans-Italic.ttf',
        }

        for font_name, filename in font_files.items():
            for font_dir in possible_dirs:
                font_path = font_dir / filename
                if font_path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                        if font_name == 'ShantellSans-Regular':
                            self.font_regular = font_name
                        elif font_name == 'ShantellSans-ExtraBold':
                            self.font_title = font_name
                        elif font_name == 'ShantellSans-Bold':
                            self.font_bold = font_name
                        elif font_name == 'ShantellSans-Italic':
                            self.font_italic = font_name
                        break
                    except Exception:
                        pass

        if self.font_regular == 'ShantellSans-Regular':
            logger.info("Successfully registered ShantellSans fonts.")
        else:
            logger.warning("Could not register ShantellSans fonts. Using Helvetica.")

    # ------------------------------------------------------------------
    # Style Definitions
    # ------------------------------------------------------------------
    def _create_styles(self):
        """Create professional readable paragraph styles."""

        self.styles.add(ParagraphStyle(
            name='CoverTitle',
            fontName=self.font_title,
            fontSize=30,
            leading=36,
            textColor=ReportColors.PRIMARY,
            spaceAfter=10,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='CoverSubtitle',
            fontName=self.font_regular,
            fontSize=13,
            leading=18,
            textColor=ReportColors.TEXT_SECONDARY,
            spaceAfter=6,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            fontName=self.font_bold,
            fontSize=20,
            leading=24,
            textColor=ReportColors.PRIMARY,
            spaceBefore=24,
            spaceAfter=4,
            allowWidows=0,
            allowOrphans=0,
        ))




        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            fontName=self.font_bold,
            fontSize=13,
            leading=17,
            textColor=ReportColors.PRIMARY_LIGHT,
            spaceBefore=12,
            spaceAfter=6,
        ))

        self.styles.add(ParagraphStyle(
            name='ReportBody',
            fontName=self.font_regular,
            fontSize=10.5,
            leading=15,
            textColor=ReportColors.TEXT_PRIMARY,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
        ))

        self.styles.add(ParagraphStyle(
            name='ReportBold',
            fontName=self.font_bold,
            fontSize=10.5,
            leading=15,
            textColor=ReportColors.TEXT_PRIMARY,
            spaceAfter=8,
        ))

        self.styles.add(ParagraphStyle(
            name='Narrative',
            fontName=self.font_regular,
            fontSize=11,
            leading=17,
            textColor=ReportColors.TEXT_PRIMARY,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
        ))

        self.styles.add(ParagraphStyle(
            name='TableCell',
            fontName=self.font_regular,
            fontSize=9.5,
            leading=13,
            textColor=ReportColors.TEXT_PRIMARY,
        ))

        self.styles.add(ParagraphStyle(
            name='TableHeader',
            fontName=self.font_bold,
            fontSize=10,
            leading=13,
            textColor=ReportColors.TEXT_ON_DARK,
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            fontName=self.font_bold,
            fontSize=22,
            leading=26,
            textColor=ReportColors.PRIMARY,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            fontName=self.font_regular,
            fontSize=9,
            leading=12,
            textColor=ReportColors.TEXT_SECONDARY,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='Caption',
            fontName=self.font_regular,
            fontSize=9,
            leading=12,
            textColor=ReportColors.TEXT_CAPTION,
            spaceAfter=6,
            alignment=TA_CENTER,
        ))

        self.styles.add(ParagraphStyle(
            name='Highlight',
            fontName=self.font_bold,
            fontSize=11,
            leading=15,
            textColor=ReportColors.ACCENT,
            spaceAfter=8,
        ))

        # Alert Box Styles (Like GitHub Alerts)
        self.styles.add(ParagraphStyle(
            name='AlertBox_Warning',
            fontName=self.font_regular,
            fontSize=10.5,
            leading=15,
            textColor=ReportColors.DANGER,
            backColor=colors.HexColor('#FEF2F2'),
            borderPadding=(10, 10, 10, 15),
            borderColor=ReportColors.DANGER,
            borderWidth=1,
            spaceBefore=12,
            spaceAfter=12,
            leftIndent=5,
            rightIndent=5,
        ))

        self.styles.add(ParagraphStyle(
            name='AlertBox_Insight',
            fontName=self.font_regular,
            fontSize=10.5,
            leading=15,
            textColor=ReportColors.SUCCESS,
            backColor=colors.HexColor('#ECFDF5'),
            borderPadding=(10, 10, 10, 15),
            borderColor=ReportColors.SUCCESS,
            borderWidth=1,
            spaceBefore=12,
            spaceAfter=12,
            leftIndent=5,
            rightIndent=5,
        ))

        self.styles.add(ParagraphStyle(
            name='RecCritical',
            fontName=self.font_bold,
            fontSize=10,
            leading=14,
            textColor=ReportColors.DANGER,
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=12,
        ))

        self.styles.add(ParagraphStyle(
            name='RecHigh',
            fontName=self.font_bold,
            fontSize=10,
            leading=14,
            textColor=ReportColors.WARNING,
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=12,
        ))

        self.styles.add(ParagraphStyle(
            name='RecNormal',
            fontName=self.font_regular,
            fontSize=10,
            leading=14,
            textColor=ReportColors.TEXT_PRIMARY,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=12,
        ))

    # ------------------------------------------------------------------
    # Helper: divider line
    # ------------------------------------------------------------------
    def _divider(self):
        return HRFlowable(
            width="100%", thickness=1,
            color=ReportColors.DIVIDER,
            spaceBefore=8, spaceAfter=8
        )

    # ------------------------------------------------------------------
    # Helper: styled table
    # ------------------------------------------------------------------
    def _styled_table(self, data, col_widths, highlight_best_row=None):
        """Create a professionally styled table with header and alternating rows."""
        t = Table(data, colWidths=col_widths)
        style_cmds = [
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), ReportColors.HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), ReportColors.TEXT_ON_DARK),
            ('FONTNAME', (0, 0), (-1, 0), self.font_bold),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            # Body
            ('FONTNAME', (0, 1), (-1, -1), self.font_regular),
            ('FONTSIZE', (0, 1), (-1, -1), 9.5),
            ('TEXTCOLOR', (0, 1), (-1, -1), ReportColors.TEXT_PRIMARY),
            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            # Grid
            ('GRID', (0, 0), (-1, 0), 0, ReportColors.HEADER_BG),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, ReportColors.PRIMARY),
            ('LINEBELOW', (0, 1), (-1, -2), 0.5, ReportColors.BORDER),
            ('LINEBELOW', (0, -1), (-1, -1), 1, ReportColors.DIVIDER),
            # Alternating rows
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [ReportColors.WHITE, ReportColors.ROW_ALT]),
        ]
        # Highlight a specific row (e.g., best model)
        if highlight_best_row is not None and highlight_best_row >= 1:
            style_cmds.append(('BACKGROUND', (0, highlight_best_row), (-1, highlight_best_row), colors.HexColor('#EBF5FB')))
            style_cmds.append(('TEXTCOLOR', (0, highlight_best_row), (-1, highlight_best_row), ReportColors.PRIMARY))
            style_cmds.append(('FONTNAME', (0, highlight_best_row), (-1, highlight_best_row), self.font_bold))

        t.setStyle(TableStyle(style_cmds))
        return t

    def _add_section_header(self, story, title):
        """Add a professional flush-left magazine-style section header."""
        header_style = ParagraphStyle(
            'SectionHeaderBar',
            parent=self.styles['SectionHeading'],
            alignment=TA_LEFT
        )
        story.append(Paragraph(title, header_style))
        
        # Add a sleek accent line underneath the header
        story.append(HRFlowable(
            width="100%", thickness=2,
            color=ReportColors.ACCENT,
            spaceBefore=0, spaceAfter=14
        ))

    def _create_side_by_side(self, left_content, right_content, widths=[3.7*inch, 3.7*inch]):
        """Create a 2-column layout using a Table."""
        data = [[left_content, right_content]]
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        return t


    # ==================================================================
    # SECTION 1: COVER PAGE
    # ==================================================================
    def _add_cover_page(self, story):
        story.append(Spacer(1, 0.8 * inch))

        # Logo
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                im = Image(self.logo_path, width=1.8 * inch, height=1.8 * inch)
                im.hAlign = 'CENTER'
                story.append(im)
            except Exception:
                pass

        story.append(Spacer(1, 0.4 * inch))

        # Title
        story.append(Paragraph(self.title, self.styles['CoverTitle']))
        story.append(Spacer(1, 0.15 * inch))

        # Subtitle line
        story.append(self._divider())
        story.append(Spacer(1, 0.1 * inch))

        # Metadata cards
        task_type = getattr(self.raw_profile, 'task_type', 'N/A') if self.raw_profile else 'N/A'
        n_rows = getattr(self.raw_profile, 'n_rows', '?') if self.raw_profile else '?'
        n_cols = getattr(self.raw_profile, 'n_columns', '?') if self.raw_profile else '?'

        meta_data = [
            [
                Paragraph("<b>Generated By</b>", self.styles['TableCell']),
                Paragraph(self.author, self.styles['TableCell']),
                Paragraph("<b>Date</b>", self.styles['TableCell']),
                Paragraph(datetime.now().strftime("%B %d, %Y at %H:%M"), self.styles['TableCell']),
            ],
            [
                Paragraph("<b>Task Type</b>", self.styles['TableCell']),
                Paragraph(str(task_type).title(), self.styles['TableCell']),
                Paragraph("<b>Dataset Size</b>", self.styles['TableCell']),
                Paragraph(f"{n_rows} rows x {n_cols} features", self.styles['TableCell']),
            ],
            [
                Paragraph("<b>Risk Score</b>", self.styles['TableCell']),
                Paragraph(f"{self.risk_score}/100 ({self.risk_category})", self.styles['TableCell']),
                Paragraph("<b>Best Model</b>", self.styles['TableCell']),
                Paragraph(str(self.best_model_name).replace('_', ' ').title(), self.styles['TableCell']),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.0 * inch, 1.3 * inch, 2.0 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('TEXTCOLOR', (0, 0), (-1, -1), ReportColors.TEXT_PRIMARY),
            ('FONTNAME', (0, 0), (-1, -1), self.font_regular),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, ReportColors.BORDER),
            ('BOX', (0, 0), (-1, -1), 1, ReportColors.DIVIDER),
        ]))
        story.append(meta_table)

        story.append(Spacer(1, 0.5 * inch))

        # Tagline
        story.append(Paragraph(
            "This report was auto-generated by OctoLearn's intelligent pipeline. "
            "It narrates the complete journey of your data -- from raw input through cleaning, "
            "profiling, model training, and actionable recommendations.",
            self.styles['Caption']
        ))

        story.append(PageBreak())

    # ==================================================================
    # SECTION 1.5: EXECUTIVE SUMMARY
    # ==================================================================
    def _add_executive_summary(self, story):
        self._add_section_header(story, "Executive Summary")
        
        task = getattr(self.raw_profile, 'task_type', 'unknown') if self.raw_profile else 'unknown'
        n_rows = getattr(self.raw_profile, 'n_rows', 0) if self.raw_profile else 0
        n_cols = getattr(self.raw_profile, 'n_columns', 0) if self.raw_profile else 0
        
        # Calculate data loss
        clean_rows = getattr(self.clean_profile, 'n_rows', n_rows) if self.clean_profile else n_rows
        dropped_rows = n_rows - clean_rows
        drop_pct = (dropped_rows / n_rows * 100) if n_rows > 0 else 0

        # Model Performance
        best_name = str(self.best_model_name).replace('_', ' ').title()
        best_score = self.model_benchmarks[0].get('score', 0) if self.model_benchmarks else 0
        metric_name = "Accuracy/F1" if task == 'classification' else "RMSE/R²"

        # Top Feature
        top_feature = "N/A"
        top_feature_impact = "N/A"
        if self.feature_importance:
            top_3 = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:1]
            if top_3:
                top_feature = top_3[0][0]
                total_importance = sum(self.feature_importance.values())
                impact_pct = (top_3[0][1] / total_importance * 100) if total_importance > 0 else 0
                top_feature_impact = f"{impact_pct:.1f}%"

        # Executive Bullet Points
        story.append(Paragraph(
            "This report details the automated transformation and modeling of your dataset. Below are the core high-level takeaways:",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.1 * inch))

        bullets = [
            f"<b>Objective Achieved</b>: Successfully trained a robust pipeline for <b>{task.title()}</b> on <b>{n_rows:,}</b> records across <b>{n_cols}</b> features.",
            f"<b>Model Performance</b>: The optimal architecture is <b>{best_name}</b>, achieving a primary validation score of <b>{best_score:.4f}</b>.",
        ]
        
        if top_feature != "N/A":
            bullets.append(f"<b>Key Business Lever</b>: The feature <b>'{top_feature}'</b> drives <b>{top_feature_impact}</b> of the prediction, marking it as the primary operational target.")
            
        if drop_pct > 0:
            bullets.append(f"<b>Data Integrity</b>: Purged <b>{dropped_rows}</b> row(s) (<b>{drop_pct:.1f}%</b> of total) to prevent hallucinations and establish algorithmic trust.")
        else:
            bullets.append(f"<b>Data Integrity</b>: The dataset passed all structural integrity checks with <b>0%</b> data loss, retaining full analytical volume.")

        for bullet in bullets:
            story.append(Paragraph(f"<font color='#94A3B8' name='Helvetica-Bold'>&gt;&gt;</font>  {bullet}", self.styles['ReportBody']))
            
        story.append(Spacer(1, 0.3 * inch))
        
        # Add Smart Alert if Data Loss is high
        if drop_pct > 20:
            story.append(Paragraph(
                f"<b>WARNING: High Data Loss Detected ({drop_pct:.1f}%)</b><br/>"
                "A significant portion of your dataset was discarded due to critical missingness or duplication. "
                "Ensure upstream data collection processes are reviewed to recover statistical power.",
                self.styles['AlertBox_Warning']
            ))
            
        # Add Insight Callout for best model
        if best_score > 0.85:
            story.append(Paragraph(
                f"<b>INSIGHT: Production Readiness</b><br/>"
                f"The champion model ({best_name}) achieved an exceptionally strong score. "
                "This indicates unambiguous dataset signals mapping variables to the target outcome. "
                "The pipeline is highly viable for production deployment.",
                self.styles['AlertBox_Insight']
            ))
            
        story.append(PageBreak())

    # ==================================================================
    # SECTION 2: DATA STORY (Narrative)
    # ==================================================================
    def _add_data_story(self, story):
        self._add_section_header(story, "The Data Story")

        n_rows = getattr(self.raw_profile, 'n_rows', 0) if self.raw_profile else 0
        n_cols = getattr(self.raw_profile, 'n_columns', 0) if self.raw_profile else 0
        task = getattr(self.raw_profile, 'task_type', 'unknown') if self.raw_profile else 'unknown'
        n_numeric = len(getattr(self.raw_profile, 'numeric_columns', [])) if self.raw_profile else 0
        n_cat = len(getattr(self.raw_profile, 'categorical_columns', [])) if self.raw_profile else 0
        n_missing_cols = sum(1 for v in getattr(self.raw_profile, 'missing_ratio', {}).values() if v > 0) if self.raw_profile else 0
        dupes = getattr(self.raw_profile, 'duplicate_rows', 0) if self.raw_profile else 0

        # Calculate data loss
        clean_rows = getattr(self.clean_profile, 'n_rows', n_rows) if self.clean_profile else n_rows
        dropped_rows = n_rows - clean_rows
        drop_pct = (dropped_rows / n_rows * 100) if n_rows > 0 else 0

        # Extract deeper insights
        high_card = getattr(self.raw_profile, 'high_cardinality_cols', []) if self.raw_profile else []
        stats = getattr(self.raw_profile, 'stats', {}) if self.raw_profile else {}
        highly_skewed = []
        for col, s in stats.items():
            if isinstance(s, dict):
                skew = s.get('skew')
                if skew is not None and not pd.isna(skew) and abs(float(skew)) > 2.0:
                    highly_skewed.append(col)

        # Opening narrative
        story.append(Paragraph(
            f"Your dataset arrived with <b>{n_rows:,}</b> records and <b>{n_cols}</b> features. "
            f"OctoLearn's intelligent compilation engine analyzed the statistical frontiers of this data "
            f"to establish a baseline for your <b>{task}</b> objective.",
            self.styles['Narrative']
        ))

        # Deep Feature composition & Data Quality Narrative (Combined)
        card_text = f" Notably, <b>{len(high_card)}</b> categorical features exhibited high cardinality, requiring advanced embedding-like encoders." if high_card else ""
        skew_text = f" We also detected significant distribution skewness (skew > 2.0) in <b>{len(highly_skewed)}</b> numeric features, prompting robust scaling." if highly_skewed else ""
        
        story.append(Paragraph(
            f"The data grammar consists of <b>{n_numeric} numeric</b> and <b>{n_cat} categorical</b> components. "
            f"Of the {n_cols} total features, <b>{n_missing_cols}</b> required mathematical imputation due to missingness. "
            f"Furthermore, <b>{dupes} duplicate rows</b> were purged.{card_text}{skew_text}",
            self.styles['Narrative']
        ))

        if drop_pct > 0:
            story.append(Paragraph(
                f"In total, the cleaning pipeline discarded <b>{dropped_rows}</b> defective rows (<b>{drop_pct:.1f}%</b> of your data). "
                f"While reducing data volume, discarding this noise is a critical prerequisite to prevent model hallucination "
                f"and ensure the predictive signals learned are structurally sound.",
                self.styles['Narrative']
            ))

        # Risk narrative
        risk_color_name = "green" if self.risk_score <= 30 else ("amber" if self.risk_score <= 60 else "red")
        story.append(Paragraph(
            f"Synthesizing these metrics, OctoLearn assigned a <b>Data Quality Risk Score of "
            f"{self.risk_score}/100</b> ({self.risk_category}), placing your dataset in the <b>{risk_color_name} zone</b>. "
            + ("This indicates exceptional structural integrity, providing a solid foundation for predictive modeling."
               if self.risk_score <= 30 else
               "This reveals structural inconsistencies that OctoLearn has automatically mitigated, though domain-expert review is advised."
               if self.risk_score <= 60 else
               "This denotes critical quality issues. While OctoLearn applied heavy automated transformations, manual intervention is highly recommended."),
            self.styles['Narrative']
        ))

        # Model narrative
        if self.model_benchmarks:
            n_models = len(self.model_benchmarks)
            best_name = str(self.best_model_name).replace('_', ' ').title()
            best_score = self.model_benchmarks[0].get('score', 0) if self.model_benchmarks else 0
            metric_name = "Accuracy/F1" if task == 'classification' else "RMSE/R²" 
            
            # Intelligent Insight based on score
            if task == 'classification':
                if best_score > 0.90:
                    insight = "This exceptional performance suggests the dataset contains unambiguous, highly separable signals mapping variables to the target outcome."
                elif best_score > 0.75:
                    insight = "This robust performance indicates a strong underlying pattern, though some classes may overlap in the feature space."
                else:
                    insight = "This moderate performance suggests a complex or noisy feature space where predictive signals are subtle or partially obscured."
            else:
                insight = "This performance establishes a strong quantitative baseline for future target forecasting."
                
            story.append(Paragraph(
                f"Powered by Joint Bayesian Inference, OctoLearn evaluated the hyperparameter spaces of <b>{n_models} model architectures</b> rapidly. "
                f"<b>{best_name}</b> ultimately dominated the Model Arena, achieving an optimized validation score of <b>{best_score:.4f}</b> against our {metric_name} baselines. "
                f"{insight}",
                self.styles['Narrative']
            ))
        else:
            story.append(Paragraph(
                "Model training was bypassed in this execution. "
                "Enable <b>train_models=True</b> in ModelingConfig to unlock automated algorithm benchmarking and Bayesian optimization.",
                self.styles['Narrative']
            ))



    # ==================================================================
    # SECTION 3: DATA HEALTH DASHBOARD
    # ==================================================================
    def _fallback_risk_card(self):
        risk_color = ReportColors.get_risk_color(self.risk_score)
        risk_label = ReportColors.get_risk_label(self.risk_score)
        risk_data = [[
            Paragraph(
                f'<font size="24" color="{risk_color.hexval()}"><b>{self.risk_score}</b></font>'
                f'<font size="12" color="{ReportColors.TEXT_SECONDARY.hexval()}">/100</font><br/>'
                f'<font size="4"> </font><br/>'
                f'<font size="11" color="{ReportColors.TEXT_PRIMARY.hexval()}"><b>{risk_label}</b></font><br/>'
                f'<font size="9" color="{ReportColors.TEXT_CAPTION.hexval()}">Data Risk Level</font>',
                ParagraphStyle('RiskCell', parent=self.styles['ReportBody'], alignment=TA_CENTER, leading=14)
            )
        ]]
        risk_table = Table(risk_data, colWidths=[2.5 * inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('BOX', (0, 0), (-1, -1), 1, risk_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return risk_table

    def _add_health_dashboard(self, story):
        self._add_section_header(story, "Data Health Dashboard")

        # --- Risk Score Infographic (Left Content) ---
        img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'icon_health_dashboard.png')
        
        risk_color = ReportColors.get_risk_color(self.risk_score)
        risk_label = ReportColors.get_risk_label(self.risk_score)
        
        risk_text = Paragraph(
            f'<font size="24" color="{risk_color.hexval()}"><b>{self.risk_score}</b></font>'
            f'<font size="12" color="{ReportColors.TEXT_SECONDARY.hexval()}">/100</font><br/>'
            f'<font size="4"> </font><br/>'
            f'<font size="11" color="{ReportColors.TEXT_PRIMARY.hexval()}"><b>{risk_label}</b></font><br/>'
            f'<font size="9" color="{ReportColors.TEXT_CAPTION.hexval()}">Data Risk Level</font>',
            ParagraphStyle('RiskCell', parent=self.styles['ReportBody'], alignment=TA_CENTER, leading=14)
        )
        
        left_items = []
        if os.path.exists(img_path):
            try:
                icon_img = Image(img_path, width=1.4 * inch, height=1.4 * inch)
                icon_img.hAlign = 'CENTER'
                left_items.append(icon_img)
            except Exception as e:
                logger.warning(f"Could not load health dashboard icon: {e}")
        
        left_items.append(risk_text)

        # Wrap in a card table
        left_content = Table([[item] for item in left_items], colWidths=[2.5 * inch])
        left_content.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('BOX', (0, 0), (-1, -1), 1, risk_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        # --- Metric Cards (Right Content) ---
        n_rows = getattr(self.raw_profile, 'n_rows', 0) if self.raw_profile else 0
        n_cols = getattr(self.raw_profile, 'n_columns', 0) if self.raw_profile else 0
        missing_pct = 0
        if self.raw_profile and hasattr(self.raw_profile, 'missing_ratio'):
            ratios = list(self.raw_profile.missing_ratio.values())
            missing_pct = round(sum(ratios) / max(len(ratios), 1) * 100, 1)
        dupes = getattr(self.raw_profile, 'duplicate_rows', 0) if self.raw_profile else 0
        dupe_pct = round(dupes / max(n_rows, 1) * 100, 1) if n_rows else 0

        # Transposed 2x2 Grid for compact side-by-side
        metric_grid = [
             [
                 Paragraph(f"<b>{n_rows:,}</b><br/><font size=8>Rows</font>", ParagraphStyle('M1', parent=self.styles['MetricValue'], leading=12)),
                 Paragraph(f"<b>{n_cols}</b><br/><font size=8>Features</font>", ParagraphStyle('M2', parent=self.styles['MetricValue'], leading=12)),
             ],
             [
                 Paragraph(f"<b>{missing_pct}%</b><br/><font size=8>Cells Missing</font>", ParagraphStyle('M3', parent=self.styles['MetricValue'], leading=12)),
                 Paragraph(f"<b>{dupe_pct}%</b><br/><font size=8>Dupes</font>", ParagraphStyle('M4', parent=self.styles['MetricValue'], leading=12)),
             ]
        ]
        
        card_table = Table(metric_grid, colWidths=[1.8 * inch, 1.8 * inch])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('GRID', (0, 0), (-1, -1), 0.5, ReportColors.BORDER),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
        ]))

        # Data Quality Score Gauge (Bottom Right)
        dq_score = max(0, 100 - self.risk_score)
        
        drawing = Drawing(3.6 * inch, 30)
        # Background bar
        drawing.add(Rect(0, 0, 3.6 * inch, 12, rx=4, ry=4, fillColor=colors.HexColor('#E2E8F0'), strokeColor=None))
        # Active bar
        bar_width = (3.6 * inch) * (dq_score / 100)
        color = ReportColors.RISK_HIGH
        if dq_score > 60: color = ReportColors.RISK_MODERATE
        if dq_score > 80: color = ReportColors.RISK_LOW
        if bar_width > 0:
            drawing.add(Rect(0, 0, bar_width, 12, rx=4, ry=4, fillColor=color, strokeColor=None))
        # Label
        drawing.add(String(1.8 * inch, 16, f"Data Quality Score: {dq_score:.1f}/100", 
                           fontSize=10, fontName=self.font_bold, 
                           textAnchor='middle', fillColor=ReportColors.TEXT_PRIMARY))

        right_items = [[card_table], [Spacer(1, 0.2 * inch)], [drawing]]
        right_content = Table(right_items, colWidths=[3.7 * inch])
        right_content.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Layout Side-by-Side
        layout = self._create_side_by_side(left_content, right_content, widths=[3.0 * inch, 3.8 * inch])
        story.append(layout)
        story.append(Spacer(1, 0.3 * inch))


        # --- Risk Factors Table ---
        if self.risk_factors:
            story.append(Paragraph("Risk Factors Identified", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.1 * inch))

            rf_data = [['Factor', 'Description', 'Severity']]

            def _get_sort_score(item):
                val = item[1]
                if isinstance(val, dict):
                    return val.get('score', 0)
                if isinstance(val, (int, float)):
                    return val
                return 0

            sorted_factors = sorted(self.risk_factors.items(), key=_get_sort_score, reverse=True)

            for factor, data in sorted_factors[:8]:
                if isinstance(data, dict):
                    score = data.get('score', 0)
                    desc = data.get('description', str(data))
                elif isinstance(data, (int, float)):
                    score = data
                    desc = str(data)
                else:
                    score = 0
                    desc = str(data)

                sev_val = score * 10
                sev_color = ReportColors.get_risk_color(sev_val)
                sev_label = "Low" if sev_val <= 30 else ("Med" if sev_val <= 60 else "High")

                rf_data.append([
                    Paragraph(str(factor).replace('_', ' ').title(), self.styles['TableCell']),
                    Paragraph(desc[:80], self.styles['TableCell']),
                    Paragraph(f'<font color="{sev_color.hexval()}"><b>{sev_label}</b></font>',
                              ParagraphStyle('Sev', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                ])

            rf_table = self._styled_table(rf_data, [1.5 * inch, 3.8 * inch, 0.8 * inch])
            story.append(rf_table)



    # ==================================================================
    # SECTION 4: BEFORE & AFTER — Data Transformation Journey
    # ==================================================================
    def _generate_before_after_plots(self) -> Optional[str]:
        """
        Generate a before/after distribution comparison figure for the top
        numeric features. Returns path to a temp PNG file, or None if
        matplotlib is unavailable or no suitable columns exist.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        if self.raw_X is None or self.clean_X is None:
            return None

        try:
            raw_df = self.raw_X
            clean_df = self.clean_X

            # Find numeric columns present in both DataFrames
            raw_num = [c for c in raw_df.columns
                       if raw_df[c].dtype.kind in ('i', 'u', 'f')]
            clean_num = [c for c in clean_df.columns
                         if clean_df[c].dtype.kind in ('i', 'u', 'f')]
            # Prefer columns that exist in both (before encoding)
            shared = [c for c in raw_num if c in clean_num]
            cols = shared[:4] if shared else raw_num[:4]
            if not cols:
                return None

            # Premium Seaborn Theme
            sns.set_theme(style="white", palette="deep", rc={
                "axes.facecolor": "#FFFFFF",
                "figure.facecolor": "#FFFFFF",
                "grid.color": "#F1F5F9",
                "axes.spines.right": False,
                "axes.spines.top": False,
            })

            n_cols = len(cols)
            fig, axes = plt.subplots(n_cols, 2, figsize=(10, 2.5 * n_cols))
            if n_cols == 1:
                axes = [axes]  # Ensure iterable

            NAVY = '#0F172A'  # Slate 900
            TEAL = '#10B981'  # Emerald 500

            for i, col in enumerate(cols):
                ax_raw = axes[i][0]
                ax_clean = axes[i][1]

                raw_data = raw_df[col].dropna()
                sns.histplot(raw_data, kde=True, ax=ax_raw, color=NAVY, alpha=0.6, linewidth=0)
                ax_raw.set_title(f'RAW: {col}', fontsize=10, fontweight='bold', color=NAVY, loc='left')
                ax_raw.set_ylabel('Density', fontsize=8, color='#64748B')
                ax_raw.set_xlabel('')
                ax_raw.tick_params(labelsize=8, colors='#64748B')

                if col in clean_df.columns:
                    clean_data = clean_df[col].dropna()
                else:
                    clean_data = raw_data  # Fallback
                    
                sns.histplot(clean_data, kde=True, ax=ax_clean, color=TEAL, alpha=0.6, linewidth=0)
                ax_clean.set_title(f'CLEANED: {col}', fontsize=10, fontweight='bold', color=TEAL, loc='left')
                ax_clean.set_ylabel('')
                ax_clean.set_xlabel('')
                ax_clean.tick_params(labelsize=8, colors='#64748B')

            fig.suptitle('Distribution Shift: The Impact of OctoLearn Cleaning',
                         fontsize=14, fontweight='bold', color=NAVY, y=1.03, ha='center')
            plt.tight_layout()

            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmp.name, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            self._temp_files.append(tmp.name)
            
            # Reset style
            sns.set_theme(style="whitegrid")
            return tmp.name

        except Exception as e:
            logger.warning(f"Before/after plot generation failed: {e}")
            return None

    def _add_before_after(self, story):
        self._add_section_header(story, "Data Transformation Journey")

        story.append(Paragraph(
            "OctoLearn's intelligent cleaning pipeline automatically transforms your raw data "
            "into a clean, model-ready dataset. Here is the before-and-after comparison:",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.15 * inch))

        raw = self.raw_profile
        clean = self.clean_profile
        if not raw or not clean:
            story.append(Paragraph("Profile data not available for comparison.", self.styles['Narrative']))
            return

        # Comparison table
        raw_missing = sum(1 for v in getattr(raw, 'missing_ratio', {}).values() if v > 0)
        clean_missing = sum(1 for v in getattr(clean, 'missing_ratio', {}).values() if v > 0)

        compare_data = [
            ['Metric', 'Before (Raw)', 'After (Clean)', 'Change'],
            [
                Paragraph('Rows', self.styles['TableCell']),
                Paragraph(f'{raw.n_rows:,}', self.styles['TableCell']),
                Paragraph(f'{clean.n_rows:,}', self.styles['TableCell']),
                Paragraph(f'{clean.n_rows - raw.n_rows:+,}', self.styles['TableCell']),
            ],
            [
                Paragraph('Features', self.styles['TableCell']),
                Paragraph(f'{raw.n_columns}', self.styles['TableCell']),
                Paragraph(f'{clean.n_columns}', self.styles['TableCell']),
                Paragraph(f'{clean.n_columns - raw.n_columns:+d}', self.styles['TableCell']),
            ],
            [
                Paragraph('Numeric Columns', self.styles['TableCell']),
                Paragraph(f'{len(raw.numeric_columns)}', self.styles['TableCell']),
                Paragraph(f'{len(clean.numeric_columns)}', self.styles['TableCell']),
                Paragraph(f'{len(clean.numeric_columns) - len(raw.numeric_columns):+d}', self.styles['TableCell']),
            ],
            [
                Paragraph('Categorical Columns', self.styles['TableCell']),
                Paragraph(f'{len(raw.categorical_columns)}', self.styles['TableCell']),
                Paragraph(f'{len(clean.categorical_columns)}', self.styles['TableCell']),
                Paragraph(f'{len(clean.categorical_columns) - len(raw.categorical_columns):+d}', self.styles['TableCell']),
            ],
            [
                Paragraph('Cols with Missing Values', self.styles['TableCell']),
                Paragraph(f'{raw_missing}', self.styles['TableCell']),
                Paragraph(f'{clean_missing}', self.styles['TableCell']),
                Paragraph(f'{clean_missing - raw_missing:+d}', self.styles['TableCell']),
            ],
        ]

        compare_table = self._styled_table(compare_data, [1.8 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch])
        story.append(compare_table)
        story.append(Spacer(1, 0.2 * inch))

        # Cleaning Log Flowchart
        if self.cleaning_log:
            story.append(Paragraph("The Data Journey", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.05 * inch))

            # Gather actions
            dupes = self.cleaning_log.get('duplicates_removed', 0)
            dropped = self.cleaning_log.get('id_columns_removed', 0) + self.cleaning_log.get('constant_columns_removed', 0) + self.cleaning_log.get('low_variance_removed', 0)
            imputed = self.cleaning_log.get('missing_imputed', 0)
            scaling = self.cleaning_log.get('scaling_method', 'none')

            img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
            icon_raw = os.path.join(img_dir, 'icon_raw_data.png')
            icon_clean = os.path.join(img_dir, 'icon_cleansed.png')
            icon_eng = os.path.join(img_dir, 'icon_engineered.png')
            icon_ready = os.path.join(img_dir, 'icon_ready.png')
            
            # Helper to safely load images
            def get_icon(path, size=0.9):
                if os.path.exists(path):
                    try:
                        img = Image(path, width=size * inch, height=size * inch)
                        img.hAlign = 'CENTER'
                        return img
                    except: pass
                # fallback text if icon fails
                return Paragraph('&#9632;', ParagraphStyle('FI', parent=self.styles['MetricLabel'], alignment=TA_CENTER))

            # Assemble visual row
            arrow_html = '<font size=18 color="#94A3B8" name="Helvetica-Bold">&gt;&gt;</font>'
            flow_data = [[
                get_icon(icon_raw),
                Paragraph(arrow_html, ParagraphStyle('FArr', parent=self.styles['MetricLabel'], alignment=TA_CENTER)),
                get_icon(icon_clean),
                Paragraph(arrow_html, ParagraphStyle('FArr', parent=self.styles['MetricLabel'], alignment=TA_CENTER)),
                get_icon(icon_eng),
                Paragraph(arrow_html, ParagraphStyle('FArr', parent=self.styles['MetricLabel'], alignment=TA_CENTER)),
                get_icon(icon_ready)
            ]]
            
            flow_table = Table(flow_data, colWidths=[1.3*inch, 0.4*inch, 1.3*inch, 0.4*inch, 1.3*inch, 0.4*inch, 1.3*inch])
            flow_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(flow_table)
            story.append(Spacer(1, 0.05 * inch))

            # Sub-descriptions
            desc_data = [[
                Paragraph(f"<font size=8 color='#1E293B'><b>RAW DATA</b></font><br/><font size=7 color='#64748B'>Initial ingestion of {raw.n_rows:,} rows</font>", self.styles['Caption']),
                Paragraph("", self.styles['Caption']),
                Paragraph(f"<font size=8 color='#1E293B'><b>CLEANSED</b></font><br/><font size=7 color='#64748B'>{dupes} dupes, {dropped} cols dropped</font>", self.styles['Caption']),
                Paragraph("", self.styles['Caption']),
                Paragraph(f"<font size=8 color='#1E293B'><b>ENGINEERED</b></font><br/><font size=7 color='#64748B'>{imputed} imputed, {scaling} scaled</font>", self.styles['Caption']),
                Paragraph("", self.styles['Caption']),
                Paragraph(f"<font size=8 color='#1E293B'><b>READY</b></font><br/><font size=7 color='#64748B'>Optimized for modeling</font>", self.styles['Caption']),
            ]]
            desc_table = Table(desc_data, colWidths=[1.3*inch, 0.4*inch, 1.3*inch, 0.4*inch, 1.3*inch, 0.4*inch, 1.3*inch])
            desc_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(desc_table)
            story.append(Spacer(1, 0.2 * inch))

        # Before/After distribution plots
        ba_plot_path = self._generate_before_after_plots()
        if ba_plot_path and os.path.exists(ba_plot_path):
            story.append(Paragraph("Distribution Topography",
                                   self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(
                "The density plots below illustrate the structural shifts in feature distributions after OctoLearn's transformations. "
                "Notice how outliers and skewness are mitigated in the cleansed density curves.",
                self.styles['Narrative']
            ))
            story.append(Spacer(1, 0.1 * inch))
            try:
                ba_img = Image(ba_plot_path, width=7.2 * inch, height=None)
                # Preserve aspect ratio
                from PIL import Image as PILImg
                with PILImg.open(ba_plot_path) as pil_img:
                    w_px, h_px = pil_img.size
                aspect = h_px / w_px
                ba_img = Image(ba_plot_path, width=7.2 * inch, height=7.2 * inch * aspect)
                ba_img.hAlign = 'CENTER'
                story.append(ba_img)
            except Exception:
                # Fallback without PIL
                try:
                    ba_img = Image(ba_plot_path, width=7.2 * inch, height=4.5 * inch)
                    ba_img.hAlign = 'CENTER'
                    story.append(ba_img)
                except Exception as e:
                    logger.warning(f"Could not embed before/after plot: {e}")



    # ==================================================================
    # SECTION 5: FEATURE INTELLIGENCE
    # ==================================================================
    def _add_feature_intelligence(self, story):
        if not self.feature_importance:
            return

        self._add_section_header(story, "Feature Intelligence")

        story.append(Paragraph(
            "OctoLearn calculated feature importance scores to identify which variables "
            "have the strongest predictive power. Higher scores indicate greater influence "
            "on the model's decisions.",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Sort features
        sorted_feats = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        limit = 10  # Show top 10 in table
        top_feats = sorted_feats[:limit]

        if not top_feats:
            return

        max_score = top_feats[0][1] if top_feats else 1

        fi_data = [['Rank', 'Feature', 'Score']]
        for rank, (feat, score) in enumerate(top_feats, 1):
            # Truncate long feature names
            feat_str = str(feat)
            if len(feat_str) > 20:
                feat_str = feat_str[:18] + ".."
            
            fi_data.append([
                Paragraph(f'<b>{rank}</b>', ParagraphStyle('R', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(feat_str, self.styles['TableCell']),
                Paragraph(f'{score:.4f}', ParagraphStyle('S', parent=self.styles['TableCell'], alignment=TA_CENTER)),
            ])

        # Left Column: Table (Compact)
        fi_table = self._styled_table(fi_data, [0.4 * inch, 1.8 * inch, 0.8 * inch])
        
        # Right Column: SHAP Plot or Placeholder
        right_content = []
        if self.shap_path and os.path.exists(self.shap_path):
            right_content.append(Paragraph("SHAP Feature Impact", self.styles['SubsectionHeading']))
            try:
                # Resize for column width (approx 3.5 inch)
                shap_img = Image(self.shap_path, width=3.8 * inch, height=2.8 * inch)
                shap_img.hAlign = 'CENTER'
                right_content.append(shap_img)
                right_content.append(Paragraph(
                    "Red = Higher prediction<br/>Blue = Lower prediction",
                    self.styles['Caption']
                ))
            except Exception:
                pass
        else:
            # Generate an inline horizontal bar chart using matplotlib
            bar_path = self._generate_importance_bar_chart(top_feats, max_score)
            if bar_path and os.path.exists(bar_path):
                try:
                    bar_img = Image(bar_path, width=3.8 * inch, height=2.8 * inch)
                    bar_img.hAlign = 'CENTER'
                    right_content.append(bar_img)
                    right_content.append(Paragraph(
                        "Feature importance scores (higher = more predictive)",
                        self.styles['Caption']
                    ))
                except Exception:
                    pass
            
            # Generate ROI Narrative
            if not right_content or len(right_content) > 0:
                # Add an ROI narrative regardless of image success
                top_3_feats = [f for f, s in top_feats[:3]]
                top_3_str = ", ".join([f'<b>{f}</b>' for f in top_3_feats])
                
                total_importance = sum(self.feature_importance.values())
                top_3_importance = sum([s for f, s in top_feats[:3]])
                impact_pct = (top_3_importance / total_importance * 100) if total_importance > 0 else 0
                
                narrative_text = "<b>Strategic Business Impact</b><br/><br/>"
                if impact_pct > 0:
                    narrative_text += (
                        f"The top 3 features ({top_3_str}) account for approximately <b>{impact_pct:.1f}%</b> of the model's predictive power. "
                        f"Focusing operational and business efforts on optimizing these specific levers will likely yield the highest direct ROI on your target outcome."
                    )
                else:
                    narrative_text += (
                        "Understanding which features drive your model's predictions is crucial "
                        "for trust and debugging. The table on the left shows the top contributors."
                    )
                
                right_content.append(Paragraph(
                    narrative_text,
                    self.styles['Narrative']
                ))

        layout = self._create_side_by_side(fi_table, right_content, widths=[3.2 * inch, 4.0 * inch])
        story.append(layout)
        story.append(Spacer(1, 0.2 * inch))




    # ==================================================================
    # SECTION 6: MODEL ARENA
    # ==================================================================
    def _add_model_arena(self, story):
        if not self.model_benchmarks:
            return

        self._add_section_header(story, "Model Arena")
        
        # Add visual performance plots if available
        self._add_performance_visuals(story)

        n_models = len(self.model_benchmarks)
        best = self.model_benchmarks[0] if self.model_benchmarks else {}
        # Try to use a pretty version of the best model name if available
        internal_best_name = best.get('model', self.best_model_name)
        best_name = str(internal_best_name).replace('_', ' ').title()
        best_score = best.get('score', 0)

        if "xgboost" in internal_best_name.lower() or "lightgbm" in internal_best_name.lower() or "gradient_boosting" in internal_best_name.lower():
            arch_insight = f"{best_name}'s dominance suggests your dataset contains strong, non-linear feature interactions that tree-based gradient boosting methods excel at capturing."
        elif "random_forest" in internal_best_name.lower():
            arch_insight = f"{best_name}'s success indicates a dataset with complex, possibly noisy relationships where ensemble bagging provides necessary variance reduction."
        elif "logistic" in internal_best_name.lower() or "linear" in internal_best_name.lower() or "ridge" in internal_best_name.lower():
            arch_insight = f"The victory of a linear model like {best_name} strongly implies the dataset's features have monotonic, proportional relationships with the target variable."
        elif "svm" in internal_best_name.lower() or "svr" in internal_best_name.lower():
            arch_insight = f"{best_name}'s performance suggests the data possesses complex boundary margins that are well-separated in high-dimensional kernel space."
        else:
            arch_insight = "This model architecture proved optimal for uncovering the underlying predictive patterns."

        story.append(Paragraph(
            f"OctoLearn explored <b>{n_models} model architectures</b> through a joint Bayesian hyperparameter space. "
            f"Evaluated rigorously on a held-out test set, the undisputed champion is <b>{best_name}</b> "
            f"with a primary score of <b>{best_score:.4f}</b>. {arch_insight}",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.15 * inch))

        # Champion card
        champion_data = [[
            Paragraph(f'<font size="10" color="{ReportColors.PRIMARY.hexval()}">CHAMPION</font><br/>'
                      f'<font size="16" color="{ReportColors.PRIMARY.hexval()}"><b>{best_name}</b></font>',
                      ParagraphStyle('Champ', parent=self.styles['Narrative'], alignment=TA_CENTER, leading=22)),
        ]]
        champ_table = Table(champion_data, colWidths=[6.2 * inch])
        champ_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF5FB')),
            ('BOX', (0, 0), (-1, -1), 2, ReportColors.PRIMARY),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(champ_table)
        story.append(Spacer(1, 0.2 * inch))

        # Determine task type from profile or benchmarks
        task_type = getattr(self.raw_profile, 'task_type', 'classification') if self.raw_profile else 'classification'
        is_regression = task_type == 'regression'

        if is_regression:
            header = ['#', 'Model', 'Score', 'RMSE', 'MAE', 'R²', 'MAPE']
        else:
            header = ['#', 'Model', 'Score', 'Accuracy', 'Precision', 'F1', 'Recall']
        bench_data = [header]

        best_row_idx = None
        limit = 5 if self.mode == 'brief' else 10
        for idx, bm in enumerate(self.model_benchmarks[:limit], 1):
            model_name = bm.get('model', 'Unknown')
            score = bm.get('score', 0)
            metrics = bm.get('metrics', {})

            if is_regression:
                col3 = metrics.get('rmse', score)
                col4 = metrics.get('mae', 0)
                col5 = metrics.get('r2', 0)
                col6 = metrics.get('mape', 0)
            else:
                col3 = metrics.get('accuracy', score)
                col4 = metrics.get('precision', 0)
                col5 = metrics.get('f1', 0)
                col6 = metrics.get('recall', 0)

            if model_name == best_name or (best_row_idx is None and idx == 1):
                best_row_idx = idx

            bench_data.append([
                Paragraph(f'{idx}', ParagraphStyle('Idx', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'<b>{model_name}</b>' if idx == 1 else model_name, self.styles['TableCell']),
                Paragraph(f'{score:.4f}', ParagraphStyle('Sc', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{col3:.4f}', ParagraphStyle('C3', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{col4:.4f}', ParagraphStyle('C4', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{col5:.4f}', ParagraphStyle('C5', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{col6:.4f}', ParagraphStyle('C6', parent=self.styles['TableCell'], alignment=TA_CENTER)),
            ])

        bench_table = self._styled_table(
            bench_data,
            [0.35 * inch, 1.6 * inch, 0.75 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 0.75 * inch],
            highlight_best_row=best_row_idx
        )
        story.append(bench_table)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            "Models are ranked by primary optimization score. The #1 model (highlighted) was selected as the best performer.",
            self.styles['Caption']
        ))



    # ==================================================================
    # SECTION 7: RECOMMENDATIONS
    # ==================================================================
    def _add_recommendations(self, story):
        if not self.recommendations:
            return

        self._add_section_header(story, "Actionable Recommendations")

        story.append(Paragraph(
            "Based on the comprehensive analysis, OctoLearn has generated the following "
            "prioritized recommendations to further improve your data and models:",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Flatten recs into priority groups
        priority_groups = {}
        if isinstance(self.recommendations, dict):
            for priority in ['critical', 'high', 'medium', 'low', 'informational']:
                if priority in self.recommendations and self.recommendations[priority]:
                    priority_groups[priority] = self.recommendations[priority]
            # Catch any non-standard keys
            for k, v in self.recommendations.items():
                if k not in priority_groups and isinstance(v, list) and v:
                    priority_groups.setdefault('medium', []).extend(v)
        elif isinstance(self.recommendations, list):
            priority_groups['medium'] = self.recommendations

        if not priority_groups:
            story.append(Paragraph("No specific recommendations at this time.", self.styles['Narrative']))
            return

        self._add_section_header(story, "Action Plan for Production")
        
        # Add Business Impact Narrative if importance is available
        if self.feature_importance:
            story.append(Paragraph("Strategic Business Insights", self.styles['SubsectionHeading']))
            # We'll use a simple logic here or import from engine if possible
            # To keep it simple and decoupled, I'll implement a local version 
            # or try to use the engine if available in context.
            # actually I can just implement a quick version here.
            top_3 = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            for feature, score in top_3:
                msg = f"<b>Key Lever: {feature}</b>"
                if "age" in feature.lower():
                    desc = "Driving factor. Age-based segmentation is critical for targeting."
                elif "score" in feature.lower() or "rating" in feature.lower():
                    desc = "Primary quality indicator. Optimization here yields immediate ROI."
                elif "amount" in feature.lower() or "price" in feature.lower():
                    desc = "Financial pivot point. Small variances here significantly impact the bottom line."
                else:
                    desc = "Significant statistical weight. Prioritize operational focus on this variable."
                
                story.append(Paragraph(f"➜ {msg} — {desc}", self.styles['Narrative']))
            story.append(Spacer(1, 0.2 * inch))

        # Priority styling
        priority_config = {
            'critical': ('[CRITICAL]', ReportColors.DANGER, self.styles['RecCritical']),
            'high':     ('[HIGH]', ReportColors.WARNING, self.styles['RecHigh']),
            'medium':   ('[MEDIUM]', ReportColors.INFO, self.styles['RecNormal']),
            'low':      ('[LOW]', ReportColors.TEXT_CAPTION, self.styles['RecNormal']),
            'informational': ('[INFO]', ReportColors.TEXT_CAPTION, self.styles['RecNormal']),
        }

        limit = 5 if self.mode == 'brief' else 15
        count = 0
        for priority, recs in priority_groups.items():
            if count >= limit:
                break
            label, color, style = priority_config.get(priority, ('[INFO]', ReportColors.TEXT_CAPTION, self.styles['RecNormal']))

            story.append(Paragraph(
                f'<font color="{color.hexval()}" size="11"><b>{priority.upper()} PRIORITY</b></font>',
                ParagraphStyle('PriorityHeader', parent=self.styles['Narrative'],
                               spaceBefore=10, spaceAfter=4)
            ))

            for rec in recs:
                if count >= limit:
                    break
                count += 1
                # Truncate long recs
                rec_text = str(rec)[:200]
                story.append(Paragraph(
                    f'<font color="{color.hexval()}"><b>{label}</b></font>  {rec_text}',
                    style
                ))

            story.append(Spacer(1, 0.05 * inch))



    # ==================================================================
    # SECTION 8: VISUAL INSIGHTS
    # ==================================================================
    def _add_visual_insights(self, story):
        """Add correlation visualization and feature distribution plots."""
        has_visuals = bool(self.dist_plots) or bool(self.heatmap_plot)
        if not has_visuals:
            return

        self._add_section_header(story, "Visual Insights")

        story.append(Paragraph(
            "The following visualizations provide deeper insight into feature distributions "
            "and correlations within your cleaned dataset.",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # ── Correlation narrative paragraph ──────────────────────────────────
        if self.corr_summary:
            story.append(Paragraph("Goal Correlation Analysis", self.styles['SubsectionHeading']))
            n_features = self.corr_summary.get('n_features', 0)
            target_name = self.corr_summary.get('target_name', 'Target')
            top_pairs = self.corr_summary.get('top_pairs', [])
            bottom_pairs = self.corr_summary.get('bottom_pairs', [])

            # Build plain-English narrative
            if n_features >= 1 and top_pairs:
                # Top correlated features with target
                top_strs = []
                for feat, target, corr in top_pairs[:3]:
                    direction = "positive" if corr > 0 else "inverse"
                    impact = "Critical" if abs(corr) > 0.7 else ("Strong" if abs(corr) > 0.5 else "Moderate")
                    top_strs.append(f"<b>{feat}</b> ({impact} {direction} relationship, Pearson r={corr:.2f})")

                top_text = "<br/>".join([f"  - {x}" for x in top_strs]) if top_strs else "no significant linear relationships identified."

                viz_note = (
                    f"Mathematically, this implies that variance in these top features "
                    f"explains a significant proportion of the variance in <b>{target_name}</b>. "
                    f"The chart below visualizes these Pearson correlation coefficients across the numeric feature space, "
                    f"allowing us to detect both multi-collinearity (redundant features) and primary predictive signals."
                )

                story.append(Paragraph(
                    f"Statistical analysis isolated the following primary linear predictors for <b>{target_name}</b>:<br/>{top_text}<br/><br/>"
                    f"{viz_note}",
                    self.styles['Narrative']
                ))
                story.append(Spacer(1, 0.1 * inch))

        # ── Correlation chart (heatmap or bar chart) ─────────────────────────
        if self.heatmap_plot and os.path.exists(self.heatmap_plot):
            try:
                strategy = self.corr_summary.get('strategy', 'heatmap') if self.corr_summary else 'heatmap'
                if strategy == 'target_correlation' or strategy == 'bar_chart':
                    # Bar chart is wider/shorter
                    hm_img = Image(self.heatmap_plot, width=7.0 * inch, height=4.5 * inch)
                    caption = (
                        "Ranked correlation chart: blue bars = positive correlation, "
                        "red bars = negative correlation. Higher absolute values "
                        "indicate features that have more linear influence on the target goal."
                    )
                else:
                    hm_img = Image(self.heatmap_plot, width=6.0 * inch, height=4.8 * inch)
                    caption = (
                        "Heatmap shows pairwise feature correlations. Strong correlations "
                        "(near ±1.0) may indicate redundant features or multicollinearity."
                    )
                hm_img.hAlign = 'CENTER'
                story.append(hm_img)
                story.append(Paragraph(caption, self.styles['Caption']))
                story.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                logger.warning(f"Could not embed correlation chart: {e}")

        # Distribution plots
        limit = 3 if self.mode == 'brief' else 8
        valid_plots = [p for p in self.dist_plots[:limit] if os.path.exists(p)]
        if valid_plots:
            story.append(Paragraph("Feature Distributions", self.styles['SubsectionHeading']))
            for plot_path in valid_plots:
                try:
                    # Source is (10, 3.5), Ratio ~2.85.
                    # Page width ~8.27 - 0.6 = ~7.6 inch.
                    # Height should be 7.6 / 2.85 = ~2.66 inch.
                    dist_img = Image(plot_path, width=7.5 * inch, height=2.8 * inch)
                    dist_img.hAlign = 'CENTER'
                    story.append(dist_img)
                    story.append(Spacer(1, 0.15 * inch))
                except Exception:
                    pass

    # ==================================================================
    # SECTION: OUTLIER & INTERACTION ANALYSIS (Detailed mode only)
    # ==================================================================
    def _add_analysis_details(self, story):
        """Outlier detection and interaction results."""
        has_outliers = bool(self.outlier_results and self.outlier_results.get('summary'))
        has_interactions = bool(self.interaction_results and self.interaction_results.get('strong_interactions'))

        if not has_outliers and not has_interactions:
            return

        self._add_section_header(story, "Advanced Analysis")

        if has_outliers:
            story.append(Paragraph("Outlier Detection", self.styles['SubsectionHeading']))
            summary = self.outlier_results.get('summary', {})
            outlier_data = [['Detection Method', 'Outliers Found']]
            for method, count in summary.items():
                outlier_data.append([
                    Paragraph(str(method).replace('_', ' ').title(), self.styles['TableCell']),
                    Paragraph(str(count), ParagraphStyle('OC', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                ])
            if len(outlier_data) > 1:
                ot = self._styled_table(outlier_data, [3.5 * inch, 2.5 * inch])
                story.append(ot)
            story.append(Spacer(1, 0.2 * inch))

        if has_interactions:
            story.append(Paragraph("Feature Interactions", self.styles['SubsectionHeading']))
            interactions = self.interaction_results.get('strong_interactions', [])
            for inter in interactions[:5]:
                if isinstance(inter, (tuple, list)) and len(inter) >= 2:
                    story.append(Paragraph(
                        f"  <font color='#94A3B8' name='Helvetica-Bold'>&gt;&gt;</font>  <b>{inter[0]}</b> &lt;-&gt; <b>{inter[1]}</b>",
                        self.styles['Narrative']
                    ))



    # ==================================================================
    # HELPER: Feature Importance Bar Chart
    # ==================================================================
    def _generate_importance_bar_chart(self, top_feats, max_score) -> Optional[str]:
        """
        Generate a horizontal bar chart of feature importance scores.
        Returns path to a temp PNG file, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE or not top_feats:
            return None
        try:
            feats = [f[:18] + '..' if len(f) > 20 else f for f, _ in top_feats]
            scores = [s for _, s in top_feats]
            n = len(feats)

            fig, ax = plt.subplots(figsize=(4.5, max(2.5, n * 0.38)))
            NAVY = '#1B3A5C'
            bars = ax.barh(range(n), scores, color=NAVY, alpha=0.85, height=0.6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(feats, fontsize=8)  # Fixed: No more [::-1] reversal
            ax.invert_yaxis()
            ax.set_xlabel('Importance Score', fontsize=8)
            ax.set_xlim(0, max_score * 1.15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7)
            # Value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_width() + max_score * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{score:.3f}', va='center', fontsize=7, color=NAVY)
            plt.tight_layout()

            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmp.name, dpi=130, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            self._temp_files.append(tmp.name)
            return tmp.name
        except Exception as e:
            logger.warning(f"Importance bar chart generation failed: {e}")
            return None

    # ==================================================================
    # HEADER / FOOTER CALLBACKS
    # ==================================================================
    def _cover_page_callback(self, canvas_obj, doc):
        """Draw logo as a large low-opacity watermark on the cover page."""
        canvas_obj.saveState()
        w, h = A4
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                # Draw large faded logo as background watermark
                logo_size = 4.5 * inch
                x = (w - logo_size) / 2
                y = (h - logo_size) / 2 - 0.5 * inch
                canvas_obj.saveState()
                canvas_obj.setFillAlpha(0.06)  # Very low opacity
                canvas_obj.drawImage(
                    self.logo_path,
                    x, y,
                    width=logo_size, height=logo_size,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                canvas_obj.restoreState()
            except Exception:
                pass
        canvas_obj.restoreState()

    def _header_footer_callback(self, canvas, doc):
        """Draw header and footer on every page (callback for onLaterPages)."""
        canvas.saveState()
        w, h = A4
        
        # --- Header bar ---
        canvas.setFillColor(ReportColors.HEADER_BG)
        canvas.rect(0, h - 35, w, 35, fill=True, stroke=False)
        
        # Logo in header (small)
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                canvas.drawImage(self.logo_path, 12, h - 32, width=26, height=26,
                               preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
                
        # Title in header
        canvas.setFont('Helvetica-Bold', 9)
        canvas.setFillColor(ReportColors.TEXT_ON_DARK)
        canvas.drawString(45, h - 24, self.title)
        
        # Date in header
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(w - 15, h - 24, datetime.now().strftime("%B %d, %Y"))
        
        # --- Footer ---
        canvas.setFillColor(ReportColors.DIVIDER)
        canvas.line(50, 30, w - 50, 30)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(ReportColors.TEXT_CAPTION)
        canvas.drawString(50, 18, "Generated by OctoLearn AutoML")
        canvas.drawRightString(w - 50, 18, f"Page {doc.page}")
        
        canvas.restoreState()

    # ==================================================================
    # GENERATE — Build the complete PDF
    # ==================================================================
    def generate(self, filename: Optional[str] = None) -> str:
        """
        Build and save the complete PDF report.

        Orchestrates all report sections in order. Each section is wrapped
        in an individual ``try/except`` block so that a failure in one section
        (e.g., malformed model benchmarks) never crashes the entire report.
        Failed sections are replaced with a graceful "Section unavailable" note.

        Parameters
        ----------
        filename : str, optional
            Output file path. If ``None``, a timestamped filename is generated
            automatically (e.g., ``octolearn_report_20260218_192700.pdf``).

        Returns
        -------
        str
            Absolute path to the generated PDF file.

        Raises
        ------
        RuntimeError
            If the PDF build itself fails (e.g., disk full, permission denied).

        Examples
        --------
        >>> pdf_path = generator.generate("my_report.pdf")
        >>> print(f"Report saved to: {pdf_path}")
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"octolearn_report_{timestamp}.pdf"

        def _safe_section(fn, name):
            """Run a section builder, log and skip on any error."""
            try:
                fn()
            except Exception as exc:
                logger.warning(f"Report section '{name}' failed: {exc}")
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph(
                    f"[{name} — section unavailable due to an internal error]",
                    self.styles.get('Caption', self.styles['Normal'])
                ))

        try:
            doc = SimpleDocTemplate(
                filename,
                pagesize=A4,
                rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                title=self.title,
                author=self.author
            )

            story = []

            # 1. Cover page (kept separate — has its own canvas callback)
            _safe_section(lambda: self._add_cover_page(story), "Cover Page")

            # 1.5 Executive Summary
            _safe_section(lambda: self._add_executive_summary(story), "Executive Summary")

            # 3. Data Story narrative
            _safe_section(lambda: self._add_data_story(story), "Data Story")
            story.append(Spacer(1, 0.4 * inch))

            # 4. Health Dashboard
            _safe_section(lambda: self._add_health_dashboard(story), "Health Dashboard")
            story.append(Spacer(1, 0.4 * inch))

            # 5. Before / After transformation journey
            _safe_section(lambda: self._add_before_after(story), "Before & After")
            story.append(Spacer(1, 0.4 * inch))

            # 6. Feature Intelligence
            _safe_section(lambda: self._add_feature_intelligence(story), "Feature Intelligence")
            story.append(Spacer(1, 0.4 * inch))

            # 7. Model Arena
            _safe_section(lambda: self._add_model_arena(story), "Model Arena")
            story.append(Spacer(1, 0.4 * inch))

            # 8. Advanced Analysis (detailed mode only)
            if self.mode == 'detailed':
                _safe_section(lambda: self._add_analysis_details(story), "Advanced Analysis")
                story.append(Spacer(1, 0.4 * inch))

            # 9. Visual Insights (correlation + distributions)
            _safe_section(lambda: self._add_visual_insights(story), "Visual Insights")

            # Build PDF with canvas callbacks
            doc.build(
                story,
                onFirstPage=self._cover_page_callback,   # Logo watermark on cover
                onLaterPages=self._header_footer_callback
            )
            logger.info(f"[OK] Report generated: {filename}")
            return filename

        except Exception as e:
            logger.error(f"[FAIL] PDF generation failed: {str(e)}")
            raise RuntimeError(f"PDF generation failed: {str(e)}")

        finally:
            # Clean up all temp files created during report generation
            for tmp_path in self._temp_files:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass
            self._temp_files.clear()



    def _add_data_quality_gauge(self, story, score: float):
        """Add a gauge for Data Quality Score."""
        drawing = Drawing(400, 50)
        
        # Gauge background
        drawing.add(Rect(50, 10, 300, 20, rx=5, ry=5, fillColor=colors.HexColor('#E2E8F0'), strokeColor=None))
        
        # Score bar
        bar_width = 300 * (score / 100)
        color = ReportColors.RISK_HIGH
        if score > 60: color = ReportColors.RISK_MODERATE
        if score > 80: color = ReportColors.RISK_LOW
            
        if bar_width > 0:
            drawing.add(Rect(50, 10, bar_width, 20, rx=5, ry=5, fillColor=color, strokeColor=None))
            
        # Label
        drawing.add(String(200, 38, f"Data Quality Score: {score:.1f}/100", 
                           fontSize=12, fontName=self.font_bold, 
                           textAnchor='middle', fillColor=ReportColors.TEXT_PRIMARY))
                           
        story.append(drawing)

    def _add_performance_visuals(self, story):
        """Add ROC Curve, Confusion Matrix, or Residual Plots."""
        visuals = []
        if self.confusion_matrix_plot and os.path.exists(self.confusion_matrix_plot):
            visuals.append((self.confusion_matrix_plot, "Confusion Matrix"))
        if self.roc_curve_plot and os.path.exists(self.roc_curve_plot):
            visuals.append((self.roc_curve_plot, "ROC Curve"))
        if self.residual_plot and os.path.exists(self.residual_plot):
            visuals.append((self.residual_plot, "Residual Analysis"))
        if not visuals:
            return
            
        story.append(Paragraph("Best Model Deep Dive", self.styles['SubsectionHeading']))
        story.append(Paragraph(
            "Detailed performance analysis of the champion model.",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # layout images side by side if 2, or grid
        images = []
        for path, title in visuals:
            try:
                img = Image(path, width=3.2 * inch, height=2.6 * inch)
                caption = Paragraph(title, ParagraphStyle('C', parent=self.styles['Caption'], alignment=TA_CENTER))
                # Keep image and caption together
                images.append([img, caption])
            except:
                pass
        
        if len(images) == 2:
            data = [[images[0][0], images[1][0]], [images[0][1], images[1][1]]]
            t = Table(data, colWidths=[3.5*inch, 3.5*inch])
            t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
            story.append(t)
        elif len(images) >= 1:
            # If 1 or more (but not exactly 2, though logic covers it), stack them or grid them
            # For now, simplistic stacking
             for img_set in images:
                 story.append(img_set[0])
                 story.append(img_set[1])
                 story.append(Spacer(1, 0.1 * inch))

        story.append(Spacer(1, 0.2 * inch))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_brief_report(raw_profile: Any, clean_profile: Any, **kwargs) -> str:
    """Create a concise 5-7 page report."""
    gen = ReportGenerator(raw_profile=raw_profile, clean_profile=clean_profile, mode='brief', **kwargs)
    return gen.generate()


def create_detailed_report(raw_profile: Any, clean_profile: Any, **kwargs) -> str:
    """Create a comprehensive 15+ page report."""
    gen = ReportGenerator(raw_profile=raw_profile, clean_profile=clean_profile, mode='detailed', **kwargs)
    return gen.generate()