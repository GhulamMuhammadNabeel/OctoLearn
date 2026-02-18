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
Version: 0.8.0 (Redesigned)
License: MIT
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import warnings

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
    """Professional light-theme color palette for readable PDF output."""

    # === Primary brand colors ===
    PRIMARY = colors.HexColor('#1B3A5C')       # Deep navy blue (headers)
    PRIMARY_LIGHT = colors.HexColor('#2D5F8A')  # Lighter navy
    ACCENT = colors.HexColor('#E74C3C')         # Warm red accent

    # === Backgrounds ===
    WHITE = colors.HexColor('#FFFFFF')
    PAGE_BG = colors.HexColor('#FFFFFF')        # White paper
    CARD_BG = colors.HexColor('#F7F9FC')        # Subtle blue-grey cards
    HEADER_BG = colors.HexColor('#1B3A5C')      # Navy header bars
    ROW_ALT = colors.HexColor('#F0F4F8')        # Alternating table rows

    # === Text colors ===
    TEXT_PRIMARY = colors.HexColor('#1A1A2E')   # Near-black for body text
    TEXT_SECONDARY = colors.HexColor('#4A5568')  # Muted grey for secondary
    TEXT_ON_DARK = colors.HexColor('#FFFFFF')    # White text on dark headers
    TEXT_CAPTION = colors.HexColor('#718096')    # Caption grey

    # === Status/Semantic colors ===
    SUCCESS = colors.HexColor('#27AE60')        # Emerald green
    WARNING = colors.HexColor('#F39C12')        # Amber
    DANGER = colors.HexColor('#E74C3C')         # Red
    INFO = colors.HexColor('#3498DB')           # Sky blue

    # === Risk gauge ===
    RISK_LOW = colors.HexColor('#27AE60')
    RISK_MODERATE = colors.HexColor('#F39C12')
    RISK_HIGH = colors.HexColor('#E74C3C')

    # === Decorative ===
    DIVIDER = colors.HexColor('#CBD5E0')        # Light divider line
    BORDER = colors.HexColor('#E2E8F0')         # Card borders

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
        company: str = "Data Science Team"
    ):
        if mode not in ['brief', 'detailed']:
            raise ValueError(f"Mode must be 'brief' or 'detailed', got '{mode}'")

        self.raw_profile = raw_profile
        self.clean_profile = clean_profile
        self.mode = mode
        self.dist_plots = dist_plots or []
        self.heatmap_plot = heatmap_plot
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
        }

        for font_name, filename in font_files.items():
            for font_dir in possible_dirs:
                font_path = font_dir / filename
                if font_path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                        if 'Regular' in font_name:
                            self.font_regular = font_name
                        elif 'ExtraBold' in font_name:
                            self.font_title = font_name
                        elif 'Bold' in font_name:
                            self.font_bold = font_name
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
            fontSize=18,
            leading=24,
            textColor=ReportColors.PRIMARY,
            spaceBefore=16,
            spaceAfter=10,
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
                Paragraph(str(self.best_model_name), self.styles['TableCell']),
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
    # SECTION 2: DATA STORY (Narrative)
    # ==================================================================
    def _add_data_story(self, story):
        story.append(Paragraph("The Data Story", self.styles['SectionHeading']))
        story.append(self._divider())

        n_rows = getattr(self.raw_profile, 'n_rows', 0) if self.raw_profile else 0
        n_cols = getattr(self.raw_profile, 'n_columns', 0) if self.raw_profile else 0
        task = getattr(self.raw_profile, 'task_type', 'unknown') if self.raw_profile else 'unknown'
        n_numeric = len(getattr(self.raw_profile, 'numeric_columns', [])) if self.raw_profile else 0
        n_cat = len(getattr(self.raw_profile, 'categorical_columns', [])) if self.raw_profile else 0
        n_missing_cols = sum(1 for v in getattr(self.raw_profile, 'missing_ratio', {}).values() if v > 0) if self.raw_profile else 0
        dupes = getattr(self.raw_profile, 'duplicate_rows', 0) if self.raw_profile else 0

        # Opening narrative
        story.append(Paragraph(
            f"Your dataset arrived with <b>{n_rows:,}</b> records and <b>{n_cols}</b> features, "
            f"setting up a <b>{task}</b> challenge. OctoLearn's 6-phase intelligent pipeline "
            f"examined every corner of this data to extract maximum insight.",
            self.styles['Narrative']
        ))

        # Feature composition
        story.append(Paragraph(
            f"The features break down into <b>{n_numeric} numeric</b> and <b>{n_cat} categorical</b> columns. "
            f"Of the {n_cols} total features, <b>{n_missing_cols}</b> contained missing values that "
            f"required intelligent imputation strategies.",
            self.styles['Narrative']
        ))

        # Data quality narrative
        if dupes > 0:
            story.append(Paragraph(
                f"OctoLearn detected <b>{dupes} duplicate rows</b> lurking in the data. "
                f"These were automatically removed during the cleaning phase to prevent "
                f"data leakage and inflated model performance.",
                self.styles['Narrative']
            ))

        # Risk narrative
        risk_color_name = "green" if self.risk_score <= 30 else ("amber" if self.risk_score <= 60 else "red")
        story.append(Paragraph(
            f"After deep analysis, OctoLearn assigned a <b>Data Quality Risk Score of "
            f"{self.risk_score}/100</b> ({self.risk_category}). "
            f"This places your dataset in the <b>{risk_color_name} zone</b> -- "
            + ("minimal risk factors were detected. Your data is well-structured and ready for modeling."
               if self.risk_score <= 30 else
               "some quality issues were found that OctoLearn addressed automatically. Review the recommendations for further improvements."
               if self.risk_score <= 60 else
               "significant quality concerns were identified. OctoLearn applied automated fixes, but manual review is strongly recommended."),
            self.styles['Narrative']
        ))

        # Model narrative
        if self.model_benchmarks:
            n_models = len(self.model_benchmarks)
            best_name = self.best_model_name
            best_score = self.model_benchmarks[0].get('score', 0) if self.model_benchmarks else 0
            story.append(Paragraph(
                f"OctoLearn trained and evaluated <b>{n_models} models</b> with hyperparameter "
                f"optimization. After rigorous benchmarking, <b>{best_name}</b> emerged as the "
                f"champion with a score of <b>{best_score:.4f}</b>. "
                f"See the Model Arena section for full comparison.",
                self.styles['Narrative']
            ))
        else:
            story.append(Paragraph(
                "Model training was not performed in this run. "
                "Set <b>train_models=True</b> in ModelingConfig to see model benchmarks.",
                self.styles['Narrative']
            ))



    # ==================================================================
    # SECTION 3: DATA HEALTH DASHBOARD
    # ==================================================================
    def _add_health_dashboard(self, story):
        story.append(Paragraph("Data Health Dashboard", self.styles['SectionHeading']))
        story.append(self._divider())

        # --- Risk Score Card ---
        risk_color = ReportColors.get_risk_color(self.risk_score)
        risk_label = ReportColors.get_risk_label(self.risk_score)

        risk_data = [[
            Paragraph(f'<font size="28" color="{risk_color.hexval()}">{self.risk_score}</font>'
                      f'<font size="12" color="{ReportColors.TEXT_SECONDARY.hexval()}">/100</font>',
                      ParagraphStyle('RiskNum', parent=self.styles['ReportBody'], alignment=TA_CENTER)),
            Paragraph(f'<font size="14" color="{risk_color.hexval()}"><b>{risk_label}</b></font><br/>'
                      f'<font size="9" color="{ReportColors.TEXT_CAPTION.hexval()}">Data Quality Risk Score</font>',
                      ParagraphStyle('RiskLabel', parent=self.styles['ReportBody'],
                                     alignment=TA_LEFT, leading=20)),
        ]]
        risk_table = Table(risk_data, colWidths=[1.5 * inch, 4.5 * inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('BOX', (0, 0), (-1, -1), 1, risk_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.2 * inch))

        # --- Metric Cards Row ---
        n_rows = getattr(self.raw_profile, 'n_rows', 0) if self.raw_profile else 0
        n_cols = getattr(self.raw_profile, 'n_columns', 0) if self.raw_profile else 0
        missing_pct = 0
        if self.raw_profile and hasattr(self.raw_profile, 'missing_ratio'):
            ratios = list(self.raw_profile.missing_ratio.values())
            missing_pct = round(sum(ratios) / max(len(ratios), 1) * 100, 1)
        dupes = getattr(self.raw_profile, 'duplicate_rows', 0) if self.raw_profile else 0
        dupe_pct = round(dupes / max(n_rows, 1) * 100, 1) if n_rows else 0

        cards = [
            [
                Paragraph(f"<b>{n_rows:,}</b>", self.styles['MetricValue']),
                Paragraph(f"<b>{n_cols}</b>", self.styles['MetricValue']),
                Paragraph(f"<b>{missing_pct}%</b>", self.styles['MetricValue']),
                Paragraph(f"<b>{dupe_pct}%</b>", self.styles['MetricValue']),
            ],
            [
                Paragraph("Total Rows", self.styles['MetricLabel']),
                Paragraph("Features", self.styles['MetricLabel']),
                Paragraph("Avg Missing", self.styles['MetricLabel']),
                Paragraph("Duplicates", self.styles['MetricLabel']),
            ],
        ]

        card_table = Table(cards, colWidths=[1.55 * inch] * 4)
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), ReportColors.CARD_BG),
            ('BOX', (0, 0), (-1, -1), 1, ReportColors.BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, ReportColors.BORDER),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, 0), 15),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 1), (-1, 1), 0),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 0.2 * inch))

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
    def _add_before_after(self, story):
        story.append(Paragraph("Data Transformation Journey", self.styles['SectionHeading']))
        story.append(self._divider())

        story.append(Paragraph(
            "OctoLearn's intelligent cleaning pipeline automatically transforms your raw data "
            "into a clean, model-ready dataset. Here is the before-and-after comparison:",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.15 * inch))

        raw = self.raw_profile
        clean = self.clean_profile
        if not raw or not clean:
            story.append(Paragraph("Profile data not available for comparison.", self.styles['BodyText']))

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

        # Cleaning Log narrative
        if self.cleaning_log:
            story.append(Paragraph("What OctoLearn Did", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.05 * inch))

            actions = []
            dupes = self.cleaning_log.get('duplicates_removed', 0)
            if dupes and dupes > 0:
                actions.append(f"Removed <b>{dupes}</b> duplicate rows to prevent data leakage")

            id_cols = self.cleaning_log.get('id_columns_removed', 0)
            if id_cols and id_cols > 0:
                actions.append(f"Dropped <b>{id_cols}</b> ID-like columns (no predictive value)")

            const_cols = self.cleaning_log.get('constant_columns_removed', 0)
            if const_cols and const_cols > 0:
                actions.append(f"Removed <b>{const_cols}</b> constant columns (zero variance)")

            low_var = self.cleaning_log.get('low_variance_removed', 0)
            if low_var and low_var > 0:
                actions.append(f"Dropped <b>{low_var}</b> low-variance features")

            imputed = self.cleaning_log.get('missing_imputed', 0)
            if imputed and imputed > 0:
                actions.append(f"Imputed missing values across <b>{imputed}</b> columns")

            if not actions:
                actions.append("Data was already clean -- no major transformations needed")

            for action in actions:
                story.append(Paragraph(f"  ->  {action}", self.styles['BodyText']))



    # ==================================================================
    # SECTION 5: FEATURE INTELLIGENCE
    # ==================================================================
    def _add_feature_intelligence(self, story):
        if not self.feature_importance:
            return

        story.append(Paragraph("Feature Intelligence", self.styles['SectionHeading']))
        story.append(self._divider())

        story.append(Paragraph(
            "OctoLearn calculated feature importance scores to identify which variables "
            "have the strongest predictive power. Higher scores indicate greater influence "
            "on the model's decisions.",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Sort features
        sorted_feats = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        limit = 8 if self.mode == 'brief' else 15
        top_feats = sorted_feats[:limit]

        if not top_feats:
            return

        max_score = top_feats[0][1] if top_feats else 1

        fi_data = [['Rank', 'Feature', 'Score', 'Relative Strength']]
        for rank, (feat, score) in enumerate(top_feats, 1):
            bar_pct = int(score / max(max_score, 0.001) * 100)
            # Use a text-based bar since reportlab doesn't support inline HTML blocks
            bar_chars = int(bar_pct / 5)
            bar_text = '|' * bar_chars

            bar_color = ReportColors.SUCCESS.hexval() if bar_pct >= 60 else (
                ReportColors.WARNING.hexval() if bar_pct >= 30 else ReportColors.TEXT_CAPTION.hexval()
            )

            fi_data.append([
                Paragraph(f'<b>{rank}</b>', ParagraphStyle('R', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(str(feat), self.styles['TableCell']),
                Paragraph(f'{score:.4f}', ParagraphStyle('S', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'<font color="{bar_color}">{bar_text}</font> {bar_pct}%',
                          self.styles['TableCell']),
            ])

        fi_table = self._styled_table(fi_data, [0.5 * inch, 2.2 * inch, 0.9 * inch, 2.6 * inch])
        story.append(fi_table)

        # SHAP plot
        if self.shap_path and os.path.exists(self.shap_path):
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("SHAP Feature Impact Analysis", self.styles['SubsectionHeading']))
            try:
                # Source (10, 6), Ratio 1.66
                # Width 7.5 inch -> Height 4.5 inch
                shap_img = Image(self.shap_path, width=7.5 * inch, height=4.5 * inch)
                shap_img.hAlign = 'CENTER'
                story.append(shap_img)
                story.append(Paragraph(
                    "SHAP values show how each feature contributes to individual predictions. "
                    "Red indicates features pushing the prediction higher, blue indicates lower.",
                    self.styles['Caption']
                ))
            except Exception:
                pass



    # ==================================================================
    # SECTION 6: MODEL ARENA
    # ==================================================================
    def _add_model_arena(self, story):
        if not self.model_benchmarks:
            return

        story.append(Paragraph("Model Arena", self.styles['SectionHeading']))
        story.append(self._divider())

        n_models = len(self.model_benchmarks)
        best = self.model_benchmarks[0] if self.model_benchmarks else {}
        best_name = best.get('model', self.best_model_name)
        best_score = best.get('score', 0)

        story.append(Paragraph(
            f"OctoLearn trained <b>{n_models} models</b> with automated hyperparameter tuning. "
            f"Each model was evaluated on a held-out test set to ensure unbiased comparison. "
            f"The champion is <b>{best_name}</b> "
            f"with a primary score of <b>{best_score:.4f}</b>.",
            self.styles['Narrative']
        ))
        story.append(Spacer(1, 0.15 * inch))

        # Champion card
        champion_data = [[
            Paragraph(f'<font size="10" color="{ReportColors.PRIMARY.hexval()}">CHAMPION</font><br/>'
                      f'<font size="16" color="{ReportColors.PRIMARY.hexval()}"><b>{best_name}</b></font>',
                      ParagraphStyle('Champ', parent=self.styles['BodyText'], alignment=TA_CENTER, leading=22)),
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

        # Benchmark table
        # Access the correct keys: 'model', 'score', 'metrics.accuracy', 'metrics.precision', 'metrics.f1'
        header = ['#', 'Model', 'Score', 'Accuracy', 'Precision', 'F1', 'Recall']
        bench_data = [header]

        best_row_idx = None
        limit = 5 if self.mode == 'brief' else 10
        for idx, bm in enumerate(self.model_benchmarks[:limit], 1):
            model_name = bm.get('model', 'Unknown')
            score = bm.get('score', 0)
            metrics = bm.get('metrics', {})
            accuracy = metrics.get('accuracy', score)
            precision = metrics.get('precision', 0)
            f1 = metrics.get('f1', 0)
            recall = metrics.get('recall', 0)

            if model_name == best_name or (best_row_idx is None and idx == 1):
                best_row_idx = idx

            bench_data.append([
                Paragraph(f'{idx}', ParagraphStyle('Idx', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'<b>{model_name}</b>' if idx == 1 else model_name, self.styles['TableCell']),
                Paragraph(f'{score:.4f}', ParagraphStyle('Sc', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{accuracy:.4f}', ParagraphStyle('Ac', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{precision:.4f}', ParagraphStyle('Pr', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{f1:.4f}', ParagraphStyle('F1', parent=self.styles['TableCell'], alignment=TA_CENTER)),
                Paragraph(f'{recall:.4f}', ParagraphStyle('Rc', parent=self.styles['TableCell'], alignment=TA_CENTER)),
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

        story.append(Paragraph("Actionable Recommendations", self.styles['SectionHeading']))
        story.append(self._divider())

        story.append(Paragraph(
            "Based on the comprehensive analysis, OctoLearn has generated the following "
            "prioritized recommendations to further improve your data and models:",
            self.styles['BodyText']
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
            story.append(Paragraph("No specific recommendations at this time.", self.styles['BodyText']))

            return

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
                ParagraphStyle('PriorityHeader', parent=self.styles['BodyText'],
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
        has_visuals = bool(self.dist_plots) or bool(self.heatmap_plot)
        if not has_visuals:
            return

        story.append(Paragraph("Visual Insights", self.styles['SectionHeading']))
        story.append(self._divider())

        story.append(Paragraph(
            "The following visualizations provide deeper insight into feature distributions "
            "and correlations within your cleaned dataset.",
            self.styles['BodyText']
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Correlation heatmap
        if self.heatmap_plot and os.path.exists(self.heatmap_plot):
            story.append(Paragraph("Correlation Matrix", self.styles['SubsectionHeading']))
            try:
                # Source is (10, 8), Ratio 1.25
                # Page width ~7.6 inch available.
                # Height should be 7.6 / 1.25 = ~6 inch.
                hm_img = Image(self.heatmap_plot, width=6.0 * inch, height=4.8 * inch)
                hm_img.hAlign = 'CENTER'
                story.append(hm_img)
                story.append(Paragraph(
                    "Heatmap shows pairwise feature correlations. Strong correlations (near +/- 1.0) "
                    "may indicate redundant features or multicollinearity.",
                    self.styles['Caption']
                ))
                story.append(Spacer(1, 0.3 * inch))
            except Exception:
                pass

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

        story.append(Paragraph("Advanced Analysis", self.styles['SectionHeading']))
        story.append(self._divider())

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
                        f"  ->  <b>{inter[0]}</b> <-> <b>{inter[1]}</b>",
                        self.styles['BodyText']
                    ))



    # ==================================================================
    # HEADER / FOOTER CALLBACKS
    # ==================================================================
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
        """Generate the complete storytelling PDF report."""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"octolearn_report_{timestamp}.pdf"

        try:
            doc = SimpleDocTemplate(
                filename,
                pagesize=A4,
                rightMargin=20, leftMargin=20,
                topMargin=30, bottomMargin=30,
                title=self.title,
                author=self.author
            )

            story = []

            # Build the narrative report
            # 1. Cover (Keep separate)
            self._add_cover_page(story)
            story.append(PageBreak()) # Cover gets its own page

            # 2. Executive Summary - Recommendations (Moved to Top)
            self._add_recommendations(story)
            story.append(Spacer(1, 0.4*inch))
            
            # Continuous Flow Section
            self._add_data_story(story)
            story.append(Spacer(1, 0.4*inch))
            
            self._add_health_dashboard(story)
            story.append(Spacer(1, 0.4*inch))
            
            self._add_before_after(story)
            story.append(Spacer(1, 0.4*inch))
            
            self._add_feature_intelligence(story)
            story.append(Spacer(1, 0.4*inch))
            
            self._add_model_arena(story)
            story.append(Spacer(1, 0.4*inch))
            
            
            # Recommendations (Action Plan) - Moved to Top
            # self._add_recommendations(story) # Removed from here
            # story.append(Spacer(1, 0.4*inch))

            # Advanced Analysis
            if self.mode == 'detailed':
                self._add_analysis_details(story)
                story.append(Spacer(1, 0.4*inch))

            # Visuals (Appendix-like but continuous)
            self._add_visual_insights(story)

            # Build PDF with standard callbacks
            doc.build(
                story,
                onFirstPage=lambda c, d: None, # No header/footer on cover
                onLaterPages=self._header_footer_callback
            )
            logger.info(f"[OK] Report generated: {filename}")
            return filename

        except Exception as e:
            logger.error(f"[FAIL] PDF generation failed: {str(e)}")
            raise RuntimeError(f"PDF generation failed: {str(e)}")


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