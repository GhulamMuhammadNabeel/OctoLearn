"""
Report Generator Module

Generates a professional PDF report summarizing the AutoML pipeline results.
Includes Data Profile, Data Quality, Recommendations, Model Benchmarks,
and Feature Importance Summary.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)

# --- CONSTANTS ---
COLOR_PRIMARY = colors.HexColor('#2E86C1')
COLOR_ACCENT = colors.HexColor('#D6EAF8')
COLOR_DARK = colors.HexColor('#1B4F72')
COLOR_TEXT = colors.HexColor('#2C3E50')

IMG_WIDTH_LARGE = 450
IMG_HEIGHT_LARGE = 250
IMG_WIDTH_MED = 400
IMG_HEIGHT_MED = 200


class ReportGenerator:
    """
    Generates a PDF report using ReportLab.
    Theme: Professional White/Blue (Classic).
    """

    def __init__(
        self,
        profile,
        dist_plots: Optional[List[str]] = None,
        heatmap_plot: Optional[str] = None,
        recommendations: Optional[List[str]] = None,
        risk_score: Optional[int] = None,
        risk_category: Optional[str] = None,
        risk_factors: Optional[Dict[str, Any]] = None,
        preprocessing_suggestions: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        shap_path: Optional[str] = None,
        model_benchmarks: Optional[List[Dict[str, Any]]] = None,
        best_model_name: Optional[str] = None
    ):
        self.profile = profile
        self.dist_plots = dist_plots or []
        self.heatmap_plot = heatmap_plot
        self.recommendations = recommendations or []
        self.risk_score = risk_score
        self.risk_category = risk_category
        self.risk_factors = risk_factors or {}
        self.preprocessing_suggestions = preprocessing_suggestions or {}
        self.feature_importance = feature_importance or {}
        self.shap_path = shap_path
        self.model_benchmarks = model_benchmarks or []
        self.best_model_name = best_model_name

        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _register_fonts(self):
        """Attempts to register Shantell Sans, falls back to Helvetica."""
        font_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        self.font_regular = "Helvetica"
        self.font_bold = "Helvetica-Bold"
        self.font_italic = "Helvetica-Oblique"
        self.font_title = "Helvetica-Bold"

        try:
            if os.path.exists(os.path.join(font_dir, 'ShantellSans-Regular.ttf')):
                pdfmetrics.registerFont(TTFont('ShantellSans-Regular', os.path.join(font_dir, 'ShantellSans-Regular.ttf')))
                pdfmetrics.registerFont(TTFont('ShantellSans-Bold', os.path.join(font_dir, 'ShantellSans-Bold.ttf')))
                pdfmetrics.registerFont(TTFont('ShantellSans-Italic', os.path.join(font_dir, 'ShantellSans-Italic.ttf')))
                pdfmetrics.registerFont(TTFont('ShantellSans-ExtraBold', os.path.join(font_dir, 'ShantellSans-ExtraBold.ttf')))
                self.font_regular = 'ShantellSans-Regular'
                self.font_bold = 'ShantellSans-Bold'
                self.font_italic = 'ShantellSans-Italic'
                self.font_title = 'ShantellSans-ExtraBold'
        except Exception as e:
            logger.warning(f"Could not load custom fonts: {e}. Using default Helvetica.")

    def _create_custom_styles(self):
        """Defines the visual theme."""
        self.styles.add(ParagraphStyle(
            name='OctoTitle',
            parent=self.styles['Heading1'],
            fontName=self.font_title,
            fontSize=26,
            leading=32,
            textColor=COLOR_DARK,
            spaceAfter=20,
            alignment=1
        ))

        self.styles.add(ParagraphStyle(
            name='OctoHeading',
            parent=self.styles['Heading2'],
            fontName=self.font_bold,
            fontSize=18,
            leading=22,
            textColor=COLOR_PRIMARY,
            spaceBefore=15,
            spaceAfter=10
        ))

        self.styles.add(ParagraphStyle(
            name='OctoNormal',
            parent=self.styles['Normal'],
            fontName=self.font_regular,
            fontSize=11,
            leading=14,
            textColor=COLOR_TEXT,
            spaceAfter=8
        ))

        self.styles.add(ParagraphStyle(
            name='OctoTableText',
            parent=self.styles['Normal'],
            fontName=self.font_regular,
            fontSize=9,
            leading=11,
            textColor=colors.black
        ))

    def generate(self, filename: Optional[str] = None) -> str:
        """Builds the PDF report."""
        self._validate_profile()

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"octolearn_report_{timestamp}.pdf"

        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )

        story = []

        story.append(Spacer(1, 40))
        story.append(Paragraph("Octolearn Analysis Report", self.styles['OctoTitle']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['OctoNormal']))
        story.append(Spacer(1, 20))

        self._add_executive_summary(story)
        self._add_data_profile(story)

        story.append(PageBreak())
        self._add_model_benchmarks(story)

        self._add_feature_importance(story)
        self._add_recommendations(story)

        story.append(PageBreak())
        self._add_visualizations(story)

        try:
            doc.build(story, onFirstPage=self._footer_page, onLaterPages=self._footer_page)
            return filename
        except Exception as e:
            logger.error(f"Failed to build PDF: {e}")
            raise RuntimeError(f"PDF generation failed: {e}")

    def _validate_profile(self):
        """Ensures profile has required attributes."""
        required_attrs = ["shape", "task_type", "numeric_columns", "categorical_columns"]
        for attr in required_attrs:
            if not hasattr(self.profile, attr):
                raise ValueError(f"Profile object missing required attribute: {attr}")

    def _footer_page(self, canvas, doc):
        """Adds footer with page number."""
        canvas.saveState()
        canvas.setFont(self.font_regular, 9)
        canvas.setFillColor(colors.grey)
        page_num = canvas.getPageNumber()
        canvas.drawRightString(A4[0] - 50, 30, f"Octolearn AutoML Report | Page {page_num}")
        canvas.restoreState()

    def _add_executive_summary(self, story):
        story.append(Paragraph("Executive Summary", self.styles['OctoHeading']))

        if self.risk_score is not None:
            health_score = max(0, 100 - self.risk_score)
            story.append(Paragraph(f"Dataset Health Score: {health_score}/100", self.styles['Heading3']))

    def _add_data_profile(self, story):
        story.append(Paragraph("Dataset Profile", self.styles['OctoHeading']))
        rows, cols = self.profile.shape
        txt = f"<b>Rows:</b> {rows:,} | <b>Columns:</b> {cols} | <b>Task Type:</b> {self.profile.task_type.title()}"
        story.append(Paragraph(txt, self.styles['OctoNormal']))
        story.append(Spacer(1, 10))

    # =========================
    # IMPROVEMENT #1
    # =========================
    def _add_model_benchmarks(self, story):
        story.append(Paragraph("Model Performance Leaderboard", self.styles['OctoHeading']))

        if not self.model_benchmarks:
            story.append(Paragraph("No models were trained or benchmarks are unavailable.", self.styles['OctoNormal']))
            story.append(Spacer(1, 15))
            return

        if self.best_model_name:
            story.append(Paragraph(f"Best Performing Model: <b>{self.best_model_name}</b>", self.styles['OctoNormal']))
            story.append(Spacer(1, 10))

        data = [['Rank', 'Model Name', 'Score', 'Best Params']]

        for idx, bench in enumerate(self.model_benchmarks):
            params_para = Paragraph(str(bench.get('params', {})), self.styles['OctoTableText'])
            data.append([
                str(idx + 1),
                bench.get('model', 'Unknown').replace('_', ' ').title(),
                f"{bench.get('score', 0):.4f}",
                params_para
            ])

        t = Table(data, colWidths=[40, 120, 80, 220])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

    # =========================
    # IMPROVEMENT #2
    # =========================
    def _add_feature_importance(self, story):
        story.append(Paragraph("Feature Importance Summary", self.styles['OctoHeading']))

        if not self.feature_importance:
            story.append(Paragraph("Feature importance could not be computed.", self.styles['OctoNormal']))
            story.append(Spacer(1, 15))
            return

        data = [['Feature', 'Importance']]
        for feat, score in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            data.append([feat, f"{score:.4f}"])

        t = Table(data, colWidths=[250, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COLOR_ACCENT),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), self.font_bold),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

    def _add_recommendations(self, story):
        story.append(Paragraph("Strategic Recommendations", self.styles['OctoHeading']))

        if self.preprocessing_suggestions:
            for cat, details in self.preprocessing_suggestions.items():
                story.append(Paragraph(f"{cat.title()}: {details}", self.styles['OctoNormal']))

        if self.recommendations:
            for rec in self.recommendations[:5]:
                story.append(Paragraph(f"- {rec}", self.styles['OctoNormal']))

        story.append(Spacer(1, 20))

    def _add_visualizations(self, story):
        story.append(Paragraph("Visual Analysis", self.styles['OctoHeading']))

        def add_img(path, width, height, title=None):
            if path and os.path.exists(path):
                if title:
                    story.append(Paragraph(title, self.styles['Heading3']))
                img = Image(path, width=width, height=height)
                story.append(img)
                story.append(Spacer(1, 15))

        if self.shap_path:
            add_img(self.shap_path, IMG_WIDTH_LARGE, IMG_HEIGHT_LARGE, "Model Explainability (SHAP)")

        if self.heatmap_plot:
            add_img(self.heatmap_plot, IMG_WIDTH_MED, 300, "Correlation Matrix")

        if self.dist_plots:
            for path in self.dist_plots[:4]:
                add_img(path, IMG_WIDTH_MED, IMG_HEIGHT_MED)
