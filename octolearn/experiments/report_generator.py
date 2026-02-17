"""
Report Generator Module - Sequential Storytelling

Generates a professional PDF report summarizing the AutoML pipeline results
with a logical flow: Profile -> Risk -> Cleaning -> Analysis -> Models.
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
COLOR_SUCCESS = colors.HexColor('#27AE60')

IMG_WIDTH_LARGE = 500
IMG_HEIGHT_LARGE = 200
IMG_WIDTH_MED = 450
IMG_HEIGHT_MED = 200


class ReportGenerator:
    """
    Generates a PDF report using ReportLab.
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
        best_model_name: Optional[str] = None,
        detail_level: str = 'detailed',
        logo_path: Optional[str] = None,
        cleaning_log: Optional[Dict] = None
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
        self.detail_level = detail_level
        self.logo_path = logo_path
        self.cleaning_log = cleaning_log or {}

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
            spaceAfter=10,
            borderPadding=5,
            borderWidth=0,
            borderColor=COLOR_PRIMARY,
            borderRadius=5
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
        """Builds the PDF report with a logical storyline."""
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

        # --- CHAPTER 0: The Cover ---
        self._add_cover_page(story)
        story.append(PageBreak())

        # --- CHAPTER 1: The Situation (Introduction) ---
        self._add_introduction(story)
        self._add_data_profile(story)
        
        # --- CHAPTER 2: The Problem (Risk Analysis) ---
        self._add_risk_analysis(story)
        
        # --- CHAPTER 3: The Fix (Cleaning) - Only Detailed ---
        if self.detail_level == 'detailed' and self.cleaning_log:
             story.append(PageBreak())
             self._add_cleaning_actions(story)

        story.append(PageBreak())

        # --- CHAPTER 4: The Investigation (Visual Analysis) ---
        self._add_visual_investigation(story)
        
        story.append(PageBreak())

        # --- CHAPTER 5: The Conclusion (Models & Recommendations) ---
        self._add_model_results(story)
        self._add_recommendations(story)

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
        
    def _add_cover_page(self, story):
        story.append(Spacer(1, 40))
        if self.logo_path and os.path.exists(self.logo_path):
             im = Image(self.logo_path, width=100, height=100)
             story.append(im)
             story.append(Spacer(1, 20))
             
        story.append(Paragraph("Octolearn Analysis Report", self.styles['OctoTitle']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['OctoNormal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary right on cover for impact
        if self.risk_score is not None:
            health_score = max(0, 100 - self.risk_score)
            color = 'green' if health_score > 70 else 'orange' if health_score > 40 else 'red'
            story.append(Paragraph(f"<b>Dataset Health Score:</b> <font color='{color}'>{health_score}/100</font>", self.styles['Heading3']))
            
        if self.best_model_name:
             story.append(Paragraph(f"<b>Best Model:</b> {self.best_model_name}", self.styles['Heading3']))

    def _add_introduction(self, story):
        story.append(Paragraph("1. Introduction", self.styles['OctoHeading']))
        story.append(Paragraph("This report provides a comprehensive analysis of your dataset, highlighting structural issues, key insights, and predictive modeling results.", self.styles['OctoNormal']))

    def _add_data_profile(self, story):
        rows, cols = self.profile.shape
        txt = f"Your dataset contains <b>{rows:,} rows</b> and <b>{cols} columns</b>. The target task has been identified as a <b>{self.profile.task_type.title()}</b> problem."
        story.append(Paragraph(txt, self.styles['OctoNormal']))
        story.append(Spacer(1, 10))

    def _add_risk_analysis(self, story):
        story.append(Paragraph("2. Risk Assessment", self.styles['OctoHeading']))
        story.append(Paragraph(f"Risk Category: <b>{self.risk_category}</b>", self.styles['OctoNormal']))
        
        if self.risk_factors:
            story.append(Paragraph("Key issues identified:", self.styles['OctoNormal']))
            for k, v in self.risk_factors.items():
                story.append(Paragraph(f"• {v}", self.styles['OctoNormal']))
        else:
             story.append(Paragraph("No significant structural risks were detected.", self.styles['OctoNormal']))

    def _add_cleaning_actions(self, story):
        story.append(Paragraph("3. Automated Cleaning Actions", self.styles['OctoHeading']))
        
        log = self.cleaning_log.get('train', {})
        actions = []
        if log.get('duplicates_removed'): actions.append(f"Removed {log['duplicates_removed']} duplicate rows.")
        if log.get('id_columns_removed'): actions.append(f"Dropped ID columns: {log['id_columns_removed']}")
        if log.get('constant_columns_removed'): actions.append(f"Dropped constant columns: {log['constant_columns_removed']}")
        
        if actions:
            for act in actions:
                story.append(Paragraph(f"✓ {act}", self.styles['OctoNormal']))
        else:
            story.append(Paragraph("No major cleaning actions were required.", self.styles['OctoNormal']))
            
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Final Feature Count:</b> {len(log.get('output_columns', []))} columns.", self.styles['OctoNormal']))

    def _add_visual_investigation(self, story):
        story.append(Paragraph("4. Visual Investigation", self.styles['OctoHeading']))
        story.append(Paragraph("We analyzed the most significant features based on their relationship with the target variable.", self.styles['OctoNormal']))
        story.append(Spacer(1, 10))

        def add_img(path, width, height, title=None):
            if path and os.path.exists(path):
                if title:
                    story.append(Paragraph(title, self.styles['Heading3']))
                # Scaling image to fit if needed
                img = Image(path, width=width, height=height)
                story.append(img)
                story.append(Spacer(1, 15))

        # Correlation Heatmap (General Overview)
        if self.heatmap_plot:
            add_img(self.heatmap_plot, IMG_WIDTH_MED, 300, "Correlation Overview")

        # Feature Dashboards (Deep Dive)
        if self.dist_plots:
            story.append(Paragraph("<b>Top Feature Analysis</b>", self.styles['Heading3']))
            for path in self.dist_plots:
                add_img(path, IMG_WIDTH_LARGE, 160)

        # SHAP (Explainability)
        if self.shap_path:
            add_img(self.shap_path, IMG_WIDTH_LARGE, IMG_HEIGHT_LARGE, "Model Decision Explanation (SHAP)")

    def _add_model_results(self, story):
        story.append(Paragraph("5. Model Performance", self.styles['OctoHeading']))

        if not self.model_benchmarks:
            story.append(Paragraph("No models were trained.", self.styles['OctoNormal']))
            return

        story.append(Paragraph(f"The champion model is <b>{self.best_model_name}</b>.", self.styles['OctoNormal']))

        data = [['Rank', 'Model Name', 'Score']]

        for idx, bench in enumerate(self.model_benchmarks):
            data.append([
                str(idx + 1),
                bench.get('model', 'Unknown').replace('_', ' ').title(),
                f"{bench.get('score', 0):.4f}"
            ])

        t = Table(data, colWidths=[50, 200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

    def _add_recommendations(self, story):
        story.append(Paragraph("6. Recommendations", self.styles['OctoHeading']))

        if self.recommendations:
            for rec in self.recommendations[:5]:
                story.append(Paragraph(f"• {rec}", self.styles['OctoNormal']))