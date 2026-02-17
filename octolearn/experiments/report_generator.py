"""
Enhanced Report Generator Module - Advanced PDF Generation with Dual Modes

Generates professional PDF reports with two operational modes:
1. BRIEF MODE (5-7 pages) - Executive summary with top insights
2. DETAILED MODE (15-20 pages) - Comprehensive analysis with all metrics

Features:
- Logo watermark/background with opacity control
- Color-coded risk levels (green/yellow/red)
- Professional typography and formatting
- Responsive charts and visualizations
- Executive summary cards
- Model benchmarking tables
- Actionable recommendations

Author: OctoLearn Development Team
Version: 0.7.4 (Patched)
License: MIT
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from io import BytesIO
import warnings

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, 
    PageBreak, KeepTogether, PageTemplate, Frame, BaseDocTemplate
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Safe import with fallback
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

warnings.filterwarnings("ignore")

# Try to import helpers, with fallback for standalone usage
try:
    from ..utils.helpers import setup_logger
except ImportError:
    import logging
    def setup_logger(name):
        return logging.getLogger(name)

logger = setup_logger(__name__)


# ============================================================================
# COLOR PALETTE - Professional Theme
# ============================================================================

class ReportColors:
    """Color scheme for reports with accessibility in mind."""
    # Primary colors
    PRIMARY = colors.HexColor('#2E86C1')      # Professional blue
    ACCENT_RED = colors.HexColor('#E74C3C')    # Warning/highlight red
    ACCENT_ORANGE = colors.HexColor('#F39C12')  # Caution orange
    ACCENT_GREEN = colors.HexColor('#27AE60')   # Success green
    
    # Backgrounds
    DARK_BG = colors.HexColor('#1B1B1B')      # Dark background
    LIGHT_BG = colors.HexColor('#F8F9F9')     # Light background
    WHITE = colors.HexColor('#FFFFFF')
    
    # Text colors
    TEXT_DARK = colors.HexColor('#2C3E50')    # Dark text
    TEXT_LIGHT = colors.HexColor('#ECF0F1')   # Light text
    
    # Risk levels
    RISK_LOW = colors.HexColor('#27AE60')     # Green
    RISK_MODERATE = colors.HexColor('#F39C12')  # Orange
    RISK_HIGH = colors.HexColor('#E74C3C')    # Red
    
    @staticmethod
    def get_risk_color(score: int) -> colors.Color:
        """Get color based on risk score (0-100)."""
        if score <= 30:
            return ReportColors.RISK_LOW
        elif score <= 60:
            return ReportColors.RISK_MODERATE
        else:
            return ReportColors.RISK_HIGH


# ============================================================================
# ENHANCED REPORT GENERATOR - MAIN CLASS
# ============================================================================

class ReportGenerator:
    """
    Advanced PDF Report Generator with dual-mode capability.
    
    Modes:
    - 'brief': 5-7 page executive summary
    - 'detailed': 15-20 page comprehensive analysis
    
    Features:
    - Logo watermark background
    - Color-coded insights
    - Professional formatting
    - Responsive layouts
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
        title: str = "OctoLearn Intelligence Report",
        author: str = "OctoLearn AutoML",
        company: str = "Data Science Team"
    ):
        """
        Initialize the enhanced report generator.
        """
        # Validate mode
        if mode not in ['brief', 'detailed']:
            raise ValueError(f"Mode must be 'brief' or 'detailed', got '{mode}'")
        
        self.raw_profile = raw_profile
        self.clean_profile = clean_profile
        self.mode = mode
        self.dist_plots = dist_plots or []
        self.heatmap_plot = heatmap_plot
        self.recommendations = recommendations or []
        self.risk_score = risk_score or 50
        self.risk_category = risk_category or "Moderate"
        self.risk_factors = risk_factors or {}
        self.preprocessing_suggestions = preprocessing_suggestions or {}
        self.feature_importance = feature_importance or {}
        self.shap_path = shap_path
        self.model_benchmarks = model_benchmarks or []
        self.best_model_name = best_model_name or "N/A"
        self.logo_path = logo_path
        self.cleaning_log = cleaning_log or {}
        self.title = title
        self.author = author
        self.company = company
        
        # Initialize
        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
        logger.info(f"Report initialized in {mode.upper()} mode")

    def _register_fonts(self):
        """Register custom fonts with safe fallback."""
        font_paths = {
            'ShantellSans-Regular': ['fonts/ShantellSans-Regular.ttf', 'ShantellSans-Regular.ttf'],
            'ShantellSans-Bold': ['fonts/ShantellSans-Bold.ttf', 'ShantellSans-Bold.ttf'],
        }
        
        self.font_regular = 'Helvetica'
        self.font_bold = 'Helvetica-Bold'
        self.font_title = 'Helvetica-Bold'
        
        for font_name, possible_paths in font_paths.items():
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        pdfmetrics.registerFont(TTFont(font_name, path))
                        if 'Regular' in font_name:
                            self.font_regular = font_name
                        elif 'Bold' in font_name:
                            self.font_bold = font_name
                            self.font_title = font_name
                        logger.debug(f"Registered font: {font_name}")
                        break
                except Exception as e:
                    logger.debug(f"Could not register {font_name}: {e}")
                    continue

    def _create_custom_styles(self):
        """Create professional paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontName=self.font_title,
            fontSize=28,
            leading=34,
            textColor=ReportColors.TEXT_DARK,
            spaceAfter=20,
            alignment=TA_CENTER,
            bold=True
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontName=self.font_bold,
            fontSize=16,
            leading=20,
            textColor=ReportColors.PRIMARY,
            spaceBefore=15,
            spaceAfter=10,
            borderPadding=10,
            borderWidth=2,
            borderColor=ReportColors.PRIMARY
        ))
        
        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontName=self.font_bold,
            fontSize=13,
            leading=16,
            textColor=ReportColors.TEXT_DARK,
            spaceBefore=10,
            spaceAfter=8
        ))
        
        # Normal text
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontName=self.font_regular,
            fontSize=11,
            leading=14,
            textColor=ReportColors.TEXT_DARK,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        ))
        
        # Table text
        self.styles.add(ParagraphStyle(
            name='TableText',
            parent=self.styles['Normal'],
            fontName=self.font_regular,
            fontSize=9,
            leading=11,
            textColor=colors.black
        ))
        
        # Highlight/insight text
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['Normal'],
            fontName=self.font_bold,
            fontSize=11,
            leading=14,
            textColor=ReportColors.ACCENT_RED,
            spaceAfter=8
        ))

    def _add_watermark_to_page(self, canvas_obj: canvas.Canvas, logo_path: str, opacity: float = 0.15):
        """Add logo as watermark to page background."""
        if not logo_path or not os.path.exists(logo_path):
            return
        
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for watermark")
            return
        
        try:
            # Open and resize logo
            img = PILImage.open(logo_path)
            img.thumbnail((400, 400), PILImage.Resampling.LANCZOS)
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Apply opacity
            alpha = img.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            img.putalpha(alpha)
            
            # Save to BytesIO
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Draw on canvas
            canvas_obj.drawImage(
                img_bytes,
                A4[0] / 2 - 150, A4[1] / 2 - 150,
                width=300, height=300,
                mask=[255, 255, 255, 255]
            )
        except Exception as e:
            logger.warning(f"Could not add watermark: {e}")

    def _add_cover_page(self, story: list):
        """Add professional cover page."""
        story.append(Spacer(1, 1 * inch))
        
        # Title
        story.append(Paragraph(
            self.title,
            self.styles['ReportTitle']
        ))
        story.append(Spacer(1, 0.3 * inch))
        
        # Subtitle
        story.append(Paragraph(
            f"Comprehensive Data Intelligence Report | {self.mode.upper()} MODE",
            self.styles['SubsectionHeading']
        ))
        story.append(Spacer(1, 0.5 * inch))
        
        # Risk Score Card
        risk_color = ReportColors.get_risk_color(self.risk_score)
        risk_table_data = [
            [
                Paragraph(
                    f"<b>RISK SCORE</b><br/><font size=24>{self.risk_score}/100</font><br/>{self.risk_category}",
                    self.styles['ReportBody']
                )
            ]
        ]
        risk_table = Table(risk_table_data, colWidths=[6 * inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), risk_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), self.font_bold),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('PADDING', (0, 0), (-1, -1), 20),
            ('ROUNDED_CORNERS', (0, 0), (-1, -1), 10),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Metadata
        metadata = f"""
        <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br/>
        <b>Company:</b> {self.company}<br/>
        <b>Author:</b> {self.author}<br/>
        <b>Dataset Rows:</b> {getattr(self.raw_profile, 'n_rows', 'N/A')}<br/>
        <b>Features Analyzed:</b> {getattr(self.raw_profile, 'n_cols', 'N/A')}
        """
        story.append(Paragraph(metadata, self.styles['ReportBody']))

    def _add_executive_summary(self, story: list):
        """Add executive summary section."""
        story.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        if self.mode == 'detailed':
            summary_text = f"""
            This comprehensive intelligence report analyzes your dataset across four dimensions: 
            <b>Data Quality</b>, <b>Statistical Properties</b>, <b>Data Cleaning Recommendations</b>, 
            and <b>Feature Engineering Insights</b>. The analysis identified {len(self.risk_factors)} 
            primary risk factors and generated {len(self.preprocessing_suggestions)} preprocessing recommendations.
            <br/><br/>
            <b>Key Metrics:</b> Your dataset contains {getattr(self.raw_profile, 'n_rows', 'N/A')} 
            records across {getattr(self.raw_profile, 'n_cols', 'N/A')} features. 
            {getattr(self.raw_profile, 'n_missing', 'Unknown')} values require imputation strategy. 
            The data quality risk score is {self.risk_score}/100 ({self.risk_category} Risk).
            """
        else:  # brief mode
            summary_text = f"""
            <b>Overview:</b> Analysis of {getattr(self.raw_profile, 'n_rows', 'N/A')} records 
            across {getattr(self.raw_profile, 'n_cols', 'N/A')} features with risk score {self.risk_score}/100.
            """
        
        story.append(Paragraph(summary_text, self.styles['ReportBody']))

    def _add_risk_analysis(self, story: list):
        """Add detailed risk analysis section with robust type handling."""
        story.append(PageBreak())
        story.append(Paragraph("Risk Assessment", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        if self.risk_factors:
            risk_data = [['Factor', 'Impact', 'Severity']]
            
            # Helper to safely get score for sorting
            def get_sort_score(item):
                val = item[1]
                if isinstance(val, dict):
                    return val.get('score', 0)
                if isinstance(val, (int, float)):
                    return val
                return 0

            # Sort by score descending
            sorted_factors = sorted(
                self.risk_factors.items(), 
                key=get_sort_score, 
                reverse=True
            )

            for factor, data in sorted_factors[:5]:
                # Extract score and description safely
                if isinstance(data, dict):
                    score = data.get('score', 0)
                    description = data.get('description', str(data))
                elif isinstance(data, (int, float)):
                    score = data
                    description = str(data)
                else:
                    # Fallback for plain strings (legacy format)
                    score = 0
                    description = str(data)

                severity_val = score * 10
                severity_color = ReportColors.get_risk_color(severity_val)
                
                risk_data.append([
                    Paragraph(str(factor).replace('_', ' ').title(), self.styles['TableText']),
                    Paragraph(description, self.styles['TableText']),
                    Paragraph('●', ParagraphStyle(
                        'SeverityIndicator',
                        parent=self.styles['TableText'],
                        textColor=severity_color,
                        alignment=TA_CENTER,
                        fontSize=14
                    ))
                ])
            
            risk_table = Table(risk_data, colWidths=[1.5 * inch, 3.5 * inch, 1 * inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ReportColors.PRIMARY),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 1), (2, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), self.font_bold),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ReportColors.LIGHT_BG])
            ]))
            story.append(risk_table)
        else:
            story.append(Paragraph("No significant risk factors identified.", self.styles['ReportBody']))

    def _add_recommendations(self, story: list):
        """Add actionable recommendations (Handles both List and Dict inputs)."""
        if not self.recommendations:
            return
        
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Flatten recommendations if they are in dictionary format (from RecommendationEngine)
        rec_list = []
        if isinstance(self.recommendations, dict):
            # Prioritize based on severity keys
            for priority in ['critical', 'high', 'medium', 'low', 'informational']:
                if priority in self.recommendations:
                    rec_list.extend(self.recommendations[priority])
            
            # If there are keys not in the standard list, add them too (just in case)
            known_keys = {'critical', 'high', 'medium', 'low', 'informational'}
            for k, v in self.recommendations.items():
                if k not in known_keys and isinstance(v, list):
                    rec_list.extend(v)
        elif isinstance(self.recommendations, list):
            rec_list = self.recommendations
        else:
            rec_list = []
        
        if not rec_list:
            return

        limit = 5 if self.mode == 'brief' else 10
        
        for idx, rec in enumerate(rec_list[:limit], 1):
            story.append(Paragraph(
                f"<b>{idx}. {rec}</b>",
                self.styles['ReportBody']
            ))
            story.append(Spacer(1, 0.1 * inch))

    def _add_model_results(self, story: list):
        """Add model training results if available."""
        if not self.model_benchmarks:
            return
        
        story.append(PageBreak())
        story.append(Paragraph("Model Benchmarks", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Model comparison table
        model_data = [['Model', 'Accuracy', 'Precision', 'F1-Score', 'Train Time']]
        for benchmark in self.model_benchmarks[:5 if self.mode == 'brief' else 10]:
            model_data.append([
                Paragraph(benchmark.get('model_name', 'N/A'), self.styles['TableText']),
                Paragraph(f"{benchmark.get('accuracy', 0):.4f}", self.styles['TableText']),
                Paragraph(f"{benchmark.get('precision', 0):.4f}", self.styles['TableText']),
                Paragraph(f"{benchmark.get('f1', 0):.4f}", self.styles['TableText']),
                Paragraph(f"{benchmark.get('train_time', 0):.2f}s", self.styles['TableText']),
            ])
        
        model_table = Table(model_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.3*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), ReportColors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), self.font_bold),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ReportColors.LIGHT_BG])
        ]))
        story.append(model_table)

    def generate(self, filename: Optional[str] = None) -> str:
        """
        Generate the complete PDF report.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"octolearn_report_{self.mode}_{timestamp}.pdf"
        
        try:
            doc = SimpleDocTemplate(
                filename,
                pagesize=A4,
                rightMargin=50, leftMargin=50,
                topMargin=50, bottomMargin=50,
                title=self.title,
                author=self.author
            )
            
            story = []
            
            # Build report based on mode
            self._add_cover_page(story)
            story.append(PageBreak())
            
            self._add_executive_summary(story)
            self._add_risk_analysis(story)
            self._add_recommendations(story)
            self._add_model_results(story)
            
            # Build PDF
            doc.build(story)
            logger.info(f"Report generated: {filename} (Mode: {self.mode})")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise RuntimeError(f"PDF generation failed: {e}")
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

    def create_brief_report(
        raw_profile: Any,
        clean_profile: Any,
        **kwargs
    ) -> str:
        """
        Convenience function to create a brief report.
        
        Parameters
        ----------
        raw_profile : DatasetProfile
            Raw data profile
        clean_profile : DatasetProfile
            Cleaned data profile
        **kwargs
            Additional arguments passed to ReportGenerator
            
        Returns
        -------
        str
            Path to generated PDF
        """
        generator = ReportGenerator(
            raw_profile=raw_profile,
            clean_profile=clean_profile,
            mode='brief',
            **kwargs
        )
        return generator.generate()


    def create_detailed_report(
        raw_profile: Any,
        clean_profile: Any,
        **kwargs
    ) -> str:
        """
        Convenience function to create a detailed report.
        
        Parameters
        ----------
        raw_profile : DatasetProfile
            Raw data profile
        clean_profile : DatasetProfile
            Cleaned data profile
        **kwargs
            Additional arguments passed to ReportGenerator
            
        Returns
        -------
        str
            Path to generated PDF
        """
        generator = ReportGenerator(
            raw_profile=raw_profile,
            clean_profile=clean_profile,
            mode='detailed',
            **kwargs
        )
        return generator.generate()