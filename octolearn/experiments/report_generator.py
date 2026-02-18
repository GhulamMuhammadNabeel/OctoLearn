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
Version: 0.7.6 (Patched)
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
    """Color scheme for reports with 'Outta World' Cyberpunk theme."""
    # Primary colors
    PRIMARY = colors.HexColor('#00F0FF')      # Neon Cyan
    ACCENT_RED = colors.HexColor('#FF0055')    # Neon Pink/Red
    ACCENT_ORANGE = colors.HexColor('#FFB800')  # Neon Orange
    ACCENT_GREEN = colors.HexColor('#00FF9F')   # Neon Green
    
    # Backgrounds
    DARK_BG = colors.HexColor('#0D0D15')      # Deep Cyberpunk Black
    LIGHT_BG = colors.HexColor('#1B1B25')     # Lighter Dark for cards
    WHITE = colors.HexColor('#FFFFFF')
    
    # Text colors
    TEXT_DARK = colors.HexColor('#E0E0E0')    # Light text for dark bg
    TEXT_LIGHT = colors.HexColor('#E0E0E0')   # Light text
    TEXT_DIM = colors.HexColor('#8888AA')     # Dimmed text
    
    # Risk levels
    RISK_LOW = colors.HexColor('#00FF9F')     # Neon Green
    RISK_MODERATE = colors.HexColor('#FFB800')  # Neon Orange
    RISK_HIGH = colors.HexColor('#FF0055')    # Neon Pink/Red
    
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
        outlier_results: Optional[Dict] = None,      # Added
        interaction_results: Optional[Dict] = None,  # Added
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
        self.outlier_results = outlier_results or {}          # Stored
        self.interaction_results = interaction_results or {}  # Stored
        self.title = title
        self.author = author
        self.company = company
        
        # Initialize
        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
        logger.info(f"Report initialized in {mode.upper()} mode")

    def _register_fonts(self):
        """Robust font loading with multi-location fallback."""
        from pathlib import Path
        
        # Default to Helvetica (always available in reportlab)
        self.font_regular = 'Helvetica'
        self.font_bold = 'Helvetica-Bold'
        self.font_title = 'Helvetica-Bold'
        
        # Check these locations in order
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
                        # Update font attribute mappings on success
                        if 'Regular' in font_name:
                            self.font_regular = font_name
                        elif 'ExtraBold' in font_name:
                            self.font_title = font_name
                        elif 'Bold' in font_name:
                            self.font_bold = font_name
                        logger.info(f"Loaded font: {font_name}")
                        break
                    except Exception as e:
                        logger.debug(f"Could not load {font_name}: {e}")

    def _create_custom_styles(self):
        """Create professional 'Outta World' styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontName='ShantellSans-ExtraBold',
            fontSize=32,
            leading=38,
            textColor=ReportColors.PRIMARY,
            spaceAfter=20,
            alignment=TA_CENTER,
            args=[('glow', ReportColors.PRIMARY, 10)] # Theoretical glow support if lib allowed
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontName='ShantellSans-Bold',
            fontSize=18,
            leading=22,
            textColor=ReportColors.PRIMARY,
            spaceBefore=20,
            spaceAfter=15,
            borderPadding=10,
            borderWidth=0, # No border, just text
            # borderColor=ReportColors.PRIMARY # Removed box border for cleaner look
        ))
        
        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontName='ShantellSans-Bold',
            fontSize=14,
            leading=18,
            textColor=ReportColors.ACCENT_ORANGE,
            spaceBefore=12,
            spaceAfter=10
        ))
        
        # Normal text
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontName='ShantellSans-Regular',
            fontSize=11,
            leading=15,
            textColor=ReportColors.TEXT_DARK,
            spaceAfter=10,
            alignment=TA_LEFT # Left align looks better in tech/code style
        ))
        
        # Table text
        self.styles.add(ParagraphStyle(
            name='TableText',
            parent=self.styles['Normal'],
            fontName='ShantellSans-Regular',
            fontSize=10,
            leading=12,
            textColor=ReportColors.TEXT_DARK
        ))
        
        # Highlight/insight text
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['Normal'],
            fontName='ShantellSans-Bold',
            fontSize=11,
            leading=15,
            textColor=ReportColors.ACCENT_RED,
            spaceAfter=10
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

    def _add_cover_page(self, story):
        """Add a professional Cyberpunk cover page."""
        story.append(Spacer(1, 2*inch))
        
        # Logo if available
        if self.logo_path and os.path.exists(self.logo_path):
            im = Image(self.logo_path, width=2*inch, height=2*inch)
            story.append(im)
        
        story.append(Spacer(1, 1*inch))
        
        # Title with Glow effect simulation (using color)
        story.append(Paragraph(self.title, self.styles['ReportTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Metadata card
        data_table = [
            ["Generated By:", self.author],
            ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Task Type:", self.raw_profile.task_type.title() if self.raw_profile else "N/A"],
            ["Dataset Shape:", f"{self.raw_profile.n_rows} rows × {self.raw_profile.n_columns} cols" if self.raw_profile else "N/A"]
        ]
        
        t = Table(data_table, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), ReportColors.LIGHT_BG),
            ('TEXTCOLOR', (0,0), (-1,-1), ReportColors.TEXT_LIGHT),
            ('FONTNAME', (0,0), (-1,-1), 'ShantellSans-Regular'),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('ALIGN', (0,0), (0,-1), 'RIGHT'),
            ('ALIGN', (1,0), (1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (0,-1), 'ShantellSans-Bold'),
            ('TEXTCOLOR', (0,0), (0,-1), ReportColors.PRIMARY),
            ('GRID', (0,0), (-1,-1), 0.5, ReportColors.DARK_BG),
        ]))
        story.append(t)
        story.append(PageBreak())

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

    def _add_preprocessing_details(self, story: list):
        """Add preprocessing strategy and cleaning log section."""
        story.append(PageBreak())
        story.append(Paragraph("Preprocessing Strategy & Cleaning Log", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        if self.cleaning_log:
            story.append(Paragraph("Data Cleaning Actions Performed:", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.1 * inch))
            
            cleaning_items = [
                ('Duplicate Rows Removed', self.cleaning_log.get('duplicates_removed', 0)),
                ('ID-Like Columns Removed', self.cleaning_log.get('id_columns_removed', 0)),
                ('Constant Columns Removed', self.cleaning_log.get('constant_columns_removed', 0)),
                ('Low Variance Columns Removed', self.cleaning_log.get('low_variance_removed', 0)),
                ('Missing Values Imputed', self.cleaning_log.get('missing_imputed', 0)),
            ]
            
            for label, count in cleaning_items:
                if count and count > 0:
                    story.append(Paragraph(f"• {label}: {count}", self.styles['ReportBody']))
            
            story.append(Spacer(1, 0.2 * inch))

    def _add_analysis_results(self, story: list):
        """Add feature analysis results section."""
        story.append(PageBreak())
        story.append(Paragraph("Feature Analysis Results", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        if self.outlier_results and self.outlier_results.get('summary'):
            story.append(Paragraph("Outlier Detection Summary:", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.1 * inch))
            
            outlier_info = self.outlier_results.get('summary', {})
            for method, count in outlier_info.items():
                story.append(Paragraph(f"• <b>{method}:</b> {count} outliers detected", self.styles['ReportBody']))
            
            story.append(Spacer(1, 0.15 * inch))
        
        if self.interaction_results and self.interaction_results.get('strong_interactions'):
            story.append(Paragraph("Strong Feature Interactions:", self.styles['SubsectionHeading']))
            story.append(Spacer(1, 0.1 * inch))
            
            interactions = self.interaction_results.get('strong_interactions', [])
            for idx, interaction in enumerate(interactions[:5], 1):
                if isinstance(interaction, (tuple, list)) and len(interaction) >= 2:
                    story.append(Paragraph(f"• <b>{interaction[0]}</b> ↔ <b>{interaction[1]}</b>", self.styles['ReportBody']))
            
            story.append(Spacer(1, 0.2 * inch))

    def _add_feature_importance(self, story: list):
        """Add feature importance section if available."""
        if not self.feature_importance:
            return
        
        story.append(PageBreak())
        story.append(Paragraph("Feature Importance", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Show top features
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        if sorted_features:
            fi_data = [['Feature', 'Importance']]
            for feat, score in sorted_features:
                fi_data.append([
                    Paragraph(str(feat), self.styles['TableText']),
                    Paragraph(f"{score:.4f}", self.styles['TableText']),
                ])
            
            fi_table = Table(fi_data, colWidths=[3.5 * inch, 2 * inch])
            fi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ReportColors.PRIMARY),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), self.font_bold),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ReportColors.LIGHT_BG])
            ]))
            story.append(fi_table)
        
        # SHAP plot if available
        if self.shap_path and os.path.exists(self.shap_path):
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("SHAP Feature Impact", self.styles['SubsectionHeading']))
            try:
                shap_img = Image(self.shap_path, width=5 * inch, height=3.5 * inch)
                story.append(shap_img)
            except Exception as e:
                logger.warning(f"Could not add SHAP plot: {e}")

    def _add_visual_insights(self, story: list):
        """Add distribution plots and heatmap visualizations."""
        if not self.dist_plots and not self.heatmap_plot:
            return
        
        story.append(PageBreak())
        story.append(Paragraph("Visual Insights", self.styles['SectionHeading']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Correlation heatmap
        if self.heatmap_plot and os.path.exists(self.heatmap_plot):
            story.append(Paragraph("Correlation Matrix", self.styles['SubsectionHeading']))
            try:
                hm_img = Image(self.heatmap_plot, width=5 * inch, height=4 * inch)
                story.append(hm_img)
                story.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                logger.warning(f"Could not add heatmap: {e}")
        
        # Distribution plots (limit to avoid oversized reports)
        limit = 3 if self.mode == 'brief' else 8
        for plot_path in self.dist_plots[:limit]:
            if os.path.exists(plot_path):
                try:
                    dist_img = Image(plot_path, width=5 * inch, height=3 * inch)
                    story.append(dist_img)
                    story.append(Spacer(1, 0.2 * inch))
                except Exception as e:
                    logger.warning(f"Could not add plot {plot_path}: {e}")

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
        """Generate the complete PDF report."""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"octolearn_report_{timestamp}.pdf"
        
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
            
            # Build comprehensive report with all sections
            self._add_cover_page(story)
            story.append(PageBreak())
            
            self._add_executive_summary(story)
            self._add_risk_analysis(story)
            self._add_preprocessing_details(story)      # NEW
            self._add_analysis_results(story)             # NEW
            self._add_feature_importance(story)
            self._add_visual_insights(story)
            self._add_model_results(story)
            self._add_recommendations(story)
            
            # Build PDF
            doc.build(story)
            logger.info(f"✓ Report generated: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"✗ Failed to generate PDF: {str(e)}")
            raise RuntimeError(f"PDF generation failed: {str(e)}")

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