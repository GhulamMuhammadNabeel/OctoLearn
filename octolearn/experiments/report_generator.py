import os
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime


class ReportGenerator:

    def __init__(self, profile, plot_paths, heatmap_path, recommendations, 
                 risk_score=None, risk_category=None, risk_factors=None,
                 preprocessing_suggestions=None, feature_importance=None, shap_path=None):
        self.profile = profile
        self.plot_paths = plot_paths
        self.heatmap_path = heatmap_path
        self.recommendations = recommendations
        self.risk_score = risk_score
        self.risk_category = risk_category
        self.risk_factors = risk_factors or {}
        self.preprocessing_suggestions = preprocessing_suggestions or {}
        self.feature_importance = feature_importance or {}
        self.shap_path = shap_path

    def generate(self):

        filename = f"octolearn_report_{self.profile.dataset_hash}.pdf"
        doc = SimpleDocTemplate(filename)
        elements = []
        styles = getSampleStyleSheet()

        # Title Page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
        )

        elements.append(Paragraph("OctoLearn Intelligence Report", title_style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Risk Score Banner
        if self.risk_score is not None:
            risk_color = colors.HexColor('#27AE60') if self.risk_score <= 30 else \
                        colors.HexColor('#F39C12') if self.risk_score <= 60 else \
                        colors.HexColor('#E74C3C')
            
            risk_banner = f"DATASET RISK SCORE: {self.risk_score}/100 - {self.risk_category}"
            elements.append(Paragraph(risk_banner, 
                ParagraphStyle('RiskBanner', parent=styles['Normal'], 
                              textColor=risk_color, fontSize=14, spaceAfter=10, 
                              alignment=1)))  # Center alignment
        
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(PageBreak())

        # Dataset Summary
        elements.append(Paragraph("Dataset Overview", styles["Heading1"]))
        elements.append(Spacer(1, 0.2 * inch))

        summary_data = [
            ["Rows", str(self.profile.n_rows)],
            ["Columns", str(self.profile.n_columns)],
            ["Task Type", self.profile.task_type],
            ["Duplicate Rows", str(self.profile.duplicate_rows)],
        ]

        table = Table(summary_data, colWidths=[2 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))

        # Feature Insights
        elements.append(Paragraph("Feature Insights", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))

        numeric_text = ", ".join(self.profile.numeric_features) if self.profile.numeric_features else "None"
        elements.append(Paragraph(
            f"<b>Numeric Features:</b> {numeric_text}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 0.2 * inch))

        categorical_text = ", ".join(self.profile.categorical_features) if self.profile.categorical_features else "None"
        elements.append(Paragraph(
            f"<b>Categorical Features:</b> {categorical_text}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 0.2 * inch))

        skewed_text = ", ".join(self.profile.skewed_columns) if self.profile.skewed_columns else "None"
        elements.append(Paragraph(
            f"<b>Skewed Columns:</b> {skewed_text}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 0.2 * inch))

        cardinality_text = ", ".join(self.profile.high_cardinality_cols) if self.profile.high_cardinality_cols else "None"
        elements.append(Paragraph(
            f"<b>High Cardinality Columns:</b> {cardinality_text}",
            styles["Normal"]
        ))

        elements.append(PageBreak())

        # Data Quality Assessment
        if self.risk_factors:
            elements.append(Paragraph("Data Quality Assessment", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))
            
            for risk_name, risk_desc in self.risk_factors.items():
                elements.append(Paragraph(f"⚠ {risk_desc}", styles["Normal"]))
                elements.append(Spacer(1, 0.15 * inch))
            
            elements.append(PageBreak())

        # Preprocessing Suggestions
        if self.preprocessing_suggestions:
            elements.append(Paragraph("Preprocessing Recommendations", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))
            
            for section, suggestions in self.preprocessing_suggestions.items():
                section_title = section.replace('_', ' ').title()
                elements.append(Paragraph(f"<b>{section_title}</b>", styles["Heading2"]))
                elements.append(Spacer(1, 0.1 * inch))
                
                if isinstance(suggestions, list):
                    for suggestion in suggestions:
                        elements.append(Paragraph(f"• {suggestion}", styles["Normal"]))
                        elements.append(Spacer(1, 0.1 * inch))
                else:
                    elements.append(Paragraph(str(suggestions), styles["Normal"]))
                
                elements.append(Spacer(1, 0.2 * inch))
            
            elements.append(PageBreak())

        # Feature Importance
        if self.feature_importance and "error" not in self.feature_importance:
            elements.append(Paragraph("Feature Importance (Baseline Model)", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))
            
            importance_data = [["Feature", "Importance"]]
            for feat, imp in list(self.feature_importance.items())[:10]:
                importance_data.append([feat, f"{imp:.4f}"])
            
            imp_table = Table(importance_data, colWidths=[3 * inch, 2 * inch])
            imp_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ]))
            
            elements.append(imp_table)
            elements.append(Spacer(1, 0.5 * inch))

        # Visual Analysis - Plots
        if self.plot_paths or self.heatmap_path or self.shap_path:
            elements.append(Paragraph("Visual Analysis - Feature Distributions", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))

            for path in self.plot_paths:
                if os.path.exists(path):
                    elements.append(Image(path, width=5 * inch, height=3 * inch))
                    elements.append(Spacer(1, 0.3 * inch))

            if self.heatmap_path and os.path.exists(self.heatmap_path):
                elements.append(Spacer(1, 0.3 * inch))
                elements.append(Paragraph("Correlation Analysis", styles["Heading2"]))
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Image(self.heatmap_path, width=5 * inch, height=4 * inch))

            if self.shap_path and os.path.exists(self.shap_path):
                elements.append(PageBreak())
                elements.append(Paragraph("SHAP Feature Importance", styles["Heading1"]))
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Image(self.shap_path, width=6 * inch, height=4 * inch))

            elements.append(PageBreak())

        # Strategic Recommendations
        elements.append(Paragraph("Strategic Recommendations", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))

        for rec in self.recommendations:
            elements.append(Paragraph(f"• {rec}", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

        doc.build(elements)

        return filename

