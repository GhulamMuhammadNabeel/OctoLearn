import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


class ReportGenerator:
    """
    Generates final PDF report for dataset profiling, visualizations,
    feature importance, risk score, preprocessing suggestions, and recommendations.
    """

    def __init__(
        self,
        profile,
        plot_paths,
        heatmap_path,
        recommendations,
        risk_score=None,
        risk_category=None,
        risk_factors=None,
        preprocessing_suggestions=None,
        feature_importance=None,
        shap_path=None,
        fonts_folder="fonts"
    ):
        self.profile = profile
        self.plot_paths = plot_paths or []
        self.heatmap_path = heatmap_path
        self.recommendations = recommendations or []
        self.risk_score = risk_score
        self.risk_category = risk_category
        self.risk_factors = risk_factors or {}
        self.preprocessing_suggestions = preprocessing_suggestions or {}
        self.feature_importance = feature_importance or {}
        self.shap_path = shap_path

        # ---------- Load fonts ----------
        self.fonts = {}
        font_files = {
            "title": os.path.join(FONTS_DIR, "ShantellSans-ExtraBold.ttf"),
            "section": os.path.join(FONTS_DIR, "ShantellSans-Bold.ttf"),
            "normal": os.path.join(FONTS_DIR, "ShantellSans-Regular.ttf"),
            "italic": os.path.join(FONTS_DIR, "ShantellSans-Italic.ttf"),
        }


        for key, file_name in font_files.items():
            font_path = os.path.join(fonts_folder, file_name)
            if os.path.exists(font_path):
                font_name = file_name.replace(".ttf", "")
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                self.fonts[key] = font_name
            else:
                print(f"Font {file_name} not found, using default Helvetica for {key}.")
                self.fonts[key] = "Helvetica"

    def generate(self):
        """
        Generate the PDF report with red font (#FF0000).

        Returns
        -------
        str : path to generated PDF file
        """
        filename = f"octolearn_report_{self.profile.dataset_hash}.pdf"
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        elements = []
        styles = getSampleStyleSheet()

        # ---------- Custom Styles ----------
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=26,
            textColor=colors.HexColor("#FF0000"),
            fontName=self.fonts["title"],
            spaceAfter=20
        )

        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#FF0000"),
            fontName=self.fonts["section"],
            spaceAfter=10
        )

        normal_style = ParagraphStyle(
            "NormalStyle",
            parent=styles["Normal"],
            fontName=self.fonts["normal"],
            fontSize=10,
            textColor=colors.HexColor("#FF0000")
        )

        # ---------- Title Page ----------
        elements.append(Paragraph("OctoLearn Intelligence Report", title_style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            normal_style
        ))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            f"Dataset Hash: {self.profile.dataset_hash}",
            normal_style
        ))

        # ---------- Risk Score ----------
        if self.risk_score is not None:
            if self.risk_score <= 30:
                risk_color = colors.HexColor("#27AE60")
            elif self.risk_score <= 60:
                risk_color = colors.HexColor("#F39C12")
            else:
                risk_color = colors.HexColor("#E74C3C")

            banner_style = ParagraphStyle(
                "RiskBanner",
                parent=normal_style,
                fontSize=14,
                textColor=risk_color,
                alignment=1
            )
            elements.append(Paragraph(
                f"DATASET RISK SCORE: {self.risk_score}/100  |  {self.risk_category}",
                banner_style
            ))

        elements.append(PageBreak())

        # ---------- Executive Summary ----------
        elements.append(Paragraph("Executive Summary", section_style))
        elements.append(Spacer(1, 0.2 * inch))

        summary_table_data = [
            ["Rows", str(self.profile.n_rows)],
            ["Columns", str(self.profile.n_columns)],
            ["Task Type", self.profile.task_type],
            ["Duplicate Rows", str(self.profile.duplicate_rows)],
        ]

        summary_table = Table(summary_table_data, colWidths=[2.5 * inch, 2.5 * inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), self.fonts["normal"]),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))

        elements.append(summary_table)
        elements.append(PageBreak())

        # ---------- Feature Overview ----------
        elements.append(Paragraph("Feature Overview", section_style))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            f"<b>Numeric Features:</b> {', '.join(self.profile.numeric_features) or 'None'}",
            normal_style
        ))
        elements.append(Spacer(1, 0.1 * inch))

        elements.append(Paragraph(
            f"<b>Categorical Features:</b> {', '.join(self.profile.categorical_features) or 'None'}",
            normal_style
        ))
        elements.append(Spacer(1, 0.1 * inch))

        elements.append(Paragraph(
            f"<b>Skewed Columns:</b> {', '.join(self.profile.skewed_columns) or 'None'}",
            normal_style
        ))

        elements.append(PageBreak())

        # ---------- Data Quality ----------
        if self.risk_factors:
            elements.append(Paragraph("Data Quality Assessment", section_style))
            elements.append(Spacer(1, 0.2 * inch))

            for desc in self.risk_factors.values():
                elements.append(Paragraph(f"• {desc}", normal_style))
                elements.append(Spacer(1, 0.1 * inch))

            elements.append(PageBreak())

        # ---------- Preprocessing ----------
        if self.preprocessing_suggestions:
            elements.append(Paragraph("Preprocessing Strategy", section_style))
            elements.append(Spacer(1, 0.2 * inch))

            for section, suggestions in self.preprocessing_suggestions.items():
                elements.append(Paragraph(
                    f"<b>{section.replace('_', ' ').title()}</b>",
                    styles["Heading2"]
                ))
                elements.append(Spacer(1, 0.1 * inch))

                for suggestion in suggestions:
                    elements.append(Paragraph(f"• {suggestion}", normal_style))
                    elements.append(Spacer(1, 0.1 * inch))

                elements.append(Spacer(1, 0.2 * inch))

            elements.append(PageBreak())

        # ---------- Feature Importance ----------
        if self.feature_importance and "error" not in self.feature_importance:

            elements.append(Paragraph("Top Feature Importance", section_style))
            elements.append(Spacer(1, 0.2 * inch))

            importance_data = [["Rank", "Feature", "Importance"]]

            sorted_feats = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for idx, (feat, imp) in enumerate(sorted_feats, 1):
                importance_data.append([str(idx), feat, f"{imp:.4f}"])

            table = Table(importance_data, colWidths=[0.8 * inch, 2.5 * inch, 1.5 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), self.fonts["normal"]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))

            elements.append(table)
            elements.append(PageBreak())

        # ---------- Visual Analysis ----------
        if self.plot_paths or self.heatmap_path or self.shap_path:

            elements.append(Paragraph("Visual Insights", section_style))
            elements.append(Spacer(1, 0.2 * inch))

            image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

            # --- Plot paths ---
            for path in self.plot_paths:
                if (
                    path
                    and os.path.exists(path)
                    and path.lower().endswith(image_extensions)
                ):
                    elements.append(Image(path, width=5.5 * inch, height=3.5 * inch))
                    elements.append(Spacer(1, 0.3 * inch))

            # --- Heatmap ---
            if (
                self.heatmap_path
                and os.path.exists(self.heatmap_path)
                and self.heatmap_path.lower().endswith(image_extensions)
            ):
                elements.append(Image(self.heatmap_path, width=5.5 * inch, height=4 * inch))
                elements.append(Spacer(1, 0.3 * inch))

            # --- SHAP ---
            if (
                self.shap_path
                and os.path.exists(self.shap_path)
                and self.shap_path.lower().endswith(image_extensions)
            ):
                elements.append(Image(self.shap_path, width=6 * inch, height=4 * inch))

            elements.append(PageBreak())

        # ---------- Strategic Recommendations ----------
        elements.append(Paragraph("Strategic Recommendations", section_style))
        elements.append(Spacer(1, 0.2 * inch))

        for rec in self.recommendations:
            elements.append(Paragraph(f"• {rec}", normal_style))
            elements.append(Spacer(1, 0.15 * inch))

        # ---------- Build ----------
        doc.build(elements)

        return filename
