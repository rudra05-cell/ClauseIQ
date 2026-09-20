"""Aggregates document-level analysis results into a risk score and exportable reports."""

import io
import csv
from datetime import datetime

RISK_WEIGHT = {"High": 3, "Medium": 1, "Low": 0}


def compute_overall_risk(results: list) -> str:
    """Simple, explainable aggregation: the document's overall risk is the
    highest individual risk level found among its flagged clauses."""
    if not results:
        return "None"
    levels = {r["level"] for r in results}
    if "High" in levels:
        return "High"
    if "Medium" in levels:
        return "Medium"
    return "Low"


def risk_breakdown(results: list) -> dict:
    return {
        "High": sum(1 for r in results if r["level"] == "High"),
        "Medium": sum(1 for r in results if r["level"] == "Medium"),
        "Low": sum(1 for r in results if r["level"] == "Low"),
    }


def build_csv(document_name: str, results: list) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Document", "Category", "Risk Level", "Reason", "Clause Text"])
    for r in results:
        writer.writerow([document_name, r["category"], r["level"], r["reason"], r["full_text"]])
    return buffer.getvalue().encode("utf-8")


def build_pdf(document_name: str, results: list, category_labels: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#1a202c"))
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#4a5568"), spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a365d"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13.5, spaceAfter=4)
    reason_style = ParagraphStyle("Reason", parent=styles["Normal"], fontSize=9, leading=12.5, textColor=colors.HexColor("#4a5568"), spaceAfter=10)

    risk_colors = {"High": colors.HexColor("#DC2626"), "Medium": colors.HexColor("#D97706"), "Low": colors.HexColor("#059669")}

    story = []
    story.append(Paragraph("Contract Risk Analysis Report", title_style))
    story.append(Paragraph(f"Document: {document_name}", meta_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#e2e8f0")))

    overall = compute_overall_risk(results)
    breakdown = risk_breakdown(results)

    story.append(Paragraph("Summary", h2))
    summary_data = [
        ["Overall Risk", overall],
        ["Flagged Clauses", str(len(results))],
        ["High Risk", str(breakdown["High"])],
        ["Medium Risk", str(breakdown["Medium"])],
        ["Low Risk", str(breakdown["Low"])],
    ]
    table = Table(summary_data, colWidths=[2.0 * inch, 3.8 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f7fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    story.append(Paragraph("Flagged Clauses", h2))
    for r in sorted(results, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x["level"]]):
        label = category_labels.get(r["category"], r["category"])
        color = risk_colors.get(r["level"], colors.grey)
        story.append(Paragraph(
            f'<b>{label}</b> &nbsp;&nbsp; <font color="{color.hexval()}"><b>{r["level"]} RISK</b></font>',
            body,
        ))
        story.append(Paragraph(r["reason"], reason_style))

    doc.build(story)
    return buffer.getvalue()
