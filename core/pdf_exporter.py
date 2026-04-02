"""
pdf_exporter.py
───────────────
Generates a professional, self-contained PDF report for the PIS-LLM
CardioDetector TDA analysis session.

Layout (A4, single continuous flow):
  ① Cover page   — title, metadata, overall status badge
  ② Signal Metrics table
  ③ Signal Overview figure (glance plot)
  ④ Topological Analysis table
  ⑤ TDA Topology figure (PCA + persistence diagram)
  ⑥ Anomaly Score Distribution table
  ⑦ Cardiovascular Metrics table
  ⑧ LLM Clinical Report (full text, Markdown → plain text)
  ⑨ Conversation History (if any)
  ⑩ Footer — disclaimer, version, timestamp

Requires: reportlab, Pillow (already in proeng env)
"""

import io
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

# ── Colour palette (matches dark-theme accent colours) ────────────────────────
C_PURPLE   = colors.HexColor("#667eea")
C_PINK     = colors.HexColor("#f093fb")
C_DARK     = colors.HexColor("#1a1a2e")
C_CARD     = colors.HexColor("#16213e")
C_MUTED    = colors.HexColor("#94a3b8")
C_TEXT     = colors.HexColor("#e2e8f0")
C_WHITE    = colors.white
C_GREEN    = colors.HexColor("#16a34a")
C_YELLOW   = colors.HexColor("#ca8a04")
C_ORANGE   = colors.HexColor("#ea580c")
C_RED      = colors.HexColor("#dc2626")
C_ACCENT   = colors.HexColor("#63b3ed")
C_HEADER_BG = colors.HexColor("#0f172a")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

# ── Helper: strip Markdown to plain text ─────────────────────────────────────

def _md_to_plain(text: str) -> str:
    """Convert Markdown to clean plain text for PDF body paragraphs."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Bold / italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"__(.+?)__",     r"\1", text)
    text = re.sub(r"_(.+?)_",       r"\1", text)
    # Headers → plain with newline
    text = re.sub(r"^#{1,6}\s+",    "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "─" * 60, text, flags=re.MULTILINE)
    # Bullet / numbered list
    text = re.sub(r"^\s*[-*+]\s+",  "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+",  "  ", text, flags=re.MULTILINE)
    # Code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`(.+?)`",        r"\1", text)
    # Links
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # Multiple blank lines → single
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _severity_color(level: str) -> colors.Color:
    l = level.lower()
    if "severe" in l:
        return C_RED
    if "moderate" in l:
        return C_ORANGE
    if "mild" in l:
        return C_YELLOW
    return C_GREEN


# ══════════════════════════════════════════════════════════════════════════════
# Style registry
# ══════════════════════════════════════════════════════════════════════════════

def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=C_WHITE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=C_MUTED,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=C_ACCENT,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=colors.HexColor("#2d3748"),
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=colors.HexColor("#1a202c"),
            leading=15,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=C_MUTED,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "chat_user": ParagraphStyle(
            "chat_user",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#2563eb"),
            spaceBefore=6,
            spaceAfter=2,
        ),
        "chat_ai": ParagraphStyle(
            "chat_ai",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#374151"),
            leading=14,
            spaceAfter=6,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=C_MUTED,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "badge_normal": ParagraphStyle(
            "badge_normal",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_GREEN,
            alignment=TA_CENTER,
        ),
        "badge_mild": ParagraphStyle(
            "badge_mild",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_YELLOW,
            alignment=TA_CENTER,
        ),
        "badge_moderate": ParagraphStyle(
            "badge_moderate",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_ORANGE,
            alignment=TA_CENTER,
        ),
        "badge_severe": ParagraphStyle(
            "badge_severe",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_RED,
            alignment=TA_CENTER,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Table helpers
# ══════════════════════════════════════════════════════════════════════════════

def _metric_table(rows: List[tuple], col_widths=None) -> Table:
    """Two-column (Label | Value) metric table with alternating row shading."""
    if col_widths is None:
        col_widths = [8 * cm, 8 * cm]
    data = [["Metric", "Value"]] + list(rows)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), C_HEADER_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),
        # Body rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ALIGN",       (1, 1), (1, -1), "RIGHT"),
        ("ALIGN",       (0, 1), (0, -1), "LEFT"),
        ("TOPPADDING",  (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.white]),
    ]
    t.setStyle(TableStyle(style))
    return t


def _four_col_table(rows: List[tuple]) -> Table:
    """Four-column metric table for compact display."""
    col_w = (PAGE_W - 2 * MARGIN) / 4
    data = [["Metric", "Value", "Metric", "Value"]] + list(rows)
    t = Table(data, colWidths=[col_w] * 4, repeatRows=1)
    style = [
        ("BACKGROUND",  (0, 0), (-1, 0), C_HEADER_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ALIGN",       (1, 1), (1, -1), "RIGHT"),
        ("ALIGN",       (3, 1), (3, -1), "RIGHT"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.white]),
    ]
    t.setStyle(TableStyle(style))
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Image helper
# ══════════════════════════════════════════════════════════════════════════════

def _embed_image(path: Optional[str], max_width: float, max_height: float) -> Optional[Image]:
    """Load an image file and scale it to fit within max dimensions."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with PILImage.open(p) as img:
            orig_w, orig_h = img.size
        scale = min(max_width / orig_w, max_height / orig_h, 1.0)
        return Image(str(p), width=orig_w * scale, height=orig_h * scale)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Cover page canvas callback
# ══════════════════════════════════════════════════════════════════════════════

def _cover_canvas(canvas, doc):
    """Draw gradient-style cover background."""
    canvas.saveState()
    # Dark gradient background
    canvas.setFillColor(C_DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Accent bar at top
    canvas.setFillColor(C_PURPLE)
    canvas.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, fill=1, stroke=0)
    # Accent bar at bottom
    canvas.setFillColor(C_PURPLE)
    canvas.rect(0, 0, PAGE_W, 4 * mm, fill=1, stroke=0)
    canvas.restoreState()


def _content_canvas(canvas, doc):
    """Draw subtle header/footer on content pages."""
    canvas.saveState()
    # Top rule
    canvas.setStrokeColor(C_PURPLE)
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN, PAGE_H - 12 * mm, PAGE_W - MARGIN, PAGE_H - 12 * mm)
    # Header text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN, PAGE_H - 10 * mm, "PIS-LLM · CardioDetector TDA System · Fudan University")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm,
                           f"Page {doc.page}")
    # Bottom rule
    canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.setFillColor(C_MUTED)
    canvas.drawCentredString(PAGE_W / 2, 8 * mm,
                             "For research purposes only. Not a medical diagnosis.")
    canvas.restoreState()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf(
    result,                          # SignalProcessingResult
    llm_report: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    filename: str = "signal.csv",
    model_name: str = "N/A",
    language: str = "English",
    session_id: str = "",
) -> bytes:
    """
    Build and return a PDF as raw bytes.

    Parameters
    ----------
    result       : SignalProcessingResult instance
    llm_report   : LLM-generated Markdown report text
    chat_history : list of {"role": "user"|"assistant", "content": str}
    filename     : original uploaded filename
    model_name   : LLM model display name
    language     : report language ("English" / "中文")
    session_id   : analysis session UUID
    """
    chat_history = chat_history or []
    buf = io.BytesIO()
    S = _build_styles()

    # ── Document setup ────────────────────────────────────────────────────────
    content_frame = Frame(
        MARGIN, 16 * mm,
        PAGE_W - 2 * MARGIN, PAGE_H - 30 * mm,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )
    cover_frame = Frame(
        MARGIN, MARGIN,
        PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    doc.addPageTemplates([
        PageTemplate(id="cover",   frames=[cover_frame],   onPage=_cover_canvas),
        PageTemplate(id="content", frames=[content_frame], onPage=_content_canvas),
    ])

    story = []

    # ── ① Cover page ──────────────────────────────────────────────────────────
    story.append(NextPageTemplate("cover"))
    story.append(Spacer(1, 5 * cm))

    story.append(Paragraph("PIS-LLM", S["title"]))
    story.append(Paragraph(
        "Persistent Homology-Informed Signal Large Language Model",
        S["subtitle"],
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Cardiovascular Signal Analysis Report",
        ParagraphStyle("cov_sub2", fontName="Helvetica", fontSize=13,
                       textColor=C_ACCENT, alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 1.2 * cm))

    # Status badge
    summary = result.summary
    anomaly_level = summary.get("anomaly_level", "N/A")
    cv_status     = summary.get("cardiovascular_status", "N/A")
    badge_style_key = (
        "badge_severe"   if "Severe"   in anomaly_level else
        "badge_moderate" if "Moderate" in anomaly_level else
        "badge_mild"     if "Mild"     in anomaly_level else
        "badge_normal"
    )
    story.append(Paragraph(f"Overall Status: {anomaly_level}", S[badge_style_key]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Cardiovascular: {cv_status}",
                            ParagraphStyle("cov_cv", fontName="Helvetica", fontSize=11,
                                           textColor=C_MUTED, alignment=TA_CENTER)))
    story.append(Spacer(1, 1.5 * cm))

    # Metadata table
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta_data = [
        ["Signal File",    filename],
        ["Analysis Time",  now],
        ["LLM Model",      model_name],
        ["Report Language", language],
        ["Session ID",     session_id or "—"],
        ["System",         "CardioDetector TDA v2.1 · Fudan University"],
    ]
    meta_t = Table(meta_data, colWidths=[5 * cm, 11 * cm])
    meta_t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), C_MUTED),
        ("TEXTCOLOR",   (1, 0), (1, -1), C_TEXT),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW",   (0, 0), (-1, -2), 0.3, colors.HexColor("#334155")),
    ]))
    story.append(meta_t)

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "⚠️  This report is generated by an AI system for research purposes only "
        "and does not constitute a medical diagnosis. Consult a qualified healthcare "
        "professional for clinical decisions.",
        ParagraphStyle("cov_disc", fontName="Helvetica-Oblique", fontSize=8,
                       textColor=colors.HexColor("#64748b"), alignment=TA_CENTER,
                       leading=13),
    ))

    # ── Switch to content template ────────────────────────────────────────────
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())

    # Shorthand
    basic   = result.basic_signal
    topo    = result.topology
    anomaly = result.anomaly
    cardio  = result.cardiovascular
    severity = cardio.get("severity_distribution", {})

    # ── ② Signal Metrics ──────────────────────────────────────────────────────
    story.append(Paragraph("① Signal Overview Metrics", S["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
    sig_rows = [
        ("Sampling Frequency (Hz)",   str(basic.get("signal_frequency_hz", "N/A"))),
        ("Signal Duration (s)",       f"{basic.get('signal_duration_seconds', 0):.1f}"),
        ("Total Samples",             str(basic.get("total_samples", "N/A"))),
        ("Dominant Frequency (Hz)",   f"{basic.get('dominant_frequency_hz', 0):.3f}"),
        ("Signal Amplitude Range",    f"{basic.get('signal_amplitude_range', 0):.4f}"),
        ("Processing Time (s)",       f"{result.processing_time:.2f}"),
    ]
    story.append(_metric_table(sig_rows))

    # ── ③ Signal Overview Figure ──────────────────────────────────────────────
    if result.plot_glance_path:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("② Signal Overview — Raw Signal · Anomaly Score · Binary Indicator", S["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
        story.append(Paragraph(
            "The three-panel figure below shows the raw cardiovascular signal (top), "
            "the continuous TDA-derived anomaly score profile (middle), and the binary "
            "anomaly indicator with threshold (bottom). Regions highlighted in red exceed "
            "the 95th-percentile anomaly threshold.",
            S["body"],
        ))
        img = _embed_image(result.plot_glance_path,
                           max_width=PAGE_W - 2 * MARGIN,
                           max_height=12 * cm)
        if img:
            story.append(img)
            story.append(Paragraph(
                "Fig 1 — Signal Overview: Raw Signal · Anomaly Score · Binary Indicator",
                S["caption"],
            ))

    # ── ④ Topological Analysis Metrics ───────────────────────────────────────
    story.append(Paragraph("③ Topological Analysis Metrics", S["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
    topo_rows = [
        ("Total Cycles Detected",    str(topo.get("total_cycles", 0))),
        ("Normal Cycles",            str(topo.get("normal_cycles", 0))),
        ("Anomaly Cycles",           str(topo.get("anomaly_cycles", 0))),
        ("Anomaly Ratio",            f"{topo.get('anomaly_ratio', 0) * 100:.1f}%"),
        ("Mean Persistence",         f"{topo.get('mean_persistence', 0):.4f}"),
        ("Max Persistence",          f"{topo.get('max_persistence', 0):.4f}"),
    ]
    story.append(_metric_table(topo_rows))

    # ── ⑤ TDA Topology Figure ────────────────────────────────────────────────
    if result.plot_tda_path:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("④ Topological Structure — PCA Point Cloud · Persistence Diagram", S["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
        story.append(Paragraph(
            "The left panel projects the Takens delay-embedded point cloud onto its first three "
            "principal components. A healthy periodic signal forms a closed loop; anomalous "
            "segments distort or break this structure. The right panel is the H₁ persistence "
            "diagram: each point represents a topological loop, and its distance above the "
            "diagonal indicates how long it persists — longer persistence implies a more "
            "robust, genuine cardiac cycle.",
            S["body"],
        ))
        img2 = _embed_image(result.plot_tda_path,
                            max_width=PAGE_W - 2 * MARGIN,
                            max_height=9 * cm)
        if img2:
            story.append(img2)
            story.append(Paragraph(
                "Fig 2 — PCA Point Cloud (H₁) · Persistence Diagram",
                S["caption"],
            ))

    # ── ⑥ Anomaly Score Distribution ─────────────────────────────────────────
    story.append(Paragraph("⑤ Anomaly Score Distribution", S["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
    anomaly_rows = [
        ("Mean Anomaly Score",           f"{anomaly.get('mean_anomaly_score', 0):.4f}"),
        ("Max Anomaly Score",            f"{anomaly.get('max_anomaly_score', 0):.4f}"),
        ("Anomaly Coverage (90th pct)",  f"{anomaly.get('anomaly_coverage_90p_percent', 0):.1f}%"),
        ("Anomaly Coverage (95th pct)",  f"{anomaly.get('anomaly_coverage_95p_percent', 0):.1f}%"),
        ("Anomaly Peak Count",           str(anomaly.get("anomaly_peak_count", 0))),
    ]
    story.append(_metric_table(anomaly_rows))

    # ── ⑦ Cardiovascular Metrics ─────────────────────────────────────────────
    story.append(Paragraph("⑥ Cardiovascular Metrics", S["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
    cv_rows = [
        ("Estimated Heart Rate (bpm)",  f"{cardio.get('estimated_heart_rate_bpm', 0):.1f}"),
        ("Mild Anomaly Proportion",     f"{severity.get('mild_percent', 0):.1f}%"),
        ("Moderate Anomaly Proportion", f"{severity.get('moderate_percent', 0):.1f}%"),
        ("Severe Anomaly Proportion",   f"{severity.get('severe_percent', 0):.1f}%"),
        ("Overall Anomaly Level",       anomaly_level),
        ("Cardiovascular Status",       cv_status),
    ]
    story.append(_metric_table(cv_rows))

    # ── ⑧ LLM Clinical Report ────────────────────────────────────────────────
    if llm_report:
        story.append(Paragraph("⑦ LLM Clinical Report", S["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
        story.append(Paragraph(
            f"Generated by {model_name} · Fudan University CardioDetector TDA System",
            ParagraphStyle("report_meta", fontName="Helvetica-Oblique", fontSize=8.5,
                           textColor=C_MUTED, spaceAfter=8),
        ))
        plain_report = _md_to_plain(llm_report)
        for para in plain_report.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("─"):
                story.append(HRFlowable(width="100%", thickness=0.3,
                                        color=colors.HexColor("#cbd5e1"), spaceAfter=4))
            elif para.startswith("•"):
                for line in para.split("\n"):
                    if line.strip():
                        story.append(Paragraph(
                            line.strip(),
                            ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5,
                                           textColor=colors.HexColor("#2d3748"),
                                           leftIndent=12, leading=14, spaceAfter=3),
                        ))
            else:
                story.append(Paragraph(para, S["body"]))

    # ── ⑨ Conversation History ───────────────────────────────────────────────
    if chat_history:
        story.append(Paragraph("⑧ Conversation History", S["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=6))
        story.append(Paragraph(
            "The following is the multi-turn Q&A session conducted after the analysis.",
            S["body"],
        ))
        for i, msg in enumerate(chat_history):
            role    = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                story.append(Paragraph(f"Q{(i // 2) + 1}: {content}", S["chat_user"]))
            else:
                plain_reply = _md_to_plain(content)
                story.append(Paragraph(plain_reply, S["chat_ai"]))
                story.append(HRFlowable(width="100%", thickness=0.3,
                                        color=colors.HexColor("#e2e8f0"), spaceAfter=2))

    # ── ⑩ Disclaimer & Footer ────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE, spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: PIS-LLM is a computational research tool. All analyses are based "
        "solely on the provided numerical signal data. No patient identity, name, or "
        "personal information is inferred or fabricated. This report does not constitute "
        "a clinical diagnosis. Always consult a qualified healthcare professional.",
        S["disclaimer"],
    ))
    story.append(Paragraph(
        f"© 2025-2026 Daomiao Wang @ Fudan University · PIS-LLM CardioDetector TDA v2.1 "
        f"· Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        S["disclaimer"],
    ))

    doc.build(story)
    return buf.getvalue()
