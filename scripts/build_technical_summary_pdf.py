#!/usr/bin/env python3
"""Render the concise final technical summary from its Markdown source."""

from __future__ import annotations

from pathlib import Path

import mistune
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from build_final_report_pdf import inline, table_rows
from build_submission_pdf import STYLES, page_decoration, table


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "FINAL_TECHNICAL_SUMMARY.md"
OUTPUT = ROOT / "output" / "pdf" / "final_technical_summary_p29.pdf"

TITLE = ParagraphStyle(
    "summary_title", parent=STYLES["title"], fontSize=19, leading=23,
    spaceAfter=2.0 * mm,
)
META = ParagraphStyle(
    "summary_meta", parent=STYLES["subtitle"], fontSize=8.3, leading=11,
    spaceAfter=2.2 * mm,
)
SECTION = ParagraphStyle(
    "summary_section", parent=STYLES["h1"], fontSize=11, leading=14,
    spaceBefore=2.5 * mm, spaceAfter=1.3 * mm, keepWithNext=True,
)
BODY = ParagraphStyle(
    "summary_body", parent=STYLES["body"], fontSize=9.2, leading=13.2,
    spaceAfter=2.2 * mm,
)


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blocks = mistune.create_markdown(renderer="ast", plugins=["table"])(
        SOURCE.read_text(encoding="utf-8")
    )
    story = [Spacer(1, 2 * mm)]
    for block in blocks:
        kind = block["type"]
        if kind == "blank_line":
            continue
        if kind == "heading":
            if inline(block["children"]) == "Main results":
                story.append(PageBreak())
            style = TITLE if block["attrs"]["level"] == 1 else SECTION
            story.append(Paragraph(inline(block["children"]), style))
        elif kind == "paragraph":
            style = META if len(story) == 2 else BODY
            story.append(Paragraph(inline(block["children"]), style))
        elif kind == "table":
            rows = table_rows(block)
            story.extend([
                table(rows, [62 * mm, 28 * mm, 28 * mm, 25 * mm, 25 * mm]),
                Spacer(1, 1.5 * mm),
            ])
        else:
            raise ValueError(f"Unsupported Markdown block: {kind}")
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=17 * mm, bottomMargin=15 * mm,
        title="Technical Summary - Video-Based People Counting (p29)",
        author="cv_people_counting project team",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()
