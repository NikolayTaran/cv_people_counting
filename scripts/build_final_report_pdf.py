#!/usr/bin/env python3
"""Render the canonical Markdown final report to a styled, paginated PDF."""

from __future__ import annotations

from html import escape
from pathlib import Path

import mistune
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from build_submission_pdf import STYLES, callout, page_decoration, table


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "final_report_p29.pdf"

TITLE = ParagraphStyle(
    "final_title", parent=STYLES["title"], fontSize=21, leading=26,
    spaceAfter=5 * mm,
)
SECTION = ParagraphStyle(
    "final_section", parent=STYLES["h1"], spaceBefore=5 * mm,
    spaceAfter=2.5 * mm, keepWithNext=True,
)
BODY = ParagraphStyle(
    "final_body", parent=STYLES["body"], fontSize=10.0, leading=14.3,
    spaceAfter=3.0 * mm,
)
META = ParagraphStyle(
    "final_meta", parent=STYLES["subtitle"], fontSize=8.5, leading=12,
    spaceAfter=4 * mm,
)


def inline(nodes: list[dict]) -> str:
    parts = []
    for node in nodes:
        kind = node["type"]
        if kind in ("text", "codespan"):
            value = escape(node.get("raw", ""))
            parts.append(f"<b>{value}</b>" if kind == "codespan" else value)
        elif kind == "strong":
            parts.append(f'<b>{inline(node.get("children", []))}</b>')
        elif kind == "emphasis":
            parts.append(f'<i>{inline(node.get("children", []))}</i>')
        elif kind == "link":
            label = inline(node.get("children", []))
            url = escape(node.get("attrs", {}).get("url", ""), quote=True)
            parts.append(f'<link href="{url}">{label}</link>')
        elif kind == "linebreak":
            parts.append("<br/>")
        elif kind == "softbreak":
            parts.append(" ")
        else:
            parts.append(inline(node.get("children", [])))
    return "".join(parts)


def table_rows(node: dict) -> list[list[str]]:
    rows = []
    for block in node["children"]:
        if block["type"] == "table_head":
            rows.append([inline(cell["children"]) for cell in block["children"]])
        elif block["type"] == "table_body":
            rows.extend([
                [inline(cell["children"]) for cell in row["children"]]
                for row in block["children"]
            ])
    return rows


def table_widths(header: list[str]) -> list[float]:
    columns = len(header)
    if columns == 4:
        return [43 * mm, 41 * mm, 42 * mm, 42 * mm]
    if columns == 5:
        return [62 * mm, 28 * mm, 28 * mm, 25 * mm, 25 * mm]
    if columns == 6 and header[0].startswith("Sequence and"):
        return [43 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm]
    if columns == 6:
        return [36 * mm, 19 * mm, 30 * mm, 27 * mm, 28 * mm, 28 * mm]
    return [168 * mm / columns] * columns


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    blocks = mistune.create_markdown(renderer="ast", plugins=["table"])(markdown)
    story = [Spacer(1, 3 * mm)]
    title_seen = False
    section_seen = False
    for block in blocks:
        kind = block["type"]
        if kind == "blank_line":
            continue
        if kind == "heading":
            level = block["attrs"]["level"]
            if level == 1:
                title_seen = True
                story.append(Paragraph(inline(block["children"]), TITLE))
            elif level == 2:
                section_seen = True
                story.append(Paragraph(inline(block["children"]), SECTION))
            else:
                story.append(Paragraph(inline(block["children"]), STYLES["h2"]))
        elif kind == "paragraph":
            value = inline(block["children"])
            if not section_seen:
                story.append(Paragraph(value, META))
            elif "All four-camera results in this report are preliminary" in value:
                story.extend([callout(value), Spacer(1, 2 * mm)])
            else:
                story.append(Paragraph(value, BODY))
        elif kind == "table":
            rows = table_rows(block)
            block_table = table(rows, table_widths(rows[0]))
            if rows[0][:2] == ["Sequence", "Frames"]:
                story.append(KeepTogether([block_table, Spacer(1, 3 * mm)]))
            else:
                story.extend([block_table, Spacer(1, 3 * mm)])
        else:
            raise ValueError(f"Unsupported Markdown block: {kind}")
    if not title_seen:
        raise ValueError("Missing report title")
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="Final Report - Video-Based People Counting (p29)",
        author="cv_people_counting project team",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()
