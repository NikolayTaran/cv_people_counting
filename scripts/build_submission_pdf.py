#!/usr/bin/env python3
"""Build the first working model report from the recorded experiment results.

Requires reportlab. The PDF deliberately distinguishes model comparison from
an untouched final test: the four camera-separated sequences informed the
choice of the submitted checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "first_working_model_report.pdf"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#247BA0")
TEAL = colors.HexColor("#2A9D8F")
TEXT = colors.HexColor("#25313C")
MUTED = colors.HexColor("#60717E")
LIGHT = colors.HexColor("#F3F7FA")
PALE_BLUE = colors.HexColor("#EAF3F8")
GRID = colors.HexColor("#D9E5EC")

arial = Path("C:/Windows/Fonts/arial.ttf")
arial_bold = Path("C:/Windows/Fonts/arialbd.ttf")
if arial.is_file() and arial_bold.is_file():
    pdfmetrics.registerFont(TTFont("Body", str(arial)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(arial_bold)))
    FONT_REGULAR, FONT_BOLD = "Body", "BodyBold"
else:
    FONT_REGULAR, FONT_BOLD = "Helvetica", "Helvetica-Bold"

STYLES = {
    "title": ParagraphStyle(
        "title", fontName=FONT_BOLD, fontSize=24, leading=29,
        textColor=NAVY, spaceAfter=6 * mm,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", fontName=FONT_REGULAR, fontSize=11, leading=15,
        textColor=MUTED, spaceAfter=5 * mm,
    ),
    "h1": ParagraphStyle(
        "h1", fontName=FONT_BOLD, fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=3 * mm, spaceAfter=2.5 * mm,
    ),
    "h2": ParagraphStyle(
        "h2", fontName=FONT_BOLD, fontSize=11, leading=14,
        textColor=BLUE, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
    ),
    "body": ParagraphStyle(
        "body", fontName=FONT_REGULAR, fontSize=9.0, leading=13.2,
        textColor=TEXT, spaceAfter=2.4 * mm,
    ),
    "small": ParagraphStyle(
        "small", fontName=FONT_REGULAR, fontSize=7.4, leading=10.2,
        textColor=TEXT,
    ),
    "table_header": ParagraphStyle(
        "table_header", fontName=FONT_BOLD, fontSize=7.4, leading=10.2,
        textColor=colors.white,
    ),
    "metric": ParagraphStyle(
        "metric", fontName=FONT_BOLD, fontSize=16, leading=19,
        textColor=TEAL, alignment=1,
    ),
    "metric_label": ParagraphStyle(
        "metric_label", fontName=FONT_REGULAR, fontSize=7.2, leading=10,
        textColor=MUTED, alignment=1,
    ),
    "caveat": ParagraphStyle(
        "caveat", fontName=FONT_BOLD, fontSize=9.0, leading=13.2,
        textColor=NAVY,
    ),
}


def paragraph(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullet(text: str) -> Paragraph:
    style = ParagraphStyle(
        "bullet", parent=STYLES["body"], leftIndent=12,
        firstLineIndent=-9, spaceAfter=1.4 * mm,
    )
    return Paragraph(f"- {text}", style)


def table(rows: list[list[str]], widths: list[float]) -> Table:
    cells = [
        [paragraph(value, "table_header" if index == 0 else "small") for value in row]
        for index, row in enumerate(rows)
    ]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def callout(text: str) -> Table:
    result = Table([[paragraph(text, "caveat")]], colWidths=[168 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return result


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, height - 10 * mm, "Computer Vision Course Project - p29")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story: list = [
        Spacer(1, 10 * mm),
        paragraph("FIRST WORKING MODEL", "title"),
        paragraph("People Counting with YOLO11 and ByteTrack", "subtitle"),
        callout(
            "Interpretation: the four camera-separated sequences were used to compare "
            "detectors and choose the submitted model. The observed CA of 0.6548 is a "
            "post-selection result, not an unbiased final test estimate."
        ),
        Spacer(1, 5 * mm),
    ]
    metrics = Table([
        [paragraph("0.6548", "metric"), paragraph("1.7021", "metric"),
         paragraph("0.4113", "metric"), paragraph("0.4324", "metric")],
        [paragraph("Observed mean CA", "metric_label"),
         paragraph("Curve MAE", "metric_label"),
         paragraph("MOTA", "metric_label"),
         paragraph("IDF1", "metric_label")],
    ], colWidths=[42 * mm] * 4)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
    ]))
    story += [
        metrics,
        Spacer(1, 4 * mm),
        paragraph("Repository", "h2"),
        paragraph("https://github.com/NikolayTaran/cv_people_counting"),
        paragraph("Project objective", "h1"),
        paragraph(
            "Count people entering and leaving a region under occlusion, crowding, "
            "and different camera views. The working system combines a YOLO11 person "
            "detector, custom ByteTrack tracker, and directional virtual-line counter."
        ),
        paragraph("Implemented architecture", "h1"),
        table([
            ["Stage", "Implementation", "Output"],
            ["Detection", "YOLO11s, person class, confidence floor 0.05", "Boxes and confidence"],
            ["Tracking", "Custom ByteTrack: Kalman + two-stage Hungarian IoU matching", "Persistent track IDs"],
            ["Counting", "Foot-point crossing, hysteresis, cooldown, gap handling", "Directional IN / OUT events"],
            ["Evaluation", "Predicted and GT trajectories use the same geometry", "CA, curve MAE, MOTA, IDF1"],
        ], [28 * mm, 100 * mm, 40 * mm]),
        PageBreak(),
        paragraph("1. Data loading and preprocessing", "h1"),
        paragraph(
            "MOTChallenge images, seqinfo metadata, and gt/gt.txt annotations are loaded "
            "in frame order. Only valid class-1 pedestrian annotations are retained. "
            "Boxes are clipped to the image; boxes under 2 px and training annotations "
            "with visibility below 0.10 are removed. Remaining boxes become normalized "
            "YOLO labels."
        ),
        table([
            ["Purpose", "Sequences", "Frames / images", "Boxes"],
            ["Train", "First 80% of MOT17-04 and MOT17-05", "1,510", "39,144"],
            ["Validation", "Final 20% of MOT17-04 and MOT17-05", "377", "11,397"],
            ["Model comparison", "MOT17-02, MOT20-01, MOT17-09, MOT17-11", "2,454", "Not trained on"],
            ["Untouched final test", "None in this submission", "-", "-"],
        ], [36 * mm, 79 * mm, 28 * mm, 25 * mm]),
        Spacer(1, 3 * mm),
        paragraph(
            "The development split is chronological to reduce mixing of near-duplicate "
            "adjacent frames. The comparison cameras never enter the training exporter. "
            "The code calls this split 'test', but its results informed model choice; it "
            "is not an independent final test. VisDrone was unavailable and has no "
            "imputed result."
        ),
        paragraph("Preprocessing and augmentation", "h2"),
        bullet("Training: 640 px letterbox. Inference: 1280 px for small/dense scenes and 640 px for the others."),
        bullet("Horizontal flip, translation, scale jitter, light mixup, and mosaic augmentation."),
        paragraph("2. Model and training setup", "h1"),
        paragraph(
            "YOLO11s starts from COCO weights and is fine-tuned with AdamW: lr0=0.001, "
            "cosine decay, weight decay 0.0005, batch 8, AMP, and seed 42. Training used "
            "an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.11.0+cu128, and Ultralytics "
            "8.4.175. Ultralytics chose the best checkpoint within each run on development "
            "validation fitness; the four comparison sequences were used to choose between "
            "detector configurations."
        ),
        table([
            ["Checkpoint", "Precision", "Recall", "mAP50", "mAP50-95"],
            ["3 epochs (submitted)", "0.9712", "0.8911", "0.9636", "0.6781"],
            ["20 epochs", "0.9644", "0.9147", "0.9643", "0.7246"],
        ], [48 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm]),
        PageBreak(),
        paragraph("3. Observed comparison results", "h1"),
        callout(
            "The same four sequences were used to compare four detectors and select "
            "the 3-epoch checkpoint. These figures describe that comparison; no "
            "untouched final test has yet been evaluated."
        ),
        Spacer(1, 4 * mm),
        paragraph(
            "The primary comparison metric is mean counting accuracy (CA), averaged "
            "equally over the four sequences. Higher CA, MOTA, and IDF1 are better; "
            "lower curve MAE is better."
        ),
        table([
            ["Detector", "Mean CA", "Curve MAE", "MOTA", "IDF1"],
            ["YOLO11n, COCO pretrained", "0.5501", "2.5913", "0.4361", "0.4446"],
            ["YOLO11s, COCO pretrained", "0.6222", "1.9747", "0.4352", "0.4671"],
            ["YOLO11s, MOT fine-tuned 3 epochs", "0.6548", "1.7021", "0.4113", "0.4324"],
            ["YOLO11s, MOT fine-tuned 20 epochs", "0.5794", "2.1542", "0.4080", "0.4277"],
        ], [65 * mm, 25 * mm, 28 * mm, 25 * mm, 25 * mm]),
        Spacer(1, 4 * mm),
        paragraph(
            "On this comparison set, the submitted checkpoint scores 0.1047 CA above "
            "YOLO11n and 0.0326 above COCO-pretrained YOLO11s. Those observed differences "
            "do not estimate expected gains on new cameras without selection bias."
        ),
        paragraph("Submitted model by comparison sequence", "h2"),
        table([
            ["Sequence", "Pred IN/OUT", "GT IN/OUT", "CA", "Curve MAE", "MOTA", "IDF1"],
            ["MOT17-02", "10 / 1", "16 / 5", "0.5238", "2.2500", "0.1797", "0.2997"],
            ["MOT20-01", "7 / 2", "21 / 5", "0.3462", "2.8750", "0.4700", "0.4433"],
            ["MOT17-09", "9 / 3", "10 / 3", "0.9231", "0.1000", "0.4770", "0.4535"],
            ["MOT17-11", "9 / 10", "11 / 12", "0.8261", "1.5833", "0.5185", "0.5331"],
        ], [32 * mm, 26 * mm, 25 * mm, 20 * mm, 25 * mm, 20 * mm, 20 * mm]),
        Spacer(1, 4 * mm),
        paragraph("Interpretation", "h2"),
        bullet("The short fine-tune has the highest observed CA on the comparison sequences."),
        bullet("The 20-epoch model raises dense MOT20-01 CA from 0.4231 to 0.6154 but scores lower overall."),
        bullet("Detection mAP and end-to-end counting quality differ: identity continuity near the line matters."),
        PageBreak(),
        paragraph("4. Problems and observations", "h1"),
        bullet(
            "An initial 20-epoch run failed inside OpenCV warpAffine with Windows data-loader "
            "workers. It resumed from last.pt with workers=0 and completed."
        ),
        bullet(
            "Development validation frames share the two training camera views, so high "
            "validation mAP did not guarantee the highest counting score across cameras."
        ),
        bullet(
            "The four camera-separated sequences were used for model choice. Their 0.6548 CA "
            "must be read as a post-selection comparison result. For an independent final "
            "estimate, freeze the pipeline now and evaluate it once on new untouched cameras."
        ),
        bullet(
            "The tracker lacks an appearance/ReID branch. Long occlusions can fragment "
            "or switch identities even with good detector recall."
        ),
        bullet(
            "Counting geometry is configured per camera. Arbitrary new video would require "
            "calibration or automatic entrance-region estimation."
        ),
        paragraph("5. Reproduction", "h1"),
        paragraph(
            "The repository includes the exporter, training script, submitted checkpoint, "
            "split-aware inference and evaluation, machine-readable results, and 39 passing "
            "unit tests. In the current CLI, '--split test' selects the four model-comparison "
            "sequences; the name does not make their result an untouched final test."
        ),
    ]
    commands = [
        "python scripts/prepare_training_data.py",
        "python scripts/train_detector.py --epochs 3 --imgsz 640 --batch 8 --workers 0 --device 0",
        "python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot.pt --device 0 --no-render --out-dir outputs/submission/runs",
        "python scripts/evaluate.py --split test --runs-dir outputs/submission/runs --results outputs/submission/test_results.json",
        "python -m pytest tests -q",
    ]
    for command in commands:
        block = Table([[paragraph(command, "small")]], colWidths=[168 * mm])
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.4, GRID),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story += [block, Spacer(1, 1.2 * mm)]
    story += [
        Spacer(1, 3 * mm),
        KeepTogether([
            paragraph("Submission artifacts", "h2"),
            bullet("Submitted model: models/people_yolo11s_mot.pt"),
            bullet("Full methodology: docs/FIRST_WORKING_MODEL.md"),
            bullet("Machine-readable comparisons and training histories: reports/"),
            bullet("Repository: https://github.com/NikolayTaran/cv_people_counting"),
        ]),
    ]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="First Working Model Submission",
        author="cv_people_counting project team",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()
