#!/usr/bin/env python3
"""Build the second-approach submission from archived evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from build_submission_pdf import (
    bullet,
    callout,
    page_decoration,
    paragraph,
    table,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "second_approach_report.pdf"
REPORTS = ROOT / "reports"


def result(name: str) -> dict:
    path = REPORTS / f"second_approach_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def overall(data: dict, field: str) -> float:
    return data["by_condition"]["overall"][field]


def ca(data: dict, name: str) -> float:
    sequence = next(item for item in data["sequences"] if item["name"].startswith(name))
    return sequence["counting"]["counting_accuracy"]


def counts(data: dict, name: str, key: str) -> str:
    sequence = next(item for item in data["sequences"] if item["name"].startswith(name))
    item = sequence["counting"]["mode_totals"][key]
    return f'{item["in"]} / {item["out"]}'


def build() -> None:
    baseline = result("baseline_comparison")
    ensemble = result("ensemble_comparison")
    development_baseline = result("baseline_development")
    development_ensemble = result("ensemble_development")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    story = [
        Spacer(1, 8 * mm),
        paragraph("SECOND APPROACH", "title"),
        paragraph("Two-detector ensemble for directional people counting", "subtitle"),
        callout(
            "Result: mean counting accuracy rises from 0.6548 to 0.6654 on four "
            "camera-separated comparison sequences. The gain is concentrated in the "
            "crowded MOT20-01 video; tracking metrics and speed worsen overall."
        ),
        Spacer(1, 5 * mm),
        paragraph("1. New approach", "h1"),
        paragraph(
            "The baseline uses one YOLO11s person detector fine-tuned for three epochs "
            "on MOT17-04/05, followed by our custom ByteTrack and directional line "
            "counter. The second approach runs that checkpoint and a COCO-pretrained "
            "YOLO11s checkpoint on every frame. It fuses their person boxes before "
            "the same tracker and counter. No additional training was needed."
        ),
        table([
            ["Stage", "First approach", "Second approach"],
            ["Detection", "Fine-tuned YOLO11s", "Fine-tuned + COCO YOLO11s"],
            ["Box processing", "Single-model NMS", "Per-model NMS, then box fusion"],
            ["Tracking/counting", "ByteTrack + virtual line", "Same ByteTrack + virtual line"],
        ], [35 * mm, 62 * mm, 71 * mm]),
        Spacer(1, 4 * mm),
        paragraph("2. Motivation", "h1"),
        paragraph(
            "The selected fine-tuned baseline reached only 0.3462 CA on crowded "
            "MOT20-01, versus 0.4231 for the COCO-pretrained YOLO11s in the first "
            "approach comparison. Fine-tuning helped other camera views. We expected "
            "the ensemble to recover additional partially occluded people near the "
            "counting line while preserving the fine-tuned detector's strengths."
        ),
        paragraph("3. Implementation differences", "h1"),
        bullet("Both detectors process the same frame at its configured resolution; person-only confidence floor is 0.05 and per-model NMS IoU is 0.70."),
        bullet("Boxes from different detectors are grouped when box IoU is at least 0.55. At most one box from each detector enters a group."),
        bullet("Coordinates are weighted by detector confidence; group confidence is the maximum. Unmatched boxes are retained."),
        bullet("Fusion runs before the unchanged tracker and counter. The CLI adds --ensemble-with and --fusion-iou; result JSON records both checkpoints."),
        PageBreak(),
        paragraph("4. Initial results", "h1"),
        paragraph(
            "Matched GPU reruns on 20 October 2026 used identical scene geometry, "
            "tracking, and evaluation settings. Mean values weight each of the four "
            "comparison sequences equally. Higher CA, MOTA and IDF1 are better; "
            "lower curve MAE is better."
        ),
    ]

    measures = [
        ("Mean counting accuracy", "mean_counting_accuracy", True),
        ("Mean curve MAE", "mean_curve_mae", False),
        ("Mean MOTA", "mean_mota", True),
        ("Mean IDF1", "mean_idf1", True),
    ]
    metric_rows = [["Metric", "Baseline", "Ensemble", "Change"]]
    for label, field, _ in measures:
        before = overall(baseline, field)
        after = overall(ensemble, field)
        metric_rows.append([label, f"{before:.4f}", f"{after:.4f}", f"{after - before:+.4f}"])
    metric_rows.append(["Processing, 2,454 frames", "125.4 s", "164.7 s", "+31% time"])
    story += [table(metric_rows, [61 * mm, 30 * mm, 31 * mm, 46 * mm]), Spacer(1, 4 * mm)]

    scene_rows = [["Sequence", "Baseline CA", "Ensemble CA", "Change", "Pred IN/OUT", "GT IN/OUT"]]
    for name in ("MOT17-02", "MOT20-01", "MOT17-09", "MOT17-11"):
        before = ca(baseline, name)
        after = ca(ensemble, name)
        scene_rows.append([
            name,
            f"{before:.4f}",
            f"{after:.4f}",
            f"{after - before:+.4f}",
            counts(ensemble, name, "pred"),
            counts(ensemble, name, "gt"),
        ])
    story += [
        paragraph("Per-sequence counting accuracy", "h2"),
        table(scene_rows, [31 * mm, 26 * mm, 27 * mm, 23 * mm, 32 * mm, 29 * mm]),
        Spacer(1, 4 * mm),
        paragraph(
            "The gain is largest on crowded MOT20-01 (+0.1923 CA). CA falls on "
            "MOT17-09 (-0.1539) and MOT17-11 (-0.0435). Mean curve MAE rises, "
            "MOTA and IDF1 fall, and the ensemble takes about 31% longer. This is "
            "a working second approach, not a universal improvement."
        ),
        callout(
            "Evaluation caveat: the four comparison videos were already used to "
            "select the baseline checkpoint in the first submission. These are "
            "exploratory post-selection results, not an unbiased final test. The "
            "development videos contain detector training frames."
        ),
        Spacer(1, 4 * mm),
        paragraph(
            "Development sanity check: mean CA was "
            f'{overall(development_baseline, "mean_counting_accuracy"):.4f} for the '
            "single detector and "
            f'{overall(development_ensemble, "mean_counting_accuracy"):.4f} for the '
            "ensemble on MOT17-04/05. These figures do not estimate generalization."
        ),
        paragraph("Reproduction and source", "h2"),
        paragraph(
            "Run scripts/run_pipeline.py --split test --ensemble-with yolo11s.pt "
            "--fusion-iou 0.55 --device 0 --no-render, then scripts/evaluate.py. "
            "The complete commands, code, and four machine-readable result files "
            "are in the repository:"
        ),
        paragraph("https://github.com/NikolayTaran/cv_people_counting"),
    ]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="Second Approach Submission",
        author="cv_people_counting project team",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()
