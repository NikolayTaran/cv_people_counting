#!/usr/bin/env python3
"""Build the progress-check PDF from recorded project results."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

from build_submission_pdf import bullet, callout, page_decoration, paragraph, table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "progress_check_report.pdf"
REPORTS = ROOT / "reports"


def read(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))


def summary(data: dict, metric: str) -> float:
    return data["by_condition"]["overall"][metric]


def scene_ca(data: dict, prefix: str) -> float:
    row = next(s for s in data["sequences"] if s["name"].startswith(prefix))
    return row["counting"]["counting_accuracy"]


def total_tracking(data: dict, metric: str) -> int:
    return sum(s["tracking"][metric] for s in data["sequences"])


def build() -> None:
    detectors = [
        ("COCO YOLO11n", read("yolo11n_baseline_test")),
        ("COCO YOLO11s", read("yolo11s_pretrained_test")),
        ("YOLO11s, 3 epochs", read("selected_3ep_test")),
        ("YOLO11s, 20 epochs", read("fine_tuned_20ep_test")),
        ("Two-detector ensemble", read("second_approach_ensemble_comparison")),
    ]
    baseline = read("second_approach_baseline_comparison")
    ensemble = detectors[-1][1]
    before_misses = total_tracking(baseline, "num_misses")
    after_misses = total_tracking(ensemble, "num_misses")
    before_fp = total_tracking(baseline, "num_false_positives")
    after_fp = total_tracking(ensemble, "num_false_positives")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    story = [
        Spacer(1, 7 * mm),
        paragraph("PROGRESS CHECK", "title"),
        paragraph("People Counting with YOLO11 and ByteTrack | p29 | 26 October 2026", "subtitle"),
        callout(
            "Current state: the end-to-end pipeline is working. Four single-detector "
            "configurations and a two-detector ensemble have been evaluated. All "
            "numbers below are preliminary comparisons, not an independent final test."
        ),
        Spacer(1, 4 * mm),
        paragraph("1. Completed experiments", "h1"),
        bullet(
            "Data: MOT17-04/05 yielded 1,510 training images (39,144 boxes) and "
            "377 chronological validation images (11,397 boxes). Four other videos "
            "from MOT17/MOT20 provided 2,454 comparison frames."
        ),
        bullet(
            "Approach 1: COCO YOLO11n and YOLO11s, plus YOLO11s fine-tuned for "
            "3 or 20 epochs, were run with the same custom ByteTrack and directional "
            "line counter. The 3-epoch model led the single-detector CA comparison."
        ),
        bullet(
            "Approach 2: the 3-epoch model and COCO YOLO11s run together; their "
            "person boxes are fused at cross-model IoU 0.55 before the unchanged "
            "tracker and counter. No extra detector training was required."
        ),
        bullet(
            "The repository includes data/training/inference/evaluation scripts, "
            "the selected checkpoint, recorded JSON results, and 42 passing tests."
        ),
        paragraph("2. Preliminary comparison", "h1"),
        paragraph(
            "Mean values weight each of the four comparison sequences equally. "
            "Higher CA, MOTA, and IDF1 are better; lower cumulative-count curve "
            "MAE is better. CA reflects errors in final IN/OUT totals."
        ),
    ]

    overall_rows = [["Configuration", "CA", "Curve MAE", "MOTA", "IDF1"]]
    for label, data in detectors:
        overall_rows.append([
            label,
            f'{summary(data, "mean_counting_accuracy"):.4f}',
            f'{summary(data, "mean_curve_mae"):.4f}',
            f'{summary(data, "mean_mota"):.4f}',
            f'{summary(data, "mean_idf1"):.4f}',
        ])
    story += [table(overall_rows, [65 * mm, 22 * mm, 33 * mm, 24 * mm, 24 * mm]),
              Spacer(1, 3 * mm)]

    scene_rows = [["Comparison video", "3-epoch CA", "Ensemble CA", "Change"]]
    for name in ("MOT17-02", "MOT20-01", "MOT17-09", "MOT17-11"):
        before = scene_ca(baseline, name)
        after = scene_ca(ensemble, name)
        scene_rows.append([name, f"{before:.4f}", f"{after:.4f}", f"{after - before:+.4f}"])
    story += [
        paragraph("Scene-level counting", "h2"),
        table(scene_rows, [50 * mm, 41 * mm, 42 * mm, 35 * mm]),
        Spacer(1, 3 * mm),
        paragraph(
            "The ensemble's mean CA is 0.6654 versus 0.6548 for the 3-epoch "
            "single model (+0.0106). Its largest gain is on crowded MOT20-01. "
            "It takes 164.7 s versus 125.4 s for the same 2,454 frames "
            "(about 31% more time)."
        ),
        PageBreak(),
        paragraph("3. Observations and difficulties", "h1"),
        bullet(
            f"The ensemble cuts tracker misses from {before_misses:,} to "
            f"{after_misses:,}, but raises false positives from {before_fp:,} "
            f"to {after_fp:,}. The trade-off helps dense-scene counting while "
            "mean MOTA and IDF1 decline."
        ),
        bullet(
            "Its final-count CA rises, but curve MAE worsens from 1.7021 to "
            "1.7858. A closer final total does not imply better counts throughout "
            "the video. CA also drops on MOT17-09 and MOT17-11."
        ),
        bullet(
            "Twenty epochs raise validation mAP50-95 from 0.6781 to 0.7246, "
            "yet comparison CA falls from 0.6548 to 0.5794. The validation "
            "frames share the two training cameras, so new-view performance "
            "cannot be inferred from that mAP alone."
        ),
        bullet(
            "Tracking has no appearance/ReID branch; long occlusions can break "
            "identities. Counting-line geometry is set manually per camera. "
            "A VisDrone aerial experiment remains unavailable."
        ),
        bullet(
            "An OpenCV warpAffine failure interrupted the 20-epoch run with "
            "Windows data-loader workers. It resumed successfully with workers=0."
        ),
        callout(
            "Protocol limitation: the four comparison videos were excluded from "
            "detector training, but their scores informed model choice. They are "
            "exploratory post-selection comparisons. The current 20% validation "
            "split also shares cameras with training."
        ),
        Spacer(1, 4 * mm),
        paragraph("4. Remaining work", "h1"),
        bullet(
            "Add camera-disjoint development data and use it to select fusion, "
            "confidence, and tracker settings without reusing the comparison set."
        ),
        bullet(
            "Review missed and extra IN/OUT events near each line; connect "
            "them to missed detections, false positives, identity switches, "
            "and direction errors."
        ),
        bullet(
            "Test track-lifetime/association changes and, if feasible, "
            "appearance features; measure throughput and memory consistently."
        ),
        bullet(
            "Obtain an aerial sequence and evaluate the existing polygon "
            "entry/exit mode with documented camera geometry."
        ),
        bullet(
            "Freeze the full pipeline, then evaluate once on untouched cameras. "
            "Report CA, curve MAE, MOTA, IDF1, speed, and failure cases."
        ),
        Spacer(1, 4 * mm),
        paragraph("Repository and evidence", "h2"),
        paragraph(
            "https://github.com/NikolayTaran/cv_people_counting | "
            "Source: docs/PROGRESS_CHECK.md and machine-readable results in reports/. "
            "The code names the comparison split 'test'; it is not an untouched final test."
        ),
    ]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="Progress Check Submission",
        author="cv_people_counting project team",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(OUTPUT)


if __name__ == "__main__":
    build()
