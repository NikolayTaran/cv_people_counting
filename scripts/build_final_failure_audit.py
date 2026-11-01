#!/usr/bin/env python3
"""Archive three event-level counting errors from the existing ensemble runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from people_counter.config import load_config  # noqa: E402
from people_counter.datasets.gt_counter import gt_count  # noqa: E402
from people_counter.datasets.mot import load_gt_tracks  # noqa: E402


CASES = [
    ("MOT20-01", 70, "gt", "in"),
    ("MOT17-09-FRCNN", 252, "pred", "out"),
    ("MOT17-11-FRCNN", 146, "gt", "in"),
]
WINDOW = 12


def event_dict(event) -> dict:
    if isinstance(event, dict):
        return {key: event[key] for key in ("frame", "track_id", "kind", "x", "y")}
    return {key: getattr(event, key) for key in ("frame", "track_id", "kind", "x", "y")}


def nearby(events: list[dict], anchor_frame_zero: int,
           frame_offset: int) -> list[dict]:
    return [event for event in events
            if abs(event["frame"] - frame_offset - anchor_frame_zero) <= WINDOW]


def box_iou(first, second) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union else 0.0


def gt_box(row) -> tuple[float, float, float, float]:
    return (row.x, row.y, row.x + row.w, row.y + row.h)


def track_diagnostic(gt, run: dict, event: dict, anchor: str) -> dict:
    frame = event["frame"]
    if anchor == "gt":
        # Run records are 0-based; MOT ground-truth frames are 1-based.
        row = gt[(gt["frame"] == frame) & (gt["id"] == event["track_id"])].iloc[0]
        tracks = run["frame_records"][frame - 1]["tracks"]
        ranked = sorted(
            ((box_iou(gt_box(row), track[1:]), int(track[0])) for track in tracks),
            reverse=True,
        )
        overlap, track_id = ranked[0] if ranked else (0.0, None)
        return {
            "gt_frame": frame,
            "best_pred_track_id": track_id,
            "best_pred_track_iou": round(overlap, 4),
        }

    samples = []
    for run_frame in (frame - 1, frame, frame + 1):
        tracks = run["frame_records"][run_frame]["tracks"]
        track = next(t for t in tracks if int(t[0]) == event["track_id"])
        rows = gt[gt["frame"] == run_frame + 1]
        ranked = sorted(
            ((box_iou(track[1:], gt_box(row)), int(row.id))
             for _, row in rows.iterrows()),
            reverse=True,
        )
        overlap, gt_id = ranked[0] if ranked else (0.0, None)
        samples.append({
            "pred_frame": run_frame,
            "aligned_gt_frame": run_frame + 1,
            "pred_track_id": event["track_id"],
            "foot_x": round((track[1] + track[3]) / 2, 2),
            "best_gt_id": gt_id,
            "best_gt_iou": round(overlap, 4),
        })
    return {"pred_track_samples": samples}


def main() -> None:
    config = {seq["name"]: seq for seq in load_config(ROOT / "configs/sequences.yaml")["sequences"]}
    runs = ROOT / "outputs" / "second_approach" / "comparison" / "runs"
    baseline_runs = ROOT / "outputs" / "second_approach" / "baseline_comparison" / "runs"
    cases = []
    for name, frame, anchor, kind in CASES:
        seq = config[name]
        run = json.loads((runs / f"{name}.json").read_text(encoding="utf-8"))
        baseline = json.loads((baseline_runs / f"{name}.json").read_text(encoding="utf-8"))
        meta = run["meta"]
        gt_path = ROOT / seq["seq_dir"] / "gt" / "gt.txt"
        gt = load_gt_tracks(gt_path, seq["dataset"])
        gt = gt[gt["frame"] <= meta["frames"]]
        gt_events = [event_dict(e) for e in gt_count(
            gt, meta["counting_cfg"], meta["width"], meta["height"]
        )["events"]]
        pred_events = [event_dict(e) for e in run["events"]]
        anchor_events = gt_events if anchor == "gt" else pred_events
        selected = next(e for e in anchor_events if e["frame"] == frame and e["kind"] == kind)
        anchor_frame_zero = frame - 1 if anchor == "gt" else frame
        cases.append({
            "sequence": name,
            "anchor": anchor,
            "anchor_event": selected,
            "window_frames": WINDOW,
            "frame_convention": "GT frames are 1-based; predicted frames are 0-based",
            "nearby_gt_events": nearby(gt_events, anchor_frame_zero, 1),
            "nearby_pred_events": nearby(pred_events, anchor_frame_zero, 0),
            "nearby_single_detector_events": nearby(
                [event_dict(e) for e in baseline["events"]], anchor_frame_zero, 0
            ),
            "track_diagnostic": track_diagnostic(gt, run, selected, anchor),
            "source_run": f"outputs/second_approach/comparison/runs/{name}.json",
            "source_single_detector_run": (
                f"outputs/second_approach/baseline_comparison/runs/{name}.json"
            ),
            "source_gt": str(gt_path.relative_to(ROOT)).replace("\\", "/"),
            "interpretation_limit": (
                "Temporal event proximity does not establish person-level correspondence "
                "or uniquely identify the causal detector/tracker error."
            ),
        })
    output = ROOT / "reports" / "final_failure_cases.json"
    output.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
