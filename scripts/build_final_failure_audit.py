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


def nearby(events: list[dict], frame: int) -> list[dict]:
    return [event for event in events if abs(event["frame"] - frame) <= WINDOW]


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
        cases.append({
            "sequence": name,
            "anchor": anchor,
            "anchor_event": selected,
            "window_frames": WINDOW,
            "nearby_gt_events": nearby(gt_events, frame),
            "nearby_pred_events": nearby(pred_events, frame),
            "nearby_single_detector_events": nearby(
                [event_dict(e) for e in baseline["events"]], frame
            ),
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
