"""Ground-truth counting: run the SAME counting geometry over GT tracks.

The reference in/out or enter/exit totals are produced by feeding annotated
trajectories through the identical counter state machines used for
predictions — so evaluation measures the detector+tracker pipeline, not a
difference in counting logic.

MOT frame indices are 1-based; the counting layer is frame-index agnostic
(it only ever looks at differences), so no offset is needed.
"""
from __future__ import annotations

from ..counting.counter import (CountingEvent, LineCounter, RegionCounter,
                                summarize_events)
from .mot import gt_feet_by_frame


def build_counter(counting_cfg: dict, width: int, height: int):
    """Instantiate a counter from a config dict with fractional geometry.

    Line endpoints and ROI vertices are given as fractions of the frame size
    so one config works for every resolution.
    """
    mode = counting_cfg.get("mode", "line")
    if mode == "line":
        (x1, y1), (x2, y2) = counting_cfg["line"]["p1"], counting_cfg["line"]["p2"]
        return LineCounter(
            p1=(float(x1) * width, float(y1) * height),
            p2=(float(x2) * width, float(y2) * height),
            min_hits=int(counting_cfg.get("min_hits", 1)),
            arm_dist=float(counting_cfg.get("arm_dist", 10.0)),
            cooldown=int(counting_cfg.get("cooldown", 8)),
            max_travel=float(counting_cfg.get("max_travel", 250.0)),
            max_gap=int(counting_cfg.get("max_gap", 5)),
            in_direction=counting_cfg["line"].get("in_direction", "positive"),
        )
    if mode == "region":
        poly = [(float(x) * width, float(y) * height)
                for x, y in counting_cfg["region"]["polygon"]]
        return RegionCounter(
            polygon=poly,
            min_hits=int(counting_cfg.get("min_hits", 1)),
            hysteresis=int(counting_cfg.get("hysteresis", 2)),
            max_gap=int(counting_cfg.get("max_gap", 5)),
        )
    raise ValueError(f"unknown counting mode: {mode}")


def gt_count(gt, counting_cfg: dict, width: int, height: int) -> dict:
    """Run the counting geometry over GT tracks -> totals + full event list."""
    counter = build_counter(counting_cfg, width, height)
    feet = gt_feet_by_frame(gt)
    events: list[CountingEvent] = []
    for frame in sorted(feet):
        for tid, x, y in feet[frame]:
            events.extend(counter.update(tid, frame, hits=10 ** 6, x=x, y=y))
    return {"events": events, "totals": summarize_events(events)}
