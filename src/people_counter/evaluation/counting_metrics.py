"""Counting accuracy metrics.

Definitions (n = number of evaluation checkpoints):
* absolute errors      e_in = |pred_in - gt_in|, e_out = |pred_out - gt_out|
* counting accuracy    CA = max(0, 1 - (e_in + e_out) / max(gt_in + gt_out, 1))
  — the standard cross-line counting accuracy from traffic analysis;
* curve MAE            mean over checkpoints t in {50, 100, ...} and both
  directions d of |cum_pred_d(t) - cum_gt_d(t)| — captures how quickly
  errors accumulate, not just the final totals.
"""
from __future__ import annotations

from ..counting.counter import CountingEvent

_DIRECTIONS = (("in", "out"), ("enter", "exit"))


def _totals_for(events: list[CountingEvent]) -> dict[str, int]:
    totals = {"in": 0, "out": 0}
    for e in events:
        if e.kind in ("in", "enter"):
            totals["in"] += 1
        elif e.kind in ("out", "exit"):
            totals["out"] += 1
    return totals


def _cumulative_counts(events: list[CountingEvent], n_frames: int,
                       checkpoints: list[int]) -> dict[int, dict[str, int]]:
    at = {cp: {"in": 0, "out": 0} for cp in checkpoints}
    for e in events:
        kind = "in" if e.kind in ("in", "enter") else "out"
        for cp in checkpoints:
            if e.frame <= cp:
                at[cp][kind] += 1
    return at


def counting_metrics(pred_events: list[CountingEvent],
                     gt_events: list[CountingEvent],
                     n_frames: int,
                     checkpoint_every: int = 50) -> dict:
    pred = _totals_for(pred_events)
    gt = _totals_for(gt_events)

    e_in = abs(pred["in"] - gt["in"])
    e_out = abs(pred["out"] - gt["out"])
    ca = max(0.0, 1.0 - (e_in + e_out) / max(gt["in"] + gt["out"], 1))

    checkpoints = [t for t in range(checkpoint_every, n_frames + 1, checkpoint_every)]
    if not checkpoints:
        checkpoints = [max(n_frames, 1)]
    pred_cum = _cumulative_counts(pred_events, n_frames, checkpoints)
    gt_cum = _cumulative_counts(gt_events, n_frames, checkpoints)
    diffs = [
        abs(pred_cum[cp][d] - gt_cum[cp][d])
        for cp in checkpoints
        for d in ("in", "out")
    ]
    curve_mae = float(sum(diffs) / len(diffs)) if diffs else 0.0

    return {
        "mode_totals": {"pred": pred, "gt": gt},
        "pred_in": pred["in"], "pred_out": pred["out"],
        "gt_in": gt["in"], "gt_out": gt["out"],
        "error_in": e_in, "error_out": e_out,
        "counting_accuracy": round(ca, 4),
        "curve_mae": round(curve_mae, 4),
    }
