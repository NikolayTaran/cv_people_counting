"""MOT metrics (MOTA, MOTP, IDF1, IDSW, ...) via the `motmetrics` package.

Protocol: greedy/Hungarian matching of predicted boxes to ground-truth boxes
per frame with IoU > 0.5, the standard CLEAR-MOT / ID metrics used on
MOTChallenge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# motmetrics 1.4 still calls np.asfarray, removed in NumPy 2.0 — provide the
# standard shim (asfarray == asarray with a float dtype) before importing it.
if not hasattr(np, "asfarray"):
    def _asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _asfarray


def _frame_rows(records: list[dict]) -> pd.DataFrame:
    """records: [{'frame': i, 'tracks': [[id, x1, y1, x2, y2], ...]}, ...]"""
    rows = []
    for rec in records:
        for tr in rec["tracks"]:
            tid, x1, y1, x2, y2 = tr
            rows.append({
                "FrameId": int(rec["frame"]) + 1,   # MOT GT frames are 1-based
                "Id": int(tid),
                "X": float(x1), "Y": float(y1),
                "Width": float(x2 - x1), "Height": float(y2 - y1),
                "Confidence": 1.0,
            })
    return pd.DataFrame(rows)


def _gt_rows(gt: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "FrameId": gt["frame"].astype(int),
        "Id": gt["id"].astype(int),
        "X": gt["x"].astype(float), "Y": gt["y"].astype(float),
        "Width": gt["w"].astype(float), "Height": gt["h"].astype(float),
    })
    return df


METRICS = ["mota", "motp", "idf1", "num_switches", "num_objects",
           "num_predictions", "num_false_positives", "num_misses",
           "mostly_tracked", "partially_tracked", "mostly_lost"]


def compute_mot_metrics(gt: pd.DataFrame, frame_records: list[dict]) -> dict:
    """Compute MOT metrics for one sequence.

    gt             : filtered GT dataframe (datasets.mot.load_gt_tracks)
    frame_records  : per-frame output of the pipeline (0-based frames)
    """
    import motmetrics as mm

    gt_df = _gt_rows(gt).set_index(["FrameId", "Id"])
    dt_df = _frame_rows(frame_records).set_index(["FrameId", "Id"])
    if gt_df.empty:
        return {k: None for k in METRICS}
    if dt_df.empty:  # motmetrics needs a matching (FrameId, Id) MultiIndex
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["FrameId", "Id"])
        dt_df = pd.DataFrame(columns=gt_df.columns, index=empty_idx)

    # clear the event-level index cache motmetrics keeps between runs
    mm.MOTAccumulator.auto_id = 0
    acc = mm.utils.compare_to_groundtruth(gt_df, dt_df, "iou", distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRICS, name="seq").to_dict("index")["seq"]

    out = {}
    for key, value in summary.items():
        if key == "num_switches":          # classic CLEAR-MOT name
            out["idsw"] = int(value)
        elif key in ("mota", "motp", "idf1"):
            out[key] = round(float(value), 4) if np.isfinite(value) else None
        elif key in ("mostly_tracked", "partially_tracked", "mostly_lost",
                     "num_objects", "num_predictions",
                     "num_false_positives", "num_misses"):
            out[key] = int(value)
        else:
            out[key] = value
    return out
