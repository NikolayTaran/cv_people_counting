"""Frame annotation: boxes, IDs, counting geometry, HUD counters."""
from __future__ import annotations

import cv2
import numpy as np

# palette (BGR) — kept in sync with the dashboard accents
C_IN = (44, 197, 129)        # emerald-500  -> "in"
C_OUT = (245, 158, 11)       # amber-500    -> "out"
C_LINE = (244, 244, 245)     # zinc-100
C_HUD_BG = (24, 24, 27)      # zinc-900
C_HUD_FG = (244, 244, 245)   # zinc-100


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct color per track id (golden-angle hue)."""
    hue = int((track_id * 137) % 256)
    m = np.zeros((1, 1, 3), dtype=np.uint8)
    m[0, 0] = (hue, 200, 255)  # HSV -> BGR via cv2
    bgr = cv2.cvtColor(m, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draw a box + id label per track (Track objects with .bbox/.track_id)."""
    for tr in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in tr.bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        color = _id_color(tr.track_id)
        thick = 2 if tr.is_confirmed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{tr.track_id}"
        if not tr.is_confirmed:
            label += "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 17, 17), 1, cv2.LINE_AA)
    return frame


def draw_counting_geometry(frame: np.ndarray, counting_cfg: dict,
                           width: int, height: int) -> np.ndarray:
    mode = counting_cfg.get("mode", "line")
    if mode == "line":
        (x1, y1), (x2, y2) = counting_cfg["line"]["p1"], counting_cfg["line"]["p2"]
        p1 = (int(x1 * width), int(y1 * height))
        p2 = (int(x2 * width), int(y2 * height))
        cv2.line(frame, p1, p2, C_LINE, 2, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(frame, p, 5, C_LINE, -1, cv2.LINE_AA)
    elif mode == "region":
        pts = np.array(
            [[int(x * width), int(y * height)]
             for x, y in counting_cfg["region"]["polygon"]],
            dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 255))
        frame[:] = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
        cv2.polylines(frame, [pts], True, C_LINE, 2, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, in_count: int, out_count: int,
             frame_idx: int, total_frames: int, label: str = "") -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w, panel_h = 300, 96
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), C_HUD_BG, -1)
    cv2.putText(frame, f"IN  {in_count}", (28, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_IN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT {out_count}", (170, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_OUT, 2, cv2.LINE_AA)
    meta = f"frame {frame_idx + 1}/{total_frames}"
    if label:
        meta = f"{label}  |  {meta}"
    cv2.putText(frame, meta, (28, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                C_HUD_FG, 1, cv2.LINE_AA)
    return frame


def annotate_frame(frame: np.ndarray, tracks: list, counting_cfg: dict,
                   in_count: int, out_count: int, frame_idx: int,
                   total_frames: int, label: str = "") -> np.ndarray:
    h, w = frame.shape[:2]
    frame = draw_counting_geometry(frame, counting_cfg, w, h)
    frame = draw_tracks(frame, tracks)
    frame = draw_hud(frame, in_count, out_count, frame_idx, total_frames, label)
    return frame
