"""ByteTrack — custom implementation of the BYTE multi-object tracker.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.

Two-stage (BYTE) association, per frame:

  1. split detections into HIGH (conf >= high_thresh) and LOW;
  2. stage 1 — Hungarian matching of ALL tracks against HIGH detections
     using IoU distance, gate ``1 - iou < match_thresh`` (i.e. IoU > 0.2);
  3. stage 2 — tracks left unmatched by stage 1 are matched against LOW
     detections with a stricter gate (``1 - iou < match_thresh_low``);
     this is what makes ByteTrack strong under occlusion: weak boxes of
     partially occluded people keep their tracks alive instead of dying;
  4. unmatched HIGH detections spawn new tracks (LOW ones never do);
  5. unmatched tracks are marked missed: tentative tracks die immediately,
     confirmed tracks survive up to ``max_age`` frames.

Optionally compensates global camera motion by estimating the inter-frame
shift with phase correlation and translating all track states by it
(BoT-SORT style), which keeps the tracker usable on moving cameras.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import Track

_EPS = 1e-6


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two xyxy box arrays, shapes (N,4) and (M,4) -> (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


def _associate(track_boxes: np.ndarray, det_boxes: np.ndarray, gate: float):
    """Hungarian assignment on IoU distance with a cost gate.

    Returns (matches, unmatched_track_idx, unmatched_det_idx); a match is
    kept only if its cost (1 - IoU) is strictly below ``gate``.
    """
    n_t, n_d = len(track_boxes), len(det_boxes)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))
    cost = 1.0 - iou_matrix(track_boxes, det_boxes)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < gate]
    matched_t = {r for r, _ in matches}
    matched_d = {c for _, c in matches}
    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_dets = [j for j in range(n_d) if j not in matched_d]
    return matches, unmatched_tracks, unmatched_dets


# --------------------------------------------------------------------------
# camera-motion compensation (phase correlation)
# --------------------------------------------------------------------------
_windows: dict[tuple[int, int], np.ndarray] = {}


def _hann_window(shape) -> np.ndarray:
    """Separable Hann window (cv2.createHannWindow is gone in OpenCV 5)."""
    key = (int(shape[0]), int(shape[1]))
    if key not in _windows:
        h, w = key
        wy = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(h) / max(h - 1, 1))
        wx = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(w) / max(w - 1, 1))
        _windows[key] = (wy[:, None] * wx[None, :]).astype(np.float32)
    return _windows[key]


def phase_shift(prev_gray, cur_gray) -> tuple[float, float]:
    """Apparent displacement of image content between two grayscale frames.

    Returns (dx, dy) such that cur(x, y) ~= prev(x - dx, y - dy), i.e. how
    much the scene appears to have moved in the image plane. The sign
    convention is pinned by tests/test_bytetrack.py::test_phase_shift_sign.
    """
    if prev_gray is None or cur_gray is None:
        return 0.0, 0.0
    if prev_gray.shape != cur_gray.shape:
        return 0.0, 0.0
    prev = np.asarray(prev_gray, dtype=np.float32)
    cur = np.asarray(cur_gray, dtype=np.float32)
    if prev.std() < 1e-6 or cur.std() < 1e-6:
        return 0.0, 0.0
    try:
        (sx, sy), _response = cv2.phaseCorrelate(prev, cur, _hann_window(prev.shape))
    except cv2.error:  # pragma: no cover - degenerate frames
        return 0.0, 0.0
    return float(sx), float(sy)


class ByteTracker:
    """Stateful BYTE tracker (one instance per video sequence)."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        match_thresh_low: float = 0.5,
        max_age: int = 30,
        n_init: int = 3,
        min_box_area: float = 10.0,
        min_conf: float = 0.05,
        motion_compensation: bool = False,
    ) -> None:
        self.high_thresh = float(high_thresh)
        self.match_thresh = float(match_thresh)
        self.match_thresh_low = float(match_thresh_low)
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.min_box_area = float(min_box_area)
        self.min_conf = float(min_conf)
        self.motion_compensation = bool(motion_compensation)

        self.tracks: list[Track] = []
        self.frame_idx = 0
        self._next_id = 1
        self._prev_gray = None

    # ------------------------------------------------------------------ helpers
    def _track_boxes(self, indices) -> np.ndarray:
        if not indices:
            return np.zeros((0, 4), dtype=np.float64)
        return np.stack([self.tracks[i].kf.bbox for i in indices])

    def _new_track(self, bbox, conf: float) -> Track:
        track = Track(bbox, conf, self._next_id, n_init=self.n_init, max_age=self.max_age)
        track.start_frame = self.frame_idx
        self._next_id += 1
        self.tracks.append(track)
        return track

    # ------------------------------------------------------------------- update
    def update(self, dets, confs, gray=None) -> list[Track]:
        """Feed one frame; returns every track matched on this frame.

        dets  : (N, 4) xyxy boxes (already class-filtered to "person")
        confs : (N,) detector confidences
        gray  : optional grayscale frame (needed for motion compensation)
        """
        self.frame_idx += 1

        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 4)
        confs = np.asarray(confs, dtype=np.float64).reshape(-1)
        if dets.size:
            keep = (
                (confs >= self.min_conf)
                & (np.clip(dets[:, 2] - dets[:, 0], 0, None)
                   * np.clip(dets[:, 3] - dets[:, 1], 0, None) >= self.min_box_area)
            )
            dets, confs = dets[keep], confs[keep]

        # 1) predict every track one frame ahead
        for t in self.tracks:
            t.predict()

        # 2) camera-motion compensation: translate predictions into the
        #    current frame's coordinate system before associating
        if self.motion_compensation and gray is not None:
            if self._prev_gray is not None and np.shape(self._prev_gray) == np.shape(gray):
                dx, dy = phase_shift(self._prev_gray, gray)
                if dx or dy:
                    for t in self.tracks:
                        t.kf.shift(dx, dy)
            self._prev_gray = np.asarray(gray).copy()

        # 3) split detections by confidence
        high = confs >= self.high_thresh if dets.size else np.zeros(0, dtype=bool)
        high_dets = dets[high] if dets.size else dets.reshape(0, 4)
        low_dets = dets[~high] if dets.size else dets.reshape(0, 4)
        high_idx = np.flatnonzero(high) if dets.size else np.array([], dtype=int)
        low_idx = np.flatnonzero(~high) if dets.size else np.array([], dtype=int)

        # 4) stage 1 — all tracks vs high-confidence detections
        all_indices = list(range(len(self.tracks)))
        m1, u_t1, u_d1 = _associate(
            self._track_boxes(all_indices), high_dets, self.match_thresh
        )
        for ti, di in m1:
            self.tracks[ti].update(high_dets[di], float(confs[high_idx[di]]))

        # 5) stage 2 — tracks unmatched in stage 1 vs low-confidence detections
        u_t1_global = [all_indices[i] for i in u_t1]
        m2, u_t2_local, _u_d2 = _associate(
            self._track_boxes(u_t1_global), low_dets, self.match_thresh_low
        )
        for ti_local, di in m2:
            ti = u_t1_global[ti_local]
            self.tracks[ti].update(low_dets[di], float(confs[low_idx[di]]))

        # 6) unmatched high detections -> new tracks; low ones are dropped
        for di in u_d1:
            self._new_track(high_dets[di], float(confs[high_idx[di]]))

        # 7) still-unmatched tracks are marked missed
        for i in u_t2_local:
            self.tracks[u_t1_global[i]].mark_missed()

        # 8) purge deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # 9) output: tracks matched on this frame (any lifecycle state —
        #    the counting layer additionally requires confirmation)
        return [t for t in self.tracks if t.time_since_update == 0]
