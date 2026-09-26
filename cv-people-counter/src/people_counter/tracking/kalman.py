"""Kalman filter for bounding-box tracking (SORT / ByteTrack formulation).

State vector (8-dim, constant-velocity model, dt = 1 frame):

    x = [cx, cy, a, h, vx, vy, va, vh]

where (cx, cy) is the box center, ``a`` the aspect ratio w/h and ``h`` the
height. The measurement vector is the first four components.

The process / measurement noise is scaled by the current box height exactly
like the reference SORT / DeepSORT implementations (std_weight_position =
1/20, std_weight_velocity = 1/10), which makes the filter adaptive to object
size: the same parameters work for a 40 px pedestrian and a 300 px one.
"""
from __future__ import annotations

import numpy as np

STD_WEIGHT_POSITION = 1.0 / 20.0
STD_WEIGHT_VELOCITY = 1.0 / 10.0

_EPS = 1e-6


def xyxy_to_zhac(bbox) -> np.ndarray:
    """(x1, y1, x2, y2) -> measurement [cx, cy, a, h]."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(x2 - x1, _EPS)
    h = max(y2 - y1, _EPS)
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    return np.array([cx, cy, w / h, h], dtype=np.float64)


def zhac_to_xyxy(zha) -> np.ndarray:
    """[cx, cy, a, h] -> (x1, y1, x2, y2)."""
    cx, cy, a, h = [float(v) for v in zha]
    a = max(a, _EPS)
    h = max(h, _EPS)
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    dtype=np.float64)


def _state_transition() -> np.ndarray:
    """F with dt = 1 (constant velocity)."""
    f = np.eye(8, dtype=np.float64)
    for i in range(4):
        f[i, i + 4] = 1.0
    return f


def _measurement_matrix() -> np.ndarray:
    """H observes [cx, cy, a, h]."""
    h = np.zeros((4, 8), dtype=np.float64)
    h[0, 0] = h[1, 1] = h[2, 2] = h[3, 3] = 1.0
    return h


class BoxKalman:
    """Stateful Kalman filter for one bounding box (SORT/ByteTrack style)."""

    _F = _state_transition()
    _H = _measurement_matrix()

    def __init__(self, bbox):
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = xyxy_to_zhac(bbox)
        h = max(float(self.mean[3]), _EPS)
        std = [
            2.0 * STD_WEIGHT_POSITION * h,
            2.0 * STD_WEIGHT_POSITION * h,
            1e-2,
            2.0 * STD_WEIGHT_POSITION * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10.0 * STD_WEIGHT_VELOCITY * h,
        ]
        self.covariance = np.diag(np.square(std))

    # ------------------------------------------------------------------ predict
    def predict(self) -> None:
        """Advance the state one frame (constant velocity)."""
        h = max(float(self.mean[3]), _EPS)
        std_pos = STD_WEIGHT_POSITION * h
        std_vel = STD_WEIGHT_VELOCITY * h
        q = [
            std_pos ** 2, std_pos ** 2, 1e-2 ** 2, std_pos ** 2,
            std_vel ** 2, std_vel ** 2, 1e-5 ** 2, std_vel ** 2,
        ]
        motion_cov = np.diag(q)
        self.mean = self._F @ self.mean
        self.covariance = self._F @ self.covariance @ self._F.T + motion_cov

    # ------------------------------------------------------------------- update
    def _project(self):
        h = max(float(self.mean[3]), _EPS)
        std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ]
        innovation_cov = np.diag(np.square(std))
        return self._H @ self.mean, self._H @ self.covariance @ self._H.T + innovation_cov

    def update(self, bbox) -> None:
        """Correct the state with an observed (x1, y1, x2, y2) box."""
        z = xyxy_to_zhac(bbox)
        projected_mean, projected_cov = self._project()
        # solve S K^T = H P  for the Kalman gain via Cholesky (fast & stable)
        chol = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (self.covariance @ self._H.T).T)
        ).T
        innovation = z - projected_mean
        self.mean = self.mean + innovation @ kalman_gain.T
        self.covariance = (
            self.covariance - kalman_gain @ projected_cov @ kalman_gain.T
        )

    # ------------------------------------------------------------- camera shift
    def shift(self, dx: float, dy: float) -> None:
        """Translate the state by (dx, dy) — camera-motion compensation."""
        self.mean[0] += float(dx)
        self.mean[1] += float(dy)

    # ------------------------------------------------------------------ helpers
    @property
    def bbox(self) -> np.ndarray:
        """Current state as (x1, y1, x2, y2)."""
        return zhac_to_xyxy(self.mean[:4])
