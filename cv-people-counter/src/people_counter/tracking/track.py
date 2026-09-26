"""Track lifecycle management (tentative -> confirmed -> deleted)."""
from __future__ import annotations

from enum import Enum

import numpy as np

from .kalman import BoxKalman


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    """A single tracked object.

    * a track is created from an unmatched *high-confidence* detection,
    * it becomes CONFIRMED after ``n_init`` consecutive hits — only confirmed
      tracks are allowed to produce counting events,
    * a tentative track dies on its first miss (guards against detector noise),
    * a confirmed track survives ``max_age`` missed frames (occlusion), during
      which it keeps being predicted and stays matchable (ByteTrack behaviour).
    """

    def __init__(self, bbox, conf: float, track_id: int,
                 n_init: int = 3, max_age: int = 30) -> None:
        self.kf = BoxKalman(bbox)
        self.track_id = track_id
        self.conf = float(conf)
        self.n_init = int(n_init)
        self.max_age = int(max_age)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame: int | None = None  # set by the tracker (frame idx)
        self.state = TrackState.CONFIRMED if n_init <= 1 else TrackState.TENTATIVE

    # ------------------------------------------------------------------ lifecycle
    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, conf: float) -> None:
        self.kf.update(bbox)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    # ------------------------------------------------------------------ helpers
    @property
    def bbox(self) -> np.ndarray:
        return self.kf.bbox

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"Track(id={self.track_id}, state={self.state.name}, "
                f"hits={self.hits}, age={self.age}, tsu={self.time_since_update})")


def foot_point(bbox: np.ndarray) -> tuple[float, float]:
    """Bottom-center of a box — the counting anchor for pedestrians."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, y2
