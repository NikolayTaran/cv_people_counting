"""YOLO11n detector wrapper (Ultralytics, COCO-pretrained, person class).

The detector is deliberately thin: the intellectual core of this project is
the tracker, the counting logic and the evaluation. We only need reliable
person boxes with raw confidence scores — ByteTrack does its own confidence
stratification, so we keep the detection threshold very low (0.05) and let
the tracker's two-stage association sort strong from weak boxes.
"""
from __future__ import annotations

import numpy as np

PERSON_CLASS = 0  # COCO


class YoloDetector:
    """Caches one Ultralytics model; imgsz / conf can be set per call."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        # imported lazily so that unit tests of tracking/counting do not pay
        # the torch import cost
        from ultralytics import YOLO  # noqa: WPS433 (local import on purpose)

        self._model = YOLO(weights)
        self.weights = weights

    def __call__(
        self,
        frame_bgr: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.05,
        iou: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect people on one BGR frame.

        Returns (boxes (N,4) float32 xyxy, confidences (N,) float32) sorted
        by confidence (descending).
        """
        results = self._model.predict(
            frame_bgr,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            classes=[PERSON_CLASS],
            device="cpu",
            verbose=False,
        )
        boxes, confs = [], []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes.append(res.boxes.xyxy.cpu().numpy())
            confs.append(res.boxes.conf.cpu().numpy())
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        all_boxes = np.concatenate(boxes).astype(np.float32)
        all_confs = np.concatenate(confs).astype(np.float32)
        order = np.argsort(-all_confs)
        return all_boxes[order], all_confs[order]
