"""YOLO11 person detector wrapper used by single-model and ensemble pipelines.

ByteTrack stratifies raw confidence scores, so inference keeps a low 0.05
confidence floor and lets the tracker's two-stage association handle weak
boxes from partially occluded pedestrians.
"""
from __future__ import annotations

import numpy as np

PERSON_CLASS = 0  # COCO


class YoloDetector:
    """Caches one Ultralytics model; imgsz / conf can be set per call."""

    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto") -> None:
        # imported lazily so that unit tests of tracking/counting do not pay
        # the torch import cost
        import torch
        from ultralytics import YOLO  # local import keeps lightweight tests fast

        self._model = YOLO(weights)
        self.weights = weights
        self.device = "0" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else str(device)
        )

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
            device=self.device,
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
