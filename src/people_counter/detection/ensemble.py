"""Two-detector box fusion before the existing tracker and counter."""

from __future__ import annotations

import numpy as np


def _iou(box: np.ndarray, other: np.ndarray) -> float:
    x1, y1 = np.maximum(box[:2], other[:2])
    x2, y2 = np.minimum(box[2:], other[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = max(0.0, float(other[2] - other[0])) * max(0.0, float(other[3] - other[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def fuse_boxes(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    iou_threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge matching boxes across models and preserve unmatched detections.

    Each cluster contains at most one box per detector, so two nearby people
    detected by the same model cannot be collapsed solely by this fusion.
    Coordinates are confidence-weighted; confidence is the maximum member
    confidence so a singleton remains usable by ByteTrack's high-score stage.
    """
    if not 0.0 < iou_threshold < 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    candidates = []
    for model_index, (boxes, scores) in enumerate((first, second)):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError("boxes and scores must have the same length")
        candidates.extend((box, float(score), model_index)
                          for box, score in zip(boxes, scores))
    if not candidates:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    candidates.sort(key=lambda item: item[1], reverse=True)
    clusters: list[dict] = []
    for box, score, model_index in candidates:
        best_index, best_iou = None, iou_threshold
        for index, cluster in enumerate(clusters):
            if model_index in cluster["models"]:
                continue
            overlap = _iou(box, cluster["box"])
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is None:
            clusters.append({
                "box": box.copy(), "weighted_sum": box * score,
                "weight": score, "score": score, "models": {model_index},
            })
        else:
            cluster = clusters[best_index]
            cluster["weighted_sum"] += box * score
            cluster["weight"] += score
            cluster["box"] = cluster["weighted_sum"] / cluster["weight"]
            cluster["score"] = max(cluster["score"], score)
            cluster["models"].add(model_index)

    clusters.sort(key=lambda cluster: cluster["score"], reverse=True)
    return (
        np.stack([cluster["box"] for cluster in clusters]).astype(np.float32),
        np.asarray([cluster["score"] for cluster in clusters], dtype=np.float32),
    )


class EnsembleDetector:
    """Call two detectors on the same frame and fuse their person boxes."""

    def __init__(self, first, second, fusion_iou: float = 0.55) -> None:
        if not 0.0 < fusion_iou < 1.0:
            raise ValueError("fusion_iou must be between zero and one")
        self.first = first
        self.second = second
        self.fusion_iou = fusion_iou
        self.device = getattr(first, "device", "unknown")

    def __call__(self, frame_bgr: np.ndarray, imgsz: int = 640,
                 conf: float = 0.05, iou: float = 0.7
                 ) -> tuple[np.ndarray, np.ndarray]:
        first = self.first(frame_bgr, imgsz=imgsz, conf=conf, iou=iou)
        second = self.second(frame_bgr, imgsz=imgsz, conf=conf, iou=iou)
        return fuse_boxes(first, second, self.fusion_iou)
