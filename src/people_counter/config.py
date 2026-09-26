"""Sequence configuration loading (configs/sequences.yaml).

A config defines global defaults plus one entry per evaluated sequence:
data location, evaluation condition, detector imgsz, tracker overrides and
the counting geometry (line endpoints / ROI vertices as frame fractions).
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {
    "detector": {
        "weights": "yolo11n.pt",
        "min_conf": 0.05,       # low floor: ByteTrack stratifies confidences itself
        "nms_iou": 0.7,
    },
    "tracker": {
        "high_thresh": 0.5,
        "match_thresh": 0.8,
        "match_thresh_low": 0.5,
        "max_age": 30,
        "n_init": 3,
        "min_box_area": 10.0,
        "motion_compensation": False,
    },
    "counting": {
        "mode": "line",
        "min_hits": 3,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load the YAML config; returns {'defaults': ..., 'sequences': [...]}."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = _merge(DEFAULTS, raw.get("defaults", {}) or {})
    sequences = []
    for seq in raw.get("sequences", []):
        merged = {
            "detector": _merge(defaults["detector"], seq.get("detector", {})),
            "tracker": _merge(defaults["tracker"], seq.get("tracker", {})),
            "counting": _merge(defaults["counting"], seq.get("counting", {})),
        }
        merged.update({k: v for k, v in seq.items()
                       if k not in ("detector", "tracker", "counting")})
        sequences.append(merged)
    return {"defaults": defaults, "sequences": sequences}


def resolve_seq_paths(cfg: dict, repo_root: Path) -> dict:
    """Make data paths in a sequence config absolute (relative to repo root)."""
    out = dict(cfg)
    for key in ("img_dir", "gt", "seqinfo", "ann_dir", "seq_dir"):
        if out.get(key):
            p = Path(out[key])
            if not p.is_absolute():
                out[key] = str(repo_root / p)
    return out
