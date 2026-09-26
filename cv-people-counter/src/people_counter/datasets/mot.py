"""Readers for MOT17 / MOT20 / VisDrone-MOT sequences.

Both datasets provide per-frame images plus ground-truth tracks:
  MOT17/MOT20 : img/%06d.jpg, gt/gt.txt, seqinfo.ini
  VisDrone    : sequences/<seq>/*.jpg + annotations/<seq>.txt

Ground-truth filtering follows the official evaluation protocols:
  * MOT: keep rows with class == 1 (pedestrian) and flag == 1; class 7
    ("static person") and flagged rows are distractors and are excluded;
  * VisDrone: keep class 1 (pedestrian) and 2 (people), drop zero-area boxes.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SequenceInfo:
    name: str
    dataset: str                      # 'MOT17' | 'MOT20' | 'VisDrone'
    img_dir: Path
    gt_path: Path | None
    width: int
    height: int
    fps: int
    length: int                       # frames on disk
    img_files: list[Path]

    def __post_init__(self) -> None:
        if self.img_files is None:  # pragma: no cover
            self.img_files = []


def parse_seqinfo(path: Path) -> dict:
    """Parse a MOT seqinfo.ini file (keys keep their original case)."""
    parser = configparser.ConfigParser()
    parser.optionxform = str                      # preserve CamelCase keys
    parser.read(path, encoding="utf-8")
    section = parser["Sequence"] if parser.has_section("Sequence") else parser[
        next(iter(parser.sections()))]
    return {k: v for k, v in section.items()}


def list_image_files(img_dir: Path) -> list[Path]:
    """Sorted image list (numeric order — MOT frames are zero-padded)."""
    files = [p for p in Path(img_dir).iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(files, key=lambda p: (len(p.stem), p.stem))


def load_mot_sequence(name: str, dataset: str, seq_dir: Path,
                      fps_default: int = 30) -> SequenceInfo:
    seq_dir = Path(seq_dir)
    info = parse_seqinfo(seq_dir / "seqinfo.ini")
    img_dir = seq_dir / info.get("imDir", "img")
    img_files = list_image_files(img_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    return SequenceInfo(
        name=name,
        dataset=dataset,
        img_dir=img_dir,
        gt_path=gt_path if gt_path.is_file() else None,
        width=int(info.get("imWidth", "0")),
        height=int(info.get("imHeight", "0")),
        fps=int(info.get("frameRate", str(fps_default))),
        length=len(img_files),
        img_files=img_files,
    )


def load_visdrone_sequence(name: str, seq_dir: Path, ann_dir: Path,
                           fps: int = 10) -> SequenceInfo:
    seq_dir = Path(seq_dir)
    img_files = list_image_files(seq_dir)
    gt_path = Path(ann_dir) / f"{seq_dir.name}.txt"
    return SequenceInfo(
        name=name,
        dataset="VisDrone",
        img_dir=seq_dir,
        gt_path=gt_path if gt_path.is_file() else None,
        width=0,                       # taken from the first frame at runtime
        height=0,
        fps=fps,
        length=len(img_files),
        img_files=img_files,
    )


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
_GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "flag", "class", "vis"]


def load_gt_tracks(gt_path: Path, dataset: str) -> pd.DataFrame:
    """Ground-truth rows (already protocol-filtered) as a DataFrame.

    Columns: frame, id, x, y, w, h, vis  (frames are 1-based in MOT files).
    """
    path = Path(gt_path)
    if not path.is_file():
        return pd.DataFrame(columns=_GT_COLUMNS)
    if dataset in ("MOT17", "MOT20"):
        df = pd.read_csv(path, header=None, names=_GT_COLUMNS)
        df = df[(df["class"] == 1) & (df["flag"] == 1)]
    elif dataset == "VisDrone":
        cols = ["frame", "id", "x", "y", "w", "h", "score", "class",
                "truncation", "occlusion"]
        df = pd.read_csv(path, header=None, names=cols)
        df = df[df["class"].isin([1, 2])]
        df = df[(df["w"] > 0) & (df["h"] > 0)]
        df["flag"] = 1
        df["vis"] = 1.0
        df = df[_GT_COLUMNS]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    df = df.astype({"frame": int, "id": int, "x": float, "y": float,
                    "w": float, "h": float, "flag": int, "class": int,
                    "vis": float})
    return df.sort_values(["frame", "id"]).reset_index(drop=True)


def gt_feet_by_frame(gt: pd.DataFrame) -> dict[int, list[tuple[int, float, float]]]:
    """Map frame -> [(track_id, foot_x, foot_y), ...] for the counting layer."""
    out: dict[int, list[tuple[int, float, float]]] = {}
    if gt.empty:
        return out
    fx = gt["x"] + gt["w"] / 2.0
    fy = gt["y"] + gt["h"]
    for frame, tid, x, y in zip(gt["frame"].to_numpy(), gt["id"].to_numpy(),
                                fx.to_numpy(), fy.to_numpy()):
        out.setdefault(int(frame), []).append((int(tid), float(x), float(y)))
    return out


def gt_boxes_by_frame(gt: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map frame -> (N, 5) array [id, x, y, w, h] for MOT metrics."""
    out: dict[int, np.ndarray] = {}
    if gt.empty:
        return out
    for frame, group in gt.groupby("frame"):
        arr = np.column_stack([
            group["id"].to_numpy(),
            group["x"].to_numpy(),
            group["y"].to_numpy(),
            group["w"].to_numpy(),
            group["h"].to_numpy(),
        ])
        out[int(frame)] = arr
    return out


def gt_people_per_frame(gt: pd.DataFrame) -> list[float]:
    """Average number of annotated pedestrians per frame (density stat)."""
    if gt.empty:
        return [0.0]
    counts = gt.groupby("frame").size()
    return [float(counts.mean())]
