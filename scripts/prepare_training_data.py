#!/usr/bin/env python3
"""Create a leakage-aware YOLO person dataset from MOT development videos.

The project protocol reserves MOT17-04 and MOT17-05 for development. Each
video is split chronologically: the first 80% is used for training and the
last 20% for validation. Test videos never enter this exporter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from people_counter.datasets.mot import load_gt_tracks, load_mot_sequence

DEFAULT_SEQUENCES = ("MOT17-04-FRCNN", "MOT17-05-FRCNN")


def _safe_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        # Hard links are space efficient on Windows. Copy only when the data
        # directory and the repository happen to live on different volumes.
        import shutil
        shutil.copy2(source, target)


def _labels_for_frame(rows, width: int, height: int, min_visibility: float) -> list[str]:
    labels: list[str] = []
    for row in rows.itertuples(index=False):
        if float(row.vis) < min_visibility:
            continue
        x1 = max(0.0, min(float(row.x), width - 1.0))
        y1 = max(0.0, min(float(row.y), height - 1.0))
        x2 = max(x1, min(float(row.x + row.w), float(width)))
        y2 = max(y1, min(float(row.y + row.h), float(height)))
        w, h = x2 - x1, y2 - y1
        if w < 2.0 or h < 2.0:
            continue
        xc = (x1 + x2) / (2.0 * width)
        yc = (y1 + y2) / (2.0 * height)
        labels.append(f"0 {xc:.6f} {yc:.6f} {w / width:.6f} {h / height:.6f}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", choices=DEFAULT_SEQUENCES)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--min-visibility", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "yolo_people")
    args = parser.parse_args()
    if not 0.05 <= args.val_fraction <= 0.5:
        parser.error("--val-fraction must be between 0.05 and 0.5")

    selected = args.sequence or list(DEFAULT_SEQUENCES)
    output = args.out.resolve()
    stats = {"train_images": 0, "val_images": 0, "train_boxes": 0,
             "val_boxes": 0, "sequences": selected,
             "val_fraction": args.val_fraction,
             "min_visibility": args.min_visibility}

    for name in selected:
        seq_dir = REPO / "data" / "MOT17" / "train" / name
        seq = load_mot_sequence(name, "MOT17", seq_dir)
        if not seq.img_files or seq.gt_path is None:
            raise FileNotFoundError(f"missing data for {name}; run download_data.py mot17")
        gt = load_gt_tracks(seq.gt_path, "MOT17")
        by_frame = {int(frame): rows for frame, rows in gt.groupby("frame")}
        split_at = round(len(seq.img_files) * (1.0 - args.val_fraction))

        for index, source in enumerate(seq.img_files, start=1):
            split = "train" if index <= split_at else "val"
            stem = f"{name}_{index:06d}"
            image_target = output / "images" / split / f"{stem}{source.suffix.lower()}"
            label_target = output / "labels" / split / f"{stem}.txt"
            _safe_hardlink(source, image_target)
            rows = by_frame.get(index, gt.iloc[0:0])
            labels = _labels_for_frame(rows, seq.width, seq.height, args.min_visibility)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(labels) + ("\n" if labels else ""),
                                    encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats[f"{split}_boxes"] += len(labels)

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }, sort_keys=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"dataset: {dataset_yaml}")


if __name__ == "__main__":
    main()
