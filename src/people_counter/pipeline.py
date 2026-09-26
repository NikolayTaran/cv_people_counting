"""End-to-end pipeline: frames -> detection -> tracking -> counting -> video."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .annotation import annotate_frame
from .config import resolve_seq_paths
from .counting.counter import CountingEvent
from .datasets.gt_counter import build_counter
from .datasets.mot import (SequenceInfo, load_mot_sequence, load_visdrone_sequence)
from .tracking.bytetrack import ByteTracker
from .tracking.track import foot_point
from .video import VideoSink


def _load_sequence(cfg: dict) -> SequenceInfo:
    dataset = cfg["dataset"]
    if dataset in ("MOT17", "MOT20"):
        return load_mot_sequence(cfg["name"], dataset, Path(cfg["seq_dir"]))
    if dataset == "VisDrone":
        return load_visdrone_sequence(
            cfg["name"], Path(cfg["seq_dir"]), Path(cfg["ann_dir"]),
            fps=int(cfg.get("fps", 10)))
    raise ValueError(f"unknown dataset: {dataset}")


def _frame_size(info: SequenceInfo) -> tuple[int, int]:
    if info.width and info.height:
        return info.width, info.height
    frame = cv2.imread(str(info.img_files[0]))  # VisDrone has no seqinfo
    h, w = frame.shape[:2]
    return w, h


def run_sequence(seq_cfg: dict, repo_root: Path, out_dir: Path,
                 detector, render: bool = True, max_frames: int | None = None,
                 video_dir: Path | None = None) -> dict:
    """Process one sequence end-to-end and persist its run artifacts.

    Returns a summary dict; detailed per-frame data is written to
    ``out_dir / '<name>.json'`` and the annotated video (optionally) to
    ``video_dir / '<name>.mp4'``.
    """
    cfg = resolve_seq_paths(seq_cfg, repo_root)
    info = _load_sequence(cfg)
    width, height = _frame_size(info)

    cap = max_frames if max_frames is not None else cfg.get("max_frames")
    n_frames = min(info.length, int(cap)) if cap else info.length
    img_files = info.img_files[:n_frames]

    tracker = ByteTracker(**{
        k: v for k, v in cfg["tracker"].items()
        if k in ("high_thresh", "match_thresh", "match_thresh_low",
                 "max_age", "n_init", "min_box_area", "motion_compensation")
    })
    counter = build_counter(cfg["counting"], width, height)

    det = cfg["detector"]
    events: list[CountingEvent] = []
    frame_records: list[dict] = []
    render_path = None
    sink = None
    if render:
        render_dir = video_dir or (out_dir.parent / "videos")
        render_path = Path(render_dir) / f"{cfg['name']}.mp4"
        sink = VideoSink(render_path, width, height, info.fps)

    t0 = time.time()
    mc = bool(cfg["tracker"].get("motion_compensation", False))
    try:
        for i, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise RuntimeError(f"could not read frame: {img_path}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if mc else None

            boxes, confs = detector(
                frame, imgsz=int(cfg.get("imgsz", 640)),
                conf=float(det["min_conf"]), iou=float(det["nms_iou"]))
            active = tracker.update(boxes, confs, gray)

            for tr in active:
                if tr.is_confirmed:
                    fx, fy = foot_point(tr.bbox)
                    events.extend(
                        counter.update(tr.track_id, i, tr.hits, fx, fy))

            frame_records.append({
                "frame": i,
                "tracks": [[int(tr.track_id),
                            float(tr.bbox[0]), float(tr.bbox[1]),
                            float(tr.bbox[2]), float(tr.bbox[3])]
                           for tr in active],
            })

            if sink is not None:
                in_c = sum(1 for e in events if e.kind in ("in", "enter"))
                out_c = sum(1 for e in events if e.kind in ("out", "exit"))
                annotated = annotate_frame(
                    frame.copy(), active, cfg["counting"],
                    in_c, out_c, i, n_frames, label=cfg["name"])
                sink.write(annotated)

            if (i + 1) % 200 == 0:
                print(f"  [{cfg['name']}] {i + 1}/{n_frames} frames, "
                      f"{len(tracker.tracks)} tracks, {len(events)} events",
                      flush=True)
    finally:
        if sink is not None:
            sink.close()

    wall = time.time() - t0
    run = {
        "meta": {
            "name": cfg["name"], "dataset": info.dataset,
            "condition": cfg.get("condition", "n/a"),
            "tags": cfg.get("tags", []),
            "width": width, "height": height, "fps": info.fps,
            "frames": n_frames, "imgsz": int(cfg.get("imgsz", 640)),
            "motion_compensation": mc,
            "counting_cfg": cfg["counting"],
            "tracker_cfg": cfg["tracker"],
            "wall_seconds": round(wall, 2),
            "processing_fps": round(n_frames / wall, 2) if wall else None,
            "video": str(render_path) if render_path else None,
        },
        "events": [e.__dict__ for e in events],
        "frame_records": frame_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg['name']}.json").write_text(
        json.dumps(run, indent=1), encoding="utf-8")

    print(f"[pipeline] {cfg['name']}: {n_frames} frames in {wall:.1f}s "
          f"({n_frames / wall:.1f} fps), {len(events)} counting events",
          flush=True)
    return run
