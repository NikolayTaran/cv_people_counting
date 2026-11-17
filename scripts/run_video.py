#!/usr/bin/env python3
"""Run the trained detector, ByteTrack, and line counter on any local video.

This is a demonstration command. An unlabelled video has no ground-truth
counts, so its output must not be treated as a quantitative evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from people_counter.annotation import annotate_frame
from people_counter.datasets.gt_counter import build_counter
from people_counter.detection.ensemble import EnsembleDetector
from people_counter.detection.yolo_detector import YoloDetector
from people_counter.tracking.bytetrack import ByteTracker
from people_counter.tracking.track import foot_point
from people_counter.video import VideoSink


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="local MP4/MOV/WebM file")
    parser.add_argument("--weights", default="models/people_yolo11s_mot.pt")
    parser.add_argument("--ensemble-with", help="optional second detector checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-frames", type=int, help="limit processing for a short demo")
    parser.add_argument("--line", nargs=4, type=float,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        default=(0.5, 0.08, 0.5, 0.92),
                        help="line endpoints as fractions of frame dimensions")
    parser.add_argument("--in-direction", choices=("positive", "negative"),
                        default="positive", help="flip to reverse IN/OUT labels")
    parser.add_argument("--out-video", type=Path,
                        default=REPO / "outputs/demo/external/annotated.mp4")
    parser.add_argument("--out-json", type=Path,
                        default=REPO / "outputs/demo/external/events.json")
    args = parser.parse_args()

    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"video not found: {source}")
    if args.out_video.resolve() == source:
        parser.error("--out-video must differ from --input")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if any(not 0 <= v <= 1 for v in args.line):
        parser.error("--line coordinates must be between zero and one")
    if args.line[:2] == args.line[2:]:
        parser.error("--line endpoints must differ")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        parser.error(f"could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(source_frames, args.max_frames) if args.max_frames and source_frames else (
        args.max_frames or source_frames)
    if width < 2 or height < 2:
        capture.release()
        parser.error("video has invalid frame dimensions")

    def weights_path(value: str) -> str:
        candidate = REPO / value
        return str(candidate) if candidate.is_file() else value

    detector = YoloDetector(weights_path(args.weights), device=args.device)
    if args.ensemble_with:
        detector = EnsembleDetector(
            detector, YoloDetector(weights_path(args.ensemble_with), device=args.device))
    tracker = ByteTracker()
    counting_cfg = {
        "mode": "line", "min_hits": 3, "arm_dist": 10, "cooldown": 8,
        "line": {"p1": args.line[:2], "p2": args.line[2:],
                 "in_direction": args.in_direction},
    }
    counter = build_counter(counting_cfg, width, height)
    events = []
    frame_records = []
    in_count = out_count = frame_idx = 0
    started = time.time()

    try:
        with VideoSink(args.out_video, width, height, fps) as sink:
            while not total_frames or frame_idx < total_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes, confs = detector(frame, imgsz=args.imgsz, conf=0.05, iou=0.7)
                active = tracker.update(boxes, confs)
                for track in active:
                    if track.is_confirmed:
                        x, y = foot_point(track.bbox)
                        for event in counter.update(track.track_id, frame_idx,
                                                    track.hits, x, y):
                            events.append(event.__dict__)
                            in_count += event.kind == "in"
                            out_count += event.kind == "out"
                frame_records.append({
                    "frame": frame_idx,
                    "tracks": [[int(t.track_id), *[float(v) for v in t.bbox]]
                               for t in active],
                })
                annotated = annotate_frame(
                    frame, active, counting_cfg, in_count, out_count,
                    frame_idx, total_frames or source_frames, label="external demo")
                sink.write(annotated)
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"[video] {frame_idx} frames, IN {in_count}, OUT {out_count}",
                          flush=True)
    finally:
        capture.release()

    run = {
        "meta": {"name": source.stem, "dataset": "unlabelled external video",
                 "source": str(source), "frames": frame_idx, "fps": fps,
                 "width": width, "height": height, "imgsz": args.imgsz,
                 "counting_cfg": counting_cfg, "wall_seconds": round(time.time() - started, 2),
                 "video": str(args.out_video.resolve())},
        "events": events, "frame_records": frame_records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"[video] done: {frame_idx} frames, IN {in_count}, OUT {out_count}")
    print(f"[video] annotated: {args.out_video.resolve()}")
    print(f"[video] events: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
