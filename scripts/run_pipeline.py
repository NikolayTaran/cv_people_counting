#!/usr/bin/env python3
"""Run the detector-and-tracker pipeline on evaluation sequences.

    python scripts/run_pipeline.py                          # all sequences
    python scripts/run_pipeline.py --seq MOT17-04-FRCNN     # one sequence
    python scripts/run_pipeline.py --max-frames 30          # smoke test
    python scripts/run_pipeline.py --no-render              # no annotated video

Outputs: outputs/runs/<name>.json (per-frame tracks + counting events) and
outputs/videos/<name>.mp4 (annotated video).
"""
# ruff: noqa: E402 -- CLI adds the repository's src directory to sys.path.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from people_counter.config import load_config
from people_counter.detection.ensemble import EnsembleDetector
from people_counter.detection.yolo_detector import YoloDetector
from people_counter.pipeline import run_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs" / "sequences.yaml"))
    parser.add_argument("--seq", action="append",
                        help="sequence name (repeatable); default: all")
    parser.add_argument("--split", choices=["development", "test"],
                        help="run only one protocol split")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--weights", help="override detector weights for every sequence")
    parser.add_argument("--ensemble-with", help="second detector weights for box fusion")
    parser.add_argument("--fusion-iou", type=float, default=0.55,
                        help="cross-model IoU needed to merge boxes (default: 0.55)")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, help="override inference size for every sequence")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out-dir", default=str(REPO / "outputs" / "runs"))
    parser.add_argument("--video-dir", default=str(REPO / "outputs" / "videos"))
    args = parser.parse_args()
    if not 0.0 < args.fusion_iou < 1.0:
        parser.error("--fusion-iou must be between zero and one")

    config = load_config(args.config)
    sequences = config["sequences"]
    if args.seq:
        names = set(args.seq)
        sequences = [s for s in sequences if s["name"] in names]
        missing = names - {s["name"] for s in sequences}
        if missing:
            parser.error(f"unknown sequence(s): {sorted(missing)}")
    if args.split:
        sequences = [s for s in sequences if s.get("split") == args.split]

    # skip sequences whose data is not on disk (e.g. VisDrone not downloaded)
    runnable = []
    for seq in sequences:
        seq_dir = Path(seq.get("seq_dir", ""))
        if seq_dir and not (REPO / seq_dir).is_dir():
            print(f"[run] SKIP {seq['name']}: {seq_dir} not found "
                  f"(run scripts/download_data.py first)")
            continue
        runnable.append(seq)
    if not runnable:
        print("[run] nothing to do")
        return

    detector_cache: dict[tuple[str, str | None], YoloDetector | EnsembleDetector] = {}
    for seq in runnable:
        if args.weights:
            seq["detector"]["weights"] = args.weights
        if args.imgsz:
            seq["imgsz"] = args.imgsz
        weights = seq["detector"]["weights"]
        candidate = REPO / weights
        if candidate.is_file():
            weights = str(candidate)
            seq["detector"]["weights"] = weights
        second_weights = args.ensemble_with
        if second_weights:
            second_candidate = REPO / second_weights
            if second_candidate.is_file():
                second_weights = str(second_candidate)
            seq["detector"]["ensemble_with"] = second_weights
            seq["detector"]["fusion_iou"] = args.fusion_iou
        cache_key = (weights, second_weights)
        if cache_key not in detector_cache:
            print(f"[run] loading detector {weights} ...", flush=True)
            first = YoloDetector(weights, device=args.device)
            if second_weights:
                print(f"[run] loading second detector {second_weights} ...", flush=True)
                second = YoloDetector(second_weights, device=args.device)
                detector_cache[cache_key] = EnsembleDetector(
                    first, second, fusion_iou=args.fusion_iou)
            else:
                detector_cache[cache_key] = first
        run_sequence(
            seq,
            repo_root=REPO,
            out_dir=Path(args.out_dir),
            detector=detector_cache[cache_key],
            render=not args.no_render,
            max_frames=args.max_frames,
            video_dir=Path(args.video_dir),
        )


if __name__ == "__main__":
    main()
