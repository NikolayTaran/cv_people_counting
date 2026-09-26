#!/usr/bin/env python3
"""Run the detector-and-tracker pipeline on evaluation sequences.

    python scripts/run_pipeline.py                          # all sequences
    python scripts/run_pipeline.py --seq MOT17-04-FRCNN     # one sequence
    python scripts/run_pipeline.py --max-frames 30          # smoke test
    python scripts/run_pipeline.py --no-render              # no annotated video

Outputs: outputs/runs/<name>.json (per-frame tracks + counting events) and
outputs/videos/<name>.mp4 (annotated video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from people_counter.config import load_config                   # noqa: E402
from people_counter.detection.yolo_detector import YoloDetector  # noqa: E402
from people_counter.pipeline import run_sequence                # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs" / "sequences.yaml"))
    parser.add_argument("--seq", action="append",
                        help="sequence name (repeatable); default: all")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out-dir", default=str(REPO / "outputs" / "runs"))
    parser.add_argument("--video-dir", default=str(REPO / "outputs" / "videos"))
    args = parser.parse_args()

    config = load_config(args.config)
    sequences = config["sequences"]
    if args.seq:
        names = set(args.seq)
        sequences = [s for s in sequences if s["name"] in names]
        missing = names - {s["name"] for s in sequences}
        if missing:
            parser.error(f"unknown sequence(s): {sorted(missing)}")

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

    detector_cache: dict[str, YoloDetector] = {}
    for seq in runnable:
        weights = seq["detector"]["weights"]
        if weights not in detector_cache:
            print(f"[run] loading detector {weights} ...", flush=True)
            detector_cache[weights] = YoloDetector(weights)
        run_sequence(
            seq,
            repo_root=REPO,
            out_dir=Path(args.out_dir),
            detector=detector_cache[weights],
            render=not args.no_render,
            max_frames=args.max_frames,
            video_dir=Path(args.video_dir),
        )


if __name__ == "__main__":
    main()
