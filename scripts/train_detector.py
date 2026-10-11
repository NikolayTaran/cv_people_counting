#!/usr/bin/env python3
"""Fine-tune YOLO11s for pedestrian detection on the MOT development split."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=REPO / "data" / "yolo_people" / "dataset.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="yolo11s_mot_people")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted Ultralytics checkpoint")
    args = parser.parse_args()

    from ultralytics import YOLO

    if not args.data.is_file():
        raise FileNotFoundError(
            f"{args.data} does not exist; run scripts/prepare_training_data.py first")

    model = YOLO(args.model)
    if args.resume:
        result = model.train(resume=True, workers=args.workers, device=args.device)
    else:
        result = model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(REPO / "outputs" / "training"),
            name=args.name,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=1e-3,
            lrf=0.01,
            weight_decay=5e-4,
            warmup_epochs=3.0,
            cos_lr=True,
            patience=8,
            close_mosaic=5,
            mosaic=0.7,
            mixup=0.05,
            translate=0.10,
            scale=0.50,
            fliplr=0.50,
            amp=True,
            cache="disk",
            seed=42,
            deterministic=True,
            plots=True,
            save=True,
            verbose=True,
        )

    run_dir = Path(result.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training finished without {best}")
    model_dir = REPO / "models"
    model_dir.mkdir(exist_ok=True)
    final = model_dir / "people_yolo11s_mot.pt"
    shutil.copy2(best, final)

    metrics = {key: float(value) for key, value in result.results_dict.items()}
    train_args = model.trainer.args
    summary = {
        "base_model": args.model,
        "final_model": str(final.relative_to(REPO)),
        "epochs_requested": int(train_args.epochs),
        "imgsz": int(train_args.imgsz),
        "batch": int(train_args.batch),
        "workers": int(train_args.workers),
        "seed": 42,
        "metrics": metrics,
        "run_dir": str(run_dir.relative_to(REPO)),
    }
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
