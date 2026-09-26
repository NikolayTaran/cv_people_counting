#!/usr/bin/env python3
"""Evaluate counting accuracy + MOT metrics for finished pipeline runs.

    python scripts/evaluate.py                 # all runs in outputs/runs
    python scripts/evaluate.py --seq MOT17-04-FRCNN

Produces:
    outputs/results.json          full results (also consumed by the dashboard)
    docs/assets/*.png             plots for the README
and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from people_counter.config import load_config                      # noqa: E402
from people_counter.counting.counter import CountingEvent          # noqa: E402
from people_counter.datasets.gt_counter import gt_count            # noqa: E402
from people_counter.datasets.mot import (                          # noqa: E402
    gt_people_per_frame, load_gt_tracks)
from people_counter.evaluation.counting_metrics import (           # noqa: E402
    counting_metrics)
from people_counter.evaluation.mot_metrics import (                # noqa: E402
    compute_mot_metrics)


def _events_from_dicts(rows: list[dict]) -> list[CountingEvent]:
    return [CountingEvent(**r) for r in rows]


def evaluate_one(seq_cfg: dict, run: dict) -> dict:
    meta = run["meta"]
    n_frames = meta["frames"]
    gt_path = Path(seq_cfg["gt"]) if seq_cfg.get("gt") else None

    result = {
        "name": meta["name"],
        "dataset": meta["dataset"],
        "condition": meta["condition"],
        "tags": meta["tags"],
        "frames": n_frames,
        "fps": meta["fps"],
        "resolution": f"{meta['width']}x{meta['height']}",
        "imgsz": meta["imgsz"],
        "motion_compensation": meta["motion_compensation"],
        "processing_fps": meta.get("processing_fps"),
        "counting": None,
        "tracking": None,
        "video": None,
    }
    video = meta.get("video")
    if video and Path(video).is_file():
        result["video"] = {
            "file": Path(video).name,
            "url": f"/api/media?file={Path(video).name}",
        }

    if gt_path is None or not gt_path.is_file():
        print(f"[eval] {meta['name']}: no ground truth — metrics skipped")
        return result

    gt = load_gt_tracks(gt_path, meta["dataset"])
    gt = gt[gt["frame"] <= n_frames]                     # cap to processed part
    density = gt_people_per_frame(gt)

    # ---- counting: same geometry over GT tracks vs predicted events -------
    gt_res = gt_count(gt, meta["counting_cfg"], meta["width"], meta["height"])
    pred_events = _events_from_dicts(run["events"])
    counting = counting_metrics(pred_events, gt_res["events"], n_frames)
    counting["mode"] = meta["counting_cfg"].get("mode", "line")
    result["counting"] = counting
    result["avg_gt_people"] = round(float(density[0]), 1)

    # ---- MOT metrics -------------------------------------------------------
    mot = compute_mot_metrics(gt, run["frame_records"])
    result["tracking"] = mot
    return result


def aggregate(results: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("counting") is not None:
            by_condition[r["condition"]].append(r)

    def _block(items: list[dict]) -> dict:
        ca = [r["counting"]["counting_accuracy"] for r in items]
        cmae = [r["counting"]["curve_mae"] for r in items]
        tot_err = [r["counting"]["error_in"] + r["counting"]["error_out"]
                   for r in items]
        idf1 = [r["tracking"]["idf1"] for r in items if r.get("tracking")]
        mota = [r["tracking"]["mota"] for r in items if r.get("tracking")]
        rmse = (sum(e * e for e in tot_err) / len(tot_err)) ** 0.5 if tot_err else 0.0
        return {
            "sequences": [r["name"] for r in items],
            "mean_counting_accuracy": round(sum(ca) / len(ca), 4),
            "mean_curve_mae": round(sum(cmae) / len(cmae), 4),
            "rmse_total_count_error": round(rmse, 4),
            "mean_idf1": round(sum(idf1) / len(idf1), 4) if idf1 else None,
            "mean_mota": round(sum(mota) / len(mota), 4) if mota else None,
        }

    out = {cond: _block(items) for cond, items in sorted(by_condition.items())}
    if results:
        out["overall"] = _block(
            [r for r in results if r.get("counting") is not None])
    return out


def make_plots(results: list[dict], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    good = [r for r in results if r.get("counting")]

    # ---- counting accuracy per sequence ------------------------------------
    if good:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        names = [r["name"].replace("-FRCNN", "") for r in good]
        values = [r["counting"]["counting_accuracy"] for r in good]
        colors = {"occlusion": "#10b981", "crowding": "#f59e0b",
                  "camera_view": "#a1a1aa"}
        bars = ax.bar(names, values,
                      color=[colors.get(r["condition"], "#71717a") for r in good])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Counting accuracy")
        ax.set_title("Counting accuracy by sequence")
        ax.tick_params(axis="x", rotation=20)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l)
                   for l, c in colors.items()]
        ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()
        path = out_dir / "counting_accuracy.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

        # ---- MOTA / IDF1 per sequence --------------------------------------
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = range(len(good))
        mota = [r["tracking"]["mota"] or 0 for r in good]
        idf1 = [r["tracking"]["idf1"] or 0 for r in good]
        ax.bar([i - 0.2 for i in x], mota, width=0.4, label="MOTA", color="#a1a1aa")
        ax.bar([i + 0.2 for i in x], idf1, width=0.4, label="IDF1", color="#10b981")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title("Tracking quality (CLEAR-MOT / ID metrics)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = out_dir / "tracking_metrics.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs" / "sequences.yaml"))
    parser.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    parser.add_argument("--seq", action="append")
    parser.add_argument("--results", default=str(REPO / "outputs" / "results.json"))
    parser.add_argument("--plots-dir", default=str(REPO / "docs" / "assets"))
    args = parser.parse_args()

    config = load_config(args.config)
    runs_dir = Path(args.runs_dir)
    results: list[dict] = []
    used_detector = None

    for seq_cfg in config["sequences"]:
        if args.seq and seq_cfg["name"] not in set(args.seq):
            continue
        run_path = runs_dir / f"{seq_cfg['name']}.json"
        if not run_path.is_file():
            print(f"[eval] SKIP {seq_cfg['name']}: no run artifact "
                  f"({run_path.name})")
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        # resolve GT path the same way the pipeline resolves seq paths
        if seq_cfg.get("gt"):
            gt = Path(seq_cfg["gt"])
            if not gt.is_absolute():
                seq_cfg = dict(seq_cfg, gt=str(REPO / gt))
        elif seq_cfg.get("seq_dir"):
            seq_dir = Path(seq_cfg["seq_dir"])
            if not seq_dir.is_absolute():
                seq_dir = REPO / seq_dir
            if seq_cfg["dataset"] in ("MOT17", "MOT20"):
                seq_cfg = dict(seq_cfg, gt=str(seq_dir / "gt" / "gt.txt"))
            else:
                ann = Path(seq_cfg["ann_dir"])
                ann = ann if ann.is_absolute() else REPO / ann
                seq_cfg = dict(seq_cfg, gt=str(ann / f"{seq_dir.name}.txt"))
        res = evaluate_one(seq_cfg, run)
        results.append(res)
        used_detector = seq_cfg["detector"]["weights"]

    if not results:
        print("[eval] no runs evaluated — run scripts/run_pipeline.py first")
        return

    payload = {
        "project": "cv-people-counter",
        "description": ("Detector-and-tracker pipeline counting people "
                        "entering/leaving a region, evaluated under crowding, "
                        "occlusion and different camera views."),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": {"model": used_detector, "pretrained": "COCO",
                     "class": "person"},
        "tracker": {"name": "ByteTrack (custom implementation)",
                    "two_stage_association": True,
                    "kalman_filter": "constant velocity, 8-dim state"},
        "sequences": results,
        "by_condition": aggregate(results),
    }
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    plots = make_plots(results, Path(args.plots_dir))

    print(f"\n[eval] wrote {args.results} and {len(plots)} plot(s)")
    header = (f"{'sequence':<20}{'condition':<13}{'pred in/out':<14}"
              f"{'gt in/out':<14}{'CA':<7}{'cMAE':<7}{'MOTA':<7}{'IDF1'}")
    print(header)
    print("-" * len(header))
    for r in results:
        c = r.get("counting")
        t = r.get("tracking") or {}
        if c:
            print(f"{r['name']:<20}{r['condition']:<13}"
                  f"{c['pred_in']}/{c['pred_out']:<10}"
                  f"{c['gt_in']}/{c['gt_out']:<10}"
                  f"{c['counting_accuracy']:<7.2f}{c['curve_mae']:<7.2f}"
                  f"{(t.get('mota') or 0):<7.2f}{t.get('idf1') or 0:.2f}")
        else:
            print(f"{r['name']:<20}{r['condition']:<13}no ground truth")
    print("\nby condition:")
    for cond, block in payload["by_condition"].items():
        print(f"  {cond:<14} CA={block['mean_counting_accuracy']:.2f} "
              f"curveMAE={block['mean_curve_mae']:.2f} "
              f"IDF1={block['mean_idf1']} MOTA={block['mean_mota']}")


if __name__ == "__main__":
    main()
