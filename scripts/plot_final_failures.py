#!/usr/bin/env python3
"""Plot the observed track overlap behind the MOT17-09 failure case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_final_failure_audit import box_iou, gt_box  # noqa: E402
from people_counter.datasets.mot import load_gt_tracks  # noqa: E402


OUT = ROOT / "docs" / "assets"
NAVY = "#17324D"
TEAL = "#188C7B"
ORANGE = "#D77829"
GRAY = "#7C8790"


def plot_identity_transfer() -> None:
    name = "MOT17-09-FRCNN"
    run = json.loads((ROOT / "outputs" / "second_approach" / "comparison" /
                      "runs" / f"{name}.json").read_text(encoding="utf-8"))
    gt = load_gt_tracks(ROOT / "data" / "MOT17" / "train" / name /
                        "gt" / "gt.txt", "MOT17")
    frames = list(range(248, 260))
    overlaps = {11: [], 12: []}
    for frame in frames:
        track = next(t for t in run["frame_records"][frame]["tracks"] if int(t[0]) == 58)
        for gt_id in overlaps:
            row = gt[(gt["frame"] == frame + 1) & (gt["id"] == gt_id)].iloc[0]
            overlaps[gt_id].append(box_iou(track[1:], gt_box(row)))

    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.plot(frames, overlaps[11], "o-", color=TEAL, linewidth=2,
            label="IoU with GT person 11")
    ax.plot(frames, overlaps[12], "s-", color=ORANGE, linewidth=2,
            label="IoU with GT person 12")
    ax.axvline(252, color=NAVY, linestyle="--", linewidth=1.2,
               label="Predicted OUT, frame 252")
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1,
               label="MOT matching threshold")
    ax.set(xlim=(247.5, 259.5), ylim=(0, 1), xlabel="Ensemble frame (0-based)",
           ylabel="Bounding-box IoU")
    ax.set_title("MOT17-09: track 58 changes its best GT match near the false OUT",
                 loc="left", fontsize=12, fontweight="bold", color=NAVY)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(color="#EDF1F4", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "final_track58_iou.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_identity_transfer()
    print(OUT / "final_track58_iou.png")


if __name__ == "__main__":
    main()
