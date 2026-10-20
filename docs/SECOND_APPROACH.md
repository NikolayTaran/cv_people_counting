# Second Approach Submission

Repository: <https://github.com/NikolayTaran/cv_people_counting>

## New approach

The first working model used one YOLO11s detector fine-tuned for three epochs,
followed by our custom ByteTrack tracker and a directional line counter. The
second approach uses **two complementary YOLO11s detectors** on every frame:

1. the same 3-epoch pedestrian checkpoint from the first submission;
2. the original COCO-pretrained YOLO11s checkpoint.

Their person boxes are merged by confidence-weighted box fusion before the
unchanged ByteTrack and counter. No additional detector training was required.
The second approach is available through `--ensemble-with yolo11s.pt`.

## Motivation

The first model scored only **0.3462 counting accuracy (CA)** on the crowded
MOT20-01 comparison sequence, while the COCO-pretrained YOLO11s scored
**0.4231** there. Short fine-tuning helped several other views, especially
MOT17-09 and MOT17-11. Combining their detections was expected to recover
additional partially occluded people in dense scenes while retaining the
fine-tuned model's camera-view performance. The hypothesis was that better
detector recall near the line would produce more correct IN/OUT events.

## Implementation differences

The first approach runs one detector. The second approach runs both detectors
at the same frame resolution and with the same `person` class, confidence
floor `0.05`, and per-model NMS IoU `0.70`. Detections from different models
with box IoU at least **0.55** are clustered. Each cluster may contain at most
one box from each model, preventing nearby people reported by one detector
from being merged with each other. The fused box coordinates are weighted by
the original detection confidences; its output confidence is the maximum of
its members. Unmatched detections are retained. The resulting boxes are sorted
by confidence and passed to the existing tracker without changing its
association thresholds, motion compensation, or counting geometry.

Code: `src/people_counter/detection/ensemble.py`, with a new
`--ensemble-with` and `--fusion-iou` option in `scripts/run_pipeline.py`.
The result JSON records both checkpoint paths and the fusion IoU. The COCO
checkpoint is downloaded by Ultralytics on first use if absent locally.

## Initial results versus the first approach

Both approaches were rerun on 20 October 2026 using the same machine, scene
configurations, tracker, counter, and evaluation code. Mean CA is the
unweighted mean over sequences; larger is better. Curve MAE is lower-is-better.

| Four camera-separated comparison sequences | Single YOLO11s | Two-model ensemble | Difference |
|---|---:|---:|---:|
| Mean counting accuracy | 0.6548 | **0.6654** | +0.0106 |
| Mean curve MAE | **1.7021** | 1.7858 | +0.0837 (worse) |
| Mean MOTA | **0.4113** | 0.3989 | -0.0124 |
| Mean IDF1 | **0.4324** | 0.4261 | -0.0063 |
| Processing time, 2,454 frames | **125.4 s** | 164.7 s | +31% |

| Sequence | Baseline CA | Ensemble CA | Difference | Baseline IN/OUT | Ensemble IN/OUT | Ground truth IN/OUT |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02, occlusion | 0.5238 | 0.5714 | +0.0476 | 10/1 | 10/2 | 16/5 |
| MOT20-01, crowding | 0.3462 | 0.5385 | **+0.1923** | 7/2 | 10/6 | 21/5 |
| MOT17-09, low-angle view | **0.9231** | 0.7692 | -0.1539 | 9/3 | 8/4 | 10/3 |
| MOT17-11, moving view | **0.8261** | 0.7826 | -0.0435 | 9/10 | 7/11 | 11/12 |

As a development sanity check, the two videos containing detector training
frames (MOT17-04 and MOT17-05) gave mean CA **0.9355** for the single model
and **0.9677** for the ensemble. These are not independent validation results:
both sequences contain training frames. The comparison table above uses
camera-separated videos, but those four videos were already used in the first
submission to choose its checkpoint. The numbers are therefore **exploratory
post-selection comparisons, not an unbiased final test**.

The ensemble meets its dense-scene objective: MOT20-01 gains 0.1923 CA. It
also loses CA on both camera-view sequences, and its MOTA/IDF1 are lower on
average. It is approximately 31% slower across the four videos. Thus it is a
second working approach with a small overall counting gain, not evidence of a
universal improvement. An independent final estimate requires freezing the
method and evaluating once on new, untouched cameras.

## Reproduce

```powershell
# Existing dataset and checkpoint preparation is documented in README.md.
python scripts/run_pipeline.py --split test --device 0 --no-render --out-dir outputs/second_approach/baseline_comparison/runs
python scripts/evaluate.py --split test --runs-dir outputs/second_approach/baseline_comparison/runs --results outputs/second_approach/baseline_comparison/results.json

python scripts/run_pipeline.py --split test --ensemble-with yolo11s.pt --fusion-iou 0.55 --device 0 --no-render --out-dir outputs/second_approach/comparison/runs
python scripts/evaluate.py --split test --runs-dir outputs/second_approach/comparison/runs --results outputs/second_approach/comparison/results.json

python -m pytest tests -q
```

Machine-readable outputs from these runs and the development sanity check
are preserved in `reports/second_approach_*.json`. VisDrone is excluded because
the configured sequence was unavailable locally.
