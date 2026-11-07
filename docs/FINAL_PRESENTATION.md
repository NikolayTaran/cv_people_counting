# Final presentation — Video-Based People Counting (p29)

Seven slides; target speaking time: about five minutes. All four-camera numbers are **preliminary post-selection comparisons**. Slide text is ready to paste into PowerPoint or Google Slides. Speaker notes are optional narration, not extra on-slide bullets.

## Slide 1 — Objective and research question (0:35)

- Count people crossing a virtual line in both directions: IN and OUT.
- Conditions: occlusion, dense crowds, different camera viewpoints.
- Research question: do pedestrian fine-tuning and detector fusion improve counting enough to justify their tracking and runtime costs?

Speaker note: The output is an event count, so better detection alone need not mean better counting.

## Slide 2 — Data and evaluation protocol (0:40)

- Detector train: first 80% of MOT17-04/05; 1,510 images, 39,144 boxes.
- Detector validation: last 20% of the same cameras; 377 images, 11,397 boxes.
- Comparison: MOT17-02, MOT20-01, MOT17-09, MOT17-11; 2,454 frames.
- Four comparison videos were excluded from training but used for model choice: results are preliminary, post-selection.

Speaker note: The code labels these videos `test`; they do not provide an untouched final test estimate.

## Slide 3 — Implemented pipeline (0:40)

```text
video → YOLO11 person boxes → custom ByteTrack → directional line counter → IN/OUT
                    ↘ optional two-detector box fusion ↗
```

- Tracker: Kalman prediction, two-stage IoU/Hungarian association.
- Counter: bottom-center crossing, hysteresis, cooldown; line set per camera.
- Second approach: fuse 3-epoch YOLO11s and COCO YOLO11s detections before the unchanged tracker.

Speaker note: A trained 3-epoch checkpoint and the complete code path are in the repository.

## Slide 4 — Comparison across five approaches (1:00)

Unweighted mean over four comparison videos; ↑ higher is better, ↓ lower is better.

| Approach | CA ↑ | Curve MAE ↓ | MOTA ↑ | IDF1 ↑ |
|---|---:|---:|---:|---:|
| COCO YOLO11n | 0.5501 | 2.5913 | **0.4361** | 0.4446 |
| COCO YOLO11s | 0.6222 | 1.9747 | 0.4352 | **0.4671** |
| YOLO11s, 3 epochs | 0.6548 | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, 20 epochs | 0.5794 | 2.1542 | 0.4080 | 0.4277 |
| Two-detector ensemble | **0.6654** | 1.7858 | 0.3989 | 0.4261 |

Speaker note: The ensemble wins the final-total CA by 0.0106 over the 3-epoch model, but loses on temporal error and tracking.

## Slide 5 — Where the approaches differ (0:45)

| Scene | 3-epoch CA | Ensemble CA | Main observation |
|---|---:|---:|---|
| MOT20-01, dense crowd | 0.3462 | **0.5385** | Fusion recovers some counts. |
| MOT17-02, occlusion | 0.5238 | **0.5714** | Small gain. |
| MOT17-09, low-angle | **0.9231** | 0.7692 | Fusion worsens counts. |
| MOT17-11, moving | **0.8261** | 0.7826 | Single detector remains better. |

- Matched no-render runtime: 125.42 s / 19.57 FPS single; 164.62 s / 14.91 FPS ensemble.

Speaker note: The ensemble is about 31% slower on this machine and scene mix.

## Slide 6 — Representative failures (0:45)

- MOT20-01, GT frame 70: an IN event is missing in both approaches; the best saved ensemble track has only 0.1135 IoU with that person. Dense overlap may break detection or association.
- MOT17-09, predicted frame 252: ensemble adds an OUT where nearby ground-truth events are IN. A saved track changes its best-overlapping person near the line, consistent with an apparent identity transfer.
- MOT17-11, GT frame 146: another IN is missing in a moving view; best saved ensemble track IoU is 0.1978.

Speaker note: These are audited event discrepancies; precise causal attribution is limited by saved diagnostics. For a visual, use `docs/assets/final_track58_iou.png` or rendered frame clips.

## Slide 7 — Conclusions and limitations (0:35)

- Working detector → tracker → counter pipeline and trained checkpoint are complete.
- Practical default: 3-epoch YOLO11s; best Curve MAE and a better quality/speed balance than fusion.
- Fusion helps the dense scene, but its small mean CA gain comes with worse MOTA, IDF1, Curve MAE, and speed.
- Limitation: comparison cameras informed model choice; independent generalization requires a frozen pipeline and new, untouched cameras.

Speaker note: This limits interpretation of the reported performance, not completeness of the implementation.
