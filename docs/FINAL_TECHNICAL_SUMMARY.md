# Technical Summary — Video-Based People Counting (p29)

Computer Vision course project · [Repository](https://github.com/NikolayTaran/cv_people_counting)

## Problem

We study directional people counting across a virtual line in video under occlusion, crowding, and changes of camera viewpoint. The research question is whether pedestrian fine-tuning or fusing two detectors improves counting over a COCO-pretrained YOLO11 baseline, and what those changes cost in tracking quality and inference time. The deliverable is a working detector–tracker–counter pipeline with reproducible evaluation, rather than a claim of deployment-level generalization.

## Dataset

MOT17-04 and MOT17-05 supplied detector training and validation: the first 80% of each sequence gave 1,510 images and 39,144 pedestrian boxes; the final 20% gave 377 images and 11,397 boxes. Labels were converted from valid MOT pedestrian annotations to YOLO format, with visibility at least 0.10 for detector training. Four other sequences—MOT17-02, MOT20-01, MOT17-09, and MOT17-11—supplied 2,454 frames for cross-camera comparison. Their frames were excluded from detector training. However, their scores informed checkpoint selection and assessment of the ensemble. Thus every four-sequence result below is a **preliminary post-selection comparison**, not an unbiased final test estimate. The chronological validation split also shares cameras with training.

## Methodology

The first stage detects the person class with COCO-pretrained YOLO11n/YOLO11s or YOLO11s fine-tuned for 3 or 20 epochs. A custom two-stage ByteTrack associates high- and low-confidence detections using an eight-dimensional constant-velocity Kalman filter, IoU cost, and Hungarian matching. MOT17-11 additionally uses camera-motion compensation. The counter tests the bottom-center point of each confirmed track against a manually configured directional line, with hysteresis and cooldown, and emits IN/OUT events. The second approach runs the 3-epoch and COCO YOLO11s detectors together, fuses overlapping person boxes at cross-model IoU 0.55, and leaves the tracker and counter unchanged. The repository includes the selected 3-epoch checkpoint.

## Experimental setup

All five configurations use the same four comparison sequences, tracking/counting settings, and ground-truth line geometry. Counting accuracy (CA) penalizes error in final directional IN/OUT totals; Curve MAE measures cumulative-count error at 50-frame checkpoints. MOTA and IDF1 assess tracking. The reported aggregates are unweighted means across sequences. The code names this comparison split `test`, but it is not an untouched final test. Runtime was measured in matched no-render runs on an NVIDIA GeForce RTX 4060 Laptop GPU. No aerial VisDrone result is included.

## Main results

| Approach | Mean CA ↑ | Curve MAE ↓ | MOTA ↑ | IDF1 ↑ |
|---|---:|---:|---:|---:|
| COCO YOLO11n | 0.5501 | 2.5913 | **0.4361** | 0.4446 |
| COCO YOLO11s | 0.6222 | 1.9747 | 0.4352 | **0.4671** |
| YOLO11s, 3 epochs | 0.6548 | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, 20 epochs | 0.5794 | 2.1542 | 0.4080 | 0.4277 |
| Two-detector ensemble | **0.6654** | 1.7858 | 0.3989 | 0.4261 |

The 3-epoch model improves observed mean CA by 0.1047 over COCO YOLO11n and has the best Curve MAE. The ensemble adds only 0.0106 mean CA over that model. Its scene-level CA is 0.5385 versus 0.3462 on crowded MOT20-01 and 0.5714 versus 0.5238 on occluded MOT17-02, but 0.7692 versus 0.9231 on MOT17-09 and 0.7826 versus 0.8261 on moving MOT17-11. The 20-epoch model is strongest on MOT20-01 (CA 0.6154) yet falls to 0.5794 mean CA. Matched runs of 2,454 frames took 125.42 s (19.57 FPS) with the 3-epoch detector and 164.62 s (14.91 FPS) with the ensemble, about 31% longer.

## Conclusions

The complete implementation produces directional counts on all four studied comparison scenes. A short YOLO11s fine-tune is the practical default: it improves observed counting over the original baseline while retaining better Curve MAE, MOTA, IDF1, and speed than the ensemble. The ensemble can help dense or occluded scenes, but its small aggregate CA gain is accompanied by worse temporal count error, tracking metrics, and runtime. Longer fine-tuning improves same-camera detector validation yet does not consistently improve cross-camera counting.

## Limitations

The four comparison cameras influenced model selection, so the reported metrics cannot establish unbiased transfer to new cameras. Training and validation share two views. Counting lines are configured manually; the tracker lacks appearance/ReID features, and occlusion can disrupt association. Three audited discrepancies illustrate missing crossings on MOT20-01 and MOT17-11 and a spurious directional event on MOT17-09; their precise detector/association causes are not fully identifiable from saved outputs. A frozen pipeline needs a once-only evaluation on previously unused cameras. This is an evaluation limitation, not an unfinished implementation.
