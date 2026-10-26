# Progress Check Submission

**Project p29:** People Counting with YOLO11 and ByteTrack

**Repository:** <https://github.com/NikolayTaran/cv_people_counting>

**Status as of:** 26 October 2026

The end-to-end pipeline is implemented: MOT data preparation, YOLO11 person
detection, custom ByteTrack tracking, directional virtual-line counting, and
counting/tracking evaluation. We have tested four single-detector configurations
and one two-detector ensemble. The results below are **preliminary comparisons**:
the four comparison videos influenced model choice and cannot serve as an
untouched final test.

## 1. Completed experiments

- **Data and protocol.** MOT17-04 and MOT17-05 supplied 1,510 training images
  (39,144 boxes) and 377 chronological validation images (11,397 boxes).
  Validation shares the training cameras. Four other sequences, MOT17-02,
  MOT20-01, MOT17-09, and MOT17-11 (2,454 frames), were excluded from detector
  training and used for configuration comparison.
- **First approach.** A YOLO11 person detector feeds custom ByteTrack and a
  line counter. We evaluated COCO-pretrained YOLO11n and YOLO11s, plus YOLO11s
  fine-tuned for 3 and 20 epochs (640-pixel training, batch 8, AdamW). The
  three-epoch checkpoint had the highest observed mean counting accuracy among
  single detectors. Tracking and counting settings were held fixed.
- **Second approach.** The selected three-epoch YOLO11s and COCO-pretrained
  YOLO11s run on each frame. Boxes from different detectors with IoU at least
  0.55 are fused using confidence-weighted coordinates; unmatched boxes remain.
  The fused detections feed the same tracker and counter. No additional
  detector training was needed.
- **Reproducibility.** The repository contains the selected checkpoint,
  preparation/training/inference/evaluation scripts, per-run JSON results,
  implementation reports, and 42 passing unit tests. The single-detector and
  ensemble comparison runs were repeated under matched settings on
  20 October 2026.

## 2. Preliminary comparison

Counting accuracy (CA) penalizes errors in the final directional IN and OUT
totals. Curve MAE measures cumulative count error over time. MOTA and IDF1
measure tracking quality. The table gives unweighted means over the same four
comparison videos; arrows show the better direction.

| Configuration | Mean CA ↑ | Curve MAE ↓ | MOTA ↑ | IDF1 ↑ |
|---|---:|---:|---:|---:|
| COCO YOLO11n | 0.5501 | 2.5913 | **0.4361** | 0.4446 |
| COCO YOLO11s | 0.6222 | 1.9747 | 0.4352 | **0.4671** |
| YOLO11s, 3 epochs | 0.6548 | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, 20 epochs | 0.5794 | 2.1542 | 0.4080 | 0.4277 |
| Two-detector ensemble | **0.6654** | 1.7858 | 0.3989 | 0.4261 |

The three-epoch model improves observed mean CA by 0.1047 over the original
YOLO11n baseline. The ensemble improves it by another 0.0106 over the
three-epoch model, but its Curve MAE, MOTA, and IDF1 are worse. Matched
processing of 2,454 frames took 125.4 seconds for the single detector and
164.7 seconds for the ensemble (approximately 31% longer).

| Comparison video | Single-detector CA | Ensemble CA | Change |
|---|---:|---:|---:|
| MOT17-02, occlusion | 0.5238 | 0.5714 | +0.0476 |
| MOT20-01, crowding | 0.3462 | 0.5385 | +0.1923 |
| MOT17-09, low-angle view | 0.9231 | 0.7692 | -0.1539 |
| MOT17-11, moving view | 0.8261 | 0.7826 | -0.0435 |

## 3. Observations and difficulties

- **Crowding gain, viewpoint loss.** The ensemble gains most on MOT20-01 but
  loses on both camera-view videos. Across the four videos, tracker misses fall
  from 26,801 to 23,491 while false positives rise from 5,812 to 9,076. This
  explains why more detections help some counts without improving tracking
  quality overall.
- **Final totals and time-series quality differ.** Ensemble mean CA increases
  while Curve MAE rises from 1.7021 to 1.7858: final directional totals can be
  closer even when cumulative counts are less accurate throughout a video.
- **More fine-tuning is not uniformly better.** The 20-epoch checkpoint raises
  validation mAP50-95 from 0.6781 to 0.7246 but lowers comparison mean CA from
  0.6548 to 0.5794. It does help dense MOT20-01 (CA 0.6154 versus 0.4231 for
  COCO YOLO11s). Validation uses the same two cameras as training, limiting
  evidence about new viewpoints.
- **Practical limitations.** Long occlusions can fragment identities because
  the tracker has no appearance/ReID features. Counting lines require manual
  camera-specific placement. The 20-epoch training run initially encountered
  an OpenCV `warpAffine` failure with Windows data-loader workers; resuming
  with `workers=0` completed it. The configured VisDrone aerial sequence was
  unavailable locally, so aerial counting remains untested.
- **Evaluation limitation.** Although the four comparison sequences were
  excluded from detector training, their scores informed the selected model
  and second approach. All reported comparison metrics are exploratory
  post-selection results, not unbiased estimates of performance on new cameras.

## 4. Remaining work

1. **Create a camera-disjoint development set.** Add new labeled viewpoints
   and crowd densities, then choose fusion IoU, confidence thresholds, and
   tracker settings using only development data. The current 20% validation
   split is useful for training checks but not camera-level generalization.
2. **Analyze counting errors.** Inspect missed and extra IN/OUT crossings near
   the line, and attribute them to detection misses, extra boxes, identity
   switches, or crossing-direction errors. Compare cumulative count curves as
   well as final totals.
3. **Test tracking and speed changes.** Evaluate association and track-lifetime
   settings and, if feasible, appearance/ReID features. Measure throughput and
   memory under the same hardware and input-resolution conditions.
4. **Extend scene coverage.** Obtain a usable aerial sequence and test the
   implemented polygon entry/exit mode. New cameras should have documented
   geometry setup.
5. **Perform an independent final evaluation.** Freeze the complete pipeline
   and analysis protocol, then evaluate once on previously unused cameras.
   Report CA, Curve MAE, MOTA, IDF1, runtime, and representative failures.

Machine-readable results: `reports/selected_3ep_test.json`,
`reports/fine_tuned_20ep_test.json`, `reports/yolo11n_baseline_test.json`,
`reports/yolo11s_pretrained_test.json`, and
`reports/second_approach_ensemble_comparison.json`. The code calls the
comparison split `test`; this report uses the more accurate term *comparison*.
