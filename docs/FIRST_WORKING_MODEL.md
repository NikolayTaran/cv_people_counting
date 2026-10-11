# First Working Model Submission

Repository: <https://github.com/NikolayTaran/cv_people_counting>

## 1. Objective and implemented pipeline

The project counts people entering and leaving a region under occlusion,
crowding, and changes in camera viewpoint. The first working system is a
complete detector-tracker-counter pipeline:

1. YOLO11 detects the `person` class in each frame.
2. A custom ByteTrack implementation associates high- and low-confidence
   boxes using an 8D Kalman filter, IoU cost, and Hungarian matching.
3. The bottom centre of each confirmed track is treated as the person's foot
   point. A hysteresis-based virtual line counter produces directional IN and
   OUT events. MOT17-11 also uses phase-correlation camera-motion compensation.
4. Predicted events are compared with events obtained by passing ground-truth
   trajectories through the same counting geometry. Tracking is evaluated
   separately with standard MOT metrics.

The primary selection metric is mean counting accuracy:

`CA = max(0, 1 - (|pred_in - gt_in| + |pred_out - gt_out|) / (gt_in + gt_out))`.

We also report cumulative-count curve MAE, MOTA, and IDF1. This prevents a
model from looking successful only because its final total happens to match.

## 2. Data loading and preprocessing

We use MOTChallenge pedestrian annotations. Data is loaded from MOT `seqinfo`
files, ordered image frames, and `gt/gt.txt`. Only valid class-1 pedestrian
annotations are retained. Detector boxes are clipped to the image, boxes under
2 px in either dimension are removed, and annotations with visibility below
0.10 are omitted from detector training. Remaining boxes are converted to
normalized YOLO `class x_center y_center width height` format.

The split is camera-separated for final evaluation:

| Purpose | Sequences | Frames/images | Boxes |
|---|---|---:|---:|
| Train | first 80% of MOT17-04 and MOT17-05 | 1,510 | 39,144 |
| Validation | final 20% of MOT17-04 and MOT17-05 | 377 | 11,397 |
| Test | MOT17-02, MOT20-01, MOT17-09, MOT17-11 | 2,454 | not used for training |

The chronological split avoids near-duplicate adjacent train/validation
frames being mixed randomly. More importantly, none of the four test cameras
is exported by `prepare_training_data.py`. VisDrone is present as an optional
configuration, but it is excluded from this report because the configured
sequence was not available through the reproducible downloader during this
run. No result is imputed for missing data.

Training preprocessing uses 640 px letterboxed input, horizontal flips,
translation, scale jitter, light mixup, and mosaic augmentation. Inference
uses 1280 px for small/dense scenes (MOT17-02 and MOT20-01) and 640 px for
MOT17-09 and MOT17-11.

## 3. Model configuration and training setup

Three working detector configurations were compared while keeping the tracker
and counter fixed:

- YOLO11n with COCO pretrained weights: original lightweight baseline.
- YOLO11s with COCO pretrained weights: higher-capacity baseline.
- YOLO11s fine-tuned on the development split: one-class pedestrian detector.

Fine-tuning uses AdamW (`lr0=0.001`, cosine decay to 1% of the initial rate,
weight decay `0.0005`), batch size 8, AMP, seed 42, deterministic mode, three
warm-up epochs, and best-checkpoint selection by Ultralytics validation
fitness. Training ran on an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch
2.11.0+cu128 and Ultralytics 8.4.175. We compared checkpoints from a short
3-epoch run and a full 20-epoch run, both at 640 px.

Detector metrics on the development validation frames were:

| Checkpoint | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **3 epochs (selected)** | **0.9712** | 0.8911 | 0.9636 | 0.6781 |
| 20 epochs | 0.9644 | **0.9147** | **0.9643** | **0.7246** |

The tracker uses `high_thresh=0.5`, a 0.05 detector confidence floor,
two-stage BYTE association, `max_age=30`, and three hits before a track is
eligible to count. The low detector threshold is deliberate: partially
occluded boxes remain available to ByteTrack's second matching stage but
cannot start tracks by themselves.

## 4. Initial quantitative results

All aggregate values below are unweighted means across the same four held-out
test sequences. Higher CA/MOTA/IDF1 and lower curve MAE are better.

| Detector | Mean CA | Curve MAE | MOTA | IDF1 |
|---|---:|---:|---:|---:|
| YOLO11n, COCO pretrained | 0.5501 | 2.5913 | **0.4361** | 0.4446 |
| YOLO11s, COCO pretrained | 0.6222 | 1.9747 | 0.4352 | **0.4671** |
| **YOLO11s, MOT fine-tuned for 3 epochs** | **0.6548** | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, MOT fine-tuned for 20 epochs | 0.5794 | 2.1542 | 0.4080 | 0.4277 |

Per-sequence results for the selected 3-epoch YOLO11s checkpoint:

| Test sequence | Pred IN/OUT | GT IN/OUT | CA | Curve MAE | MOTA | IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 10 / 1 | 16 / 5 | 0.5238 | 2.2500 | 0.1797 | 0.2997 |
| MOT20-01 | 7 / 2 | 21 / 5 | 0.3462 | 2.8750 | 0.4700 | 0.4433 |
| MOT17-09 | 9 / 3 | 10 / 3 | **0.9231** | **0.1000** | 0.4770 | 0.4535 |
| MOT17-11 | 9 / 10 | 11 / 12 | 0.8261 | 1.5833 | **0.5185** | **0.5331** |

The selected checkpoint improves mean CA by 0.1047 absolute over the original
YOLO11n baseline and by 0.0326 over the larger pretrained baseline. Its main
gain is on the two camera-view sequences. The full 20-epoch fine-tune improves
the difficult dense MOT20-01 sequence from 0.4231 to 0.6154 CA, but lowers
aggregate CA because it specializes to the two fixed development views.

Machine-readable outputs are stored in `reports/`.

## 5. Problems and observations

- A 20-epoch run initially failed when OpenCV's `warpAffine` was called from
  Windows data-loader worker processes. Training resumed from `last.pt` with
  `workers=0` and completed normally. The reproduction command uses this
  stable setting.
- High validation mAP did not guarantee the best end-to-end counting score.
  The validation frames share the same cameras as training; their appearance
  distribution is narrower than the camera-separated test split.
- Long fine-tuning substantially helps extreme crowding but loses robustness
  under the unseen occlusion and low-angle views. Future training should use
  more cameras and camera-stratified validation, then tune the epoch count on
  development data only.
- The tracker has no appearance/ReID branch. Long occlusions can fragment or
  switch identities even when detector recall is good.
- The counting line is manually configured for each camera. A calibrated
  ground plane or learned entrance region would be required for deployment on
  arbitrary unseen video.

## 6. Implementation status and reproduction

The repository now includes:

- leak-aware MOT-to-YOLO export (`scripts/prepare_training_data.py`);
- deterministic GPU fine-tuning and checkpoint export
  (`scripts/train_detector.py`);
- automatic CUDA/CPU device selection;
- CLI overrides for weights, device, resolution, and data split;
- explicit development/test sequence labels;
- 39 unit tests for tracking, counting, geometry, metrics, and readers.

Main commands:

```powershell
python scripts/prepare_training_data.py
python scripts/train_detector.py --epochs 3 --imgsz 640 --batch 8 --workers 0 --device 0
python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot.pt --device 0 --no-render --out-dir outputs/submission/runs
python scripts/evaluate.py --split test --runs-dir outputs/submission/runs --results outputs/submission/test_results.json --plots-dir outputs/submission/test_plots
python -m pytest tests -q
```

The trained model is exported to `models/people_yolo11s_mot.pt`. Full raw
datasets and temporary inference outputs are intentionally ignored by Git.
