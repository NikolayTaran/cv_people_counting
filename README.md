# People Counting with YOLO11 and ByteTrack

End-to-end computer-vision pipeline for counting people who cross a virtual
line. It contains dataset preparation, YOLO11 detector training, a custom
ByteTrack implementation, directional counting, MOT/counting evaluation, and
reproducible experiment reports.

## Final submission package

- [Two-page technical summary](docs/FINAL_TECHNICAL_SUMMARY.md) and its [PDF](output/pdf/final_technical_summary_p29.pdf)
- [Seven-slide presentation text](docs/FINAL_PRESENTATION.md)
- [Two-minute demo instructions](docs/FINAL_DEMO_PLAN.md)
- [Team contributions template](docs/TEAM_CONTRIBUTIONS_TEMPLATE.md) — proposed roles must be confirmed by the team
- [Full final report](docs/FINAL_REPORT.md) and [PDF](output/pdf/final_report_p29.pdf)

The code, trained checkpoint, evaluation outputs, and documentation are ready.
The comparison cameras informed model selection, so the numerical results are
**preliminary post-selection comparisons**; this is an evaluation limitation,
not an incomplete pipeline.

## Current result

The camera-separated comparison set contains 2,454 frames from MOT17-02,
MOT20-01, MOT17-09, and MOT17-11. The primary comparison metric is mean
counting accuracy (CA); higher is better. All candidates use the same tracker,
counting geometry, and sequences. The code labels this split `test`, but its
results were used to choose the submitted checkpoint. Consequently, these
numbers are **post-selection results, not an unbiased final test estimate**.

| Detector | Mean CA | Curve MAE | MOTA | IDF1 |
|---|---:|---:|---:|---:|
| YOLO11n, COCO pretrained | 0.5501 | 2.5913 | **0.4361** | 0.4446 |
| YOLO11s, COCO pretrained | 0.6222 | 1.9747 | 0.4352 | **0.4671** |
| **YOLO11s, 3-epoch MOT fine-tune** | **0.6548** | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, 20-epoch MOT fine-tune | 0.5794 | 2.1542 | 0.4080 | 0.4277 |
| **Two-detector ensemble** | **0.6654** | 1.7858 | 0.3989 | 0.4261 |

The selected 3-epoch fine-tune has the best observed aggregate counting result
among single detectors on this comparison set: CA `0.6548`, 0.1047 absolute
above the original baseline. The ensemble reaches `0.6654` but has worse
Curve MAE, tracking metrics, and runtime. A longer 20-epoch run
reaches `mAP50=0.9643` and `mAP50-95=0.7246` on development validation and
raises CA on dense MOT20-01 from 0.4231 to 0.6154, but it specializes to the
two development views and scores lower on the comparison sequences. This is
why the short checkpoint is shipped in `models/people_yolo11s_mot.pt`. A new,
untouched set of cameras is required to estimate its final generalization.

See [the first working model report](docs/FIRST_WORKING_MODEL.md) for the full
protocol, per-sequence results, training setup, and limitations. A ready-to-
submit [PDF version](output/pdf/first_working_model_report.pdf) is included.

## Second approach

The [second approach report](docs/SECOND_APPROACH.md) evaluates a two-detector
ensemble. It combines the submitted 3-epoch YOLO11s checkpoint with the
COCO-pretrained YOLO11s detector using confidence-weighted box fusion, then
feeds the boxes to the unchanged ByteTrack and counting pipeline. On the same
four camera-separated comparison sequences, mean CA is **0.6654** versus
**0.6548** for the single detector. This is a small observed gain with lower
tracking metrics and longer processing time. The four sequences have already
been used for model comparison, so these are exploratory results rather than
an independent final test.

```powershell
python scripts/run_pipeline.py --split test --ensemble-with yolo11s.pt --device 0 --no-render --out-dir outputs/second_approach/comparison/runs
python scripts/evaluate.py --split test --runs-dir outputs/second_approach/comparison/runs --results outputs/second_approach/comparison/results.json
```

## Progress check

The [progress check report](docs/PROGRESS_CHECK.md) summarizes all completed
experiments, the five-configuration comparison, key observations, and the
remaining work. A [PDF for submission](output/pdf/progress_check_report.pdf)
is included.

## Final course report

The [final report](docs/FINAL_REPORT.md) compares all five tested approaches,
documents matched runtime measurements and three event-level failure cases,
and states the post-selection evaluation limitation. A [submission PDF](output/pdf/final_report_p29.pdf)
is included. The [failure audit](reports/final_failure_cases.json) records the
ground-truth and predicted events behind those examples. A clear case table
summarizes the local counting discrepancies, while the [track-overlap
plot](docs/assets/final_track58_iou.png) shows the apparent identity transfer
on MOT17-09.

## Architecture

```text
video frames -> YOLO11 person detector -> custom ByteTrack -> line/ROI counter
                    |                         |                    |
                 boxes/confidence       Kalman + BYTE        IN/OUT events
                                                                  |
GT trajectories --------------------------------------------------+
                    -> counting metrics + MOT metrics + plots
```

- **Detector:** Ultralytics YOLO11, person class only, low confidence floor so
  ByteTrack can recover partially occluded detections.
- **Tracker:** custom two-stage ByteTrack with an 8D constant-velocity Kalman
  filter, Hungarian matching, track lifecycle management, and optional global
  camera-motion compensation.
- **Counter:** foot-point line crossing with hysteresis, cooldown, and gap
  handling; polygon region mode is also implemented.
- **Evaluation:** counting accuracy, cumulative-count curve MAE, MOTA, MOTP,
  IDF1, identity switches, false positives, and misses.

## Data protocol

The raw datasets are not committed. `scripts/download_data.py` downloads the
required MOT17/MOT20 sequences.

- **Development:** MOT17-04 and MOT17-05. Each sequence is split
  chronologically: first 80% train, final 20% validation. This produces 1,510
  train images / 39,144 boxes and 377 validation images / 11,397 boxes.
- **Camera-separated comparison:** MOT17-02, MOT20-01, MOT17-09, and MOT17-11.
  These sequences never enter the training exporter, but were used for model
  selection; the implementation still calls this split `test`.
- **Untouched final test:** none in this submission.
- Pedestrian ground truth uses MOT class 1, valid annotations, visibility at
  least 0.10 for detector training, clipped boxes, and YOLO normalized labels.
- VisDrone remains configured as an optional future aerial test, but is not
  included in the reported numbers because that sequence was unavailable in
  the reproducible download used for this submission.

## Setup

Python 3.10+ and a CUDA-capable GPU are recommended for training.

```powershell
git clone https://github.com/NikolayTaran/cv_people_counting.git
cd cv_people_counting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the matching PyTorch wheel first as described in
`requirements.txt`.

## Reproduce

```powershell
# Download and verify the selected sequences
python scripts/download_data.py mot17
python scripts/download_data.py mot20
python scripts/download_data.py --verify

# Export the leak-free detector dataset
python scripts/prepare_training_data.py

# Fine-tune on an NVIDIA GPU (the selected checkpoint uses three epochs)
python scripts/train_detector.py --epochs 3 --imgsz 640 --batch 8 --workers 0 --device 0

# Evaluate any detector on the camera-separated comparison split
python scripts/run_pipeline.py --split test --weights yolo11s.pt --device 0 --no-render --out-dir outputs/yolo11s_pretrained/runs
python scripts/evaluate.py --split test --runs-dir outputs/yolo11s_pretrained/runs --results outputs/yolo11s_pretrained/test_results.json --plots-dir outputs/yolo11s_pretrained/test_plots

# Evaluate the trained checkpoint
python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot.pt --device 0 --no-render --out-dir outputs/trained/runs
python scripts/evaluate.py --split test --runs-dir outputs/trained/runs --results outputs/trained/test_results.json --plots-dir outputs/trained/test_plots

# Unit tests
python -m pytest tests -q
```

### Reproduce the five-way comparison

The commands below use the repository's comparison split, called `test` by the
CLI. It is **post-selection**, because these videos were used to select the
checkpoint. Download the data first using the commands above. The 3-epoch
checkpoint is committed; retraining is optional for reproducing inference.
Ultralytics downloads COCO checkpoints on first use. The 20-epoch checkpoint
is not committed. Generate it separately so the submitted 3-epoch checkpoint
is preserved; the archived 20-epoch metrics are in
`reports/fine_tuned_20ep_test.json`.

```powershell
# COCO baselines
python scripts/run_pipeline.py --split test --weights yolo11n.pt --device 0 --no-render --out-dir outputs/final/yolo11n/runs
python scripts/evaluate.py --split test --runs-dir outputs/final/yolo11n/runs --results outputs/final/yolo11n/results.json --plots-dir outputs/final/yolo11n/plots
python scripts/run_pipeline.py --split test --weights yolo11s.pt --device 0 --no-render --out-dir outputs/final/yolo11s/runs
python scripts/evaluate.py --split test --runs-dir outputs/final/yolo11s/runs --results outputs/final/yolo11s/results.json --plots-dir outputs/final/yolo11s/plots

# Selected 3-epoch model
python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot.pt --device 0 --no-render --out-dir outputs/final/three_epoch/runs
python scripts/evaluate.py --split test --runs-dir outputs/final/three_epoch/runs --results outputs/final/three_epoch/results.json --plots-dir outputs/final/three_epoch/plots

# Two-detector ensemble (same tracker and line settings)
python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot.pt --ensemble-with yolo11s.pt --fusion-iou 0.55 --device 0 --no-render --out-dir outputs/final/ensemble/runs
python scripts/evaluate.py --split test --runs-dir outputs/final/ensemble/runs --results outputs/final/ensemble/results.json --plots-dir outputs/final/ensemble/plots

# Optional: rebuild and evaluate the 20-epoch checkpoint (requires training time)
python scripts/train_detector.py --epochs 20 --imgsz 640 --batch 8 --workers 0 --device 0 --name yolo11s_mot_people_20ep --output-checkpoint models/people_yolo11s_mot_20ep.pt
python scripts/run_pipeline.py --split test --weights models/people_yolo11s_mot_20ep.pt --device 0 --no-render --out-dir outputs/final/twenty_epoch/runs
python scripts/evaluate.py --split test --runs-dir outputs/final/twenty_epoch/runs --results outputs/final/twenty_epoch/results.json --plots-dir outputs/final/twenty_epoch/plots
```

`reports/` contains the archived five-way results, including the 20-epoch
run. Do not compare new timings with the archived timings without matching
hardware, input sizes, rendering mode, and software environment.

### Prepare demo clips

These commands render annotated MP4 files with person boxes, track IDs, line,
and directional count overlay. `--max-frames` starts from frame zero; the
MOT17-09 command includes the audited event near predicted frame 252.

```powershell
python scripts/run_pipeline.py --seq MOT20-01 --weights models/people_yolo11s_mot.pt --max-frames 180 --device 0 --out-dir outputs/demo/mot20_single/runs --video-dir outputs/demo/mot20_single/videos
python scripts/run_pipeline.py --seq MOT20-01 --weights models/people_yolo11s_mot.pt --ensemble-with yolo11s.pt --max-frames 180 --device 0 --out-dir outputs/demo/mot20_ensemble/runs --video-dir outputs/demo/mot20_ensemble/videos
python scripts/run_pipeline.py --seq MOT17-09-FRCNN --weights models/people_yolo11s_mot.pt --ensemble-with yolo11s.pt --max-frames 270 --device 0 --out-dir outputs/demo/mot17_09_ensemble/runs --video-dir outputs/demo/mot17_09_ensemble/videos
```

The MP4s are written to each `outputs/demo/.../videos/` directory. These
short clips are for presentation; full-sequence results are in `reports/`.

`run_pipeline.py` also accepts `--seq`, `--imgsz`, and `--max-frames`.
Omit `--no-render` to create annotated videos.

To regenerate the PDF reports after editing their sources:

```powershell
pip install reportlab mistune
python scripts/build_submission_pdf.py
python scripts/build_second_approach_pdf.py
python scripts/build_progress_check_pdf.py
python scripts/build_final_report_pdf.py
python scripts/build_technical_summary_pdf.py
python scripts/build_final_failure_audit.py
python scripts/plot_final_failures.py
```

## Repository layout

```text
configs/sequences.yaml             experiment splits and per-scene geometry
scripts/prepare_training_data.py   MOT -> YOLO person dataset exporter
scripts/train_detector.py          deterministic YOLO11 fine-tuning
scripts/run_pipeline.py            detector -> tracker -> counter
src/people_counter/detection/ensemble.py  two-detector box fusion
scripts/evaluate.py                counting and MOT metrics
src/people_counter/                pipeline implementation
models/                             selected trained checkpoint
reports/                            machine-readable comparisons and protocol note
docs/FIRST_WORKING_MODEL.md         submission report
docs/SECOND_APPROACH.md             second approach and measured comparison
docs/PROGRESS_CHECK.md              current status and remaining work
docs/FINAL_REPORT.md                 final report and failure analysis
docs/FINAL_TECHNICAL_SUMMARY.md       two-page technical handout
docs/FINAL_PRESENTATION.md            seven-slide presentation text
docs/FINAL_DEMO_PLAN.md               two-minute demo instructions
docs/TEAM_CONTRIBUTIONS_TEMPLATE.md   unconfirmed contribution template
scripts/build_submission_pdf.py     reproducible PDF source
scripts/build_second_approach_pdf.py  second submission PDF source
scripts/build_progress_check_pdf.py  progress-check PDF source
scripts/build_final_report_pdf.py   final report PDF source
scripts/build_technical_summary_pdf.py  technical-summary PDF source
scripts/build_final_failure_audit.py  event-level failure audit source
scripts/plot_final_failures.py      diagnostic figure source
output/pdf/                         ready-to-submit PDF report
tests/                              unit tests
```

## Known limitations

The training set contains only two fixed surveillance views, so a long
fine-tune specializes to their scale and perspective. The tracker uses IoU and
motion without appearance embeddings, which allows identity switches during
long occlusions. Counting geometry is configured per scene. A stronger next
experiment would add more training cameras, use camera-stratified development
validation for model selection, and evaluate the chosen pipeline once on new,
untouched cameras. The current four comparison sequences cannot be reused as
an independent final test after selecting the checkpoint with their scores.
