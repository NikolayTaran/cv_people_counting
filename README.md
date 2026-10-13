# People Counting with YOLO11 and ByteTrack

End-to-end computer-vision pipeline for counting people who cross a virtual
line. It contains dataset preparation, YOLO11 detector training, a custom
ByteTrack implementation, directional counting, MOT/counting evaluation, and
reproducible experiment reports.

## Current result

The camera-separated comparison set contains 2,454 frames from MOT17-02,
MOT20-01, MOT17-09, and MOT17-11. The primary comparison metric is mean
counting accuracy (CA); higher is better. All candidates use the same tracker,
counting geometry, and sequences. The code labels this split `test`, but its
results were used to choose the submitted checkpoint. Consequently, these
numbers are **post-selection results, not an unbiased final test estimate**.

| Detector | Mean CA | Curve MAE | MOTA | IDF1 |
|---|---:|---:|---:|---:|
| YOLO11n, COCO pretrained | 0.5501 | 2.5913 | 0.4361 | 0.4446 |
| YOLO11s, COCO pretrained | 0.6222 | 1.9747 | **0.4352** | **0.4671** |
| **YOLO11s, 3-epoch MOT fine-tune** | **0.6548** | **1.7021** | 0.4113 | 0.4324 |
| YOLO11s, 20-epoch MOT fine-tune | 0.5794 | 2.1542 | 0.4080 | 0.4277 |

The selected 3-epoch fine-tune has the best observed aggregate counting result
on this comparison set: CA `0.6548`, 0.1047 absolute above the original
baseline. A longer 20-epoch run
reaches `mAP50=0.9643` and `mAP50-95=0.7246` on development validation and
raises CA on dense MOT20-01 from 0.4231 to 0.6154, but it specializes to the
two development views and scores lower on the comparison sequences. This is
why the short checkpoint is shipped in `models/people_yolo11s_mot.pt`. A new,
untouched set of cameras is required to estimate its final generalization.

See [the first working model report](docs/FIRST_WORKING_MODEL.md) for the full
protocol, per-sequence results, training setup, and limitations. A ready-to-
submit [PDF version](output/pdf/first_working_model_report.pdf) is included.

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

`run_pipeline.py` also accepts `--seq`, `--imgsz`, and `--max-frames`.
Omit `--no-render` to create annotated videos.

To regenerate the PDF report after editing its source:

```powershell
pip install reportlab
python scripts/build_submission_pdf.py
```

## Repository layout

```text
configs/sequences.yaml             experiment splits and per-scene geometry
scripts/prepare_training_data.py   MOT -> YOLO person dataset exporter
scripts/train_detector.py          deterministic YOLO11 fine-tuning
scripts/run_pipeline.py            detector -> tracker -> counter
scripts/evaluate.py                counting and MOT metrics
src/people_counter/                pipeline implementation
models/                             selected trained checkpoint
reports/                            machine-readable comparisons and protocol note
docs/FIRST_WORKING_MODEL.md         submission report
scripts/build_submission_pdf.py     reproducible PDF source
output/pdf/                         ready-to-submit PDF report
tests/                              39 unit tests
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
