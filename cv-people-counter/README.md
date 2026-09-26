# People Counter — Detector-and-Tracker Pipeline

A complete computer-vision pipeline that **counts people entering or leaving a
region**, and a rigorous evaluation of **counting accuracy under crowding,
occlusion and different camera views** (static elevated, low-angle, moving
handheld, aerial drone).

The system combines a **YOLO11n detector** (COCO-pretrained, `person` class)
with a **custom from-scratch ByteTrack tracker** (Kalman filter + two-stage
BYTE association + phase-correlation camera-motion compensation), and two
counting modes — a **directional virtual line** and a **region-of-interest
polygon**. Evaluation runs on 7 real surveillance sequences from
**MOT17, MOT20 and VisDrone**, comparing predicted counts against
ground-truth counts generated from annotated trajectories through the exact
same counting geometry, plus standard **MOT metrics** (MOTA, MOTP, IDF1,
IDSW) via `motmetrics`.

```
video frames ──► YOLO11n (person) ──► custom ByteTrack ──► counting layer ──► events
                                                     │                         │
                                        phase-correlation            GT tracks (same geometry)
                                        camera-motion comp.                     │
                                                     ▼                         ▼
                                          MOT metrics (MOTA/IDF1)    counting metrics (CA/MAE)
                                                                ▼
                                            annotated MP4 · results.json · plots · web dashboard
```

---

## Results at a glance

**Overall (7 sequences, 4 690 frames):** counting accuracy **0.59**, curve
MAE **3.04**, MOTA **0.39**, IDF1 **0.45**.

| Condition | Sequences | mean CA | curve MAE | RMSE total | mean MOTA | mean IDF1 |
|---|---|---|---|---|---|---|
| Occlusion | MOT17-02, MOT17-05 | **0.63** | 2.39 | 9.22 | 0.38 | 0.47 |
| Crowding | MOT17-04, MOT20-01 | **0.58** | 2.30 | 12.81 | 0.46 | 0.48 |
| Camera views | MOT17-09, MOT17-11, VisDrone | **0.58** | 3.98 | 13.83 | 0.34 | 0.42 |

![Counting accuracy by sequence](docs/assets/counting_accuracy.png)
![Tracking metrics](docs/assets/tracking_metrics.png)

**Annotated output** — left: virtual-line counting on MOT17-04 (crowded
street), right: region-of-interest counting on the aerial VisDrone sequence:

| Line mode | Region mode |
|---|---|
| ![line mode](docs/assets/demo_line_mode.png) | ![region mode](docs/assets/demo_region_mode.png) |

Full per-sequence numbers and analysis: [Results & analysis](#results--analysis).

---

## Repository layout

```
├── configs/
│   └── sequences.yaml         # evaluation protocol: sequences, conditions, geometry
├── src/people_counter/
│   ├── detection/
│   │   └── yolo_detector.py   # YOLO11n wrapper (person class, low conf floor)
│   ├── tracking/
│   │   ├── kalman.py          # 8-dim constant-velocity Kalman filter (SORT form)
│   │   ├── track.py           # track lifecycle: tentative → confirmed → deleted
│   │   └── bytetrack.py       # BYTE two-stage association + IoU + Hungarian + MC
│   ├── counting/
│   │   ├── geometry.py        # signed line distance, point-in-polygon
│   │   └── counter.py         # LineCounter (Schmitt trigger) & RegionCounter (hysteresis)
│   ├── datasets/
│   │   ├── mot.py             # MOT17/MOT20/VisDrone readers + GT protocol filtering
│   │   └── gt_counter.py      # ground-truth counting via the SAME geometry
│   ├── evaluation/
│   │   ├── counting_metrics.py# CA, abs errors, curve MAE
│   │   └── mot_metrics.py     # MOTA/MOTP/IDF1/IDSW via motmetrics
│   ├── annotation.py          # boxes + IDs + geometry + IN/OUT HUD rendering
│   ├── video.py               # frames → ffmpeg pipe → H.264 MP4
│   ├── config.py              # YAML loading with defaults + per-sequence overrides
│   └── pipeline.py            # end-to-end sequence runner
├── scripts/
│   ├── download_data.py       # selective dataset download (HTTP range tricks) + verify
│   ├── run_pipeline.py        # run detection → tracking → counting → video
│   └── evaluate.py            # GT counting + all metrics + plots + results.json
├── tests/                     # 39 unit tests (Kalman, BYTE, counting, metrics, readers)
├── outputs/                   # generated: runs/*.json, videos/*.mp4, results.json
├── docs/assets/               # generated plots + demo frames (referenced by README)
├── data/                      # downloaded datasets (git-ignored)
├── requirements.txt
└── README.md
```

---

## How it works

### 1. Detection — YOLO11n (`src/people_counter/detection/yolo_detector.py`)

* **Model**: `yolo11n.pt`, pretrained on COCO, class `person` only. The
  detector is deliberately off-the-shelf: the engineering value of this
  project lives in the tracker, the counting logic and the evaluation.
* **Low confidence floor** (`conf = 0.05`): ByteTrack performs its own
  confidence stratification — weak boxes of occluded people are exactly what
  its second association stage needs, so we don't throw them away early.
* **Per-sequence inference resolution** (`imgsz`): 640 by default, 1280 for
  scenes with small/dense people (this matters a lot — see
  [tuning findings](#detector-resolution-matters)).

### 2. Tracking — custom ByteTrack (`src/people_counter/tracking/`)

Implemented from scratch per the ECCV 2022 paper
(*Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box"*).

**Kalman filter** (`kalman.py`) — one filter per track, state

```
x = [cx, cy, a, h, vx, vy, va, vh]ᵀ        a = box width/height
```

constant-velocity transition with dt = 1 frame; process and measurement
noise are scaled by the current box height (`std_weight_position = 1/20`,
`std_weight_velocity = 1/10`, SORT convention), which makes the same
parameters work for a 20 px distant pedestrian and a 300 px close one.

**Track lifecycle** (`track.py`):

* every unmatched **high-confidence** detection spawns a *tentative* track;
* `n_init = 3` consecutive hits → *confirmed* (only confirmed tracks may
  produce counting events — phantom tracks never count);
* a tentative track dies on its first miss (guards against detector noise);
* a confirmed track survives `max_age = 30` missed frames (≈1 s of
  occlusion), staying predictable and matchable throughout.

**BYTE association** (`bytetrack.py`), per frame:

1. split detections into **HIGH** (`conf ≥ 0.5`) and **LOW**;
2. stage 1 — Hungarian assignment (`scipy.linear_sum_assignment`) of *all*
   tracks vs HIGH detections on IoU distance, gate `1 − IoU < 0.8`;
3. stage 2 — tracks unmatched in stage 1 vs LOW detections with a stricter
   gate (`1 − IoU < 0.5`) — **this is what makes ByteTrack robust under
   occlusion**: the weak, partial box of a half-hidden person keeps its
   track alive instead of spawning a duplicate;
4. unmatched HIGH detections → new tracks; unmatched LOW detections are
   dropped (weak evidence alone never creates identity);
5. still-unmatched tracks are marked missed (see lifecycle).

**Camera-motion compensation** (optional, enabled for MOT17-11): the global
inter-frame shift is estimated with `cv2.phaseCorrelate` (FFT phase
correlation on grayscale frames with a Hann window) and every track's
predicted position is translated by it before association — a lightweight
BoT-SORT-style global motion model that keeps the constant-velocity Kalman
usable on moving cameras. The sign convention of `phaseCorrelate` is pinned
by a unit test (`tests/test_bytetrack.py::test_phase_shift_sign`).

### 3. Counting (`src/people_counter/counting/`)

The counting anchor is the **foot point** (bottom-center of the box) — the
natural contact point with the ground plane.

**Line mode** (`LineCounter`) — a directional virtual line. A **Schmitt
trigger** per track: the trigger "arms" only after the foot point is at
least `arm_dist` (10 px) from the line, and a crossing registers only on a
full transition to the armed state on the other side. Jitter around the
line therefore produces **zero** events. Additional guards:
per-track `cooldown` (8 frames) after an event, `max_travel` (250 px) to
reject teleports caused by ID switches, and `max_gap` (5 frames) — after a
longer observation gap at most **one** transition event is emitted.

**Region mode** (`RegionCounter`) — a polygonal ROI with **hysteresis**:
the inside/outside state flips only after `hysteresis` (2) consecutive
contradicting observations, which suppresses boundary flicker; gap handling
mirrors the line mode.

**Ground truth for counting** (`datasets/gt_counter.py`): the *identical*
counter state machines run over the annotated trajectories (foot points of
GT boxes), so prediction and reference differ only by the
detector+tracker, never by counting logic. GT tracks are "always eligible"
(`hits = 10⁶`).

### 4. Evaluation (`src/people_counter/evaluation/`)

**Counting metrics** (per sequence):

* `error_in = |pred_in − gt_in|`, `error_out` — absolute errors;
* **counting accuracy** `CA = max(0, 1 − (error_in + error_out) / max(gt_in + gt_out, 1))`
  (the standard cross-line counting accuracy from traffic analysis);
* **curve MAE** — mean of `|cum_pred(t) − cum_gt(t)|` over both directions
  at checkpoints every 50 frames — captures *how early* errors accumulate,
  not only the final totals.

**MOT metrics** via `motmetrics` (CLEAR-MOT + ID measures): MOTA, MOTP
(mean `1 − IoU` over matches — lower is better), IDF1, IDSW, FP, FN, etc.
Protocol: per-frame Hungarian matching, IoU > 0.5, GT filtered per the
official dataset rules (below).

### 5. Annotation & video (`annotation.py`, `video.py`)

Each frame is rendered with per-track colored boxes + IDs, the counting
geometry (line / translucent ROI), and an IN/OUT HUD, then piped as raw
BGR into **ffmpeg** → H.264 `yuv420p` + `faststart` (plays in every
browser; this is why the dashboard can stream them).

---

## Datasets & evaluation protocol

Datasets are **not** in the repository — `scripts/download_data.py` fetches
them and `data/` is git-ignored.

| Sequence | Dataset | Condition | Frames | Resolution@fps | Avg people | Counting mode |
|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | MOT17 | occlusion | 600 | 1920×1080@30 | 31.0 | line (x = 0.5W) |
| MOT17-05-FRCNN | MOT17 | occlusion | 837 | 640×480@14 | 8.3 | line (x = 0.5W) |
| MOT17-04-FRCNN | MOT17 | crowding | 1050 | 1920×1080@30 | 45.3 | line (x = 0.5W) |
| MOT20-01 | MOT20 | crowding (extreme) | 429 | 1920×1080@25 | 46.3 | line (x = 0.5W) |
| MOT17-09-FRCNN | MOT17 | camera view (static, low-angle) | 525 | 1920×1080@30 | 10.1 | line (x = 0.5W) |
| MOT17-11-FRCNN | MOT17 | camera view (moving handheld) | 900 | 1920×1080@30 | 10.5 | line + **motion compensation** |
| VisDrone-uav0000117_02622_v | VisDrone | camera view (aerial drone) | 349 | 2720×1530@10 | 27.7 | **region (ROI)** |

* **MOT17 / MOT20** — [motchallenge.net](https://motchallenge.net). Street
  and mall surveillance; MOT20 contains extreme densities (up to 226 people
  per frame in the full set). License: CC BY-NC-SA 4.0 (academic use).
* **VisDrone2019-MOT** — [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset),
  drone-mounted cameras over Chinese urban streets. License: for academic
  research only.

**Ground-truth protocol**: MOT — keep rows with `class == 1` (pedestrian)
and flag `== 1` (class 7 "static person" and flagged distractor rows are
excluded, per the official MOTChallenge evaluation); VisDrone — keep
classes 1 (pedestrian) and 2 (people), drop zero-area boxes. The simplified
VisDrone protocol does not implement the official ignore-region machinery,
which slightly inflates FP on the aerial view — noted in the analysis.

**How the downloader stays small**: MOT17.zip is 5.9 GB and MOT20.zip
5.0 GB, but only ~700 MB of members are needed. The script reads the zip
**central directory** via HTTP range requests (`remotezip`), merges the
wanted members into contiguous byte spans, fetches each span with a single
large range request and parses the local zip headers itself — a handful of
requests instead of thousands (per-file fetching measured ~0.35 files/s on
a high-latency link; span fetching is ~100× faster). VisDrone comes from
the official Google Drive via `gdown` (1.48 GB, optional — the pipeline
skips it gracefully when absent; in a sandbox with exhausted Drive quota
the same sequence was sourced from a HuggingFace mirror of the official
archive, `AndriiDemk/visDrone_copy`, using the same range-span trick).

---

## Installation

Requirements: **Python 3.10+** (tested on 3.12), ~3 GB disk for deps+data,
no GPU needed (everything runs on CPU by design).

```bash
git clone <this-repo>
cd cv-people-counter

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

* `requirements.txt` pins **minimum bounds**, and torch is installed from
  the **CPU-only wheel index** (`--extra-index-url` in the file) — no CUDA
  bundles, ~500 MB instead of ~2.5 GB.
* `ultralytics` pulls in `opencv-python` (GUI build). On a headless
  machine re-install the headless build afterwards so `import cv2` does not
  require `libGL`:

  ```bash
  pip install --force-reinstall --no-deps opencv-python-headless
  ```

* The YOLO11n weights (`yolo11n.pt`, 5.4 MB) are auto-downloaded by
  ultralytics on first use.

## Usage

```bash
# 1. download the evaluation data (~700 MB MOT + optional 1.5 GB VisDrone)
python scripts/download_data.py mot17
python scripts/download_data.py mot20
python scripts/download_data.py visdrone       # optional; safe to skip
python scripts/download_data.py --verify       # integrity check

# 2. run the test suite (39 tests: Kalman, ByteTrack, counting, metrics)
python -m pytest tests/ -q

# 3. run the pipeline (detector → tracker → counter → annotated video)
python scripts/run_pipeline.py                       # all sequences
python scripts/run_pipeline.py --seq MOT17-04-FRCNN  # one sequence
python scripts/run_pipeline.py --max-frames 30       # quick smoke test
python scripts/run_pipeline.py --no-render           # skip video output

# 4. evaluate: counting + MOT metrics, plots, results.json
python scripts/evaluate.py
```

Outputs:

* `outputs/runs/<seq>.json` — per-frame track boxes + counting events
* `outputs/videos/<seq>.mp4` — annotated H.264 video
* `outputs/results.json` — the full result document (consumed by the dashboard)
* `docs/assets/*.png` — plots used in this README

Expected runtime on 2 CPU cores: ~25 min for all 7 sequences (4 690 frames,
~4–14 fps depending on `imgsz` and crowd density). A GPU is not required.

### Configuration (`configs/sequences.yaml`)

Global defaults + per-sequence overrides for the detector (`weights`,
`min_conf`, `nms_iou`), the tracker (`high_thresh`, `match_thresh`,
`match_thresh_low`, `max_age`, `n_init`, `min_box_area`,
`motion_compensation`) and the counting geometry. Geometry is expressed as
**fractions of the frame size**, so one config works at any resolution:

```yaml
- name: MOT17-04-FRCNN
  dataset: MOT17
  condition: crowding            # occlusion | crowding | camera_view
  seq_dir: data/MOT17/train/MOT17-04-FRCNN
  imgsz: 1280                    # inference resolution (see tuning notes)
  counting:
    mode: line                   # line | region
    line: { p1: [0.50, 0.08], p2: [0.50, 0.92], in_direction: positive }
    arm_dist: 12                 # Schmitt-trigger arming distance, px
    cooldown: 8                  # frames to ignore a track after an event
```

`in_direction: positive` counts motion towards the positive side of the
p1→p2 vector (for a top→bottom vertical line: rightward = IN).

---

## Results & analysis

Per-sequence results (final configuration):

| Sequence | Condition | Pred in/out | GT in/out | CA | curve MAE | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | occlusion | 9 / 1 | 16 / 5 | 0.48 | 2.33 | 0.29 | 0.34 | 87 |
| MOT17-05-FRCNN | occlusion | 16 / 8 | 19 / 12 | 0.77 | 2.44 | 0.48 | 0.59 | 92 |
| MOT17-04-FRCNN | crowding | 3 / 8 | 4 / 9 | **0.85** | 0.79 | 0.47 | 0.51 | 151 |
| MOT20-01 | crowding | 4 / 4 | 21 / 5 | 0.31 | 3.81 | 0.44 | 0.45 | 214 |
| MOT17-09-FRCNN | camera view | 7 / 3 | 10 / 3 | 0.77 | 0.65 | 0.49 | 0.47 | 58 |
| MOT17-11-FRCNN | camera view (MC) | 5 / 9 | 11 / 12 | 0.61 | 3.69 | **0.53** | 0.52 | 31 |
| VisDrone-uav0000117 | camera view (aerial) | 3 / 10 | 14 / 21 | 0.37 | 7.58 | −0.00 | 0.27 | 173 |

What the numbers say:

* **Best case** — MOT17-04: in a dense street crowd the counter is almost
  exact (CA 0.85, curve MAE 0.79) because the dominant flow crosses the
  line cleanly and the tracker holds identities well (IDF1 0.51 at 45
  people/frame).
* **Occlusion** costs mostly *recall of crossings*: MOT17-02 loses 7 of 16
  entries (heavy partial occlusion behind a street barrier → missed
  detections, CA 0.48), while the sparser MOT17-05 stays strong (0.77).
* **Extreme crowding** (MOT20-01) is the hardest street scene: heavy
  inter-person occlusion fragments tracks and hides crossings (CA 0.31
  despite MOTA 0.44 — tracking is decent, but the *entries* that go missing
  are precisely the heavily-occluded ones near the line).
* **Camera views** degrade gracefully from static low-angle (0.77) to a
  moving handheld camera (0.61, best MOTA of all street scenes thanks to
  motion compensation) to the **aerial drone view (0.37)** — the honest
  headline of this evaluation: a COCO-pretrained detector collapses on
  10–30 px people seen from above (MOTA ≈ 0), so region counting
  undercounts badly. Fine-tuning the detector on aerial data is the
  obvious next step (see Limitations).
* Counting is consistently *harder* than tracking quality alone would
  suggest: a single missed or fragmented track at the line is one lost
  event, while MOTA averages over all boxes. Curve MAE exposes this —
  errors accumulate early where occlusion sits on the counting geometry.

### Detector resolution matters (tuning findings)

Sweeps on 200-frame slices, MOTA:

| Sequence | imgsz 640 | imgsz 1280 |
|---|---|---|
| MOT17-02 (small distant people) | 0.18 | **0.29** |
| MOT20-01 (extreme density) | 0.26 | **0.49** |
| MOT17-11 (moving camera) | **0.41** | 0.35 |

Dense/small-people scenes gain massively from 1280-pixel inference
(adopted in the final config); on large-people scenes 640 is already
sufficient and even better (fewer false positives at higher resolution
noise). For the aerial view, higher resolution *hurts* — it multiplies
false detections faster than it recovers tiny true positives (FP 818 →
1723 while FN only 2044 → 1933 on a 100-frame slice), so VisDrone keeps
1280 with a raised `high_thresh` of 0.40 instead.

### Ablation: camera-motion compensation (MOT17-11, moving handheld)

| Config | MOTA | IDF1 | IDSW | Counting CA |
|---|---|---|---|---|
| phase-correlation MC **on** | **0.535** | **0.524** | **31** | **0.61** |
| MC off | 0.532 | 0.509 | 32 | 0.57 |

The gain is modest but consistent — MOT17-11 pans slowly, so the global
shift per frame is small; on faster cameras the effect grows (the synthetic
unit test shows MC fully preserving identity where a panning camera would
otherwise break tracking).

---

## Dashboard

A Next.js dashboard visualizes `outputs/results.json` and streams the
annotated videos (it lives in the host Next.js app of this workspace;
only the `/` route is user-visible):

* `GET /api/results` — the result document (read at request time)
* `GET /api/media?file=<name>.mp4` — video streaming with HTTP Range
  support (seekable `<video>` players)

It shows overall KPIs, per-condition tables, accuracy/MOTA/IDF1 charts and
one annotated video per sequence.

---

## Testing

`python -m pytest tests/ -q` — 39 tests, all green, no network or datasets
required:

* `test_kalman.py` — coordinate round-trips, static-box stability,
  constant-velocity prediction accuracy (<1.5 px error), covariance
  contraction, camera-shift translation;
* `test_bytetrack.py` — IoU matrix against hand-computed values, stable
  IDs on crossing trajectories, occlusion survival with identity
  retention, stage-2 matching of low-confidence boxes (no duplicate
  tracks), "low confidence alone spawns no track", tentative-death on
  miss, empty-frame robustness, **phase-correlation sign convention** and
  end-to-end identity preservation on a synthetic panning camera;
* `test_counting.py` — exact in/out on controlled crossings, direction
  flip, jitter immunity at the line, alternating crossings, cooldown,
  teleport/gap rules, min-hits gate, region enter/exit with hysteresis,
  boundary flicker suppression, gap transitions;
* `test_core.py` — geometry primitives (convex/concave polygons), counting
  metrics math against hand-computed values, GT protocol filtering for
  MOT and VisDrone, motmetrics integration (perfect tracking → MOTA/IDF1
  1.0; known FP/FN → exact MOTA).

The test suite caught three real bugs before delivery: an incorrect
Kalman covariance update (Joseph form misusing the innovation covariance),
the OpenCV-5 removal of `createHannWindow`, and the `motmetrics`
`idsw`→`num_switches` rename plus its NumPy-2 `np.asfarray` incompatibility.

## Reproducibility

* CPU inference is deterministic; Hungarian assignment, the Kalman filter
  and the counters are pure deterministic code — re-running a sequence
  reproduces identical outputs.
* Exact versions used for the reported numbers: Python 3.12, torch
  2.14.0+cpu, ultralytics 8.4.159, opencv-python-headless 5.0.0,
  scipy 1.18.1, pandas 3.0.6, motmetrics 1.4.0 (NumPy 2.5.3).
* `outputs/runs/<seq>.json` stores every per-frame box, so evaluation
  (`scripts/evaluate.py`) can be re-run without re-running inference.

## Limitations & future work

* **Aerial view needs a fine-tuned detector** — the COCO-pretrained
  YOLO11n is not the right tool for 10–30 px people; fine-tuning on
  VisDrone-DET or switching to a tile-based inference scheme is the
  highest-leverage improvement.
* **No re-identification (ReID)** — BYTE associates by IoU only; tracks
  that swap identities in dense crowds stay swapped. A appearance model
  (e.g. a light OSNet embedding) would cut IDSW in crowds.
* **Simplified VisDrone protocol** — ignore-regions are not masked, which
  inflates FP on the aerial sequence (documented above).
* **Counting geometry is hand-placed** per sequence (a single vertical
  line / a hand-drawn ROI); a calibration step (ground-plane homography)
  would make line placement view-independent.
* **Single-threaded CPU processing** at 4–14 fps; batching frames and
  using ONNX/INT8 would give real-time throughput on one core.
