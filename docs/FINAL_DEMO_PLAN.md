# Two-minute live demo plan (instructions only)

Primary live demonstration: run the pipeline on a fixed-camera street video outside MOT17/MOT20, then show its annotated output. The prepared source is [Omar Ramadan's Pexels clip](https://www.pexels.com/video/people-crossing-the-street-5631418/); the local 600-frame sample shows both predicted IN and OUT changes. Keep the existing MOT20-01 and MOT17-09 clips as backups. The external clip has no ground truth: do not present its counts as an accuracy result.

## Before presenting

1. Save the external MP4 locally and confirm `models/people_yolo11s_mot.pt` exists.
2. Run `scripts/run_video.py` using the command in the README. Verify the output MP4 opens and shows boxes, IDs, the line, and changing IN/OUT totals.
3. Keep the output MP4 and `docs/FINAL_PRESENTATION.md` locally. Avoid an internet download during the presentation.

## Speaking order (2:00)

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Pipeline slide or terminal command | “YOLO11 detects people, ByteTrack connects detections into tracks, and a line counter emits IN/OUT events.” |
| 0:20–1:15 | External street video, annotated | Point out that this clip is outside the evaluation datasets. Show boxes, IDs, the virtual line, and the changing predicted directional totals. |
| 1:15–1:40 | Run command or event JSON | Show that the same trained checkpoint processes a normal MP4; the line is configurable. |
| 1:40–2:00 | Results slide | State that the external clip has no GT metrics. The reported CA/MOTA/IDF1 remain the preliminary post-selection MOT comparison. |

## If something fails

Use the pre-rendered external MP4 if live inference stalls. If it cannot open, use the pre-rendered MOT20-01 clip. State which footage is external and which belongs to the comparison dataset. Do not change model settings on stage.
