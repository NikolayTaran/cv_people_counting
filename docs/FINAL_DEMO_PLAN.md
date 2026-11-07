# Two-minute live demo plan (instructions only)

Prepare two short annotated clips in advance: MOT20-01 for dense crowding and MOT17-09 for the low-angle view. Use the committed 3-epoch checkpoint for the first clip; optionally show the ensemble on the same MOT20-01 segment to contrast its recovery of counts. Use MOT17-09 to point out that extra fused detections can also produce a wrong-direction event. Do not promise that every short segment will contain the cited event unless the clip has been checked.

## Before presenting

1. Download MOT17 and MOT20 with the README commands and confirm `models/people_yolo11s_mot.pt` exists.
2. Pre-render a short segment of each sequence using the README's demo commands. Check that the videos open and visibly show person boxes, track IDs, the line, and IN/OUT totals.
3. Keep the saved clips and `docs/FINAL_PRESENTATION.md` available locally. Avoid depending on an internet download during the presentation.

## Speaking order (2:00)

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Pipeline slide or terminal command | “YOLO11 detects people, ByteTrack connects detections into tracks, and a line counter emits IN/OUT events.” |
| 0:20–1:05 | MOT20-01 annotated clip | Point out the dense crowd, IDs, line, and changing directional totals. State the measured full-sequence CA: 0.3462 single versus 0.5385 ensemble. |
| 1:05–1:35 | MOT17-09 annotated clip | Show the alternative camera view. Explain that ensemble CA is 0.7692 versus 0.9231 for the single detector; one audited extra OUT is consistent with apparent identity transfer. |
| 1:35–2:00 | Results slide | Conclude: the 3-epoch detector is the practical default; fusion is an optional density-sensitive trade-off. Call all comparison scores post-selection. |

## If something fails

Use the pre-rendered clip if live inference or video playback stalls. If no clip opens, show the saved frame/track diagnostic `docs/assets/final_track58_iou.png` and the scene-level results table on slide 5. State that this fallback illustrates saved evidence; it is not a live inference run. Do not change model settings on stage.
