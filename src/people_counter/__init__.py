"""people_counter — detector-and-tracker pipeline for counting people
entering or leaving a region.

Modules
-------
detection    YOLO11n detector wrapper (person class)
tracking     Custom ByteTrack: Kalman filter, track management, BYTE association
counting     Virtual-line and region (ROI) counters with anti-jitter logic
datasets     MOT17 / MOT20 / VisDrone readers + ground-truth generation
evaluation   Counting metrics (MAE, CA, curve-MAE) and MOT metrics (MOTA, IDF1)
pipeline     End-to-end sequence runner
"""

__version__ = "1.0.0"
