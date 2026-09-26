"""Geometry primitives and dataset readers + metrics."""
import pandas as pd

from people_counter.counting.geometry import (line_signed_distance,
                                              point_in_polygon, polygon_area)
from people_counter.counting.counter import CountingEvent
from people_counter.evaluation.counting_metrics import counting_metrics


# ---------------------------------------------------------------- geometry
def test_line_signed_distance_sides():
    # line top->bottom at x=100
    d_right = line_signed_distance((130, 100), (100, 10), (100, 290))
    d_left = line_signed_distance((70, 100), (100, 10), (100, 290))
    assert d_right > 0 and d_left < 0
    assert abs(d_right - 30) < 1e-9 and abs(d_left + 30) < 1e-9


def test_line_signed_distance_degenerate_line():
    assert line_signed_distance((5, 5), (1, 1), (1, 1)) == 0.0


def test_point_in_polygon_square():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert not point_in_polygon((-1, 5), square)


def test_point_in_polygon_concave():
    concave = [(0, 0), (10, 0), (10, 10), (5, 5), (0, 10)]  # notch at top
    assert point_in_polygon((2, 2), concave)
    assert not point_in_polygon((5, 8), concave)   # inside the notch


def test_polygon_area():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert abs(polygon_area(square) - 100) < 1e-9


# ------------------------------------------------------------ counting metrics
def test_counting_metrics_perfect():
    pred = [CountingEvent(10, 1, "in", 0, 0), CountingEvent(20, 2, "out", 0, 0)]
    gt = [CountingEvent(12, 1, "in", 0, 0), CountingEvent(18, 2, "out", 0, 0)]
    m = counting_metrics(pred, gt, n_frames=100)
    assert m["counting_accuracy"] == 1.0
    assert m["curve_mae"] == 0.0
    assert m["pred_in"] == 1 and m["gt_out"] == 1


def test_counting_metrics_errors():
    pred = [CountingEvent(10, 1, "in", 0, 0),
            CountingEvent(20, 2, "in", 0, 0),
            CountingEvent(30, 3, "in", 0, 0)]
    gt = [CountingEvent(12, 1, "in", 0, 0)]
    m = counting_metrics(pred, gt, n_frames=100)
    # e_in = 2, e_out = 0, gt total = 1 -> CA = max(0, 1 - 2/1) = 0
    assert m["counting_accuracy"] == 0.0
    assert m["error_in"] == 2
    # curve MAE: checkpoints 50, 100; all three pred events and the single
    # gt event happen before frame 50 -> in diffs |3-1| = 2 at both
    # checkpoints, out diffs 0 -> mean = (2 + 2 + 0 + 0) / 4 = 1.0
    assert m["curve_mae"] == 1.0


def test_counting_metrics_empty_gt():
    m = counting_metrics([], [], n_frames=10)
    assert m["counting_accuracy"] == 1.0   # vacuous truth, no error
    assert m["curve_mae"] == 0.0


# ------------------------------------------------------------------ datasets
def test_mot_gt_filtering(tmp_path):
    from people_counter.datasets.mot import load_gt_tracks

    gt_file = tmp_path / "gt.txt"
    rows = [
        "1,1,10,10,20,40,1,1,0.9",    # pedestrian, active -> keep
        "1,2,50,10,20,40,1,7,0.5",    # static person -> drop
        "2,1,80,10,20,40,0,1,0.9",    # flag 0 (ignore) -> drop
        "2,3,90,10,20,40,1,1,0.8",    # pedestrian -> keep
    ]
    gt_file.write_text("\n".join(rows) + "\n")
    gt = load_gt_tracks(gt_file, "MOT17")
    assert list(gt["id"]) == [1, 3]
    assert list(gt["frame"]) == [1, 2]


def test_visdrone_gt_filtering(tmp_path):
    from people_counter.datasets.mot import load_gt_tracks

    gt_file = tmp_path / "ann.txt"
    rows = [
        "1,1,10,10,20,40,0,1,0,0",    # pedestrian -> keep
        "1,2,50,10,20,40,0,2,0,0",    # people -> keep
        "1,3,80,10,20,40,0,3,0,0",    # bicycle -> drop
        "2,4,90,10,0,0,0,1,0,0",      # zero area -> drop
    ]
    gt_file.write_text("\n".join(rows) + "\n")
    gt = load_gt_tracks(gt_file, "VisDrone")
    assert sorted(gt["id"]) == [1, 2]


def test_gt_feet_by_frame(tmp_path):
    from people_counter.datasets.mot import gt_feet_by_frame

    gt = pd.DataFrame({
        "frame": [1, 1, 2], "id": [1, 2, 1],
        "x": [10.0, 50.0, 12.0], "y": [10.0, 10.0, 10.0],
        "w": [20.0, 20.0, 20.0], "h": [40.0, 40.0, 40.0],
    })
    feet = gt_feet_by_frame(gt)
    assert set(feet) == {1, 2}
    assert feet[1][0] == (1, 20.0, 50.0)     # x + w/2, y + h
    assert len(feet[2]) == 1


# ----------------------------------------------------------------- mot metrics
def test_mot_metrics_perfect_tracking():
    from people_counter.evaluation.mot_metrics import compute_mot_metrics

    gt = pd.DataFrame({
        "frame": [1, 1, 2, 2], "id": [1, 2, 1, 2],
        "x": [10, 100, 12, 102], "y": [10, 10, 10, 10],
        "w": [20, 20, 20, 20], "h": [40, 40, 40, 40],
    })
    records = [
        {"frame": 0, "tracks": [[7, 10, 10, 30, 50], [8, 100, 10, 120, 50]]},
        {"frame": 1, "tracks": [[7, 12, 10, 32, 50], [8, 102, 10, 122, 50]]},
    ]
    m = compute_mot_metrics(gt, records)
    assert m["mota"] == 1.0
    assert m["idf1"] == 1.0
    assert m["idsw"] == 0
    assert m["num_objects"] == 4


def test_mot_metrics_with_fp_and_miss():
    from people_counter.evaluation.mot_metrics import compute_mot_metrics

    gt = pd.DataFrame({
        "frame": [1, 2], "id": [1, 1],
        "x": [10, 12], "y": [10, 10], "w": [20, 20], "h": [40, 40],
    })
    records = [
        {"frame": 0, "tracks": [[1, 10, 10, 30, 50], [2, 500, 500, 520, 540]]},  # 1 FP
        {"frame": 1, "tracks": []},                                              # 1 miss
    ]
    m = compute_mot_metrics(gt, records)
    # MOTA = 1 - (FP + FN + IDSW) / GT = 1 - (1 + 1 + 0)/2 = 0
    assert m["mota"] == 0.0
    assert m["num_false_positives"] == 1
    assert m["num_misses"] == 1
