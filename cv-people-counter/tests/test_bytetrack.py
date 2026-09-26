"""ByteTrack: association, lifecycle, occlusion handling, motion compensation."""
import numpy as np

from people_counter.tracking.bytetrack import (ByteTracker, iou_matrix,
                                               phase_shift)


def _boxes_at(cx, cy, w=30, h=60):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([
        [0, 0, 10, 10],    # identical -> 1.0
        [5, 5, 15, 15],    # 25/175 overlap
        [20, 20, 30, 30],  # disjoint -> 0.0
    ])
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 3)
    assert abs(iou[0, 0] - 1.0) < 1e-9
    assert abs(iou[0, 1] - 25 / 175) < 1e-9
    assert iou[0, 2] == 0.0


def test_two_objects_stable_ids():
    tracker = ByteTracker(n_init=2)
    ids_over_time = []
    for frame in range(25):
        dets = np.stack([
            _boxes_at(100 + 4 * frame, 100 + 2 * frame),
            _boxes_at(400 - 3 * frame, 300 - 1 * frame),
        ])
        confs = np.array([0.9, 0.9])
        active = tracker.update(dets, confs)
        ids_over_time.append(sorted(t.track_id for t in active))
    # after warm-up: exactly two confirmed tracks with identical ids all along
    for ids in ids_over_time[3:]:
        assert ids == ids_over_time[3]
        assert len(ids) == 2


def test_occlusion_survival_same_id():
    """Track must survive a 5-frame dropout with the same id (max_age=30)."""
    tracker = ByteTracker(n_init=2, max_age=30)
    first_id = None
    for frame in range(30):
        visible = not (10 <= frame < 15)   # occluded frames 10..14
        dets = [_boxes_at(200 + 3 * frame, 200)] if visible else []
        confs = np.array([0.9] * len(dets))
        active = tracker.update(np.array(dets).reshape(-1, 4), confs)
        if frame in (5,):
            first_id = active[0].track_id
        if frame == 20:
            assert any(t.track_id == first_id for t in active), \
                "track id changed across occlusion"


def test_low_conf_detections_keep_track_alive():
    """A weak box (below high_thresh) must match in stage 2, not spawn a track."""
    tracker = ByteTracker(high_thresh=0.5, match_thresh_low=0.5, n_init=2)
    dets = np.array([_boxes_at(150, 150)]).reshape(-1, 4)
    tracker.update(dets, np.array([0.9]))
    tid = tracker.tracks[0].track_id
    # now only a weak detection of the same object
    active = tracker.update(
        np.array([_boxes_at(152, 150)]).reshape(-1, 4), np.array([0.3]))
    assert len(tracker.tracks) == 1, "weak det must not create a second track"
    assert active and active[0].track_id == tid


def test_low_conf_alone_creates_no_track():
    tracker = ByteTracker(high_thresh=0.5, n_init=2)
    tracker.update(np.array([_boxes_at(150, 150)]).reshape(-1, 4),
                   np.array([0.3]))
    assert tracker.tracks == [] or all(
        t.hits == 0 for t in tracker.tracks) or tracker.tracks == []
    # BYTE: only HIGH detections spawn tracks
    assert len(tracker.tracks) == 0


def test_tentative_track_dies_on_miss():
    tracker = ByteTracker(high_thresh=0.5, n_init=3)
    tracker.update(np.array([_boxes_at(50, 50)]).reshape(-1, 4),
                   np.array([0.9]))
    assert len(tracker.tracks) == 1
    tracker.update(np.zeros((0, 4)), np.zeros((0,)))   # miss
    assert len(tracker.tracks) == 0                    # tentative -> deleted


def test_empty_frame_is_fine():
    tracker = ByteTracker()
    active = tracker.update(np.zeros((0, 4)), np.zeros((0,)))
    assert active == []


def test_phase_shift_sign():
    """Pins the sign convention: content moving right/down must report +dx/+dy."""
    base = np.zeros((120, 160), dtype=np.uint8)
    base[30:90, 30:130] = 200          # bright block with texture
    base[50:70, 40:60] = 255
    base[50:70, 90:120] = 60
    prev = base
    dx_true, dy_true = 6, 4
    cur = np.roll(np.roll(base, dy_true, axis=0), dx_true, axis=1)
    dx, dy = phase_shift(prev, cur)
    assert abs(dx - dx_true) <= 1.0, f"dx={dx}, expected ~{dx_true}"
    assert abs(dy - dy_true) <= 1.0, f"dy={dy}, expected ~{dy_true}"


def test_motion_compensation_keeps_tracks_on_moving_camera():
    """Simulated camera pan must not destroy tracking when MC is enabled."""
    import cv2

    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (5, 5), 0)   # natural-ish spectrum

    def frame_with_object(pan_x, obj_x, obj_y):
        img = np.roll(scene, pan_x, axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[obj_y - 15:obj_y + 15, obj_x - 8:obj_x + 8] = (0, 0, 255)
        return img

    def run(with_mc: bool) -> list[int]:
        tracker = ByteTracker(n_init=2, motion_compensation=with_mc)
        ids = []
        obj_x, obj_y = 120, 120
        for f in range(20):
            pan = 3 * f                     # camera pans right steadily
            ox = obj_x                      # object static in the world
            img = frame_with_object(pan, ox, obj_y)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = np.array([[ox - 8, obj_y - 15, ox + 8, obj_y + 15]],
                            dtype=float)
            active = tracker.update(dets, np.array([0.9]), gray)
            ids.extend(t.track_id for t in active)
        return ids

    ids_mc = run(True)
    assert len(set(ids_mc[4:])) == 1, \
        f"with MC the track must stay single, got {sorted(set(ids_mc[4:]))}"
