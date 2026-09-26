"""Kalman filter behaviour: static boxes, constant velocity, compensation."""
import numpy as np

from people_counter.tracking.kalman import BoxKalman, xyxy_to_zhac, zhac_to_xyxy


def test_roundtrip_conversion():
    bbox = (10.0, 20.0, 60.0, 120.0)
    z = xyxy_to_zhac(bbox)
    back = zhac_to_xyxy(z)
    assert np.allclose(back, bbox, atol=1e-6)


def test_static_box_prediction_stays_put():
    kf = BoxKalman((100, 100, 150, 150))
    for _ in range(10):
        kf.predict()
    box = kf.bbox
    assert abs(box[0] - 100) < 8 and abs(box[1] - 100) < 8  # grows slowly
    assert abs(box[2] - 150) < 5 and abs(box[3] - 150) < 5


def test_constant_velocity_tracking():
    """After a few updates the filter must predict motion accurately."""
    vx, vy = 5.0, 3.0
    x, y = 100.0, 200.0
    kf = BoxKalman((x, y, x + 40, y + 80))
    box = np.array([x, y, x + 40, y + 80])
    for _ in range(15):
        kf.predict()
        box = box + np.array([vx, vy, vx, vy])
        kf.update(box)
    kf.predict()
    predicted = kf.bbox
    true_next = box + np.array([vx, vy, vx, vy])
    assert np.abs(predicted[:2] - true_next[:2]).max() < 1.5


def test_covariance_shrinks_with_updates():
    """Repeated observation of a STATIC box must reduce total uncertainty."""
    kf = BoxKalman((50, 50, 90, 130))
    before = np.trace(kf.covariance)
    for _ in range(10):
        kf.predict()
        kf.update((50, 50, 90, 130))     # static object
    after = np.trace(kf.covariance)
    assert after < before


def test_shift_moves_state():
    kf = BoxKalman((10, 10, 50, 60))
    kf.predict()
    kf.shift(7.0, -3.0)
    box = kf.bbox
    assert abs(box[0] - 17) < 1e-6 and abs(box[1] - 7) < 1e-6
