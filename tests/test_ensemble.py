"""Box-fusion checks for the second detector approach."""

import numpy as np
import pytest

from people_counter.detection.ensemble import fuse_boxes


def test_matching_boxes_are_confidence_weighted():
    first = (np.array([[0, 0, 10, 10]], dtype=np.float32),
             np.array([0.9], dtype=np.float32))
    second = (np.array([[1, 0, 11, 10]], dtype=np.float32),
              np.array([0.6], dtype=np.float32))

    boxes, scores = fuse_boxes(first, second)

    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [0.4, 0, 10.4, 10], atol=1e-5)
    np.testing.assert_allclose(scores, [0.9])


def test_nearby_boxes_from_one_detector_remain_distinct():
    first = (np.array([[0, 0, 10, 10], [2, 0, 12, 10]], dtype=np.float32),
             np.array([0.9, 0.8], dtype=np.float32))
    empty = (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))

    boxes, scores = fuse_boxes(first, empty)

    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(scores, [0.9, 0.8])


def test_empty_and_invalid_input():
    empty = (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
    boxes, scores = fuse_boxes(empty, empty)
    assert boxes.shape == (0, 4) and scores.shape == (0,)
    with pytest.raises(ValueError, match="same length"):
        fuse_boxes((np.zeros((1, 4)), np.empty(0)), empty)
