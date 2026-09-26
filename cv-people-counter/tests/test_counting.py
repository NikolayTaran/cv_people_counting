"""Counting layer: line crossings, ROI enter/exit, jitter and gap handling."""
from people_counter.counting.counter import (LineCounter, RegionCounter,
                                             summarize_events)


def feed_line(counter, track_id, frames_xy, hits=10):
    events = []
    for frame, x, y in frames_xy:
        events.extend(counter.update(track_id, frame, hits, x, y))
    return events


def test_single_left_to_right_crossing_is_in():
    # vertical line x=100 from y=10 to y=290 (top -> bottom):
    # right of the direction vector = larger x -> walking right = IN
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3)
    events = feed_line(counter, 1, [(f, 100 - 40 + 6 * f, 150) for f in range(15)])
    kinds = [e.kind for e in events]
    assert kinds.count("in") == 1 and "out" not in kinds


def test_right_to_left_crossing_is_out():
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3)
    events = feed_line(counter, 7, [(f, 100 + 40 - 6 * f, 150) for f in range(15)])
    kinds = [e.kind for e in events]
    assert kinds.count("out") == 1 and "in" not in kinds


def test_in_direction_flip():
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3,
                          in_direction="negative")
    events = feed_line(counter, 1, [(f, 100 - 40 + 6 * f, 150) for f in range(15)])
    assert [e.kind for e in events] == ["out"]


def test_jitter_at_the_line_is_ignored():
    """Oscillation of +-3 px around the line must produce zero events."""
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3, arm_dist=10)
    frames = [(f, 100 + (3 if f % 2 else -3), 150) for f in range(30)]
    assert feed_line(counter, 1, frames) == []


def test_back_and_forth_counts_each_crossing():
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3,
                          arm_dist=10, cooldown=0)
    path = []
    x = 40
    for f in range(60):
        x += 12
        if x > 170:
            x = 40
        path.append((f, x, 150))
    events = feed_line(counter, 3, path)
    kinds = [e.kind for e in events]
    assert kinds.count("in") >= 3 and kinds.count("out") >= 3
    # strictly alternating after the first event
    seq = [k for k in kinds]
    for a, b in zip(seq, seq[1:]):
        assert a != b


def test_cooldown_prevents_immediate_recount():
    """Same-direction double crossing within cooldown must count once."""
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3,
                          arm_dist=10, cooldown=15)
    path = []
    x = 40
    for f in range(40):
        x += 15 if f < 12 else 0        # cross once, then hover past the line
        path.append((f, x, 150))
    events = feed_line(counter, 5, path)
    assert [e.kind for e in events].count("in") == 1


def test_teleport_across_frame_not_double_counted():
    """A long gap with a side change yields exactly one transition event."""
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=3, max_gap=5)
    frames = [(0, 40, 150), (1, 40, 150), (2, 40, 150),
              (30, 300, 150), (31, 300, 150)]   # gap of 28 frames
    events = feed_line(counter, 9, frames)
    assert len(events) == 1


def test_min_hits_gate():
    """Tracks with fewer hits than min_hits never produce events."""
    counter = LineCounter(p1=(100, 10), p2=(100, 290), min_hits=5)
    events = feed_line(counter, 1, [(f, 100 - 40 + 6 * f, 150)
                                    for f in range(15)], hits=3)
    assert events == []


def test_region_enter_exit_with_hysteresis():
    poly = [(50, 50), (200, 50), (200, 200), (50, 200)]
    counter = RegionCounter(polygon=poly, min_hits=1, hysteresis=2)
    path = [(0, 20, 100), (1, 60, 100), (2, 70, 100), (3, 70, 100),
            (10, 220, 100), (11, 220, 100), (12, 220, 100)]
    events = feed_line(counter, 1, path)
    kinds = [e.kind for e in events]
    assert kinds == ["enter", "exit"]


def test_region_flicker_suppressed():
    poly = [(50, 50), (200, 50), (200, 200), (50, 200)]
    counter = RegionCounter(polygon=poly, min_hits=1, hysteresis=2)
    # alternating single frames in/out never sustain 2 consecutive frames
    frames = [(f, 110 if f % 2 == 0 else 45, 100) for f in range(16)]
    events = feed_line(counter, 2, frames)
    assert events == []


def test_region_gap_transition():
    poly = [(50, 50), (200, 50), (200, 200), (50, 200)]
    counter = RegionCounter(polygon=poly, min_hits=1, hysteresis=2, max_gap=5)
    frames = [(0, 100, 100), (1, 100, 100), (2, 100, 100),
              (40, 250, 100), (41, 250, 100)]
    events = feed_line(counter, 4, frames)
    assert [e.kind for e in events] == ["exit"]


def test_summarize_events():
    from people_counter.counting.counter import CountingEvent
    events = [
        CountingEvent(1, 1, "in", 0, 0),
        CountingEvent(2, 2, "in", 0, 0),
        CountingEvent(3, 3, "out", 0, 0),
    ]
    assert summarize_events(events) == {"in": 2, "out": 1}
