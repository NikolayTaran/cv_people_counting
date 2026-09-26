"""People counting: virtual line (directional in/out) and region (ROI).

Design goals — robustness against the three failure modes of naive counting:

* detection jitter at the boundary  -> Schmitt-trigger "arming" (line) and
  hysteresis (region): a crossing / region change only registers once the
  anchor has moved a configurable distance past the boundary;
* double counting                   -> per-track cooldown after an event;
* teleports from ID switches        -> a maximum travel distance per step and
  a maximum frame gap between consecutive anchor observations.

Both counters are pure-python state machines over per-track anchor points
(foot points), so the very same classes produce the ground-truth counts from
annotated tracks — evaluation and prediction share one code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import line_signed_distance, point_in_polygon

EventKind = str  # 'in' | 'out' | 'enter' | 'exit'


@dataclass(frozen=True)
class CountingEvent:
    frame: int
    track_id: int
    kind: EventKind
    x: float
    y: float


@dataclass
class _LineTrackState:
    last_frame: int = -1
    armed_side: int = 0          # -1 / +1 — side the track armed on
    last_d: float = 0.0
    cooldown_until: int = -1


@dataclass
class LineCounter:
    """Counts directional crossings of a virtual line.

    ``in`` is a crossing towards the positive side of p1->p2 (see
    geometry.line_signed_distance); set ``in_direction='negative'`` to flip.
    """

    p1: tuple[float, float]
    p2: tuple[float, float]
    min_hits: int = 3            # required tracker hits before events count
    arm_dist: float = 10.0       # px from the line to (re)arm the trigger
    cooldown: int = 8            # frames to ignore a track after an event
    max_travel: float = 250.0    # max |d_prev| + |d_now| per crossing
    max_gap: int = 5             # max frame gap between anchor observations
    in_direction: str = "positive"

    state: dict[int, _LineTrackState] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.in_direction not in ("positive", "negative"):
            raise ValueError("in_direction must be 'positive' or 'negative'")

    def update(self, track_id: int, frame: int, hits: int,
               x: float, y: float) -> list[CountingEvent]:
        """Feed one anchor observation; returns events triggered by it."""
        st = self.state.get(track_id)
        d = line_signed_distance((x, y), self.p1, self.p2)
        if st is None:
            self.state[track_id] = _LineTrackState(last_frame=frame, last_d=d)
            return []

        events: list[CountingEvent] = []
        gap = frame - st.last_frame
        eligible = hits >= self.min_hits

        # --- re-sync after a long gap: at most one transition event --------
        if gap > self.max_gap:
            if eligible and st.armed_side != 0:
                side_now = 1 if d >= self.arm_dist else (-1 if d <= -self.arm_dist else 0)
                if side_now != 0 and side_now != st.armed_side:
                    events.append(self._event(frame, track_id, side_now, x, y))
            self.state[track_id] = _LineTrackState(last_frame=frame, last_d=d)
            return events

        # --- regular Schmitt-trigger crossing ------------------------------
        if eligible and frame > st.cooldown_until:
            side_now = 1 if d >= self.arm_dist else (-1 if d <= -self.arm_dist else 0)
            if (
                st.armed_side != 0
                and side_now != 0
                and side_now != st.armed_side
                and abs(d) + abs(st.last_d) <= self.max_travel
            ):
                events.append(self._event(frame, track_id, side_now, x, y))
                st.cooldown_until = frame + self.cooldown

        st.last_frame = frame
        st.last_d = d
        if side_now_ := (1 if d >= self.arm_dist else (-1 if d <= -self.arm_dist else 0)):
            st.armed_side = side_now_
        return events

    def _event(self, frame: int, track_id: int, side_now: int,
               x: float, y: float) -> CountingEvent:
        if (side_now == 1) == (self.in_direction == "positive"):
            kind = "in"
        else:
            kind = "out"
        return CountingEvent(frame=frame, track_id=track_id, kind=kind, x=x, y=y)


@dataclass
class _RegionTrackState:
    last_frame: int = -1
    inside: bool | None = None
    pending: int = 0          # consecutive observations of the opposite state
    pending_kind: EventKind | None = None


@dataclass
class RegionCounter:
    """Counts enter/exit events of a polygonal region with hysteresis."""

    polygon: list[tuple[float, float]]
    min_hits: int = 3
    hysteresis: int = 2        # consecutive frames required to flip state
    max_gap: int = 5           # frames of absence before re-syncing

    state: dict[int, _RegionTrackState] = field(default_factory=dict, repr=False)

    def update(self, track_id: int, frame: int, hits: int,
               x: float, y: float) -> list[CountingEvent]:
        st = self.state.get(track_id)
        inside_now = point_in_polygon((x, y), self.polygon)
        if st is None:
            self.state[track_id] = _RegionTrackState(
                last_frame=frame, inside=inside_now)
            return []

        events: list[CountingEvent] = []
        gap = frame - st.last_frame
        eligible = hits >= self.min_hits

        # --- re-sync after a long absence: emit one transition if the state
        #     actually changed while the track was unobserved
        if gap > self.max_gap:
            if eligible and st.inside is not None and st.inside != inside_now:
                events.append(CountingEvent(
                    frame=frame, track_id=track_id,
                    kind="enter" if inside_now else "exit", x=x, y=y))
            self.state[track_id] = _RegionTrackState(
                last_frame=frame, inside=inside_now)
            return events

        if st.inside is None:      # first definite observation
            st.inside = inside_now
            st.pending, st.pending_kind = 0, None
        elif inside_now != st.inside:
            kind = "enter" if inside_now else "exit"
            if st.pending_kind == kind:
                st.pending += 1
            else:
                st.pending, st.pending_kind = 1, kind
            if eligible and st.pending >= self.hysteresis:
                events.append(CountingEvent(
                    frame=frame, track_id=track_id, kind=kind, x=x, y=y))
                st.inside = inside_now
                st.pending, st.pending_kind = 0, None
        else:                      # back to the stable state — reset pending
            st.pending, st.pending_kind = 0, None

        st.last_frame = frame
        return events


def summarize_events(events: list[CountingEvent]) -> dict[str, int]:
    """Totals per event kind ('in'/'out' for lines, 'enter'/'exit' for ROI)."""
    totals: dict[str, int] = {}
    for e in events:
        totals[e.kind] = totals.get(e.kind, 0) + 1
    return totals
