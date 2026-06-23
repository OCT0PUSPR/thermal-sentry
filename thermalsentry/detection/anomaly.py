"""Anomaly / alert rules.

Each rule inspects the current frame, detections, and tracks and emits zero or
more :class:`Alert` objects. Rules implemented:

* ``overheat``       -- any blob/pixel peak above the overheat threshold.
* ``rapid_rise``     -- scene-max temperature rose sharply over a time window.
* ``zone_intrusion`` -- a person centroid inside a restricted polygon.
* ``loitering``      -- a tracked person dwelt longer than a threshold.
* ``crowding``       -- live person count exceeded a threshold.

Coordinates for zones are NORMALISED (0..1). Detections/tracks expose pixel
centroids plus the frame size so the engine can normalise them.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from ..config import AnomalySettings
from .detector import Detection
from .tracker import Track

Severity = str  # "info" | "warning" | "critical"


@dataclass
class Alert:
    """A single anomaly alert."""

    rule: str
    severity: Severity
    message: str
    key: str  # debounce key (e.g. "loitering:track=5")
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "key": self.key,
            "timestamp": self.timestamp,
            "data": self.data,
        }


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test. ``polygon`` is a list of (x, y)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class AnomalyEngine:
    """Evaluate anomaly rules against the live pipeline state."""

    def __init__(self, settings: Optional[AnomalySettings] = None) -> None:
        self.settings = settings or AnomalySettings()
        # History of (timestamp, scene_max_c) for the rapid-rise rule.
        self._scene_max_history: Deque[Tuple[float, float]] = deque(maxlen=512)

    def set_zones(self, zones: List[List[Tuple[float, float]]]) -> None:
        """Replace the restricted zones (normalised polygons)."""
        self.settings.restricted_zones = zones

    def evaluate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        frame_w: int,
        frame_h: int,
        now: Optional[float] = None,
        mono_now: Optional[float] = None,
    ) -> List[Alert]:
        """Run all rules and return the alerts raised this frame.

        ``now`` is a wall-clock timestamp used for the rapid-rise window.
        ``mono_now`` is the monotonic clock used to compute track dwell time and
        MUST match the clock used to set ``Track.first_seen`` (the tracker uses
        ``time.monotonic()``). If omitted it defaults to ``now`` so tests that
        drive both with a single injected clock keep working.
        """
        wall_now = time.time() if now is None else now
        dwell_now = wall_now if mono_now is None else mono_now
        alerts: List[Alert] = []

        scene_max = float(np.max(frame)) if frame.size else 0.0
        person_tracks = [t for t in tracks if t.label == "person"]

        alerts.extend(self._rule_overheat(frame, detections, scene_max))
        alerts.extend(self._rule_rapid_rise(scene_max, wall_now))
        alerts.extend(self._rule_zone_intrusion(person_tracks, frame_w, frame_h))
        alerts.extend(self._rule_loitering(person_tracks, dwell_now))
        alerts.extend(self._rule_crowding(person_tracks))
        return alerts

    # -- individual rules -----------------------------------------------------

    def _rule_overheat(
        self, frame: np.ndarray, detections: List[Detection], scene_max: float
    ) -> List[Alert]:
        thr = self.settings.overheat_temp_c
        if scene_max < thr:
            return []
        hottest = max(
            detections, key=lambda d: d.peak_temp_c, default=None
        )
        peak = hottest.peak_temp_c if hottest else scene_max
        return [
            Alert(
                rule="overheat",
                severity="critical",
                message=f"Overheat / fire risk: peak {peak:.1f} C (>= {thr:.1f} C)",
                key="overheat",
                data={"peak_temp_c": round(peak, 2), "threshold_c": thr},
            )
        ]

    def _rule_rapid_rise(self, scene_max: float, now: float) -> List[Alert]:
        s = self.settings
        self._scene_max_history.append((now, scene_max))
        window_start = now - s.rapid_rise_window_s
        past = [v for (t, v) in self._scene_max_history if t >= window_start]
        if len(past) < 2:
            return []
        delta = scene_max - min(past)
        if delta >= s.rapid_rise_delta_c:
            return [
                Alert(
                    rule="rapid_rise",
                    severity="warning",
                    message=(
                        f"Rapid temperature rise: +{delta:.1f} C in "
                        f"{s.rapid_rise_window_s:.0f}s"
                    ),
                    key="rapid_rise",
                    data={"delta_c": round(delta, 2)},
                )
            ]
        return []

    def _rule_zone_intrusion(
        self, person_tracks: List[Track], frame_w: int, frame_h: int
    ) -> List[Alert]:
        zones = self.settings.restricted_zones
        if not zones:
            return []
        alerts: List[Alert] = []
        for track in person_tracks:
            cx, cy = track.centroid
            nx = cx / frame_w if frame_w else 0.0
            ny = cy / frame_h if frame_h else 0.0
            for zi, poly in enumerate(zones):
                if point_in_polygon(nx, ny, poly):
                    alerts.append(
                        Alert(
                            rule="zone_intrusion",
                            severity="critical",
                            message=f"Person #{track.id} entered restricted zone {zi}",
                            key=f"zone_intrusion:zone={zi}:track={track.id}",
                            data={"track_id": track.id, "zone": zi},
                        )
                    )
                    break
        return alerts

    def _rule_loitering(self, person_tracks: List[Track], now: float) -> List[Alert]:
        thr = self.settings.loiter_seconds
        alerts: List[Alert] = []
        for track in person_tracks:
            dwell = track.dwell_seconds(now)
            if dwell >= thr:
                alerts.append(
                    Alert(
                        rule="loitering",
                        severity="warning",
                        message=f"Person #{track.id} loitering for {dwell:.0f}s",
                        key=f"loitering:track={track.id}",
                        data={"track_id": track.id, "dwell_s": round(dwell, 1)},
                    )
                )
        return alerts

    def _rule_crowding(self, person_tracks: List[Track]) -> List[Alert]:
        count = len(person_tracks)
        if count > self.settings.max_person_count:
            return [
                Alert(
                    rule="crowding",
                    severity="warning",
                    message=(
                        f"Person count {count} exceeds limit "
                        f"{self.settings.max_person_count}"
                    ),
                    key="crowding",
                    data={"count": count},
                )
            ]
        return []
