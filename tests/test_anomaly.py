"""Tests for the anomaly rule engine."""

from __future__ import annotations

import numpy as np

from thermalsentry.config import AnomalySettings
from thermalsentry.detection.anomaly import AnomalyEngine, point_in_polygon
from thermalsentry.detection.detector import Detection
from thermalsentry.detection.tracker import Track

FRAME_W, FRAME_H = 320, 240


def _person_track(tid, cx, cy, first_seen=0.0, last_seen=0.0, peak=34.0):
    return Track(
        id=tid,
        centroid=(cx, cy),
        bbox=(int(cx - 10), int(cy - 10), int(cx + 10), int(cy + 10)),
        label="person",
        peak_temp_c=peak,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def _flat_frame(temp):
    return np.full((FRAME_H, FRAME_W), temp, dtype=np.float32)


def test_point_in_polygon():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon(0.5, 0.5, sq)
    assert not point_in_polygon(1.5, 0.5, sq)


def test_overheat_triggers():
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=50.0))
    frame = _flat_frame(22.0)
    frame[10:20, 10:20] = 60.0
    det = Detection((15, 15), (10, 10, 20, 20), 100, 60.0, 55.0, "hotspot")
    alerts = eng.evaluate(frame, [det], [], FRAME_W, FRAME_H, now=1.0)
    rules = {a.rule for a in alerts}
    assert "overheat" in rules
    over = next(a for a in alerts if a.rule == "overheat")
    assert over.severity == "critical"


def test_overheat_not_triggered_when_cool():
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=50.0))
    alerts = eng.evaluate(_flat_frame(30.0), [], [], FRAME_W, FRAME_H, now=1.0)
    assert all(a.rule != "overheat" for a in alerts)


def test_rapid_rise_triggers():
    eng = AnomalyEngine(
        AnomalySettings(
            overheat_temp_c=999, rapid_rise_delta_c=8.0, rapid_rise_window_s=5.0
        )
    )
    eng.evaluate(_flat_frame(22.0), [], [], FRAME_W, FRAME_H, now=0.0)
    alerts = eng.evaluate(_flat_frame(34.0), [], [], FRAME_W, FRAME_H, now=2.0)
    assert any(a.rule == "rapid_rise" for a in alerts)


def test_zone_intrusion_triggers():
    zone = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=999, restricted_zones=[zone]))
    # Centroid at (32, 24) -> normalised (0.1, 0.1), inside the zone.
    track = _person_track(1, 32, 24)
    alerts = eng.evaluate(_flat_frame(30.0), [], [track], FRAME_W, FRAME_H, now=1.0)
    assert any(a.rule == "zone_intrusion" for a in alerts)


def test_zone_intrusion_outside_zone():
    zone = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=999, restricted_zones=[zone]))
    track = _person_track(1, 300, 220)  # normalised ~(0.94, 0.92), outside.
    alerts = eng.evaluate(_flat_frame(30.0), [], [track], FRAME_W, FRAME_H, now=1.0)
    assert all(a.rule != "zone_intrusion" for a in alerts)


def test_loitering_triggers():
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=999, loiter_seconds=20.0))
    track = _person_track(1, 100, 100, first_seen=0.0)
    alerts = eng.evaluate(_flat_frame(30.0), [], [track], FRAME_W, FRAME_H, now=25.0)
    assert any(a.rule == "loitering" for a in alerts)


def test_loitering_not_yet():
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=999, loiter_seconds=20.0))
    track = _person_track(1, 100, 100, first_seen=0.0)
    alerts = eng.evaluate(_flat_frame(30.0), [], [track], FRAME_W, FRAME_H, now=5.0)
    assert all(a.rule != "loitering" for a in alerts)


def test_crowding_triggers():
    eng = AnomalyEngine(AnomalySettings(overheat_temp_c=999, max_person_count=2))
    tracks = [_person_track(i, 20 * i, 20 * i) for i in range(1, 5)]
    alerts = eng.evaluate(_flat_frame(30.0), [], tracks, FRAME_W, FRAME_H, now=1.0)
    assert any(a.rule == "crowding" for a in alerts)
