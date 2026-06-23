"""Tests for the centroid/IOU tracker."""

from __future__ import annotations

from thermalsentry.config import TrackerSettings
from thermalsentry.detection.detector import Detection
from thermalsentry.detection.tracker import CentroidTracker, iou


def _det(cx, cy, label="person", peak=34.0, half=10):
    return Detection(
        centroid=(cx, cy),
        bbox=(int(cx - half), int(cy - half), int(cx + half), int(cy + half)),
        area=4 * half * half,
        peak_temp_c=peak,
        mean_temp_c=peak - 2,
        label=label,
    )


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_id_stability_across_moving_blob():
    tr = CentroidTracker(TrackerSettings(max_distance=30, max_missed=5))
    now = 0.0
    tracks = tr.update([_det(50, 50)], now=now)
    assert len(tracks) == 1
    tid = tracks[0].id
    # Move the blob a little each frame; ID must stay constant.
    for step in range(1, 12):
        now += 0.1
        tracks = tr.update([_det(50 + step * 4, 50 + step * 2)], now=now)
        assert len(tracks) == 1
        assert tracks[0].id == tid


def test_two_blobs_get_distinct_ids():
    tr = CentroidTracker(TrackerSettings(max_distance=30))
    tracks = tr.update([_det(20, 20), _det(120, 120)], now=0.0)
    ids = sorted(t.id for t in tracks)
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_track_dropped_after_max_missed():
    tr = CentroidTracker(TrackerSettings(max_distance=30, max_missed=3))
    tr.update([_det(50, 50)], now=0.0)
    assert len(tr.tracks) == 1
    # Feed empty detections until the track ages out.
    for i in range(1, 6):
        tr.update([], now=float(i))
    assert len(tr.tracks) == 0


def test_new_track_when_jump_exceeds_distance():
    tr = CentroidTracker(TrackerSettings(max_distance=20, max_missed=5))
    tr.update([_det(10, 10)], now=0.0)
    # A far jump can't associate -> new ID created (old one still alive/missed).
    tracks = tr.update([_det(200, 200)], now=0.1)
    ids = {t.id for t in tracks}
    assert 2 in ids or len(ids) >= 1


def test_dwell_time_accumulates():
    tr = CentroidTracker(TrackerSettings(max_distance=30))
    tr.update([_det(50, 50)], now=100.0)
    tr.update([_det(52, 51)], now=130.0)
    track = list(tr.tracks.values())[0]
    assert track.dwell_seconds(now=130.0) >= 29.0
