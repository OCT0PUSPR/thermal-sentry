"""Simple multi-object tracker (pure numpy).

Associates detections across frames using a greedy nearest-centroid match with
an optional IOU gate, assigning stable integer IDs. Tracks accumulate a creation
time so the anomaly engine can compute dwell time (loitering).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TrackerSettings
from .detector import Detection


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    """A tracked object with a stable ID and history."""

    id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    label: str
    peak_temp_c: float
    first_seen: float
    last_seen: float
    age: int = 1  # number of frames matched
    missed: int = 0  # consecutive frames unmatched

    def dwell_seconds(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.first_seen)

    def as_dict(self, frame_w: int, frame_h: int, now: Optional[float] = None) -> dict:
        cx, cy = self.centroid
        x0, y0, x1, y1 = self.bbox
        return {
            "id": self.id,
            "label": self.label,
            "centroid": [round(cx, 2), round(cy, 2)],
            "centroid_norm": [
                round(cx / frame_w, 4) if frame_w else 0.0,
                round(cy / frame_h, 4) if frame_h else 0.0,
            ],
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_norm": [
                round(x0 / frame_w, 4) if frame_w else 0.0,
                round(y0 / frame_h, 4) if frame_h else 0.0,
                round(x1 / frame_w, 4) if frame_w else 0.0,
                round(y1 / frame_h, 4) if frame_h else 0.0,
            ],
            "peak_temp_c": round(self.peak_temp_c, 2),
            "age": self.age,
            "dwell_s": round(self.dwell_seconds(now), 1),
        }


class CentroidTracker:
    """Greedy nearest-centroid tracker with an optional IOU gate."""

    def __init__(self, settings: Optional[TrackerSettings] = None) -> None:
        self.settings = settings or TrackerSettings()
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    def update(
        self, detections: List[Detection], now: Optional[float] = None
    ) -> List[Track]:
        """Update tracks with the current frame's detections.

        ``now`` (monotonic seconds) is injectable for deterministic tests.
        Returns the list of currently live tracks.
        """
        now = time.monotonic() if now is None else now

        if not self.tracks:
            for det in detections:
                self._spawn(det, now)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        # Build the candidate match list (distance-gated, optionally IOU-gated).
        candidates: List[Tuple[float, int, int]] = []  # (distance, track_id, det_idx)
        for tid in track_ids:
            track = self.tracks[tid]
            tx, ty = track.centroid
            for di, det in enumerate(detections):
                dx, dy = det.centroid
                dist = float(np.hypot(tx - dx, ty - dy))
                if dist > self.settings.max_distance:
                    continue
                if self.settings.min_iou > 0:
                    if iou(track.bbox, det.bbox) < self.settings.min_iou:
                        continue
                candidates.append((dist, tid, di))

        # Greedy assignment: closest pairs first.
        candidates.sort(key=lambda c: c[0])
        for _dist, tid, di in candidates:
            if tid in unmatched_tracks and di in unmatched_dets:
                self._match(tid, detections[di], now)
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(di)

        # Unmatched detections -> new tracks.
        for di in unmatched_dets:
            self._spawn(detections[di], now)

        # Unmatched tracks -> increment missed, drop if stale.
        for tid in unmatched_tracks:
            track = self.tracks[tid]
            track.missed += 1
            if track.missed > self.settings.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())

    # -- internals ------------------------------------------------------------

    def _spawn(self, det: Detection, now: float) -> None:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(
            id=tid,
            centroid=det.centroid,
            bbox=det.bbox,
            label=det.label,
            peak_temp_c=det.peak_temp_c,
            first_seen=now,
            last_seen=now,
        )

    def _match(self, tid: int, det: Detection, now: float) -> None:
        track = self.tracks[tid]
        track.centroid = det.centroid
        track.bbox = det.bbox
        track.label = det.label
        track.peak_temp_c = det.peak_temp_c
        track.last_seen = now
        track.age += 1
        track.missed = 0
