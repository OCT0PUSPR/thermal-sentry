"""Classical thermal blob detector.

Pipeline:

1. Threshold the (upscaled) frame to a binary hot mask, using a fixed and/or
   adaptive (background-mean + k*std) threshold.
2. Label connected components -- ``scipy.ndimage.label`` when available, with a
   pure-numpy iterative flood-fill fallback so the package works with numpy only.
3. For each component compute centroid, area, bbox, and peak/mean temperature.
4. Classify each blob as ``person`` / ``animal`` / ``hotspot`` / ``object`` by
   simple, transparent area+temperature heuristics.

The detector is tuned to find the right blobs on the synthetic data produced by
:class:`thermalsentry.sensors.simulator.SyntheticThermalSource`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..config import DetectionSettings


@dataclass
class Detection:
    """A single detected thermal blob (coordinates in upscaled-pixel space)."""

    centroid: Tuple[float, float]  # (x, y) = (col, row)
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    area: int
    peak_temp_c: float
    mean_temp_c: float
    label: str  # person | animal | hotspot | object
    confidence: float = 1.0

    def as_dict(self, frame_w: int, frame_h: int) -> dict:
        """Serialise, adding normalised (0..1) coordinates for the dashboard."""
        x0, y0, x1, y1 = self.bbox
        cx, cy = self.centroid
        return {
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
            "area": int(self.area),
            "peak_temp_c": round(self.peak_temp_c, 2),
            "mean_temp_c": round(self.mean_temp_c, 2),
            "label": self.label,
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Connected-components labelling
# ---------------------------------------------------------------------------


def _label_scipy(mask: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    try:
        from scipy.ndimage import label  # type: ignore

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labels, n = label(mask, structure=structure)
        return labels.astype(np.int32), int(n)
    except Exception:
        return None


def _label_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pure-numpy 8-connected component labelling via iterative flood fill.

    A small BFS over a stack; adequate for 24x32-derived masks even when
    upscaled because the number of foreground pixels is modest.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    # 8-connected neighbour offsets.
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1
                stack = [(sy, sx)]
                labels[sy, sx] = current
                while stack:
                    y, x = stack.pop()
                    for dy, dx in neighbours:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((ny, nx))
    return labels, current


def label_connected_components(
    mask: np.ndarray, prefer_scipy: bool = True
) -> Tuple[np.ndarray, int]:
    """Label connected components in a boolean mask. 8-connectivity.

    Returns ``(labels, count)`` where ``labels`` is an int32 array (0 = bg).
    """
    if prefer_scipy:
        res = _label_scipy(mask)
        if res is not None:
            return res
    return _label_numpy(mask)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass
class ThermalDetector:
    """Detect and classify hot blobs in a thermal frame."""

    settings: DetectionSettings = field(default_factory=DetectionSettings)
    prefer_scipy: bool = True

    def compute_threshold(self, frame: np.ndarray) -> float:
        """Compute the working hot-pixel threshold (deg C) for ``frame``."""
        thr = self.settings.hot_threshold_c
        if self.settings.adaptive:
            bg_mean = float(np.median(frame))
            bg_std = float(np.std(frame))
            adaptive_thr = bg_mean + self.settings.adaptive_k * bg_std
            thr = max(thr, adaptive_thr)
        return thr

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return detections for an (upscaled or raw) thermal frame.

        ``frame`` is a 2-D deg-C array. Detections use the frame's own pixel
        coordinate space, so callers usually pass the *upscaled* frame.
        """
        if frame.ndim != 2:
            raise ValueError("detect() expects a 2-D frame")

        thr = self.compute_threshold(frame)
        mask = frame >= thr
        if not mask.any():
            return []

        labels, n = label_connected_components(mask, prefer_scipy=self.prefer_scipy)
        detections: List[Detection] = []

        for lab in range(1, n + 1):
            ys, xs = np.where(labels == lab)
            area = int(ys.size)
            if area < self.settings.min_area:
                continue
            comp_temps = frame[ys, xs]
            peak = float(np.max(comp_temps))
            mean = float(np.mean(comp_temps))
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            x0, y0 = int(np.min(xs)), int(np.min(ys))
            x1, y1 = int(np.max(xs)) + 1, int(np.max(ys)) + 1

            label_name, conf = self._classify(area, peak, mean)
            detections.append(
                Detection(
                    centroid=(cx, cy),
                    bbox=(x0, y0, x1, y1),
                    area=area,
                    peak_temp_c=peak,
                    mean_temp_c=mean,
                    label=label_name,
                    confidence=conf,
                )
            )

        # Largest / hottest first.
        detections.sort(key=lambda d: (d.area, d.peak_temp_c), reverse=True)
        return detections

    def _classify(self, area: int, peak: float, mean: float) -> Tuple[str, float]:
        """Heuristic classification by area + peak temperature."""
        s = self.settings
        if peak >= s.hotspot_temp_c:
            return "hotspot", 0.95
        in_person_temp = s.person_min_temp_c <= peak <= s.person_max_temp_c
        if in_person_temp and s.person_min_area <= area <= s.person_max_area:
            # Confidence grows toward the centre of the person area band.
            mid = 0.5 * (s.person_min_area + s.person_max_area)
            span = max(1.0, 0.5 * (s.person_max_area - s.person_min_area))
            conf = float(np.clip(1.0 - abs(area - mid) / (2.0 * span), 0.5, 0.99))
            return "person", conf
        if in_person_temp and area < s.person_min_area:
            return "animal", 0.6
        return "object", 0.4
