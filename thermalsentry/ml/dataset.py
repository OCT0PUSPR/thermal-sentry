"""Synthetic + augmented thermal datasets for the from-scratch CNN.

Two dataset builders, both pure NumPy (no torch needed to *build* the data, so
tests + the representative-quantisation set use them freely):

* :func:`generate_dataset` -- small single-channel *crops* (default 24x24) for a
  quick per-blob classifier sanity check (kept for backward-compatibility and
  used by the crop classifier backend / unit tests).

* :func:`generate_frame_dataset` -- full up-scaled *frames* (48x64) reusing the
  project's :class:`SyntheticThermalSource` plus heavy augmentation (rotation,
  flip, sensor noise, thermal-gradient shifts, varying body counts/temps, and
  injected overheat anomalies). Each frame carries a **class label** (dominant
  scene content) and a **center heatmap** (warm-body centers), i.e. labels for
  the model's two heads. This is the dataset the real training run uses.

Crops/frames are normalised into ``[0, 1]`` with a fixed temperature band so the
INT8 quantiser sees a stable input range.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .. import FRAME_COLS, FRAME_ROWS
from ..sensors.simulator import SyntheticThermalSource
from .labels import (
    HEATMAP_H,
    HEATMAP_STRIDE,
    HEATMAP_W,
    MODEL_IN_H,
    MODEL_IN_W,
    NUM_CLASSES,
)

# Temperature normalisation band used to map deg C -> [0, 1] for the model.
TEMP_NORM_MIN = 15.0
TEMP_NORM_MAX = 60.0


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Map a deg-C array into [0, 1] using the fixed training band."""
    c = (crop.astype(np.float32) - TEMP_NORM_MIN) / (TEMP_NORM_MAX - TEMP_NORM_MIN)
    return np.clip(c, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Crop dataset (kept for the crop classifier backend + existing tests)
# ---------------------------------------------------------------------------


def _gaussian_blob(size, cy, cx, sy, sx, amp):
    yy = np.arange(size)[:, None]
    xx = np.arange(size)[None, :]
    return amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def make_sample(label_idx: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize one deg-C crop for the given class index."""
    ambient = rng.uniform(19.0, 24.0)
    gy = np.linspace(-1, 1, size)[:, None] * rng.uniform(-1.0, 1.0)
    gx = np.linspace(-1, 1, size)[None, :] * rng.uniform(-1.0, 1.0)
    crop = np.full((size, size), ambient, dtype=np.float32) + (gy + gx)

    cy = size / 2 + rng.uniform(-2, 2)
    cx = size / 2 + rng.uniform(-2, 2)

    if label_idx == 1:  # person: medium, taller-than-wide, 30-36 C
        peak = rng.uniform(30.0, 36.0)
        sy = rng.uniform(size * 0.22, size * 0.32)
        sx = rng.uniform(size * 0.14, size * 0.22)
        crop += _gaussian_blob(size, cy, cx, sy, sx, peak - ambient)
    elif label_idx == 2:  # animal: small, low, 28-33 C
        peak = rng.uniform(28.0, 33.0)
        s = rng.uniform(size * 0.08, size * 0.15)
        crop += _gaussian_blob(size, cy, cx, s, s * rng.uniform(0.8, 1.4), peak - ambient)
    elif label_idx == 3:  # hotspot: compact, hot >=45 C
        peak = rng.uniform(46.0, 75.0)
        s = rng.uniform(size * 0.06, size * 0.16)
        crop += _gaussian_blob(size, cy, cx, s, s, peak - ambient)
    # label_idx == 0 (background): nothing added.

    crop += rng.normal(0.0, rng.uniform(0.2, 0.6), size=crop.shape)
    if rng.random() < 0.5:
        crop = crop[:, ::-1].copy()
    crop += rng.uniform(-1.0, 1.0)  # ambient offset jitter
    return crop.astype(np.float32)


def generate_dataset(
    n_per_class: int = 600, size: int = 24, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a balanced *crop* dataset.

    Returns ``X`` (N, 1, size, size) float32 in [0, 1] and ``y`` (N,) int64.
    """
    rng = np.random.default_rng(seed)
    crops = []
    labels = []
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            crop = make_sample(cls, size, rng)
            crops.append(normalize_crop(crop))
            labels.append(cls)
    X = np.stack(crops, axis=0)[:, None, :, :].astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Full-frame dataset (the real training data)
# ---------------------------------------------------------------------------


def _resize_bilinear(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Local bilinear resize (avoids importing the display preprocess module)."""
    from ..processing.preprocess import _bilinear_numpy

    return _bilinear_numpy(frame.astype(np.float32), out_h, out_w)


def _rotate90(frame: np.ndarray, k: int) -> np.ndarray:
    """Rotate by k*90 degrees but keep the 24x32 aspect by resizing back.

    For non-square frames a true 90-degree rotation swaps the axes; we rotate
    then resize back to (24, 32) so the geometry stays sensor-consistent. This
    still meaningfully augments the gradient + body layout.
    """
    if k % 4 == 0:
        return frame
    rot = np.rot90(frame, k)
    if rot.shape != frame.shape:
        rot = _resize_bilinear(rot, frame.shape[0], frame.shape[1])
    return rot.astype(np.float32)


def _inject_overheat(frame: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Add a compact, very hot source (an overheat anomaly) at a random cell."""
    h, w = frame.shape
    r = float(rng.uniform(3, h - 4))
    c = float(rng.uniform(3, w - 4))
    peak = float(rng.uniform(52.0, 85.0))
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    sig = rng.uniform(0.8, 1.6)
    blob = peak * np.exp(-0.5 * (((rows - r) / sig) ** 2 + ((cols - c) / sig) ** 2))
    out = np.maximum(frame, blob)
    return out.astype(np.float32), (r, c)


def _augment_raw(frame: np.ndarray, rng: np.random.Generator, centers: List[Tuple[float, float]]):
    """Apply flips/rotation/noise/gradient shift to a raw 24x32 frame + centers.

    Returns ``(frame, centers)`` with center (row, col) coordinates transformed
    consistently with the geometric augmentation.
    """
    h, w = frame.shape
    out = frame.astype(np.float32).copy()
    cs = list(centers)

    # Horizontal flip.
    if rng.random() < 0.5:
        out = out[:, ::-1].copy()
        cs = [(r, (w - 1) - c) for (r, c) in cs]
    # Vertical flip.
    if rng.random() < 0.5:
        out = out[::-1, :].copy()
        cs = [((h - 1) - r, c) for (r, c) in cs]
    # 180-degree rotation (aspect-preserving for a non-square frame).
    if rng.random() < 0.25:
        out = np.rot90(out, 2).copy()
        cs = [((h - 1) - r, (w - 1) - c) for (r, c) in cs]

    # Thermal-gradient shift: add a random smooth gradient + global offset.
    rr = np.linspace(-1.0, 1.0, h)[:, None]
    cc = np.linspace(-1.0, 1.0, w)[None, :]
    grad = (rng.uniform(-2.0, 2.0) * rr + rng.uniform(-2.0, 2.0) * cc)
    out = out + grad.astype(np.float32) + float(rng.uniform(-2.0, 2.0))

    # Extra sensor noise.
    out = out + rng.normal(0.0, rng.uniform(0.0, 0.5), size=out.shape).astype(np.float32)
    return out.astype(np.float32), cs


def _make_heatmap(centers: List[Tuple[float, float]]) -> np.ndarray:
    """Build a gaussian center heatmap at the model's heatmap resolution.

    ``centers`` are (row, col) in the raw 24x32 grid; they are scaled to the
    heatmap grid (HEATMAP_H x HEATMAP_W) and splatted with a small gaussian.
    """
    heat = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
    if not centers:
        return heat
    sy = HEATMAP_H / FRAME_ROWS
    sx = HEATMAP_W / FRAME_COLS
    rows = np.arange(HEATMAP_H)[:, None]
    cols = np.arange(HEATMAP_W)[None, :]
    sigma = 0.9
    for (r, c) in centers:
        hy = r * sy
        hx = c * sx
        g = np.exp(-0.5 * (((rows - hy) / sigma) ** 2 + ((cols - hx) / sigma) ** 2))
        np.maximum(heat, g.astype(np.float32), out=heat)
    return heat


def _add_animal(frame: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Add a small, low-temperature warm body (an animal) at a random cell."""
    h, w = frame.shape
    r = float(rng.uniform(3, h - 4))
    c = float(rng.uniform(3, w - 4))
    peak = float(rng.uniform(27.0, 32.0))
    ambient = float(np.median(frame))
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    sig = rng.uniform(0.7, 1.2)
    blob = ambient + (peak - ambient) * np.exp(
        -0.5 * (((rows - r) / sig) ** 2 + ((cols - c) / sig) ** 2)
    )
    out = np.maximum(frame, blob)
    return out.astype(np.float32), (r, c)


def make_frame_sample(
    rng: np.random.Generator, scene: Optional[int] = None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Synthesize one augmented full-frame sample.

    ``scene`` (0..3) forces the dominant scene class for class balance; if None a
    class is drawn uniformly. Returns ``(frame_in, class_idx, heatmap)`` where
    ``frame_in`` is the normalised (1, MODEL_IN_H, MODEL_IN_W) model input,
    ``class_idx`` the classification label, and ``heatmap`` the center target.
    """
    if scene is None:
        scene = int(rng.integers(0, NUM_CLASSES))

    # Person scenes have 1..4 bodies; other scenes may also carry incidental
    # people, but the dominant label is the requested ``scene``.
    if scene == 1:  # person-dominant
        n_bodies = int(rng.integers(1, 5))
    elif scene == 0:  # background
        n_bodies = 0
    else:  # animal / hotspot scenes: occasionally a person is present too
        n_bodies = int(rng.integers(0, 2))

    body_temp = float(rng.uniform(31.0, 37.0))
    ambient = float(rng.uniform(18.0, 26.0))
    noise = float(rng.uniform(0.2, 0.7))
    src = SyntheticThermalSource(
        num_bodies=n_bodies,
        ambient_c=ambient,
        body_temp_c=body_temp,
        noise_std=noise,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    for _ in range(int(rng.integers(1, 6))):
        frame = src.read()
    centers = [(r, c) for (r, c) in src.body_positions()]

    if scene == 3:  # hotspot/overheat anomaly
        frame, oc = _inject_overheat(frame, rng)
        centers = centers + [oc]
    elif scene == 2:  # animal
        frame, ac = _add_animal(frame, rng)
        centers = centers + [ac]

    # Geometric + photometric augmentation (consistent with centers).
    frame, centers = _augment_raw(frame, rng, centers)

    heat = _make_heatmap(centers)

    up = _resize_bilinear(frame, MODEL_IN_H, MODEL_IN_W)
    x = normalize_crop(up)[None, :, :].astype(np.float32)
    return x, int(scene), heat


def generate_frame_dataset(
    n: int = 3000, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a full-frame dataset for the two-head model.

    Returns
    -------
    X    : (N, 1, MODEL_IN_H, MODEL_IN_W) float32 in [0, 1]
    y    : (N,) int64 class labels
    H    : (N, HEATMAP_H, HEATMAP_W) float32 center heatmaps in [0, 1]
    """
    rng = np.random.default_rng(seed)
    xs = np.empty((n, 1, MODEL_IN_H, MODEL_IN_W), dtype=np.float32)
    ys = np.empty((n,), dtype=np.int64)
    hs = np.empty((n, HEATMAP_H, HEATMAP_W), dtype=np.float32)
    for i in range(n):
        # Round-robin the scene class for a balanced dataset across all heads.
        scene = i % NUM_CLASSES
        x, cls, heat = make_frame_sample(rng, scene=scene)
        xs[i] = x
        ys[i] = cls
        hs[i] = heat
    # Shuffle so batches mix classes.
    perm = rng.permutation(n)
    return xs[perm], ys[perm], hs[perm]


def heatmap_peaks(
    heat: np.ndarray, threshold: float = 0.3, min_distance: int = 1
) -> List[Tuple[int, int, float]]:
    """Extract local-maxima peaks from a heatmap.

    Returns a list of ``(row, col, score)`` in heatmap-grid coordinates. A cell
    is a peak if it exceeds ``threshold`` and is >= all of its 8 neighbours.
    """
    h, w = heat.shape
    peaks: List[Tuple[int, int, float]] = []
    for r in range(h):
        for c in range(w):
            v = float(heat[r, c])
            if v < threshold:
                continue
            is_max = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and heat[nr, nc] > v:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                peaks.append((r, c, v))
    # Greedy non-max suppression by distance.
    peaks.sort(key=lambda p: p[2], reverse=True)
    kept: List[Tuple[int, int, float]] = []
    for r, c, v in peaks:
        if all((r - kr) ** 2 + (c - kc) ** 2 >= min_distance ** 2 for kr, kc, _ in kept):
            kept.append((r, c, v))
    return kept


def heatmap_grid_to_frame(r: int, c: int) -> Tuple[float, float]:
    """Map a heatmap-grid cell back to raw 24x32 (row, col) coordinates."""
    fr = (r + 0.5) * FRAME_ROWS / HEATMAP_H
    fc = (c + 0.5) * FRAME_COLS / HEATMAP_W
    return float(fr), float(fc)


__all__ = [
    "TEMP_NORM_MIN",
    "TEMP_NORM_MAX",
    "normalize_crop",
    "make_sample",
    "generate_dataset",
    "make_frame_sample",
    "generate_frame_dataset",
    "heatmap_peaks",
    "heatmap_grid_to_frame",
    "HEATMAP_STRIDE",
]
