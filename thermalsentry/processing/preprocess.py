"""Thermal frame preprocessing.

Everything here is pure NumPy so it is fast, dependency-light, and unit-testable:

* :func:`bilinear_upscale` -- resize a 24x32 frame to a larger grid (scipy is
  used when available for speed, but a pure-numpy fallback is the default).
* :func:`normalize_temperature` -- map deg C onto a normalised 0..1 range.
* :func:`apply_colormap` -- apply an 'ironbow', 'inferno', or 'grayscale' LUT.
* :func:`frame_to_rgb` -- the convenience pipeline used by the dashboard.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Colormap lookup tables (LUTs). Each is a (256, 3) uint8 array. Anchor colors
# are defined and linearly interpolated to 256 entries with pure numpy so the
# package has no hard dependency on matplotlib / opencv for display.
# ---------------------------------------------------------------------------


def _interp_lut(anchors: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``(K, 3)`` anchor colors to a ``(256, 3)`` uint8 LUT."""
    k = anchors.shape[0]
    xp = np.linspace(0.0, 1.0, k)
    x = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(x, xp, anchors[:, c]) for c in range(3)]
    lut = np.stack(channels, axis=1)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


# "Ironbow" / "iron" palette: black -> purple -> red -> orange -> yellow -> white.
_IRONBOW_ANCHORS = np.array(
    [
        [0, 0, 0],
        [30, 0, 60],
        [90, 0, 120],
        [160, 20, 90],
        [220, 60, 30],
        [255, 130, 0],
        [255, 200, 40],
        [255, 245, 160],
        [255, 255, 255],
    ],
    dtype=np.float64,
)

# "Inferno"-like palette: black -> purple -> magenta -> orange -> yellow.
_INFERNO_ANCHORS = np.array(
    [
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ],
    dtype=np.float64,
)

_GRAYSCALE_ANCHORS = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)

_LUTS = {
    "ironbow": _interp_lut(_IRONBOW_ANCHORS),
    "iron": _interp_lut(_IRONBOW_ANCHORS),
    "inferno": _interp_lut(_INFERNO_ANCHORS),
    "grayscale": _interp_lut(_GRAYSCALE_ANCHORS),
    "gray": _interp_lut(_GRAYSCALE_ANCHORS),
}


def get_lut(name: str) -> np.ndarray:
    """Return the ``(256, 3)`` uint8 LUT for ``name`` (defaults to ironbow)."""
    return _LUTS.get(name.lower(), _LUTS["ironbow"])


# ---------------------------------------------------------------------------
# Upscaling
# ---------------------------------------------------------------------------


def _bilinear_numpy(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-numpy bilinear resize (align_corners style) of a 2-D array."""
    in_h, in_w = frame.shape
    if out_h == in_h and out_w == in_w:
        return frame.astype(np.float32)

    # Sample coordinates in input space using align_corners semantics.
    if out_h > 1:
        ys = np.linspace(0, in_h - 1, out_h)
    else:
        ys = np.zeros(out_h)
    if out_w > 1:
        xs = np.linspace(0, in_w - 1, out_w)
    else:
        xs = np.zeros(out_w)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    f = frame.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def bilinear_upscale(
    frame: np.ndarray, factor: int = 20, use_scipy: bool = False
) -> np.ndarray:
    """Bilinearly upscale a 2-D thermal frame by an integer ``factor``.

    Parameters
    ----------
    frame:
        2-D input array (deg C).
    factor:
        Per-axis upscale factor (>= 1).
    use_scipy:
        If True and scipy is importable, use :func:`scipy.ndimage.zoom`
        (order=1). Otherwise the pure-numpy implementation is used.
    """
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2-D frame, got shape {frame.shape}")
    if factor < 1:
        raise ValueError("factor must be >= 1")

    out_h = frame.shape[0] * factor
    out_w = frame.shape[1] * factor

    if use_scipy:
        try:
            from scipy.ndimage import zoom  # type: ignore

            return zoom(frame.astype(np.float32), factor, order=1).astype(np.float32)
        except Exception:
            pass  # fall through to the numpy implementation
    return _bilinear_numpy(frame, out_h, out_w)


# ---------------------------------------------------------------------------
# Normalisation + colormapping
# ---------------------------------------------------------------------------


def normalize_temperature(
    frame: np.ndarray,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> np.ndarray:
    """Map temperatures (deg C) into ``[0, 1]``.

    If ``tmin`` / ``tmax`` are omitted they are taken from the frame's own
    min/max (per-frame auto-scaling).
    """
    f = frame.astype(np.float32)
    lo = float(np.min(f)) if tmin is None else float(tmin)
    hi = float(np.max(f)) if tmax is None else float(tmax)
    if hi - lo < 1e-6:
        return np.zeros_like(f)
    norm = (f - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def apply_colormap(norm: np.ndarray, colormap: str = "ironbow") -> np.ndarray:
    """Map a normalised ``[0, 1]`` array to an RGB uint8 image via a LUT."""
    lut = get_lut(colormap)
    idx = np.clip(np.round(norm * 255.0), 0, 255).astype(np.intp)
    return lut[idx]


def frame_to_rgb(
    frame: np.ndarray,
    factor: int = 20,
    colormap: str = "ironbow",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    use_scipy: bool = False,
) -> np.ndarray:
    """Full display pipeline: upscale -> normalise -> colormap -> RGB uint8.

    Returns an ``(H, W, 3)`` uint8 image.
    """
    up = bilinear_upscale(frame, factor=factor, use_scipy=use_scipy)
    norm = normalize_temperature(up, tmin=tmin, tmax=tmax)
    return apply_colormap(norm, colormap=colormap)
