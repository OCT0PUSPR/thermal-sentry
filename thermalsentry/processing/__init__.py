"""Frame preprocessing: upscaling, normalisation, and colormapping."""

from __future__ import annotations

from .preprocess import (
    apply_colormap,
    bilinear_upscale,
    frame_to_rgb,
    normalize_temperature,
)

__all__ = [
    "apply_colormap",
    "bilinear_upscale",
    "frame_to_rgb",
    "normalize_temperature",
]
