"""Tests for upscaling, normalisation, and colormap LUTs."""

from __future__ import annotations

import numpy as np

from thermalsentry.processing.preprocess import (
    apply_colormap,
    bilinear_upscale,
    frame_to_rgb,
    get_lut,
    normalize_temperature,
)


def test_upscale_shape():
    frame = np.zeros((24, 32), dtype=np.float32)
    up = bilinear_upscale(frame, factor=10)
    assert up.shape == (240, 320)


def test_upscale_preserves_range():
    frame = np.linspace(20, 40, 24 * 32).reshape(24, 32).astype(np.float32)
    up = bilinear_upscale(frame, factor=8)
    assert up.min() >= frame.min() - 1e-3
    assert up.max() <= frame.max() + 1e-3


def test_upscale_factor_one_is_identity():
    frame = np.random.default_rng(0).normal(25, 2, (24, 32)).astype(np.float32)
    up = bilinear_upscale(frame, factor=1)
    np.testing.assert_allclose(up, frame, rtol=1e-5)


def test_normalize_range():
    frame = np.array([[18.0, 40.0], [29.0, 22.0]], dtype=np.float32)
    norm = normalize_temperature(frame, tmin=18.0, tmax=40.0)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert np.all((norm >= 0) & (norm <= 1))


def test_normalize_constant_frame():
    frame = np.full((4, 4), 25.0, dtype=np.float32)
    norm = normalize_temperature(frame)
    assert np.all(norm == 0.0)


def test_lut_shape_and_endpoints():
    for name in ("ironbow", "inferno", "grayscale"):
        lut = get_lut(name)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8
    # Ironbow/inferno start dark and end bright.
    iron = get_lut("ironbow")
    assert iron[0].sum() < 60
    assert iron[-1].sum() > 600


def test_colormap_monotonic_brightness():
    # Brightness (sum of channels) should be non-decreasing across the LUT for
    # ironbow / inferno (perceptually hot = bright).
    for name in ("ironbow", "inferno", "grayscale"):
        lut = get_lut(name).astype(int)
        bright = lut.sum(axis=1)
        # Allow tiny dips from rounding but require a strong overall increase.
        assert bright[-1] > bright[0]
        assert np.mean(np.diff(bright) >= -2) > 0.95


def test_apply_colormap_shape():
    norm = np.linspace(0, 1, 24 * 32).reshape(24, 32)
    rgb = apply_colormap(norm, "ironbow")
    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8


def test_frame_to_rgb_pipeline():
    frame = np.linspace(18, 40, 24 * 32).reshape(24, 32).astype(np.float32)
    rgb = frame_to_rgb(frame, factor=10, colormap="ironbow", tmin=18, tmax=40)
    assert rgb.shape == (240, 320, 3)
    # Hottest pixel should be brighter than coldest.
    assert rgb.reshape(-1, 3)[-1].sum() > rgb.reshape(-1, 3)[0].sum()
