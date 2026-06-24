"""Tests for the optional Pi Camera fusion source (laptop / non-Pi path)."""

from __future__ import annotations

import numpy as np
import pytest

from thermalsentry.sensors.camera import PiCameraSource, overlay_thermal_on_visible


def test_overlay_blends_thermal_over_visible():
    visible = np.zeros((8, 8, 3), dtype=np.uint8)
    thermal = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = overlay_thermal_on_visible(visible, thermal, alpha=0.5)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    # Blend of black (0) and 200 at alpha 0.5 -> ~100.
    assert out.max() <= 200
    assert out.mean() > 0


def test_overlay_alpha_zero_returns_visible():
    visible = np.full((6, 6, 3), 120, dtype=np.uint8)
    thermal = np.zeros((3, 3, 3), dtype=np.uint8)
    out = overlay_thermal_on_visible(visible, thermal, alpha=0.0)
    np.testing.assert_array_equal(out, visible)


def test_pi_camera_source_raises_off_pi():
    # picamera2 is unavailable on this machine -> a clear RuntimeError.
    with pytest.raises(RuntimeError, match="picamera2"):
        PiCameraSource()
