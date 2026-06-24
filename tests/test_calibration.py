"""Tests for thermal sensor calibration (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from thermalsentry import FRAME_COLS, FRAME_ROWS
from thermalsentry.sensors.calibration import (
    Calibration,
    apply_calibration,
    calibrate_offset,
)


def _uniform(value: float) -> np.ndarray:
    return np.full((FRAME_ROWS, FRAME_COLS), value, dtype=np.float32)


def test_apply_scalar_offset_only():
    cal = Calibration(offset=2.0, emissivity=1.0)
    frame = _uniform(20.0)
    out = cal.apply(frame)
    np.testing.assert_allclose(out, 22.0)
    assert out.dtype == np.float32


def test_apply_emissivity_rescale():
    cal = Calibration(offset=0.0, emissivity=0.5, reference_ambient_c=20.0)
    frame = _uniform(30.0)
    out = cal.apply(frame)
    # 20 + (30 - 20) / 0.5 = 40
    np.testing.assert_allclose(out, 40.0)


def test_apply_offset_map():
    omap = np.ones((FRAME_ROWS, FRAME_COLS), dtype=np.float32) * 3.0
    cal = Calibration(offset=1.0, offset_map=omap.tolist(), emissivity=1.0)
    out = cal.apply(_uniform(10.0))
    # 10 + 3 (map) + 1 (scalar) = 14
    np.testing.assert_allclose(out, 14.0)


def test_apply_offset_map_wrong_shape_is_ignored():
    cal = Calibration(offset=1.0, offset_map=[[0.0, 0.0]], emissivity=1.0)
    out = cal.apply(_uniform(10.0))
    # Mismatched map is skipped; only the scalar offset applies.
    np.testing.assert_allclose(out, 11.0)


def test_to_dict_save_load_roundtrip(tmp_path):
    cal = Calibration(offset=1.5, emissivity=0.9, reference_ambient_c=21.0)
    d = cal.to_dict()
    assert d["offset"] == 1.5
    assert d["emissivity"] == 0.9
    path = tmp_path / "sub" / "cal.json"
    cal.save(str(path))
    assert path.exists()
    loaded = Calibration.load(str(path))
    assert loaded.offset == cal.offset
    assert loaded.emissivity == cal.emissivity
    assert loaded.reference_ambient_c == cal.reference_ambient_c


def test_calibrate_offset_per_pixel():
    frames = np.stack([_uniform(18.0), _uniform(20.0)], axis=0)  # mean 19
    cal = calibrate_offset(frames, known_temp_c=25.0, per_pixel=True)
    assert cal.offset_map is not None
    omap = np.asarray(cal.offset_map)
    np.testing.assert_allclose(omap, 6.0)  # 25 - 19


def test_calibrate_offset_scalar():
    frames = np.stack([_uniform(18.0), _uniform(22.0)], axis=0)  # mean 20
    cal = calibrate_offset(frames, known_temp_c=25.0, per_pixel=False)
    assert cal.offset_map is None
    assert cal.offset == pytest.approx(5.0)


def test_calibrate_offset_accepts_single_frame():
    cal = calibrate_offset(_uniform(20.0), known_temp_c=21.0, per_pixel=False)
    assert cal.offset == pytest.approx(1.0)


def test_calibrate_offset_shape_validation():
    bad = np.zeros((2, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        calibrate_offset(bad, known_temp_c=20.0)


def test_apply_calibration_passthrough_when_none():
    frame = _uniform(20.0)
    out = apply_calibration(frame, None)
    assert out is frame


def test_apply_calibration_applies_when_present():
    cal = Calibration(offset=5.0, emissivity=1.0)
    out = apply_calibration(_uniform(10.0), cal)
    np.testing.assert_allclose(out, 15.0)
