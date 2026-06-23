"""Tests for the synthetic thermal source."""

from __future__ import annotations

import numpy as np

from thermalsentry import FRAME_COLS, FRAME_ROWS
from thermalsentry.sensors.simulator import SyntheticThermalSource


def test_frame_shape_and_dtype():
    src = SyntheticThermalSource(num_bodies=2, seed=1)
    frame = src.read()
    assert frame.shape == (FRAME_ROWS, FRAME_COLS)
    assert frame.dtype == np.float32


def test_temperature_ranges_are_physical():
    src = SyntheticThermalSource(num_bodies=3, ambient_c=22.0, body_temp_c=34.0, seed=7)
    mins, maxs = [], []
    for _ in range(30):
        f = src.read()
        mins.append(float(f.min()))
        maxs.append(float(f.max()))
    # Background near ambient; warm bodies clearly above it but not absurd.
    assert min(mins) > 16.0
    assert min(mins) < 26.0
    assert max(maxs) > 28.0
    assert max(maxs) < 45.0


def test_determinism_same_seed():
    a = SyntheticThermalSource(num_bodies=2, seed=123)
    b = SyntheticThermalSource(num_bodies=2, seed=123)
    for _ in range(10):
        np.testing.assert_allclose(a.read(), b.read())


def test_different_seeds_differ():
    a = SyntheticThermalSource(num_bodies=2, seed=1)
    b = SyntheticThermalSource(num_bodies=2, seed=2)
    fa, fb = a.read(), b.read()
    assert not np.allclose(fa, fb)


def test_bodies_move():
    src = SyntheticThermalSource(num_bodies=1, seed=5)
    start = src.body_positions()[0]
    for _ in range(20):
        src.read()
    end = src.body_positions()[0]
    assert (abs(start[0] - end[0]) + abs(start[1] - end[1])) > 0.5


def test_no_noise_is_smooth():
    src = SyntheticThermalSource(num_bodies=1, noise_std=0.0, seed=3)
    f1 = src.read()
    # With zero noise the background pixels equal ambient + gradient exactly,
    # so a corner far from the body should be very close to ambient.
    corner = f1[0, 0]
    assert 19.0 < corner < 25.0
