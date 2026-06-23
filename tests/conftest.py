"""Shared test fixtures and synthetic-frame helpers."""

from __future__ import annotations

import numpy as np
import pytest

from thermalsentry import FRAME_COLS, FRAME_ROWS


def make_frame(bodies, ambient=22.0, sigma=2.4, noise=0.0, seed=0):
    """Build a synthetic 24x32 frame with gaussian warm bodies.

    ``bodies`` is a list of (row, col, peak_c) tuples.
    """
    rng = np.random.default_rng(seed)
    frame = np.full((FRAME_ROWS, FRAME_COLS), ambient, dtype=np.float64)
    rows = np.arange(FRAME_ROWS)[:, None]
    cols = np.arange(FRAME_COLS)[None, :]
    for (r, c, peak) in bodies:
        dr = (rows - r) / sigma
        dc = (cols - c) / sigma
        frame += (peak - ambient) * np.exp(-0.5 * (dr * dr + dc * dc))
    if noise > 0:
        frame += rng.normal(0.0, noise, size=frame.shape)
    return frame.astype(np.float32)


@pytest.fixture
def two_body_frame():
    return make_frame([(6, 8, 34.0), (16, 22, 35.0)])
