"""Shared test fixtures and synthetic-frame helpers."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from thermalsentry import FRAME_COLS, FRAME_ROWS


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    """Guarantee a current event loop on the main thread for each test.

    On Python 3.9 ``asyncio.run`` closes its loop and leaves the thread with no
    current loop, so any later code that constructs an ``asyncio`` primitive
    (e.g. ``AsyncRuntime`` builds an ``asyncio.Event`` in ``__init__``) outside a
    running loop raises ``RuntimeError``. We install a fresh loop before each test
    and tidy it up afterwards. Tests that call ``asyncio.run`` still work because
    ``asyncio.run`` manages its own loop and restores ours via this fixture.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield
    # Re-establish a usable loop if the test (e.g. via asyncio.run) closed it.
    try:
        current = asyncio.get_event_loop()
        if current.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


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
