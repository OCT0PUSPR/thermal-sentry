"""Tests for the threaded ThermalSentryApp pipeline runner."""

from __future__ import annotations

import time

import numpy as np

from thermalsentry.app import ThermalSentryApp
from thermalsentry.config import SourceType, get_settings


def _settings(**kw):
    base = dict(source=SourceType.SIMULATE, sim_num_bodies=2, sim_seed=42, upscale=8, fps=50.0)
    base.update(kw)
    return get_settings(**base)


def test_process_once_returns_state():
    app = ThermalSentryApp(settings=_settings())
    st = app.process_once(now=1000.0)
    assert st.display_w == 32 * 8
    assert st.display_h == 24 * 8
    assert st.thermal_rgb_base64.startswith("data:image/png;base64,")
    assert st.stats["frames_processed"] == 1
    app.stop()


def test_subscribe_callback_fires():
    app = ThermalSentryApp(settings=_settings())
    payloads = []
    app.subscribe(payloads.append)
    app.process_once(now=1000.0)
    assert len(payloads) == 1
    assert "stats" in payloads[0]
    app.stop()


def test_subscriber_exception_is_isolated():
    app = ThermalSentryApp(settings=_settings())

    def boom(_payload):
        raise RuntimeError("subscriber failure")

    app.subscribe(boom)
    # Must not propagate out of process_once.
    st = app.process_once(now=1000.0)
    assert st.frame_index == 1
    app.stop()


def test_get_state_and_stats():
    app = ThermalSentryApp(settings=_settings())
    app.process_once(now=1000.0)
    state = app.get_state()
    stats = app.get_stats()
    assert state.frame_index == 1
    assert stats["frames_processed"] == 1
    app.stop()


def test_set_zones_does_not_raise():
    app = ThermalSentryApp(settings=_settings())
    app.set_zones([[(0.1, 0.1), (0.4, 0.1), (0.4, 0.5)]])
    app.process_once(now=1000.0)
    app.stop()


def test_threaded_start_stop_processes_frames():
    app = ThermalSentryApp(settings=_settings())
    app.start()
    # Calling start twice is a no-op (already running).
    app.start()
    deadline = time.monotonic() + 1.0
    while app.get_stats().get("frames_processed", 0) < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    app.stop()
    assert app.get_stats()["frames_processed"] >= 1


def test_record_writes_npy(tmp_path):
    rec = tmp_path / "rec.npy"
    app = ThermalSentryApp(settings=_settings(), record_path=str(rec))
    for i in range(3):
        app.process_once(now=1000.0 + i)
    app.stop()
    arr = np.load(rec)
    assert arr.shape == (3, 24, 32)


def test_stop_without_start_is_safe():
    app = ThermalSentryApp(settings=_settings())
    # No frames, no thread -> stop just tears down cleanly.
    app.stop()
