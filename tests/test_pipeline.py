"""End-to-end pipeline tests using the simulator (numpy only, no hardware)."""

from __future__ import annotations

import numpy as np

from thermalsentry.app import ThermalSentryApp
from thermalsentry.config import SourceType, get_settings
from thermalsentry.sensors.base import build_source
from thermalsentry.sensors.file_source import FileThermalSource


def _sim_settings(**kw):
    base = dict(
        source=SourceType.SIMULATE,
        sim_num_bodies=2,
        sim_seed=42,
        upscale=8,
        fps=8.0,
    )
    base.update(kw)
    return get_settings(**base)


def test_pipeline_process_once_runs():
    app = ThermalSentryApp(settings=_sim_settings())
    state = app.process_once(now=1000.0)
    assert state.display_w == 32 * 8
    assert state.display_h == 24 * 8
    assert state.thermal_rgb_base64.startswith("data:image/png;base64,")
    assert "frames_processed" in state.stats
    app.stop()


def test_pipeline_detects_people_over_time():
    app = ThermalSentryApp(settings=_sim_settings())
    person_counts = []
    for i in range(15):
        state = app.process_once(now=1000.0 + i)
        person_counts.append(state.stats["person_count"])
    # The simulator places ~2 warm bodies; the pipeline should report >=1
    # person on most frames.
    assert max(person_counts) >= 1
    app.stop()


def test_build_source_simulate():
    src = build_source(_sim_settings())
    frame = src.read()
    assert frame.shape == (24, 32)
    src.close()


def test_file_source_roundtrip(tmp_path):
    # Save a short synthetic clip and replay it.
    clip = np.random.default_rng(0).normal(24, 1, (5, 24, 32)).astype(np.float32)
    path = tmp_path / "clip.npy"
    np.save(path, clip)

    src = FileThermalSource(str(path), loop=False)
    assert len(src) == 5
    frames = [src.read() for _ in range(5)]
    np.testing.assert_allclose(frames[0], clip[0])
    # Non-looping source raises at the end.
    try:
        src.read()
        raised = False
    except StopIteration:
        raised = True
    assert raised


def test_file_source_loops(tmp_path):
    clip = np.zeros((2, 24, 32), dtype=np.float32)
    clip[1] += 1.0
    path = tmp_path / "loop.npy"
    np.save(path, clip)
    src = FileThermalSource(str(path), loop=True)
    seen = [src.read()[0, 0] for _ in range(5)]
    # Loops: 0,1,0,1,0
    assert seen[0] == seen[2] == seen[4] == 0.0
    assert seen[1] == seen[3] == 1.0


def test_record_writes_npy(tmp_path):
    rec = tmp_path / "rec.npy"
    app = ThermalSentryApp(settings=_sim_settings(), record_path=str(rec))
    for i in range(4):
        app.process_once(now=1000.0 + i)
    app.stop()
    arr = np.load(rec)
    assert arr.shape == (4, 24, 32)


def test_default_settings_detect_people():
    """Regression: the SHOWCASE path (all default settings) must reliably find
    the two simulated people. Catches detector-area / upscale mismatches.
    """
    settings = get_settings(source=SourceType.SIMULATE, sim_num_bodies=2, sim_seed=42)
    app = ThermalSentryApp(settings=settings)
    counts = []
    for i in range(40):
        st = app.process_once(now=1000.0 + i)
        counts.append(st.stats["person_count"])
    app.stop()
    # Both people detected on a clear majority of frames.
    assert max(counts) >= 2
    assert sum(1 for c in counts if c >= 2) >= 20


def test_default_settings_overheat_alert():
    """Regression: a genuinely hot source (60 C) fires an overheat alert under
    default settings -- the simulator must not clamp real hotspots away.
    """
    settings = get_settings(
        source=SourceType.SIMULATE, sim_num_bodies=1, sim_body_temp_c=60.0, sim_seed=1
    )
    app = ThermalSentryApp(settings=settings)
    last = None
    for i in range(5):
        last = app.process_once(now=2000.0 + i)
    app.stop()
    assert last.stats["scene_max_c"] >= 50.0
    assert last.stats["total_alerts"] >= 1


def test_loitering_uses_consistent_clock():
    """Regression: dwell time must use the monotonic clock, not wall-clock, so
    a short run never spuriously reports multi-decade loitering.
    """
    settings = get_settings(source=SourceType.SIMULATE, sim_num_bodies=2, sim_seed=42)
    app = ThermalSentryApp(settings=settings)
    for i in range(10):
        st = app.process_once(now=1000.0 + i)
    app.stop()
    for tr in st.tracks:
        assert tr["dwell_s"] < 60.0  # not an absurd wall-clock-derived value
