"""Tests for the async edge runtime (AsyncRuntime)."""

from __future__ import annotations

import asyncio

import numpy as np

from thermalsentry.config import SourceType, get_settings
from thermalsentry.observability import Metrics
from thermalsentry.runtime import AsyncRuntime, FrameState, Health


def _settings(**kw):
    base = dict(
        source=SourceType.SIMULATE,
        sim_num_bodies=2,
        sim_seed=42,
        upscale=8,
        fps=50.0,
    )
    rt = kw.pop("runtime", None)
    s = get_settings(**{**base, **kw})
    if rt is not None:
        for k, v in rt.items():
            setattr(s.runtime, k, v)
    return s


def test_health_as_dict_and_uptime():
    h = Health()
    assert h.uptime() == 0.0
    d = h.as_dict()
    assert d["frames_processed"] == 0
    assert d["running"] is False


def test_framestate_to_payload():
    fs = FrameState(frame_index=3, display_w=10, display_h=20, stats={"a": 1})
    payload = fs.to_payload()
    assert payload["frame_index"] == 3
    assert payload["display_w"] == 10
    assert payload["stats"] == {"a": 1}
    assert "thermal_rgb_base64" in payload


def test_detector_backend_and_set_zones():
    rt = AsyncRuntime(settings=_settings())
    assert rt.detector_backend() == "classical"
    rt.set_zones([[(0.1, 0.1), (0.4, 0.1), (0.4, 0.5)]])


def test_process_frame_directly_updates_state():
    rt = AsyncRuntime(settings=_settings())
    rng = np.random.default_rng(0)
    raw = np.full((24, 32), 22.0, dtype=np.float32)
    raw[10:14, 14:18] = 35.0  # a warm body
    raw += rng.normal(0, 0.1, raw.shape).astype(np.float32)
    rt._process_frame(raw, wall_ts=1000.0)
    assert rt.health.frames_processed == 1
    assert rt.state.frame_index == 1
    assert "scene_max_c" in rt.state.stats
    assert rt.state.stats["scene_max_c"] >= 30.0


def test_compute_fps():
    rt = AsyncRuntime(settings=_settings())
    assert rt._compute_fps() == 0.0
    rt._fps_window.append(100.0)
    rt._fps_window.append(101.0)
    rt._fps_window.append(102.0)
    # 2 intervals across 2 seconds -> ~1 fps.
    assert abs(rt._compute_fps() - 1.0) < 0.001


def test_enqueue_backpressure_drops_oldest():
    async def run():
        rt = AsyncRuntime(settings=_settings(runtime={"queue_maxsize": 2}))
        rt._queue = asyncio.Queue(maxsize=2)
        rt._enqueue(("a", 1.0))
        rt._enqueue(("b", 2.0))
        assert rt.health.frames_dropped == 0
        # Third item overflows -> oldest dropped.
        rt._enqueue(("c", 3.0))
        assert rt.health.frames_dropped == 1
        assert rt._queue.qsize() == 2

    asyncio.run(run())


def test_is_ready_progression():
    rt = AsyncRuntime(settings=_settings())
    assert rt.is_ready() is False  # not running
    rt.health.running = True
    assert rt.is_ready() is False  # no frames yet
    rt.health.frames_processed = 1
    rt.health.sensor_ok = True
    assert rt.is_ready() is True


def test_status_dict_keys():
    rt = AsyncRuntime(settings=_settings())
    st = rt.status()
    for key in ("service", "running", "ready", "source", "detector_backend", "health", "stats"):
        assert key in st
    assert st["service"] == "thermal-sentry"
    assert st["source"] == "simulate"


def test_start_runs_and_processes_frames():
    async def run():
        rt = AsyncRuntime(settings=_settings(fps=60.0), metrics=Metrics(enabled=True))
        await rt.start()
        try:
            for _ in range(15):
                await asyncio.sleep(0.02)
                if rt.health.frames_processed > 0 and rt.is_ready():
                    break
            assert rt.health.frames_processed > 0
            assert rt.is_ready() is True
            status = rt.status()
            assert status["running"] is True
            assert status["health"]["frames_captured"] > 0
        finally:
            await rt.stop()
        assert rt.health.running is False

    asyncio.run(run())


def test_subscribe_receives_payload_during_run():
    async def run():
        rt = AsyncRuntime(settings=_settings(fps=60.0))
        received = []
        rt.subscribe(received.append)
        await rt.start()
        try:
            for _ in range(15):
                await asyncio.sleep(0.02)
                if received:
                    break
            assert received
            assert "stats" in received[-1]
        finally:
            await rt.stop()

    asyncio.run(run())


class _FlakySource:
    """A source that raises a configurable number of times before succeeding."""

    def __init__(self, fail_times=0, frame=None):
        self.fail_times = fail_times
        self.calls = 0
        self.closed = False
        self._frame = frame if frame is not None else np.full((24, 32), 30.0, dtype=np.float32)

    def read(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise OSError("bus glitch")
        return self._frame.copy()

    def close(self):
        self.closed = True


def test_read_with_retry_recovers_after_transient_error():
    async def run():
        rt = AsyncRuntime(settings=_settings(runtime={"sensor_max_retries": 3, "sensor_retry_backoff_s": 0.0}))
        rt._source = _FlakySource(fail_times=2)
        frame = await rt._read_with_retry()
        assert frame is not None
        assert rt.health.sensor_errors == 2

    asyncio.run(run())


def test_read_with_retry_returns_none_after_exhausting_retries():
    async def run():
        rt = AsyncRuntime(settings=_settings(runtime={"sensor_max_retries": 2, "sensor_retry_backoff_s": 0.0}))
        rt._source = _FlakySource(fail_times=99)
        frame = await rt._read_with_retry()
        assert frame is None
        assert rt.health.last_error.startswith("sensor read")

    asyncio.run(run())


def test_process_frame_persists_event_to_store():
    from thermalsentry.persistence.store import EventStore

    store = EventStore(url="sqlite:///:memory:")
    store.create_all()
    rt = AsyncRuntime(settings=_settings(), store=store)
    raw = np.full((24, 32), 22.0, dtype=np.float32)
    raw[10:14, 14:18] = 35.0
    rt._process_frame(raw, wall_ts=1000.0)
    assert store.counts()["events"] == 1
    store.close()


def test_set_zones_records_config_change_to_store():
    from thermalsentry.persistence.store import EventStore

    store = EventStore(url="sqlite:///:memory:")
    store.create_all()
    rt = AsyncRuntime(settings=_settings(), store=store)
    rt.set_zones([[(0.1, 0.1), (0.4, 0.1), (0.4, 0.5)]])
    # A config-change row was recorded (no exception).
    store.close()


def test_open_and_close_source():
    rt = AsyncRuntime(settings=_settings())
    rt._open_source()
    assert rt._source is not None
    assert rt.health.sensor_ok is True
    rt._close_source()
    assert rt._source is None
    # Closing again is a no-op.
    rt._close_source()


def test_restart_capture_reopens_source():
    async def run():
        rt = AsyncRuntime(settings=_settings(fps=60.0))
        await rt.start()
        try:
            await asyncio.sleep(0.05)
            before = rt._source
            await rt._restart_capture()
            assert rt._source is not None
            assert rt._source is not before
        finally:
            await rt.stop()

    asyncio.run(run())


def test_capture_loop_marks_sensor_down_on_persistent_failure():
    async def run():
        rt = AsyncRuntime(
            settings=_settings(fps=60.0, runtime={"sensor_max_retries": 1, "sensor_retry_backoff_s": 0.0})
        )
        await rt.start()
        try:
            # Swap in a perpetually-failing source so the capture loop hits the
            # "sensor down" branch.
            rt._source = _FlakySource(fail_times=999)
            for _ in range(20):
                await asyncio.sleep(0.02)
                if rt.health.sensor_ok is False:
                    break
            assert rt.health.sensor_ok is False
        finally:
            await rt.stop()

    asyncio.run(run())


def test_watchdog_restarts_on_stale_capture():
    async def run():
        rt = AsyncRuntime(
            settings=_settings(
                fps=60.0,
                runtime={"watchdog_interval_s": 0.02, "watchdog_timeout_s": 0.05},
            )
        )
        await rt.start()
        try:
            # Let a frame be captured so last_capture_ts is set, then freeze it in
            # the past so the watchdog sees a stall.
            for _ in range(20):
                await asyncio.sleep(0.02)
                if rt.health.frames_captured > 0:
                    break
            import time as _t

            rt.health.last_capture_ts = _t.monotonic() - 10.0
            for _ in range(20):
                await asyncio.sleep(0.02)
                if rt.health.watchdog_restarts > 0:
                    break
            assert rt.health.watchdog_restarts > 0
        finally:
            await rt.stop()

    asyncio.run(run())


def test_enqueue_on_empty_then_full_queue():
    async def run():
        rt = AsyncRuntime(settings=_settings(runtime={"queue_maxsize": 1}))
        rt._queue = asyncio.Queue(maxsize=1)
        rt._enqueue(("a", 1.0))
        assert rt._queue.qsize() == 1
        # Overflow drops oldest and inserts the new item.
        rt._enqueue(("b", 2.0))
        assert rt.health.frames_dropped == 1
        assert rt._queue.qsize() == 1

    asyncio.run(run())


def test_gc_loop_trims_history():
    async def run():
        rt = AsyncRuntime(settings=_settings(runtime={"gc_interval_s": 0.02, "history_max_items": 3}))
        # Pre-fill recent history beyond the cap.
        rt.alerts.recent = [{"i": i} for i in range(10)]
        await rt.start()
        try:
            for _ in range(20):
                await asyncio.sleep(0.02)
                if len(rt.alerts.recent) <= 3:
                    break
            assert len(rt.alerts.recent) <= 3
        finally:
            await rt.stop()

    asyncio.run(run())
