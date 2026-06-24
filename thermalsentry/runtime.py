"""Async edge runtime with backpressure, watchdog, and recovery.

Architecture (two cooperating asyncio tasks + helpers):

  capture_task:  reads frames from the sensor (retries on error), pushes onto a
                 BOUNDED queue. If the queue is full it drops the OLDEST frame
                 (backpressure) so the processor never falls irrecoverably behind.

  process_task:  pulls frames, runs preprocess -> detect -> track -> rules ->
                 alerts -> publish, updates health + metrics.

  watchdog:      if no frame has been captured within ``watchdog_timeout_s`` the
                 capture loop is restarted (sensor recovery). Restarts are counted.

  gc loop:       periodically trims in-memory history + runs gc to keep memory
                 bounded over multi-day runs.

The runtime owns sensor lifecycle and exposes a health snapshot for /health,
/ready and /status. It degrades to the classical detector if the ML model is
absent and never crashes the process on a transient sensor fault.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Deque, List, Optional

import numpy as np

from .alerts.manager import AlertManager
from .config import Settings, get_settings
from .detection.anomaly import AnomalyEngine
from .detection.detector import ThermalDetector
from .detection.tracker import CentroidTracker
from .ml.backends import build_backend, build_detector_backend
from .observability import get_logger, get_metrics
from .processing.preprocess import apply_colormap, bilinear_upscale, normalize_temperature
from .sensors.base import build_source

logger = get_logger("thermalsentry.runtime")


def _rgb_to_png_base64(rgb: np.ndarray) -> str:
    try:
        from PIL import Image

        img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # pragma: no cover
        return ""


@dataclass
class Health:
    """Live health snapshot used by /health, /ready, /status."""

    started_at: float = 0.0
    last_capture_ts: float = 0.0
    last_process_ts: float = 0.0
    frames_captured: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    sensor_errors: int = 0
    watchdog_restarts: int = 0
    queue_depth: int = 0
    running: bool = False
    sensor_ok: bool = False
    last_error: str = ""

    def uptime(self) -> float:
        return max(0.0, time.monotonic() - self.started_at) if self.started_at else 0.0

    def as_dict(self) -> dict:
        return {
            "running": self.running,
            "sensor_ok": self.sensor_ok,
            "uptime_s": round(self.uptime(), 1),
            "frames_captured": self.frames_captured,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "sensor_errors": self.sensor_errors,
            "watchdog_restarts": self.watchdog_restarts,
            "queue_depth": self.queue_depth,
            "last_error": self.last_error,
        }


@dataclass
class FrameState:
    """Latest pipeline output snapshot (read by the web layer)."""

    frame_index: int = 0
    timestamp: float = 0.0
    thermal_rgb_base64: str = ""
    display_w: int = 0
    display_h: int = 0
    tracks: List[dict] = field(default_factory=list)
    detections: List[dict] = field(default_factory=list)
    alerts: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "thermal_rgb_base64": self.thermal_rgb_base64,
            "display_w": self.display_w,
            "display_h": self.display_h,
            "detections": self.detections,
            "tracks": self.tracks,
            "alerts": self.alerts,
            "stats": self.stats,
        }


class AsyncRuntime:
    """The async edge pipeline runtime."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store=None,
        metrics=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.metrics = metrics or get_metrics(self.settings.observability.metrics_enabled)

        # Sensor + frame queue are created lazily on the running loop in start();
        # typed as Any so the hot-path accesses (guaranteed set before the loops
        # run) don't trip the optional-attr checker.
        self._source: Any = None
        self.classifier = build_backend(self.settings.ml)
        self.ml_detector = build_detector_backend(self.settings.ml)
        self.detector = ThermalDetector(
            settings=self.settings.detection,
            classifier=self.classifier,
            detector_backend=self.ml_detector,
        )
        self.tracker = CentroidTracker(settings=self.settings.tracker)
        self.anomaly = AnomalyEngine(settings=self.settings.anomaly)
        self.alerts = AlertManager(
            settings=self.settings.alerts, store=store, metrics=self.metrics
        )

        self.health = Health()
        self.state = FrameState()
        # The queue is created in start() on the running loop. Binding it here
        # (when there may be no loop, e.g. under a test portal) would bind it to
        # the wrong loop and the consumer would never wake.
        self._queue: Any = None  # asyncio.Queue[tuple], created in start()
        self._subscribers: List[Callable[[dict], None]] = []
        self._tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()
        # Dedicated executor so blocking sensor reads + processing never starve
        # each other (and don't depend on the host loop's default executor,
        # which is single-capacity under anyio test portals).
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ts-pipe")
        self._fps_window: Deque[float] = deque(maxlen=self.settings.runtime.history_max_items)
        self._total_alerts = 0

    # -- subscriptions / control ---------------------------------------------

    def subscribe(self, cb: Callable[[dict], None]) -> None:
        self._subscribers.append(cb)

    def set_zones(self, zones) -> None:
        self.anomaly.set_zones(zones)
        if self.store is not None:
            try:
                self.store.record_config_change("api", "zones", {"zones": zones})
            except Exception:
                pass

    def detector_backend(self) -> str:
        if self.ml_detector is not None and getattr(self.ml_detector, "available", lambda: False)():
            return "ml"
        return getattr(self.classifier, "name", "classical")

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        self._stop.clear()
        # Create the bounded queue on the running loop so the consumer wakes.
        self._queue = asyncio.Queue(maxsize=self.settings.runtime.queue_maxsize)
        self.health.started_at = time.monotonic()
        self.health.running = True
        self._open_source()
        self._tasks = [
            asyncio.create_task(self._capture_loop(), name="capture"),
            asyncio.create_task(self._process_loop(), name="process"),
            asyncio.create_task(self._watchdog_loop(), name="watchdog"),
            asyncio.create_task(self._gc_loop(), name="gc"),
        ]
        logger.info("runtime_started", source=self.settings.source.value, backend=self.detector_backend())

    async def stop(self) -> None:
        self._stop.set()
        self.health.running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        self._close_source()
        self.alerts.close()
        self._executor.shutdown(wait=False)
        logger.info("runtime_stopped")

    def _open_source(self) -> None:
        self._source = build_source(self.settings)
        self.health.sensor_ok = True

    def _close_source(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                pass
            self._source = None

    # -- capture --------------------------------------------------------------

    async def _capture_loop(self) -> None:
        period = 1.0 / max(0.1, self.settings.fps)
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            t0 = loop.time()
            frame = await self._read_with_retry()
            if frame is None:
                # Persistent failure: mark sensor down; watchdog will recover.
                self.health.sensor_ok = False
                await asyncio.sleep(min(1.0, period))
                continue
            self.health.sensor_ok = True
            self.health.frames_captured += 1
            self.health.last_capture_ts = time.monotonic()
            self._enqueue((frame, time.time()))
            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _read_with_retry(self) -> Optional[np.ndarray]:
        rt = self.settings.runtime
        for attempt in range(1, rt.sensor_max_retries + 1):
            try:
                # source.read may block briefly; run in executor to stay async.
                return await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._source.read
                )
            except StopIteration:
                raise
            except Exception as exc:
                self.health.sensor_errors += 1
                self.health.last_error = f"sensor read: {exc}"
                if self.metrics is not None:
                    try:
                        self.metrics.sensor_errors_total.inc()
                    except Exception:
                        pass
                if attempt < rt.sensor_max_retries:
                    await asyncio.sleep(rt.sensor_retry_backoff_s * attempt)
        return None

    def _enqueue(self, item: tuple) -> None:
        """Non-blocking enqueue; drop the OLDEST frame on backpressure."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # drop oldest
                self.health.frames_dropped += 1
                if self.metrics is not None:
                    try:
                        self.metrics.frames_dropped_total.inc()
                    except Exception:
                        pass
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
        self.health.queue_depth = self._queue.qsize()
        if self.metrics is not None:
            try:
                self.metrics.queue_depth.set(self._queue.qsize())
            except Exception:
                pass

    # -- process --------------------------------------------------------------

    async def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            frame, wall_ts = item
            try:
                # Processing is fast pure-numpy (sub-ms..few-ms at typical upscale)
                # so we run it inline on the loop. This keeps frame ordering, avoids
                # executor starvation, and yields between frames via the queue await.
                self._process_frame(frame, wall_ts)
            except Exception as exc:  # pragma: no cover - defensive
                self.health.last_error = f"process: {exc}"
                logger.warning("process_error", error=str(exc))
            self.health.queue_depth = self._queue.qsize()
            # Cooperative yield so other tasks (ws broadcast, watchdog) run.
            await asyncio.sleep(0)

    def _process_frame(self, raw: np.ndarray, wall_ts: float) -> None:
        t0 = time.perf_counter()
        up = bilinear_upscale(raw, factor=self.settings.upscale)
        disp_h, disp_w = up.shape

        detections = self.detector.detect(up)
        mono = time.monotonic()
        tracks = self.tracker.update(detections, now=mono)

        raised = self.anomaly.evaluate(
            up, detections, tracks, disp_w, disp_h, now=wall_ts, mono_now=mono
        )
        dispatched = self.alerts.dispatch(raised, now=wall_ts)
        self._total_alerts += len(dispatched)

        norm = normalize_temperature(
            up, tmin=self.settings.temp_display_min_c, tmax=self.settings.temp_display_max_c
        )
        rgb = apply_colormap(norm, colormap=self.settings.colormap)
        rgb_b64 = _rgb_to_png_base64(rgb)

        self.health.frames_processed += 1
        self.health.last_process_ts = time.monotonic()
        self._fps_window.append(time.monotonic())
        fps = self._compute_fps()

        scene_max = float(np.max(raw))
        scene_min = float(np.min(raw))
        person_count = sum(1 for t in tracks if t.label == "person")
        latency = time.perf_counter() - t0

        if self.metrics is not None:
            self.metrics.observe_frame(latency, fps, person_count, scene_max)

        stats = {
            "frames_processed": self.health.frames_processed,
            "frames_dropped": self.health.frames_dropped,
            "fps_actual": round(fps, 2),
            "detect_latency_ms": round(latency * 1000, 2),
            "person_count": person_count,
            "track_count": len(tracks),
            "scene_max_c": round(scene_max, 2),
            "scene_min_c": round(scene_min, 2),
            "total_alerts": self._total_alerts,
            "source": self.settings.source.value,
            "detector_backend": self.detector_backend(),
        }

        self.state = FrameState(
            frame_index=self.health.frames_processed,
            timestamp=wall_ts,
            thermal_rgb_base64=rgb_b64,
            display_w=disp_w,
            display_h=disp_h,
            detections=[d.as_dict(disp_w, disp_h) for d in detections],
            tracks=[t.as_dict(disp_w, disp_h, now=mono) for t in tracks],
            alerts=self.alerts.recent[-20:],
            stats=stats,
        )

        # Persist a sampled event (every frame here; sampling configurable upstream).
        if self.store is not None:
            try:
                self.store.record_event(
                    frame_index=self.health.frames_processed,
                    person_count=person_count,
                    track_count=len(tracks),
                    max_temp_c=scene_max,
                    min_temp_c=scene_min,
                    source=self.settings.source.value,
                    detector_backend=self.detector_backend(),
                )
            except Exception:
                pass

        payload = self.state.to_payload()
        for cb in self._subscribers:
            try:
                cb(payload)
            except Exception:  # pragma: no cover
                pass

    def _compute_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    # -- watchdog -------------------------------------------------------------

    async def _watchdog_loop(self) -> None:
        rt = self.settings.runtime
        while not self._stop.is_set():
            await asyncio.sleep(rt.watchdog_interval_s)
            if self.health.last_capture_ts == 0:
                continue
            stale = time.monotonic() - self.health.last_capture_ts
            if stale > rt.watchdog_timeout_s and self.health.running:
                logger.warning("watchdog_restart", stale_s=round(stale, 1))
                self.health.watchdog_restarts += 1
                if self.metrics is not None:
                    try:
                        self.metrics.watchdog_restarts_total.inc()
                    except Exception:
                        pass
                await self._restart_capture()

    async def _restart_capture(self) -> None:
        """Recreate the sensor + capture task after a stall."""
        # Cancel the existing capture task.
        for t in list(self._tasks):
            if t.get_name() == "capture":
                t.cancel()
                self._tasks.remove(t)
                break
        self._close_source()
        try:
            self._open_source()
        except Exception as exc:
            self.health.last_error = f"sensor reopen: {exc}"
            self.health.sensor_ok = False
        self.health.last_capture_ts = time.monotonic()  # reset stall timer
        self._tasks.append(asyncio.create_task(self._capture_loop(), name="capture"))

    # -- periodic GC ----------------------------------------------------------

    async def _gc_loop(self) -> None:
        rt = self.settings.runtime
        while not self._stop.is_set():
            await asyncio.sleep(rt.gc_interval_s)
            # Trim history buffers + run gc to keep RSS bounded over days.
            if len(self.alerts.recent) > rt.history_max_items:
                self.alerts.recent = self.alerts.recent[-rt.history_max_items :]
            gc.collect()
            logger.debug("gc_run", history=len(self.alerts.recent))

    # -- health views ---------------------------------------------------------

    def is_ready(self) -> bool:
        """Ready = running and we have processed at least one frame recently."""
        if not self.health.running:
            return False
        if self.health.frames_processed == 0:
            return False
        return self.health.sensor_ok

    def status(self) -> dict:
        return {
            "service": "thermal-sentry",
            "running": self.health.running,
            "ready": self.is_ready(),
            "source": self.settings.source.value,
            "detector_backend": self.detector_backend(),
            "health": self.health.as_dict(),
            "stats": dict(self.state.stats),
        }
