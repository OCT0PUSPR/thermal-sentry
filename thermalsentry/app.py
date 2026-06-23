"""Runtime pipeline.

Wires the full edge pipeline together:

    source.read -> preprocess -> detect -> track -> anomaly rules -> alerts
                -> publish frame + detections to subscribers (the dashboard)

The :class:`ThermalSentryApp` runs the loop on a background thread and keeps a
thread-safe snapshot of the latest state that the FastAPI server reads to push
over the WebSocket. Optionally records raw frames to a ``.npy`` sequence.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .alerts.manager import AlertManager
from .config import Settings, get_settings
from .detection.anomaly import AnomalyEngine
from .detection.detector import ThermalDetector
from .detection.tracker import CentroidTracker
from .processing.preprocess import (
    apply_colormap,
    bilinear_upscale,
    normalize_temperature,
)
from .sensors.base import build_source

logger = logging.getLogger("thermalsentry.app")


def _rgb_to_png_base64(rgb: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` uint8 RGB array as a base64 data URI (PNG)."""
    try:
        from PIL import Image

        img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception as exc:  # pragma: no cover - PIL always present in reqs
        logger.warning("PNG encoding failed: %s", exc)
        return ""


@dataclass
class FrameState:
    """Thread-safe snapshot of the latest pipeline output."""

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


class ThermalSentryApp:
    """The edge pipeline runtime."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        record_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.source = build_source(self.settings)
        self.detector = ThermalDetector(settings=self.settings.detection)
        self.tracker = CentroidTracker(settings=self.settings.tracker)
        self.anomaly = AnomalyEngine(settings=self.settings.anomaly)
        self.alerts = AlertManager(settings=self.settings.alerts)

        self._state = FrameState()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribers: List[Callable[[dict], None]] = []

        self._record_path = record_path
        self._recorded: List[np.ndarray] = []

        # Running stats.
        self._frames_processed = 0
        self._started_at = 0.0
        self._total_alerts = 0

    # -- subscriptions --------------------------------------------------------

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Register a callback invoked with the payload dict each frame."""
        self._subscribers.append(callback)

    def get_state(self) -> FrameState:
        with self._state_lock:
            return self._state

    def get_stats(self) -> dict:
        with self._state_lock:
            return dict(self._state.stats)

    def set_zones(self, zones) -> None:
        self.anomaly.set_zones(zones)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run_loop, name="ts-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._teardown()

    def _teardown(self) -> None:
        try:
            self.source.close()
        except Exception:
            pass
        self.alerts.close()
        if self._record_path and self._recorded:
            self._flush_recording()

    def _flush_recording(self) -> None:
        path = Path(self._record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(self._recorded, axis=0).astype(np.float32)
        np.save(path, arr)
        logger.info("Recorded %d frames to %s", arr.shape[0], path)

    # -- core loop ------------------------------------------------------------

    def _run_loop(self) -> None:
        target_dt = 1.0 / max(0.1, self.settings.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.process_once()
            except StopIteration:
                logger.info("Source exhausted; stopping loop.")
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Pipeline error: %s", exc)
                time.sleep(0.5)
                continue
            elapsed = time.monotonic() - t0
            sleep = target_dt - elapsed
            if sleep > 0:
                self._stop.wait(sleep)

    def process_once(self, now: Optional[float] = None) -> FrameState:
        """Run one full pipeline iteration and update the shared state.

        Exposed (and side-effect-light enough) for unit testing the pipeline.
        """
        wall_now = time.time() if now is None else now
        raw = self.source.read()  # (24, 32) deg C
        if self._record_path is not None:
            self._recorded.append(raw.astype(np.float32).copy())

        # Preprocess: upscale once and reuse for detection + display.
        up = bilinear_upscale(raw, factor=self.settings.upscale)
        disp_h, disp_w = up.shape

        # Detection on the upscaled frame.
        detections = self.detector.detect(up)

        # Tracking (monotonic clock for dwell logic).
        mono = time.monotonic()
        tracks = self.tracker.update(detections, now=mono)

        # Anomaly evaluation + alert dispatch. Pass both clocks: wall-clock for
        # the rapid-rise window and the monotonic clock for track dwell (which is
        # what the tracker used to stamp ``first_seen``).
        raised = self.anomaly.evaluate(
            up, detections, tracks, disp_w, disp_h, now=wall_now, mono_now=mono
        )
        delivered = self.alerts.dispatch(raised, now=wall_now)
        self._total_alerts += len(delivered)

        # Display image.
        norm = normalize_temperature(
            up,
            tmin=self.settings.temp_display_min_c,
            tmax=self.settings.temp_display_max_c,
        )
        rgb = apply_colormap(norm, colormap=self.settings.colormap)
        rgb_b64 = _rgb_to_png_base64(rgb)

        self._frames_processed += 1
        scene_max = float(np.max(raw))
        scene_min = float(np.min(raw))
        person_count = sum(1 for t in tracks if t.label == "person")
        runtime = max(1e-6, time.monotonic() - self._started_at)

        stats = {
            "frames_processed": self._frames_processed,
            "fps_actual": round(self._frames_processed / runtime, 2),
            "person_count": person_count,
            "track_count": len(tracks),
            "scene_max_c": round(scene_max, 2),
            "scene_min_c": round(scene_min, 2),
            "total_alerts": self._total_alerts,
            "source": self.settings.source.value,
        }

        state = FrameState(
            frame_index=self._frames_processed,
            timestamp=wall_now,
            thermal_rgb_base64=rgb_b64,
            display_w=disp_w,
            display_h=disp_h,
            detections=[d.as_dict(disp_w, disp_h) for d in detections],
            tracks=[t.as_dict(disp_w, disp_h, now=mono) for t in tracks],
            alerts=self.alerts.recent[-20:],
            stats=stats,
        )

        with self._state_lock:
            self._state = state

        payload = state.to_payload()
        for cb in self._subscribers:
            try:
                cb(payload)
            except Exception:  # pragma: no cover - subscriber isolation
                logger.debug("Subscriber callback raised; ignoring.", exc_info=True)
        return state
