"""Observability: structured JSON logging (structlog) + Prometheus metrics.

Importing this module is cheap and side-effect free until :func:`configure_logging`
or :func:`get_metrics` is called. Metrics use a private registry so multiple test
instances do not clash with the global default registry.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .config import ObservabilitySettings

try:
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - structlog is a core dep
    structlog = None  # type: ignore
    _HAS_STRUCTLOG = False

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False


def configure_logging(settings: Optional["ObservabilitySettings"] = None) -> None:
    """Configure stdlib logging + structlog for JSON or pretty console output."""
    level_name = (settings.log_level if settings else "INFO").upper()
    fmt = (settings.log_format if settings else "json").lower()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    if not _HAS_STRUCTLOG:  # pragma: no cover
        return

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "thermalsentry"):
    """Return a structlog logger (or a stdlib logger if structlog is absent)."""
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)  # pragma: no cover


class Metrics:
    """Prometheus metrics for the pipeline. Uses an isolated registry."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and _HAS_PROM
        if not self.enabled:
            self.registry = None
            return
        self.registry = CollectorRegistry()
        self.frame_rate = Gauge(
            "thermal_sentry_frame_rate", "Processed frames per second", registry=self.registry
        )
        self.detect_latency = Histogram(
            "thermal_sentry_detect_latency_seconds",
            "Per-frame detection latency",
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.person_count = Gauge(
            "thermal_sentry_person_count", "Current tracked person count", registry=self.registry
        )
        self.max_temp_c = Gauge(
            "thermal_sentry_max_temp_c", "Current scene max temperature (C)", registry=self.registry
        )
        self.queue_depth = Gauge(
            "thermal_sentry_queue_depth", "Frame queue depth", registry=self.registry
        )
        self.frames_total = Counter(
            "thermal_sentry_frames_total", "Total frames processed", registry=self.registry
        )
        self.frames_dropped_total = Counter(
            "thermal_sentry_frames_dropped_total",
            "Frames dropped due to backpressure",
            registry=self.registry,
        )
        self.sensor_errors_total = Counter(
            "thermal_sentry_sensor_errors_total",
            "Total sensor read errors",
            registry=self.registry,
        )
        self.watchdog_restarts_total = Counter(
            "thermal_sentry_watchdog_restarts_total",
            "Capture-loop restarts triggered by the watchdog",
            registry=self.registry,
        )
        self.alerts_total = Counter(
            "thermal_sentry_alerts_total",
            "Total alerts dispatched",
            ["severity", "rule"],
            registry=self.registry,
        )
        self.alert_delivery_failures_total = Counter(
            "thermal_sentry_alert_delivery_failures_total",
            "Alert channel delivery failures",
            ["channel"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Render metrics in Prometheus text exposition format."""
        if not self.enabled or self.registry is None:
            return b"# metrics disabled\n"
        return generate_latest(self.registry)

    # Convenience no-op-safe helpers ------------------------------------------

    def observe_frame(self, latency_s: float, fps: float, persons: int, max_temp: float) -> None:
        if not self.enabled:
            return
        self.detect_latency.observe(latency_s)
        self.frame_rate.set(fps)
        self.person_count.set(persons)
        self.max_temp_c.set(max_temp)
        self.frames_total.inc()

    def record_alert(self, severity: str, rule: str) -> None:
        if not self.enabled:
            return
        self.alerts_total.labels(severity=severity, rule=rule).inc()


_GLOBAL_METRICS: Optional[Metrics] = None


def get_metrics(enabled: bool = True) -> Metrics:
    """Return a process-wide :class:`Metrics` (created once)."""
    global _GLOBAL_METRICS
    if _GLOBAL_METRICS is None:
        _GLOBAL_METRICS = Metrics(enabled=enabled)
    return _GLOBAL_METRICS


def reset_metrics() -> None:
    """Drop the global metrics instance (tests use this for isolation)."""
    global _GLOBAL_METRICS
    _GLOBAL_METRICS = None
