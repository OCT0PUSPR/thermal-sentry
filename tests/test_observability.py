"""Tests for structured logging + Prometheus metrics."""

from __future__ import annotations

from thermalsentry.config import ObservabilitySettings
from thermalsentry.observability import (
    Metrics,
    configure_logging,
    get_logger,
    get_metrics,
    reset_metrics,
)


def test_configure_logging_json_and_console():
    configure_logging(ObservabilitySettings(log_format="json", log_level="DEBUG"))
    configure_logging(ObservabilitySettings(log_format="console", log_level="INFO"))
    # No settings -> defaults (json/INFO).
    configure_logging(None)


def test_get_logger_returns_usable_logger():
    log = get_logger("thermalsentry.test")
    # structlog (or stdlib) logger exposes an ``info`` callable.
    assert callable(getattr(log, "info", None))
    log.info("hello", extra_field=1)


def test_get_metrics_is_singleton():
    reset_metrics()
    m1 = get_metrics(True)
    m2 = get_metrics(True)
    assert m1 is m2
    reset_metrics()


def test_metrics_enabled_render_contains_names():
    m = Metrics(enabled=True)
    assert m.enabled is True
    m.observe_frame(latency_s=0.01, fps=8.0, persons=2, max_temp=34.0)
    m.record_alert("critical", "overheat")
    out = m.render()
    assert isinstance(out, bytes)
    assert b"thermal_sentry_frames_total" in out
    assert b"thermal_sentry_alerts_total" in out
    assert b"thermal_sentry_max_temp_c" in out


def test_metrics_disabled_render():
    m = Metrics(enabled=False)
    assert m.enabled is False
    assert m.registry is None
    # Disabled helpers are no-ops and never raise.
    m.observe_frame(0.1, 1.0, 0, 0.0)
    m.record_alert("info", "noop")
    assert m.render() == b"# metrics disabled\n"


def test_reset_metrics_clears_singleton():
    reset_metrics()
    first = get_metrics(True)
    reset_metrics()
    second = get_metrics(True)
    assert first is not second
    reset_metrics()
