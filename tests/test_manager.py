"""Tests for AlertManager channel-building, routing, retries, dead-letter."""

from __future__ import annotations

import json

from thermalsentry.alerts.manager import AlertManager
from thermalsentry.config import AlertSettings
from thermalsentry.detection.anomaly import Alert


def _alert(key="overheat", rule="overheat", sev="critical"):
    return Alert(rule=rule, severity=sev, message="test", key=key)


class _FailingChannel:
    name = "boom"

    def __init__(self):
        self.attempts = 0

    def send(self, alert):
        self.attempts += 1
        raise RuntimeError("delivery failed")


class _RecordingChannel:
    name = "rec"

    def __init__(self):
        self.received = []

    def send(self, alert):
        self.received.append(alert)


def test_build_channels_from_settings(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    settings = AlertSettings(
        console=True,
        jsonl_path=str(jsonl),
        webhook_url="http://hook",
        email_enabled=True,
        smtp_host="h",
        email_to="t",
        email_from="f",
        mqtt_enabled=True,
        mqtt_host="broker",
        telegram_enabled=True,
        telegram_bot_token="x",
        telegram_chat_id="y",
        dead_letter_path=None,
    )
    mgr = AlertManager(settings=settings)
    assert set(mgr.channels) == {"console", "jsonl", "webhook", "email", "mqtt", "telegram"}


def test_routing_by_severity():
    rec = _RecordingChannel()
    other = _RecordingChannel()
    other.name = "other"
    settings = AlertSettings(
        console=False, jsonl_path=None, dead_letter_path=None, debounce_seconds=0,
        route_critical=["rec"],
    )
    mgr = AlertManager(settings=settings, channels={"rec": rec, "other": other})
    mgr.dispatch([_alert(sev="critical")], now=1.0)
    # Only the routed channel got the alert.
    assert len(rec.received) == 1
    assert len(other.received) == 0


def test_empty_route_uses_all_channels():
    a = _RecordingChannel()
    b = _RecordingChannel()
    b.name = "b"
    settings = AlertSettings(console=False, jsonl_path=None, dead_letter_path=None, debounce_seconds=0)
    mgr = AlertManager(settings=settings, channels={"a": a, "b": b})
    mgr.dispatch([_alert(sev="info")], now=1.0)
    assert len(a.received) == 1
    assert len(b.received) == 1


def test_delivery_retries_then_dead_letters(tmp_path):
    dead = tmp_path / "dead.jsonl"
    failing = _FailingChannel()
    settings = AlertSettings(
        console=False, jsonl_path=None, dead_letter_path=str(dead),
        debounce_seconds=0, max_retries=2, retry_backoff_s=0.0,
    )
    mgr = AlertManager(settings=settings, channels={"boom": failing})
    mgr.dispatch([_alert()], now=1.0)
    assert failing.attempts == 2  # retried up to max_retries
    lines = dead.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["channel"] == "boom"
    assert rec["alert"]["rule"] == "overheat"


def test_dead_letter_disabled_when_no_path():
    failing = _FailingChannel()
    settings = AlertSettings(
        console=False, jsonl_path=None, dead_letter_path=None,
        debounce_seconds=0, max_retries=1, retry_backoff_s=0.0,
    )
    mgr = AlertManager(settings=settings, channels={"boom": failing})
    # No dead-letter path -> no file written, no exception.
    delivered = mgr.dispatch([_alert()], now=1.0)
    assert len(delivered) == 1


def test_metrics_record_alert_called():
    class _Metrics:
        def __init__(self):
            self.alerts = []

        def record_alert(self, severity, rule):
            self.alerts.append((severity, rule))

    metrics = _Metrics()
    settings = AlertSettings(console=False, jsonl_path=None, dead_letter_path=None, debounce_seconds=0)
    rec = _RecordingChannel()
    mgr = AlertManager(settings=settings, channels={"rec": rec}, metrics=metrics)
    mgr.dispatch([_alert(sev="warning", rule="loitering")], now=1.0)
    assert ("warning", "loitering") in metrics.alerts


def test_persist_to_store():
    from thermalsentry.persistence.store import EventStore

    store = EventStore(url="sqlite:///:memory:")
    store.create_all()
    settings = AlertSettings(console=False, jsonl_path=None, dead_letter_path=None, debounce_seconds=0)
    rec = _RecordingChannel()
    mgr = AlertManager(settings=settings, channels={"rec": rec}, store=store)
    mgr.dispatch([_alert()], now=1.0)
    rows = store.query_alerts()
    assert len(rows) == 1
    assert rows[0]["delivered"] is True
    assert "rec" in rows[0]["delivery_channels"]
    store.close()


def test_close_calls_channel_close():
    class _Closeable:
        name = "c"

        def __init__(self):
            self.closed = False

        def send(self, alert):
            pass

        def close(self):
            self.closed = True

    ch = _Closeable()
    settings = AlertSettings(console=False, jsonl_path=None, dead_letter_path=None)
    mgr = AlertManager(settings=settings, channels={"c": ch})
    mgr.close()
    assert ch.closed is True
