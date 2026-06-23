"""Tests for the alert manager, especially debouncing."""

from __future__ import annotations

import json

from thermalsentry.alerts.manager import AlertManager
from thermalsentry.config import AlertSettings
from thermalsentry.detection.anomaly import Alert


def _alert(key="overheat", rule="overheat", sev="critical"):
    return Alert(rule=rule, severity=sev, message="test", key=key)


def test_debounce_blocks_repeat(capsys):
    mgr = AlertManager(AlertSettings(console=True, jsonl_path=None, debounce_seconds=15))
    delivered = mgr.dispatch([_alert()], now=100.0)
    assert len(delivered) == 1
    # Same key within the debounce window is suppressed.
    delivered = mgr.dispatch([_alert()], now=105.0)
    assert delivered == []
    # After the window it fires again.
    delivered = mgr.dispatch([_alert()], now=200.0)
    assert len(delivered) == 1


def test_distinct_keys_not_debounced():
    mgr = AlertManager(AlertSettings(console=False, jsonl_path=None, debounce_seconds=15))
    delivered = mgr.dispatch(
        [_alert(key="a"), _alert(key="b")], now=100.0
    )
    assert len(delivered) == 2


def test_jsonl_written(tmp_path):
    path = tmp_path / "alerts.jsonl"
    mgr = AlertManager(
        AlertSettings(console=False, jsonl_path=str(path), debounce_seconds=0)
    )
    mgr.dispatch([_alert(key="x", rule="loitering")], now=1.0)
    mgr.dispatch([_alert(key="y", rule="crowding")], now=2.0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["rule"] == "loitering"
    assert rec["severity"] == "critical"


def test_recent_buffer_tracks_delivered():
    mgr = AlertManager(AlertSettings(console=False, jsonl_path=None, debounce_seconds=0))
    for i in range(5):
        mgr.dispatch([_alert(key=f"k{i}")], now=float(i))
    assert len(mgr.recent) == 5
    assert mgr.recent[-1]["key"] == "k4"


def test_console_output(capsys):
    mgr = AlertManager(AlertSettings(console=True, jsonl_path=None, debounce_seconds=0))
    mgr.dispatch([_alert(rule="overheat", sev="critical")], now=1.0)
    out = capsys.readouterr().out
    assert "ALERT" in out
    assert "CRITICAL" in out
    assert "overheat" in out
