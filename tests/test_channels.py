"""Unit tests for alert delivery channels with injected fakes (no network)."""

from __future__ import annotations

import json

import pytest

from thermalsentry.alerts.channels import (
    ConsoleChannel,
    EmailChannel,
    JsonlChannel,
    MqttChannel,
    TelegramChannel,
    WebhookChannel,
)
from thermalsentry.config import AlertSettings

ALERT = {"rule": "overheat", "severity": "critical", "message": "Fire 60C", "key": "overheat"}


# -- console ------------------------------------------------------------------


def test_console_channel_uses_injected_printer():
    captured = []
    ch = ConsoleChannel(printer=captured.append)
    assert ch.name == "console"
    ch.send(ALERT)
    assert len(captured) == 1
    assert "[ALERT][CRITICAL]" in captured[0]
    assert "overheat" in captured[0]


# -- jsonl --------------------------------------------------------------------


def test_jsonl_channel_writes_and_reads_back(tmp_path):
    path = tmp_path / "nested" / "alerts.jsonl"
    ch = JsonlChannel(str(path))
    ch.send(ALERT)
    ch.send({**ALERT, "key": "second"})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["rule"] == "overheat"
    assert json.loads(lines[1])["key"] == "second"


# -- webhook ------------------------------------------------------------------


class _FakeResp:
    def __init__(self, raised: bool = False):
        self._raised = raised
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True
        if self._raised:
            raise RuntimeError("http 500")


class _FakeHttpClient:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []
        self.closed = False

    def post(self, url, json=None):
        self.calls.append((url, json))
        return self._resp

    def close(self):
        self.closed = True


def test_webhook_channel_posts_and_checks_status():
    resp = _FakeResp()
    client = _FakeHttpClient(resp)
    ch = WebhookChannel("http://hook", client=client)
    ch.send(ALERT)
    assert client.calls[0][0] == "http://hook"
    assert client.calls[0][1] == ALERT
    assert resp.raise_called is True
    ch.close()
    assert client.closed is True


def test_webhook_channel_raises_on_bad_status():
    client = _FakeHttpClient(_FakeResp(raised=True))
    ch = WebhookChannel("http://hook", client=client)
    with pytest.raises(RuntimeError):
        ch.send(ALERT)


# -- email --------------------------------------------------------------------


class _FakeSMTP:
    def __init__(self):
        self.events = []

    def starttls(self, context=None):
        self.events.append("starttls")

    def login(self, user, password):
        self.events.append(("login", user, password))

    def sendmail(self, from_addr, to_addrs, msg):
        self.events.append(("sendmail", from_addr, to_addrs))

    def quit(self):
        self.events.append("quit")


def test_email_channel_full_flow():
    smtp = _FakeSMTP()
    settings = AlertSettings(
        email_enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        email_to="ops@example.com",
        email_from="bot@example.com",
        smtp_user="user",
        smtp_password="pass",
        smtp_starttls=True,
    )
    ch = EmailChannel(settings, smtp_factory=lambda host, port: smtp)
    ch.send(ALERT)
    assert "starttls" in smtp.events
    assert ("login", "user", "pass") in smtp.events
    assert any(e[0] == "sendmail" for e in smtp.events if isinstance(e, tuple))
    assert "quit" in smtp.events


def test_email_channel_no_starttls_no_login():
    smtp = _FakeSMTP()
    settings = AlertSettings(
        email_enabled=True,
        smtp_host="smtp.example.com",
        email_to="ops@example.com",
        email_from="bot@example.com",
        smtp_starttls=False,
    )
    ch = EmailChannel(settings, smtp_factory=lambda host, port: smtp)
    ch.send(ALERT)
    assert "starttls" not in smtp.events
    assert not any(isinstance(e, tuple) and e[0] == "login" for e in smtp.events)


def test_email_channel_incomplete_config_raises():
    settings = AlertSettings(email_enabled=True, smtp_host=None)
    ch = EmailChannel(settings, smtp_factory=lambda host, port: _FakeSMTP())
    with pytest.raises(ValueError):
        ch.send(ALERT)


# -- mqtt ---------------------------------------------------------------------


class _FakeMqttInfo:
    def __init__(self, rc):
        self.rc = rc


class _FakeMqttClient:
    def __init__(self, rc=0):
        self.rc = rc
        self.published = []
        self.stopped = False

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload, qos))
        return _FakeMqttInfo(self.rc)

    def loop_stop(self):
        self.stopped = True

    def disconnect(self):
        pass


def test_mqtt_channel_publishes_success():
    client = _FakeMqttClient(rc=0)
    settings = AlertSettings(mqtt_enabled=True, mqtt_host="broker", mqtt_topic="t/alerts", mqtt_qos=1)
    ch = MqttChannel(settings, client=client)
    ch.send(ALERT)
    assert client.published[0][0] == "t/alerts"
    assert json.loads(client.published[0][1])["rule"] == "overheat"
    ch.close()


def test_mqtt_channel_bad_rc_raises():
    client = _FakeMqttClient(rc=1)
    settings = AlertSettings(mqtt_enabled=True, mqtt_host="broker")
    ch = MqttChannel(settings, client=client)
    with pytest.raises(RuntimeError):
        ch.send(ALERT)


def test_mqtt_channel_no_host_raises():
    settings = AlertSettings(mqtt_enabled=True, mqtt_host=None)
    ch = MqttChannel(settings, client=_FakeMqttClient())
    with pytest.raises(ValueError):
        ch.send(ALERT)


# -- telegram -----------------------------------------------------------------


def test_telegram_channel_sends():
    resp = _FakeResp()
    client = _FakeHttpClient(resp)
    settings = AlertSettings(
        telegram_enabled=True, telegram_bot_token="abc", telegram_chat_id="123"
    )
    ch = TelegramChannel(settings, client=client)
    ch.send(ALERT)
    url, body = client.calls[0]
    assert "api.telegram.org/botabc/sendMessage" in url
    assert body["chat_id"] == "123"
    assert resp.raise_called is True
    ch.close()
    assert client.closed is True


def test_telegram_channel_incomplete_config_raises():
    settings = AlertSettings(telegram_enabled=True, telegram_bot_token=None)
    ch = TelegramChannel(settings, client=_FakeHttpClient(_FakeResp()))
    with pytest.raises(ValueError):
        ch.send(ALERT)
