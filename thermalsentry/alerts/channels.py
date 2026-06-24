"""Alert delivery channels.

Each channel implements ``send(alert_dict) -> None`` and raises on failure (the
manager handles retries / dead-lettering). Transports are injectable so tests can
mock them without real network/SMTP/MQTT. No secrets are hardcoded -- every
credential comes from settings (which come from env).
"""

from __future__ import annotations

import json
import smtplib
import ssl
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from ..config import AlertSettings


class Channel(Protocol):
    name: str

    def send(self, alert: dict) -> None:
        ...


class ConsoleChannel:
    name = "console"

    def __init__(self, printer: Callable[[str], None] = print) -> None:
        self._print = printer

    def send(self, alert: dict) -> None:
        sev = str(alert.get("severity", "info")).upper()
        self._print(f"[ALERT][{sev}] {alert.get('rule')}: {alert.get('message')}")


class JsonlChannel:
    name = "jsonl"

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert) + "\n")


class WebhookChannel:
    name = "webhook"

    def __init__(self, url: str, client=None, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout
        self._client = client  # injectable httpx.Client-like

    def _ensure_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def send(self, alert: dict) -> None:
        client = self._ensure_client()
        resp = client.post(self.url, json=alert)
        # httpx Response exposes raise_for_status; a mock may omit it.
        raise_for = getattr(resp, "raise_for_status", None)
        if callable(raise_for):
            raise_for()

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()


class EmailChannel:
    name = "email"

    def __init__(self, settings: "AlertSettings", smtp_factory: Optional[Callable] = None) -> None:
        self.s = settings
        # smtp_factory(host, port) -> an smtplib.SMTP-like object (injectable).
        self._smtp_factory = smtp_factory or (lambda host, port: smtplib.SMTP(host, port, timeout=10))

    def send(self, alert: dict) -> None:
        s = self.s
        if not (s.smtp_host and s.email_to and s.email_from):
            raise ValueError("SMTP config incomplete (host/to/from required)")
        msg = MIMEText(str(alert.get("message", "")))
        msg["Subject"] = f"[thermal-sentry] {str(alert.get('severity','')).upper()}: {alert.get('rule')}"
        msg["From"] = s.email_from
        msg["To"] = s.email_to
        server = self._smtp_factory(s.smtp_host, s.smtp_port)
        try:
            if s.smtp_starttls:
                server.starttls(context=ssl.create_default_context())
            if s.smtp_user and s.smtp_password:
                server.login(s.smtp_user, s.smtp_password)
            server.sendmail(s.email_from, [s.email_to], msg.as_string())
        finally:
            try:
                server.quit()
            except Exception:
                pass


class MqttChannel:
    name = "mqtt"

    def __init__(self, settings: "AlertSettings", client=None) -> None:
        self.s = settings
        self._client = client  # injectable paho client

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        import paho.mqtt.client as mqtt

        client = mqtt.Client()
        if self.s.mqtt_username:
            client.username_pw_set(self.s.mqtt_username, self.s.mqtt_password or "")
        if self.s.mqtt_tls:
            client.tls_set()
        client.connect(self.s.mqtt_host, self.s.mqtt_port, keepalive=30)
        client.loop_start()
        self._client = client
        return client

    def send(self, alert: dict) -> None:
        if not self.s.mqtt_host:
            raise ValueError("mqtt_host not configured")
        client = self._ensure_client()
        info = client.publish(self.s.mqtt_topic, json.dumps(alert), qos=self.s.mqtt_qos)
        # paho returns an MQTTMessageInfo with rc; 0 == success.
        rc = getattr(info, "rc", 0)
        if rc not in (0, None):
            raise RuntimeError(f"MQTT publish failed rc={rc}")

    def close(self) -> None:
        if self._client is not None:
            try:  # pragma: no cover - network teardown
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


class TelegramChannel:
    name = "telegram"

    def __init__(self, settings: "AlertSettings", client=None, timeout: float = 5.0) -> None:
        self.s = settings
        self.timeout = timeout
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def send(self, alert: dict) -> None:
        s = self.s
        if not (s.telegram_bot_token and s.telegram_chat_id):
            raise ValueError("Telegram bot token / chat id not configured")
        text = (
            f"\U0001F6F0 thermal-sentry [{str(alert.get('severity','')).upper()}]\n"
            f"{alert.get('rule')}: {alert.get('message')}"
        )
        url = f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage"
        client = self._ensure_client()
        resp = client.post(url, json={"chat_id": s.telegram_chat_id, "text": text})
        raise_for = getattr(resp, "raise_for_status", None)
        if callable(raise_for):
            raise_for()

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
