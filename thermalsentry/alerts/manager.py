"""Production alert manager: routing, debounce, retries, dead-letter, persistence.

Pipeline per alert:
  1. Debounce by ``alert.key`` within ``debounce_seconds``.
  2. Resolve the channels for the alert's severity (per-severity routing).
  3. Deliver to each channel with bounded retries + exponential backoff.
  4. Permanently-failed deliveries go to a JSONL dead-letter log.
  5. Optionally persist the alert (delivered flag + channels) to the event store.

The manager is synchronous and thread-safe enough to be called from the pipeline
thread. Channels and their transports are injectable for testing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ..config import AlertSettings
from ..detection.anomaly import Alert
from ..observability import get_logger
from .channels import (
    ConsoleChannel,
    EmailChannel,
    JsonlChannel,
    MqttChannel,
    TelegramChannel,
    WebhookChannel,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..persistence.store import EventStore

logger = get_logger("thermalsentry.alerts")


class AlertManager:
    """Route :class:`Alert` objects to channels with debounce + retries."""

    def __init__(
        self,
        settings: Optional[AlertSettings] = None,
        channels: Optional[Dict[str, object]] = None,
        store: "Optional[EventStore]" = None,
        metrics=None,
    ) -> None:
        self.settings = settings or AlertSettings()
        self.store = store
        self.metrics = metrics
        self._last_fired: Dict[str, float] = {}
        self.recent: List[dict] = []
        self._recent_max = 100

        # Build channels from settings (or accept injected ones for tests).
        self.channels: Dict[str, object] = channels if channels is not None else self._build_channels()

        if self.settings.dead_letter_path:
            Path(self.settings.dead_letter_path).parent.mkdir(parents=True, exist_ok=True)

    # -- construction ---------------------------------------------------------

    def _build_channels(self) -> Dict[str, object]:
        s = self.settings
        ch: Dict[str, object] = {}
        if s.console:
            ch["console"] = ConsoleChannel()
        if s.jsonl_path:
            ch["jsonl"] = JsonlChannel(s.jsonl_path)
        if s.webhook_url:
            ch["webhook"] = WebhookChannel(s.webhook_url)
        if s.email_enabled:
            ch["email"] = EmailChannel(s)
        if s.mqtt_enabled:
            ch["mqtt"] = MqttChannel(s)
        if s.telegram_enabled:
            ch["telegram"] = TelegramChannel(s)
        return ch

    def _channels_for(self, severity: str) -> List[str]:
        """Resolve the channel names for a severity (empty route == all)."""
        route = {
            "info": self.settings.route_info,
            "warning": self.settings.route_warning,
            "critical": self.settings.route_critical,
        }.get(severity, [])
        names = route if route else list(self.channels.keys())
        # Only keep channels that actually exist.
        return [n for n in names if n in self.channels]

    # -- public API -----------------------------------------------------------

    def dispatch(self, alerts: List[Alert], now: Optional[float] = None) -> List[Alert]:
        """Deliver alerts that pass the debounce filter. Returns those dispatched."""
        now = time.time() if now is None else now
        dispatched: List[Alert] = []
        for alert in alerts:
            if self._debounced(alert, now):
                continue
            self._last_fired[alert.key] = now
            channels_used = self._deliver(alert)
            self._remember(alert)
            if self.metrics is not None:
                self.metrics.record_alert(alert.severity, alert.rule)
            if self.store is not None:
                try:
                    self.store.record_alert(
                        rule=alert.rule,
                        severity=alert.severity,
                        message=alert.message,
                        key=alert.key,
                        data=alert.data,
                        delivered=bool(channels_used),
                        delivery_channels=",".join(channels_used),
                    )
                except Exception as exc:  # pragma: no cover - DB error path
                    logger.warning("alert_persist_failed", error=str(exc))
            dispatched.append(alert)
        return dispatched

    def close(self) -> None:
        for ch in self.channels.values():
            closer = getattr(ch, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    # -- debouncing -----------------------------------------------------------

    def _debounced(self, alert: Alert, now: float) -> bool:
        last = self._last_fired.get(alert.key)
        if last is None:
            return False
        return (now - last) < self.settings.debounce_seconds

    def _remember(self, alert: Alert) -> None:
        self.recent.append(alert.as_dict())
        if len(self.recent) > self._recent_max:
            self.recent = self.recent[-self._recent_max :]

    # -- delivery -------------------------------------------------------------

    def _deliver(self, alert: Alert) -> List[str]:
        """Deliver to routed channels with retries. Returns channels that succeeded."""
        payload = alert.as_dict()
        succeeded: List[str] = []
        for name in self._channels_for(alert.severity):
            channel = self.channels[name]
            if self._deliver_with_retry(name, channel, payload):
                succeeded.append(name)
            else:
                self._dead_letter(name, payload)
                if self.metrics is not None and hasattr(self.metrics, "alert_delivery_failures_total"):
                    try:
                        self.metrics.alert_delivery_failures_total.labels(channel=name).inc()
                    except Exception:
                        pass
        return succeeded

    def _deliver_with_retry(self, name: str, channel: object, payload: dict) -> bool:
        attempts = self.settings.max_retries
        for attempt in range(1, attempts + 1):
            try:
                channel.send(payload)  # type: ignore[attr-defined]
                return True
            except Exception as exc:
                logger.warning(
                    "alert_delivery_attempt_failed",
                    channel=name,
                    attempt=attempt,
                    max_attempts=attempts,
                    error=str(exc),
                )
                if attempt < attempts:
                    time.sleep(self.settings.retry_backoff_s * (2 ** (attempt - 1)))
        return False

    def _dead_letter(self, channel: str, payload: dict) -> None:
        if not self.settings.dead_letter_path:
            return
        try:
            import json

            record = {"channel": channel, "alert": payload, "ts": time.time()}
            with open(self.settings.dead_letter_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem error path
            logger.warning("dead_letter_write_failed", error=str(exc))
