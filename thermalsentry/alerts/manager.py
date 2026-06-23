"""Alert manager with debouncing and pluggable channels.

Channels:

* ``console``  -- print to stdout (always safe).
* ``jsonl``    -- append each alert as a JSON line to a file.
* ``webhook``  -- HTTP POST the alert JSON (e.g. a Slack incoming webhook).
* ``email``    -- SMTP stub; credentials come from env, never hardcoded.

Alerts are debounced per ``alert.key``: an identical key will not fire again
within ``debounce_seconds``. The manager is sync-friendly and can be driven from
either a threaded loop or an asyncio task.
"""

from __future__ import annotations

import json
import logging
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

from ..config import AlertSettings
from ..detection.anomaly import Alert

logger = logging.getLogger("thermalsentry.alerts")


class AlertManager:
    """Route :class:`Alert` objects to configured channels with debouncing."""

    def __init__(self, settings: Optional[AlertSettings] = None) -> None:
        self.settings = settings or AlertSettings()
        self._last_fired: Dict[str, float] = {}
        # Ring buffer of recently delivered alerts for the dashboard feed.
        self.recent: List[dict] = []
        self._recent_max = 100
        self._http_client = None  # lazily created httpx.Client

        if self.settings.jsonl_path:
            Path(self.settings.jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    # -- public API -----------------------------------------------------------

    def dispatch(self, alerts: List[Alert], now: Optional[float] = None) -> List[Alert]:
        """Deliver alerts that pass the debounce filter. Returns those delivered."""
        now = time.time() if now is None else now
        delivered: List[Alert] = []
        for alert in alerts:
            if self._debounced(alert, now):
                continue
            self._last_fired[alert.key] = now
            self._deliver(alert)
            delivered.append(alert)
            self._remember(alert)
        return delivered

    def close(self) -> None:
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

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

    def _deliver(self, alert: Alert) -> None:
        if self.settings.console:
            self._to_console(alert)
        if self.settings.jsonl_path:
            self._to_jsonl(alert)
        if self.settings.webhook_url:
            self._to_webhook(alert)
        if self.settings.email_enabled:
            self._to_email(alert)

    def _to_console(self, alert: Alert) -> None:
        sev = alert.severity.upper()
        print(f"[ALERT][{sev}] {alert.rule}: {alert.message}")

    def _to_jsonl(self, alert: Alert) -> None:
        try:
            with open(self.settings.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.as_dict()) + "\n")
        except OSError as exc:  # pragma: no cover - filesystem error path
            logger.warning("Failed to write alert to %s: %s", self.settings.jsonl_path, exc)

    def _to_webhook(self, alert: Alert) -> None:
        url = self.settings.webhook_url
        if not url:
            return
        try:
            import httpx  # imported lazily so the package stays light

            if self._http_client is None:
                self._http_client = httpx.Client(timeout=5.0)
            self._http_client.post(url, json=alert.as_dict())
        except Exception as exc:  # pragma: no cover - network error path
            logger.warning("Webhook delivery failed: %s", exc)

    def _to_email(self, alert: Alert) -> None:
        s = self.settings
        if not (s.smtp_host and s.email_to and s.email_from):
            logger.warning("Email enabled but SMTP config incomplete; skipping.")
            return
        try:  # pragma: no cover - requires a live SMTP server
            msg = MIMEText(alert.message)
            msg["Subject"] = f"[thermal-sentry] {alert.severity.upper()}: {alert.rule}"
            msg["From"] = s.email_from
            msg["To"] = s.email_to
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=10) as server:
                server.starttls()
                if s.smtp_user and s.smtp_password:
                    server.login(s.smtp_user, s.smtp_password)
                server.sendmail(s.email_from, [s.email_to], msg.as_string())
        except Exception as exc:  # pragma: no cover - network error path
            logger.warning("Email delivery failed: %s", exc)
