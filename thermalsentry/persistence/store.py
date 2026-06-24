"""Event store: a thin synchronous facade over SQLAlchemy.

Thread-safe enough for the pipeline (one writer) + the web API (readers) because
each operation uses a short-lived session. Retention deletes old rows.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import AlertRecord, Base, ConfigHistory, EventRecord, TrackRecord

if TYPE_CHECKING:  # pragma: no cover
    from ..config import DatabaseSettings


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class EventStore:
    """Persistence facade for events, alerts, tracks and config history."""

    def __init__(self, url: str = "sqlite:///captures/thermal_sentry.db", echo: bool = False):
        # check_same_thread=False so the pipeline thread and request handlers can
        # both use the engine; sessions are still short-lived per call.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        engine_kwargs = {}
        is_memory = url in ("sqlite:///:memory:", "sqlite://")
        if is_memory:
            # A shared in-memory DB across threads/sessions needs StaticPool, else
            # every connection gets its own (empty) database.
            engine_kwargs["poolclass"] = StaticPool
        elif url.startswith("sqlite:///"):
            from pathlib import Path

            db_path = url.replace("sqlite:///", "", 1)
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            url, echo=echo, future=True, connect_args=connect_args, **engine_kwargs
        )
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    @classmethod
    def from_settings(cls, settings: "DatabaseSettings") -> "EventStore":
        return cls(url=settings.url)

    def create_all(self) -> None:
        """Create tables if they do not exist (idempotent; Alembic in prod)."""
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self._Session()

    # -- writes ---------------------------------------------------------------

    def record_event(self, **fields) -> int:
        with self._Session() as s:
            ev = EventRecord(**fields)
            s.add(ev)
            s.commit()
            return ev.id

    def record_alert(
        self,
        rule: str,
        severity: str,
        message: str,
        key: str,
        data: Optional[dict] = None,
        delivered: bool = False,
        delivery_channels: str = "",
        clip_path: str = "",
    ) -> int:
        with self._Session() as s:
            rec = AlertRecord(
                rule=rule,
                severity=severity,
                message=message,
                key=key,
                data=data or {},
                delivered=delivered,
                delivery_channels=delivery_channels,
                clip_path=clip_path,
            )
            s.add(rec)
            s.commit()
            return rec.id

    def record_track(self, **fields) -> int:
        with self._Session() as s:
            rec = TrackRecord(**fields)
            s.add(rec)
            s.commit()
            return rec.id

    def record_config_change(self, actor: str, kind: str, payload: dict) -> int:
        with self._Session() as s:
            rec = ConfigHistory(actor=actor, kind=kind, payload=payload)
            s.add(rec)
            s.commit()
            return rec.id

    def acknowledge_alert(self, alert_id: int, actor: str) -> bool:
        with self._Session() as s:
            rec = s.get(AlertRecord, alert_id)
            if rec is None:
                return False
            rec.acknowledged = True
            rec.acknowledged_by = actor
            rec.acknowledged_at = _utcnow()
            s.commit()
            return True

    # -- reads ----------------------------------------------------------------

    def query_alerts(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        rule: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        since: Optional[dt.datetime] = None,
    ) -> List[dict]:
        with self._Session() as s:
            stmt = select(AlertRecord).order_by(AlertRecord.ts.desc())
            if severity:
                stmt = stmt.where(AlertRecord.severity == severity)
            if rule:
                stmt = stmt.where(AlertRecord.rule == rule)
            if acknowledged is not None:
                stmt = stmt.where(AlertRecord.acknowledged == acknowledged)
            if since:
                stmt = stmt.where(AlertRecord.ts >= since)
            stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._alert_to_dict(r) for r in rows]

    def query_events(
        self, limit: int = 500, since: Optional[dt.datetime] = None
    ) -> List[dict]:
        with self._Session() as s:
            stmt = select(EventRecord).order_by(EventRecord.ts.desc())
            if since:
                stmt = stmt.where(EventRecord.ts >= since)
            stmt = stmt.limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [self._event_to_dict(r) for r in rows]

    def counts(self) -> dict:
        with self._Session() as s:
            return {
                "events": s.execute(select(func.count(EventRecord.id))).scalar_one(),
                "alerts": s.execute(select(func.count(AlertRecord.id))).scalar_one(),
                "tracks": s.execute(select(func.count(TrackRecord.id))).scalar_one(),
            }

    # -- retention ------------------------------------------------------------

    def apply_retention(self, retention_days: int) -> dict:
        """Delete events/alerts/tracks older than ``retention_days``. Returns counts."""
        if retention_days <= 0:
            return {"events": 0, "alerts": 0, "tracks": 0}
        cutoff = _utcnow() - dt.timedelta(days=retention_days)
        deleted = {}
        with self._Session() as s:
            # ``execute(delete(...))`` returns a CursorResult exposing rowcount at
            # runtime (the base Result stub does not declare it).
            deleted["events"] = s.execute(
                delete(EventRecord).where(EventRecord.ts < cutoff)
            ).rowcount  # type: ignore[attr-defined]
            deleted["alerts"] = s.execute(
                delete(AlertRecord).where(AlertRecord.ts < cutoff)
            ).rowcount  # type: ignore[attr-defined]
            deleted["tracks"] = s.execute(
                delete(TrackRecord).where(TrackRecord.last_seen < cutoff)
            ).rowcount  # type: ignore[attr-defined]
            s.commit()
        return deleted

    def close(self) -> None:
        self.engine.dispose()

    # -- serialisers ----------------------------------------------------------

    @staticmethod
    def _alert_to_dict(r: AlertRecord) -> dict:
        return {
            "id": r.id,
            "ts": r.ts.isoformat() if r.ts else None,
            "rule": r.rule,
            "severity": r.severity,
            "message": r.message,
            "key": r.key,
            "data": r.data,
            "delivered": r.delivered,
            "delivery_channels": r.delivery_channels,
            "acknowledged": r.acknowledged,
            "acknowledged_by": r.acknowledged_by,
            "clip_path": r.clip_path,
        }

    @staticmethod
    def _event_to_dict(r: EventRecord) -> dict:
        return {
            "id": r.id,
            "ts": r.ts.isoformat() if r.ts else None,
            "frame_index": r.frame_index,
            "person_count": r.person_count,
            "track_count": r.track_count,
            "max_temp_c": r.max_temp_c,
            "min_temp_c": r.min_temp_c,
            "source": r.source,
            "detector_backend": r.detector_backend,
        }
