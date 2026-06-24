"""SQLAlchemy ORM models for the thermal-sentry event store."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all models."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class EventRecord(Base):
    """A per-frame (or sampled) pipeline event snapshot."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    person_count: Mapped[int] = mapped_column(Integer, default=0)
    track_count: Mapped[int] = mapped_column(Integer, default=0)
    max_temp_c: Mapped[float] = mapped_column(Float, default=0.0)
    min_temp_c: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="simulate")
    detector_backend: Mapped[str] = mapped_column(String(16), default="classical")

    __table_args__ = (Index("ix_events_ts_frame", "ts", "frame_index"),)


class AlertRecord(Base):
    """A dispatched (or attempted) alert."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    rule: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str] = mapped_column(Text)
    key: Mapped[str] = mapped_column(String(128), index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    delivery_channels: Mapped[str] = mapped_column(String(256), default="")
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_by: Mapped[str] = mapped_column(String(64), default="")
    acknowledged_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    clip_path: Mapped[str] = mapped_column(String(256), default="")


class TrackRecord(Base):
    """A completed track summary (for counting / dwell analytics)."""

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[str] = mapped_column(String(32), default="person")
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    dwell_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    peak_temp_c: Mapped[float] = mapped_column(Float, default=0.0)
    frames: Mapped[int] = mapped_column(Integer, default=0)


class ConfigHistory(Base):
    """An audit log of config changes (zones, rules, runtime tweaks)."""

    __tablename__ = "config_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    kind: Mapped[str] = mapped_column(String(32), default="zones")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
