"""Tests for the SQLAlchemy event store + retention."""

from __future__ import annotations

import datetime as dt

import pytest

from thermalsentry.persistence.store import EventStore


@pytest.fixture
def store():
    s = EventStore(url="sqlite:///:memory:")
    s.create_all()
    yield s
    s.close()


def test_record_and_query_alert(store):
    aid = store.record_alert(
        rule="overheat", severity="critical", message="Fire 60C", key="overheat",
        data={"peak": 60.0}, delivered=True, delivery_channels="webhook,mqtt",
    )
    assert aid > 0
    rows = store.query_alerts()
    assert len(rows) == 1
    assert rows[0]["rule"] == "overheat"
    assert rows[0]["delivered"] is True
    assert rows[0]["delivery_channels"] == "webhook,mqtt"


def test_query_filters(store):
    store.record_alert(rule="overheat", severity="critical", message="m", key="a")
    store.record_alert(rule="loitering", severity="warning", message="m", key="b")
    assert len(store.query_alerts(severity="critical")) == 1
    assert len(store.query_alerts(rule="loitering")) == 1
    assert len(store.query_alerts()) == 2


def test_acknowledge(store):
    aid = store.record_alert(rule="overheat", severity="critical", message="m", key="a")
    assert store.acknowledge_alert(aid, "operator") is True
    assert store.acknowledge_alert(99999, "operator") is False
    acked = store.query_alerts(acknowledged=True)
    assert len(acked) == 1
    assert acked[0]["acknowledged_by"] == "operator"


def test_events_and_counts(store):
    for i in range(5):
        store.record_event(frame_index=i, person_count=i % 3, max_temp_c=30 + i, source="simulate")
    assert store.counts()["events"] == 5
    evs = store.query_events(limit=10)
    assert len(evs) == 5
    assert evs[0]["frame_index"] == 4  # newest first


def test_config_history(store):
    cid = store.record_config_change("api", "zones", {"zones": [[[0.1, 0.1]]]})
    assert cid > 0


def test_retention_deletes_old_rows(store):
    # Insert an alert then force its timestamp into the past.
    from sqlalchemy import update

    from thermalsentry.persistence.models import AlertRecord

    store.record_alert(rule="old", severity="info", message="m", key="old")
    store.record_alert(rule="new", severity="info", message="m", key="new")
    old_ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=100)
    with store.session() as s:
        s.execute(update(AlertRecord).where(AlertRecord.rule == "old").values(ts=old_ts))
        s.commit()

    deleted = store.apply_retention(retention_days=30)
    assert deleted["alerts"] == 1
    remaining = [a["rule"] for a in store.query_alerts()]
    assert remaining == ["new"]


def test_retention_zero_keeps_all(store):
    store.record_alert(rule="r", severity="info", message="m", key="k")
    deleted = store.apply_retention(retention_days=0)
    assert deleted == {"events": 0, "alerts": 0, "tracks": 0}
    assert len(store.query_alerts()) == 1
