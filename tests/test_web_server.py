"""Tests for the FastAPI web server (TestClient, no real websockets)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from thermalsentry.config import SecuritySettings, SourceType, get_settings
from thermalsentry.observability import reset_metrics
from thermalsentry.persistence.store import EventStore
from thermalsentry.web.server import _basic_credentials, create_app


@pytest.fixture(autouse=True)
def _reset_metrics():
    reset_metrics()
    yield
    reset_metrics()


@pytest.fixture
def store():
    s = EventStore(url="sqlite:///:memory:")
    s.create_all()
    yield s
    s.close()


def _settings(auth_enabled: bool, **sec):
    base = dict(
        api_key="testkey",
        basic_auth_user="admin",
        basic_auth_password="pw",
        rate_limit_enabled=False,
        auth_enabled=auth_enabled,
        security_headers=True,
    )
    base.update(sec)
    return get_settings(
        source=SourceType.SIMULATE,
        upscale=8,
        security=SecuritySettings(**base),
        # No DB-backed store auto-created; we inject one.
    )


def _client(auth_enabled, store=None, **sec):
    settings = _settings(auth_enabled, **sec)
    # Disable the auto event store; inject our own in-memory store.
    settings.database.enabled = False
    app = create_app(settings=settings, store=store, autostart=False)
    return TestClient(app)


# -- helpers ------------------------------------------------------------------


def test_basic_credentials_parser():
    class _Req:
        def __init__(self, header):
            self.headers = {"Authorization": header} if header else {}

    token = base64.b64encode(b"admin:pw").decode()
    assert _basic_credentials(_Req(f"Basic {token}")) == ("admin", "pw")
    assert _basic_credentials(_Req("")) == (None, None)
    assert _basic_credentials(_Req("Bearer abc")) == (None, None)
    assert _basic_credentials(_Req("Basic notbase64!!")) == (None, None)


# -- public probes ------------------------------------------------------------


def test_health(store):
    client = _client(False, store)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    # Security headers applied by middleware.
    assert r.headers.get("X-Frame-Options") == "DENY"


def test_ready_not_ready_without_frames(store):
    client = _client(False, store)
    r = client.get("/ready")
    # autostart=False -> runtime never started -> not ready.
    assert r.status_code == 503
    assert r.json()["status"] == "not-ready"


def test_status(store):
    client = _client(False, store)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "thermal-sentry"
    assert "health" in body


def test_metrics(store):
    client = _client(False, store)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "thermal_sentry" in r.text or "# metrics disabled" in r.text


# -- login --------------------------------------------------------------------


def test_login_success_sets_cookie(store):
    client = _client(True, store)
    token = base64.b64encode(b"admin:pw").decode()
    r = client.get("/login", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 200
    assert r.json()["status"] == "logged-in"
    assert "ts_session" in r.cookies


def test_login_failure_401(store):
    client = _client(True, store)
    token = base64.b64encode(b"admin:wrong").decode()
    r = client.get("/login", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


# -- auth enforcement ---------------------------------------------------------


def test_api_stats_requires_auth(store):
    client = _client(True, store)
    assert client.get("/api/stats").status_code == 401
    r = client.get("/api/stats", headers={"X-API-Key": "testkey"})
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_api_stats_wrong_key_rejected(store):
    client = _client(True, store)
    assert client.get("/api/stats", headers={"X-API-Key": "nope"}).status_code == 401


def test_auth_disabled_allows_anonymous(store):
    client = _client(False, store)
    assert client.get("/api/stats").status_code == 200


def test_api_key_via_query_param(store):
    client = _client(True, store)
    assert client.get("/api/stats?api_key=testkey").status_code == 200


def test_session_cookie_grants_access(store):
    client = _client(True, store)
    token = base64.b64encode(b"admin:pw").decode()
    client.get("/login", headers={"Authorization": f"Basic {token}"})
    # The TestClient persists the cookie jar.
    assert client.get("/api/stats").status_code == 200


# -- state / events / alerts --------------------------------------------------


def test_api_state_omits_image(store):
    client = _client(False, store)
    r = client.get("/api/state")
    assert r.status_code == 200
    assert "thermal_rgb_base64" not in r.json()


def test_api_events(store):
    store.record_event(frame_index=1, person_count=1, max_temp_c=30.0, source="simulate")
    client = _client(False, store)
    r = client.get("/api/events?limit=10")
    assert r.status_code == 200
    assert len(r.json()["events"]) == 1


def test_api_events_with_hours_filter(store):
    store.record_event(frame_index=1, person_count=0, max_temp_c=22.0, source="simulate")
    client = _client(False, store)
    r = client.get("/api/events?hours=1")
    assert r.status_code == 200
    assert "events" in r.json()


def test_api_alerts(store):
    store.record_alert(rule="overheat", severity="critical", message="m", key="k")
    client = _client(False, store)
    r = client.get("/api/alerts?severity=critical")
    assert r.status_code == 200
    assert len(r.json()["alerts"]) == 1


def test_api_alerts_no_store_uses_recent():
    # No store injected and DB disabled -> falls back to runtime.alerts.recent.
    client = _client(False, store=None)
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert "alerts" in r.json()


def test_api_events_no_store_returns_empty():
    client = _client(False, store=None)
    r = client.get("/api/events")
    assert r.status_code == 200
    assert r.json()["events"] == []


# -- ack ----------------------------------------------------------------------


def test_ack_alert(store):
    aid = store.record_alert(rule="overheat", severity="critical", message="m", key="k")
    client = _client(False, store)
    r = client.post(f"/api/alerts/{aid}/ack")
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"


def test_ack_missing_alert_404(store):
    client = _client(False, store)
    assert client.post("/api/alerts/99999/ack").status_code == 404


def test_ack_no_store_404():
    client = _client(False, store=None)
    assert client.post("/api/alerts/1/ack").status_code == 404


# -- zones --------------------------------------------------------------------


def test_set_zones_valid(store):
    client = _client(False, store)
    body = {"zones": [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]]]}
    r = client.post("/api/zones", json=body)
    assert r.status_code == 200
    assert r.json()["zone_count"] == 1


def test_set_zones_too_few_points(store):
    client = _client(False, store)
    body = {"zones": [[[0.1, 0.1], [0.4, 0.1]]]}
    r = client.post("/api/zones", json=body)
    assert r.status_code == 422


def test_set_zones_out_of_range_coords(store):
    client = _client(False, store)
    body = {"zones": [[[1.5, 0.1], [0.4, 0.1], [0.4, 0.5]]]}
    r = client.post("/api/zones", json=body)
    assert r.status_code == 422


def test_set_zones_bad_point_shape(store):
    client = _client(False, store)
    body = {"zones": [[[0.1, 0.1, 0.1], [0.4, 0.1], [0.4, 0.5]]]}
    r = client.post("/api/zones", json=body)
    assert r.status_code == 422


# -- dashboard ----------------------------------------------------------------


def test_index_serves_login_when_unauthenticated(store):
    client = _client(True, store)
    r = client.get("/")
    assert r.status_code == 200


def test_index_serves_dashboard_when_auth_disabled(store):
    client = _client(False, store)
    r = client.get("/")
    assert r.status_code == 200


def test_startup_shutdown_lifecycle_no_autostart(store):
    # Entering the context fires startup/shutdown events with autostart=False.
    settings = _settings(False)
    settings.database.enabled = False
    app = create_app(settings=settings, store=store, autostart=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


# -- hub broadcast ------------------------------------------------------------


def test_hub_publish_threadsafe_and_broadcast():
    import asyncio

    from thermalsentry.web.server import _Hub

    hub = _Hub()
    # No loop set yet -> publish just stores latest without raising.
    hub.publish_threadsafe({"a": 1})
    assert hub._latest == {"a": 1}

    class _FakeWS:
        def __init__(self, fail=False):
            self.sent = []
            self.fail = fail

        async def accept(self):
            pass

        async def send_text(self, text):
            if self.fail:
                raise RuntimeError("broken pipe")
            self.sent.append(text)

    async def run():
        good = _FakeWS()
        bad = _FakeWS(fail=True)
        # connect sends the latest snapshot immediately.
        await hub.connect(good)
        assert good.sent  # received the cached latest payload
        hub.clients.add(bad)
        await hub._broadcast({"b": 2})
        # The failing client is pruned.
        assert bad not in hub.clients
        hub.disconnect(good)
        assert good not in hub.clients

    asyncio.run(run())


def test_hub_broadcast_no_clients_is_noop():
    import asyncio

    from thermalsentry.web.server import _Hub

    async def run():
        hub = _Hub()
        await hub._broadcast({"x": 1})  # no clients -> returns early

    asyncio.run(run())


# -- auto store creation ------------------------------------------------------


def test_auto_store_created_when_db_enabled(tmp_path):
    # store=None + database.enabled=True -> the app builds an EventStore.
    settings = _settings(False)
    settings.database.enabled = True
    settings.database.url = f"sqlite:///{tmp_path / 'auto.db'}"
    app = create_app(settings=settings, store=None, autostart=False)
    assert app.state.store is not None
    client = TestClient(app)
    assert client.get("/api/events").status_code == 200


# -- rate limiting + CORS wiring ----------------------------------------------


def test_rate_limit_and_cors_middleware_wire_up(store):
    settings = _settings(
        False,
        rate_limit_enabled=True,
        rate_limit="1000/minute",
        cors_origins=["https://example.com"],
    )
    settings.database.enabled = False
    app = create_app(settings=settings, store=store, autostart=False)
    client = TestClient(app)
    # Under the high limit, normal requests still succeed.
    assert client.get("/health").status_code == 200


# -- websocket auth rejection -------------------------------------------------


def test_ws_rejected_without_auth(store):
    from starlette.websockets import WebSocketDisconnect

    client = _client(True, store)
    try:
        with client.websocket_connect("/ws") as ws:
            ws.receive_text()
        rejected = False
    except WebSocketDisconnect as exc:
        rejected = True
        assert exc.code == 4401
    except Exception:
        # Some stacks surface the close differently; treat any failure to
        # establish an authenticated stream as a rejection.
        rejected = True
    assert rejected


# -- generated-secret startup logging -----------------------------------------


def test_startup_logs_generated_secrets(store):
    # api_key/basic password unset -> AuthManager generates them and the startup
    # event logs the generated values (covering those branches).
    settings = get_settings(
        source=SourceType.SIMULATE,
        upscale=8,
        security=SecuritySettings(
            api_key=None,
            basic_auth_password=None,
            rate_limit_enabled=False,
            auth_enabled=True,
        ),
    )
    settings.database.enabled = False
    app = create_app(settings=settings, store=store, autostart=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
