"""FastAPI server: dashboard, WebSocket feed, control + history API, security.

Endpoints
---------
Public:
  GET  /health        liveness probe
  GET  /ready         readiness probe (running + processed a frame + sensor ok)
  GET  /status        full health + stats snapshot
  GET  /metrics       Prometheus exposition
  GET  /login         HTTP Basic login -> sets a signed session cookie
Authenticated (API key OR session cookie when auth is enabled):
  GET  /              dashboard
  WS   /ws            live frame/detection/alert stream
  GET  /api/stats     latest stats
  GET  /api/state     latest payload (image omitted)
  GET  /api/events    historical events from the DB (filterable)
  GET  /api/alerts    historical alerts from the DB (filterable)
  POST /api/alerts/{id}/ack   acknowledge an alert
  POST /api/zones     set restricted-zone polygons (persisted)

Security: API-key / session auth, CORS allowlist, rate limiting (slowapi),
strict security headers. Secrets come from settings (env), never hardcoded.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import datetime as dt
import json
from pathlib import Path
from typing import List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..observability import get_logger, get_metrics
from ..runtime import AsyncRuntime
from .security import SECURITY_HEADERS, AuthManager

logger = get_logger("thermalsentry.web")
STATIC_DIR = Path(__file__).parent / "static"


class ZonesPayload(BaseModel):
    zones: List[List[List[float]]]


class _Hub:
    """Tracks WebSocket clients and broadcasts payloads from the pipeline thread."""

    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._latest: Optional[dict] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        if self._latest is not None:
            try:
                await ws.send_text(json.dumps(self._latest))
            except Exception:
                pass

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def publish_threadsafe(self, payload: dict) -> None:
        self._latest = payload
        if self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)
        except RuntimeError:  # pragma: no cover
            pass

    async def _broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(payload)
        dead: List[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def _basic_credentials(request: Request):
    header = request.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return None, None
    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
        user, _, pw = decoded.partition(":")
        return user, pw
    except (binascii.Error, ValueError):
        return None, None


def create_app(
    settings: Optional[Settings] = None,
    runtime: Optional[AsyncRuntime] = None,
    store=None,
    autostart: bool = True,
) -> FastAPI:
    """Build the FastAPI app wired to an :class:`AsyncRuntime`."""
    settings = settings or get_settings()
    metrics = get_metrics(settings.observability.metrics_enabled)

    # Event store (created here if not injected and DB is enabled).
    if store is None and settings.database.enabled:
        from ..persistence.store import EventStore

        store = EventStore.from_settings(settings.database)
        store.create_all()

    runtime = runtime or AsyncRuntime(settings=settings, store=store, metrics=metrics)
    hub = _Hub()
    runtime.subscribe(hub.publish_threadsafe)
    auth = AuthManager(settings.security)

    api = FastAPI(title="thermal-sentry", version="0.2.0")

    # --- CORS allowlist (never "*" by default) ---
    if settings.security.cors_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # --- rate limiting (slowapi) ---
    limiter = None
    if settings.security.rate_limit_enabled:
        try:
            from slowapi import Limiter, _rate_limit_exceeded_handler
            from slowapi.errors import RateLimitExceeded
            from slowapi.util import get_remote_address

            limiter = Limiter(key_func=get_remote_address, default_limits=[settings.security.rate_limit])
            api.state.limiter = limiter
            api.add_exception_handler(
                RateLimitExceeded, _rate_limit_exceeded_handler  # type: ignore[arg-type]
            )

            from slowapi.middleware import SlowAPIMiddleware

            api.add_middleware(SlowAPIMiddleware)
        except Exception as exc:  # pragma: no cover - slowapi optional
            logger.warning("rate_limit_unavailable", error=str(exc))

    # --- security headers middleware ---
    @api.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        if settings.security.security_headers:
            for k, v in SECURITY_HEADERS.items():
                response.headers.setdefault(k, v)
        return response

    # --- auth dependency ---
    def require_auth(request: Request) -> str:
        if not settings.security.auth_enabled:
            return "anonymous"
        # API key (header or query).
        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if auth.check_api_key(key):
            return "api-key"
        # Session cookie.
        token = request.cookies.get(AuthManager.COOKIE_NAME)
        if auth.verify_session(token):
            return "session"
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    # --- lifecycle ---
    @api.on_event("startup")
    async def _startup() -> None:
        hub.set_loop(asyncio.get_running_loop())
        if auth._generated_api_key:
            logger.warning("generated_api_key", api_key=auth.api_key)
        if auth._generated_password:
            logger.warning("generated_basic_password", user=settings.security.basic_auth_user, password=auth.basic_password)
        if autostart:
            await runtime.start()

    @api.on_event("shutdown")
    async def _shutdown() -> None:
        await runtime.stop()
        if store is not None:
            store.close()

    # ----- public probes -----
    @api.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "running": runtime.health.running})

    @api.get("/ready")
    async def ready() -> JSONResponse:
        if runtime.is_ready():
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not-ready"}, status_code=503)

    @api.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(runtime.status())

    @api.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(metrics.render().decode("utf-8"))

    # ----- login -----
    @api.get("/login")
    async def login(request: Request) -> Response:
        user, pw = _basic_credentials(request)
        if not auth.check_basic(user, pw):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": "Basic realm=thermal-sentry"},
                content="Authentication required",
            )
        token = auth.issue_session(user)
        resp = JSONResponse({"status": "logged-in"})
        resp.set_cookie(
            AuthManager.COOKIE_NAME, token,
            httponly=True, samesite="strict",
            max_age=settings.security.session_ttl_seconds,
        )
        return resp

    # ----- authenticated API -----
    @api.get("/api/stats")
    async def stats(_: str = Depends(require_auth)) -> JSONResponse:
        return JSONResponse(dict(runtime.state.stats))

    @api.get("/api/state")
    async def state(_: str = Depends(require_auth)) -> JSONResponse:
        payload = runtime.state.to_payload()
        payload.pop("thermal_rgb_base64", None)
        return JSONResponse(payload)

    @api.get("/api/events")
    async def events(
        limit: int = 200, hours: Optional[float] = None, _: str = Depends(require_auth)
    ) -> JSONResponse:
        if store is None:
            return JSONResponse({"events": []})
        since = None
        if hours:
            since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
        return JSONResponse({"events": store.query_events(limit=min(limit, 2000), since=since)})

    @api.get("/api/alerts")
    async def alerts(
        limit: int = 100,
        severity: Optional[str] = None,
        rule: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        _: str = Depends(require_auth),
    ) -> JSONResponse:
        if store is None:
            return JSONResponse({"alerts": runtime.alerts.recent[-limit:]})
        rows = store.query_alerts(
            limit=min(limit, 1000), severity=severity, rule=rule, acknowledged=acknowledged
        )
        return JSONResponse({"alerts": rows})

    @api.post("/api/alerts/{alert_id}/ack")
    async def ack_alert(alert_id: int, actor: str = Depends(require_auth)) -> JSONResponse:
        if store is None:
            raise HTTPException(status_code=404, detail="No event store")
        ok = store.acknowledge_alert(alert_id, actor)
        if not ok:
            raise HTTPException(status_code=404, detail="Alert not found")
        return JSONResponse({"status": "acknowledged", "id": alert_id})

    @api.post("/api/zones")
    async def set_zones(payload: ZonesPayload, _: str = Depends(require_auth)) -> JSONResponse:
        # Validate normalised coordinate ranges.
        zones = []
        for poly in payload.zones:
            if len(poly) < 3:
                raise HTTPException(status_code=422, detail="Each zone needs >= 3 points")
            clean = []
            for pt in poly:
                if len(pt) != 2:
                    raise HTTPException(status_code=422, detail="Points must be [x, y]")
                x, y = float(pt[0]), float(pt[1])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise HTTPException(status_code=422, detail="Coords must be normalised 0..1")
                clean.append((x, y))
            zones.append(clean)
        runtime.set_zones(zones)
        return JSONResponse({"status": "ok", "zone_count": len(zones)})

    # ----- websocket -----
    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if settings.security.auth_enabled:
            key = ws.query_params.get("api_key")
            token = ws.cookies.get(AuthManager.COOKIE_NAME)
            if not (auth.check_api_key(key) or auth.verify_session(token)):
                await ws.close(code=4401)
                return
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(ws)
        except Exception:
            hub.disconnect(ws)

    # ----- dashboard -----
    @api.get("/")
    async def index(request: Request) -> Response:
        # Allow the page to load; client-side auth uses the session/api key.
        if settings.security.auth_enabled:
            token = request.cookies.get(AuthManager.COOKIE_NAME)
            key = request.query_params.get("api_key")
            if not (auth.verify_session(token) or auth.check_api_key(key)):
                return FileResponse(str(STATIC_DIR / "login.html"))
        return FileResponse(str(STATIC_DIR / "index.html"))

    if STATIC_DIR.exists():
        api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    api.state.runtime = runtime
    api.state.hub = hub
    api.state.auth = auth
    api.state.store = store
    return api
