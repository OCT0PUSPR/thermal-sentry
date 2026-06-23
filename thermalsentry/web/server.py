"""FastAPI server: dashboard, WebSocket feed, and control API.

Endpoints
---------
* ``GET  /``            -- the operations dashboard (static HTML/JS/CSS).
* ``WS   /ws``          -- pushes ``{thermal_rgb_base64, detections, tracks,
                           alerts, stats}`` JSON each frame.
* ``GET  /health``      -- liveness probe.
* ``GET  /api/stats``   -- latest pipeline stats.
* ``GET  /api/state``   -- the full latest frame payload (sans image churn).
* ``POST /api/zones``   -- set restricted-zone polygons (normalised coords).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..app import ThermalSentryApp
from ..config import Settings, get_settings

logger = logging.getLogger("thermalsentry.web")

STATIC_DIR = Path(__file__).parent / "static"


class ZonesPayload(BaseModel):
    """Restricted zones as a list of normalised polygons (list of [x, y])."""

    zones: List[List[List[float]]]


class _Hub:
    """Tracks connected WebSocket clients and broadcasts payloads to them."""

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
        """Called from the pipeline thread; schedules a broadcast on the loop."""
        self._latest = payload
        if self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)
        except RuntimeError:  # pragma: no cover - loop closing
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


def create_app(
    settings: Optional[Settings] = None,
    app_runtime: Optional[ThermalSentryApp] = None,
    record_path: Optional[str] = None,
    autostart: bool = True,
) -> FastAPI:
    """Build the FastAPI app bound to a :class:`ThermalSentryApp` runtime."""
    settings = settings or get_settings()
    runtime = app_runtime or ThermalSentryApp(settings=settings, record_path=record_path)
    hub = _Hub()
    runtime.subscribe(hub.publish_threadsafe)

    api = FastAPI(title="thermal-sentry", version="0.1.0")

    @api.on_event("startup")
    async def _on_startup() -> None:
        hub.set_loop(asyncio.get_running_loop())
        if autostart:
            runtime.start()
            logger.info("Pipeline started (source=%s)", settings.source.value)

    @api.on_event("shutdown")
    async def _on_shutdown() -> None:
        runtime.stop()

    @api.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "source": settings.source.value})

    @api.get("/api/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(runtime.get_stats())

    @api.get("/api/state")
    async def state() -> JSONResponse:
        st = runtime.get_state()
        payload = st.to_payload()
        # Drop the (large) image from this JSON endpoint.
        payload.pop("thermal_rgb_base64", None)
        return JSONResponse(payload)

    @api.post("/api/zones")
    async def set_zones(payload: ZonesPayload) -> JSONResponse:
        zones = [[(float(x), float(y)) for x, y in poly] for poly in payload.zones]
        runtime.set_zones(zones)
        return JSONResponse({"status": "ok", "zone_count": len(zones)})

    @api.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                # We only push; keep the socket open and ignore inbound pings.
                await ws.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(ws)
        except Exception:
            hub.disconnect(ws)

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    if STATIC_DIR.exists():
        api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Expose runtime for tests / external control.
    api.state.runtime = runtime
    api.state.hub = hub
    return api
