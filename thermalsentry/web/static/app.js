/* thermal-sentry dashboard client (vanilla JS over WebSocket). */
(function () {
  "use strict";

  const thermalCanvas = document.getElementById("thermal");
  const overlayCanvas = document.getElementById("overlay");
  const tctx = thermalCanvas.getContext("2d");
  const octx = overlayCanvas.getContext("2d");

  const els = {
    connDot: document.getElementById("conn-dot"),
    connLabel: document.getElementById("conn-label"),
    sourceLabel: document.getElementById("source-label"),
    people: document.getElementById("g-people"),
    maxtemp: document.getElementById("g-maxtemp"),
    tracks: document.getElementById("g-tracks"),
    fps: document.getElementById("g-fps"),
    frames: document.getElementById("s-frames"),
    min: document.getElementById("s-min"),
    alertsStat: document.getElementById("s-alerts"),
    source: document.getElementById("s-source"),
    alertsList: document.getElementById("alerts-list"),
    ts: document.getElementById("ts"),
  };

  // ---- zone drawing state --------------------------------------------------
  let drawing = false;
  let currentPoly = [];   // points in normalised coords [x,y]
  let zones = [];         // committed polygons
  const thermalImg = new Image();
  let lastDetections = [];
  let lastTracks = [];

  const seenAlertKeys = new Set();

  // ---- WebSocket -----------------------------------------------------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
      setConn(true);
      // keep-alive ping so the server's receive loop stays happy
      setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 15000);
    };
    ws.onclose = () => { setConn(false); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      try { onPayload(JSON.parse(ev.data)); } catch (e) { /* ignore */ }
    };
  }

  function setConn(ok) {
    els.connDot.className = "conn-dot " + (ok ? "on" : "off");
    els.connLabel.textContent = ok ? "live" : "disconnected";
  }

  // ---- payload handling ----------------------------------------------------
  function onPayload(p) {
    if (p.thermal_rgb_base64) {
      thermalImg.onload = () => {
        tctx.drawImage(thermalImg, 0, 0, thermalCanvas.width, thermalCanvas.height);
        drawOverlay();
      };
      thermalImg.src = p.thermal_rgb_base64;
    }
    lastDetections = p.detections || [];
    lastTracks = p.tracks || [];
    drawOverlay();
    updateStats(p.stats || {});
    updateAlerts(p.alerts || []);
    if (p.timestamp) {
      els.ts.textContent = new Date(p.timestamp * 1000).toLocaleTimeString();
    }
  }

  function updateStats(s) {
    setText(els.people, s.person_count != null ? s.person_count : 0);
    setText(els.maxtemp, s.scene_max_c != null ? s.scene_max_c.toFixed(1) + "°" : "—");
    setText(els.tracks, s.track_count != null ? s.track_count : 0);
    setText(els.fps, s.fps_actual != null ? s.fps_actual : 0);
    setText(els.frames, s.frames_processed != null ? s.frames_processed : 0);
    setText(els.min, s.scene_min_c != null ? s.scene_min_c.toFixed(1) + "°" : "—");
    setText(els.alertsStat, s.total_alerts != null ? s.total_alerts : 0);
    setText(els.source, s.source || "—");
    if (s.source) els.sourceLabel.textContent = s.source;
    // Tint the max-temp gauge by severity.
    if (s.scene_max_c >= 50) els.maxtemp.style.color = "var(--crit)";
    else if (s.scene_max_c >= 40) els.maxtemp.style.color = "var(--warn)";
    else els.maxtemp.style.color = "var(--accent-2)";
  }

  function updateAlerts(alerts) {
    const list = els.alertsList;
    const fresh = alerts.filter(a => {
      const id = a.key + ":" + Math.round(a.timestamp);
      if (seenAlertKeys.has(id)) return false;
      seenAlertKeys.add(id);
      return true;
    });
    if (!fresh.length) return;
    const empty = list.querySelector(".alert-empty");
    if (empty) empty.remove();
    fresh.forEach(a => {
      const li = document.createElement("li");
      li.className = a.severity || "info";
      const t = new Date((a.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
      li.innerHTML =
        `<span class="alert-time">${t}</span>` +
        `<div class="alert-rule ${a.severity}">${a.rule}</div>` +
        `<div class="alert-msg">${escapeHtml(a.message)}</div>`;
      list.insertBefore(li, list.firstChild);
    });
    while (list.children.length > 40) list.removeChild(list.lastChild);
  }

  // ---- overlay rendering ---------------------------------------------------
  function drawOverlay() {
    const W = overlayCanvas.width, H = overlayCanvas.height;
    octx.clearRect(0, 0, W, H);

    // Restricted zones.
    drawZones(W, H);
    if (drawing && currentPoly.length) drawPoly(currentPoly, W, H, "rgba(255,140,26,0.9)", "rgba(255,140,26,0.12)");

    // Tracks (preferred) else detections.
    const items = lastTracks.length ? lastTracks : lastDetections;
    items.forEach(d => {
      const b = d.bbox_norm;
      if (!b) return;
      const x = b[0] * W, y = b[1] * H, w = (b[2] - b[0]) * W, h = (b[3] - b[1]) * H;
      const color = colorFor(d.label);
      octx.lineWidth = 2;
      octx.strokeStyle = color;
      octx.strokeRect(x, y, w, h);
      const temp = d.peak_temp_c != null ? `${d.peak_temp_c.toFixed(1)}°` : "";
      const id = d.id != null ? `#${d.id} ` : "";
      const label = `${id}${d.label} ${temp}`;
      octx.font = "12px monospace";
      const tw = octx.measureText(label).width + 8;
      octx.fillStyle = color;
      octx.fillRect(x, y - 16, tw, 16);
      octx.fillStyle = "#0a0e14";
      octx.fillText(label, x + 4, y - 4);
    });
  }

  function colorFor(label) {
    switch (label) {
      case "person": return "#2ee6c6";
      case "hotspot": return "#ff3b58";
      case "animal": return "#ffb020";
      default: return "#9aa7bd";
    }
  }

  function drawZones(W, H) {
    zones.forEach(p => drawPoly(p, W, H, "rgba(255,59,88,0.9)", "rgba(255,59,88,0.12)"));
  }

  function drawPoly(poly, W, H, stroke, fill) {
    if (poly.length < 1) return;
    octx.beginPath();
    poly.forEach((pt, i) => {
      const x = pt[0] * W, y = pt[1] * H;
      if (i === 0) octx.moveTo(x, y); else octx.lineTo(x, y);
    });
    if (poly.length > 2) octx.closePath();
    octx.fillStyle = fill; octx.fill();
    octx.lineWidth = 2; octx.strokeStyle = stroke; octx.stroke();
    poly.forEach(pt => {
      octx.beginPath();
      octx.arc(pt[0] * W, pt[1] * H, 4, 0, Math.PI * 2);
      octx.fillStyle = stroke; octx.fill();
    });
  }

  // ---- zone drawing interactions ------------------------------------------
  document.getElementById("zone-toggle").addEventListener("click", (e) => {
    drawing = !drawing;
    e.target.classList.toggle("active", drawing);
    e.target.textContent = drawing ? "Finish Zone" : "Draw Zone";
    if (!drawing && currentPoly.length >= 3) {
      zones.push(currentPoly.slice());
      pushZones();
    }
    currentPoly = [];
    drawOverlay();
  });

  document.getElementById("zone-clear").addEventListener("click", () => {
    zones = []; currentPoly = []; pushZones(); drawOverlay();
  });

  document.getElementById("alerts-clear").addEventListener("click", () => {
    els.alertsList.innerHTML = '<li class="alert-empty">No alerts yet.</li>';
  });

  overlayCanvas.addEventListener("click", (e) => {
    if (!drawing) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    currentPoly.push([+x.toFixed(4), +y.toFixed(4)]);
    drawOverlay();
  });

  function pushZones() {
    fetch("/api/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zones: zones }),
    }).catch(() => {});
  }

  // ---- utils ---------------------------------------------------------------
  function setText(el, v) { if (el) el.textContent = v; }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  connect();
})();
