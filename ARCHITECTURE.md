# thermal-sentry — Architecture

This document describes the runtime architecture, the **from-scratch thermal
CNN** (model, training, metrics, quantization), and the production-hardening
layer. For setup/usage see [`README.md`](README.md).

---

## 1. System overview

```
 ThermalSource ──▶ Preprocess ──▶ Detector ──▶ Tracker ──▶ AnomalyEngine ──▶ AlertManager ──▶ channels
 (MLX90640 /        (upscale,      (classical                                                  (console, jsonl,
  simulate /         normalize,     CC  *or*                                                    webhook, email,
  file replay)       colormap)      ML detector)                                                MQTT, Telegram)
                                       │                                          │
                                       ▼                                          ▼
                                  EventStore (SQLAlchemy + Alembic)        Prometheus /metrics
                                       │
                                       ▼
                                  FastAPI server  ──▶  WebSocket  ──▶  dashboard (browser)
```

Two runtimes share the same pipeline stages:

| Runtime | File | Use |
|---|---|---|
| **`ThermalSentryApp`** | `thermalsentry/app.py` | Simple threaded loop for headless/CLI runs + tests. |
| **`AsyncRuntime`** | `thermalsentry/runtime.py` | Production async runtime: bounded queue with backpressure (drop-oldest), sensor-read retries, a **watchdog** that restarts a stalled capture loop, periodic GC, health/readiness snapshots, metrics + event persistence. Served by the FastAPI app. |

### Frame geometry

The MLX90640 is a **24×32** far-infrared array. The display pipeline upscales by
`TS_UPSCALE` (default 20 → 480×640) for the dashboard and for the classical
detector's connected-components. The ML model uses its own light ×2 upscale
(→ **48×64**) as input.

---

## 2. Detection backends (selectable)

Localisation/classification is **pluggable** via `TS_ML_BACKEND`. The classical
detector is always the fallback — every ML path degrades to it gracefully when a
model is missing or fails to load.

| `TS_ML_BACKEND` | Localisation | Classification | Notes |
|---|---|---|---|
| `classical` (default) | threshold + connected components | area/temperature heuristics | pure-numpy, no model |
| `onnx` / `tflite` | **classical** | learned **classification head** *refines* each blob's label from its thermal crop | crop-classifier refinement |
| `ml` | learned **center-heatmap head** | learned **classification head** | full-frame two-head CNN localises **and** classifies |

The classical detector lives in `thermalsentry/detection/detector.py`. When an
`MLDetectorBackend` is attached, `ThermalDetector.detect()` routes to
`_detect_ml()`: heatmap peaks → detection centroids, with a bbox/area estimated
by growing a window over the local hot region so the **tracker, anomaly rules,
and dashboard keep working unchanged**.

---

## 3. The from-scratch thermal CNN

Package: `thermalsentry/ml/`. Built from **primitive `conv`/`BN`/`ReLU`/`pool`
blocks** — no pretrained backbones, no model-zoo wrappers (torch + numpy only).
`torch` is imported lazily everywhere, so the package (and the numpy-only CI
path / the Pi runtime) import without it.

### 3.1 Architecture — two heads

`thermalsentry/ml/model.py` → `ThermalNet`:

```
 input  (B, 1, 48, 64)        up-scaled, normalised thermal frame
   │
 ConvBlock(1→16)  + maxpool → 24×32      (conv 3×3, BN, ReLU, pool)
 ConvBlock(16→32) + maxpool → 12×16      ← heatmap stride = 4
 ConvBlock(32→48)           → 12×16      shared feature map
   ├───────────────────────────────┬───────────────────────────────┐
   ▼                               ▼
 CLASSIFICATION head            DETECTION head
 GAP → FC(48→48) → FC(48→4)     1×1 conv(48→24) → 1×1 conv(24→1)
 → class logits (B, 4)          → center heatmap logits (B, 12, 16)
```

* **Classification head** — 4 classes: `background`, `person`, `animal`,
  `hotspot`. (`hotspot` is the anomaly readout used for overheat/fire scenes.)
* **Detection head** — a single-channel **center heatmap** at stride 4
  (12×16). Peaks (local maxima above a threshold) are the warm-body centers and
  are mapped back to the 24×32 grid for downstream stages.

**~22.5K trainable parameters** → a sub-100 KB FP32 ONNX, a ~39 KB INT8 model.

### 3.2 Synthetic dataset + augmentation

`thermalsentry/ml/dataset.py` reuses the project's `SyntheticThermalSource` and
generates a **class-balanced** full-frame dataset (`generate_frame_dataset`,
round-robin over the 4 scene types). Each sample carries a **class label** and a
gaussian **center heatmap** (labels for both heads). Augmentation:

* horizontal/vertical flips and 180° rotation (centers transformed consistently),
* random **thermal-gradient shift** + global offset (room/wall non-uniformity),
* extra per-pixel **sensor noise**,
* **varying body counts** (0–4) and **body temperatures** (~31–37 °C),
* injected **overheat anomalies** (compact 52–85 °C sources) and small low-temp
  **animal** bodies (~27–32 °C).

Everything is **fully local and pure-numpy** (no torch needed to build the data),
so the same generator feeds training, evaluation, and the INT8 calibration set.

### 3.3 Training pipeline

`thermalsentry/ml/train.py` (+ `scripts/train.py`):

* DataLoader over the synthetic tensors; **AdamW** + a **cosine LR schedule with
  warmup**; device auto-select **MPS > CUDA > CPU**.
* Joint loss: **cross-entropy** (classification) + **weighted BCE** on the soft
  gaussian heatmap (center cells up-weighted to keep recall on sparse positives).
* Best-validation **checkpointing** (combined F1 / accuracy / localisation
  score), with a held-out split (different seed) for honest metrics.

### 3.4 Real measured metrics

Trained locally on **Apple Silicon (MPS)**, **6,000** synthetic train frames /
**1,500** held-out val frames, **30 epochs**, **~35 s**. Held-out results
(`models/thermal_cnn.report.json`):

| Metric | Value |
|---|---|
| Classification **accuracy** | **0.9913** |
| Classification **macro-F1** | **0.9913** |
| Per-class acc | background 1.000 · person 1.000 · animal 0.965 · hotspot 1.000 |
| Localisation **recall** (hit-rate) | **0.9884** |
| Localisation **precision** | **0.9924** |
| Parameters | 22,517 |

> Hit-rate = fraction of ground-truth warm-body centers matched by a predicted
> heatmap peak within a 3-pixel (24×32-grid) tolerance.

### 3.5 Export + INT8 quantization

`thermalsentry/ml/export.py` (+ `scripts/export.py`):

1. **FP32 ONNX** (`models/thermal_cnn.onnx`) — the laptop/CI ML backend.
2. **INT8 ONNX** (`models/thermal_cnn_int8.onnx`) — **static QDQ** quantization
   via onnxruntime with a synthetic **calibration reader**. Static (not dynamic)
   quant is used deliberately: it emits `QLinearConv`/`QuantizeLinear` ops the
   CPU execution provider can actually run (dynamic quant emits `ConvInteger`
   with no CPU kernel).
3. **INT8 TFLite** — attempted only if the heavy `tensorflow`+`onnx2tf` tooling
   imports. **On this macOS build it did not run**, so the **INT8 ONNX is the
   shipped edge artefact** and the script states this plainly. The INT8 ONNX
   runs identically on the Pi's CPU.

Measured on a fresh held-out 2,000-frame split (`models/thermal_cnn.eval.json`):

| Model | Size | Accuracy | Macro-F1 | Loc recall | Loc precision |
|---|---:|---:|---:|---:|---:|
| FP32 ONNX | **90.7 KB** | 0.9880 | 0.9880 | 0.9898 | 0.9917 |
| **INT8 ONNX** | **39.3 KB** | 0.9855 | 0.9855 | 0.9902 | 0.9619 |
| **Δ (INT8 − FP32)** | **−56.7 %** | **−0.0025** | −0.0025 | +0.0004 | −0.0298 |

INT8 shrinks the model **56.7 %** for a **0.25 pp** accuracy cost — negligible.

### 3.6 Scale-up / production retrain

`scripts/retrain_scaleup.py` trains a larger model (more synthetic frames, more
epochs, `--device cuda`) and can **weak-label real recorded `.npy` thermal
clips** with the classical detector to fine-tune on a specific sensor/room. See
the README "Scale-up & retraining" section. After retraining, re-run
`scripts/export.py` to regenerate the FP32 + INT8 artefacts.

---

## 4. Production hardening

| Concern | Implementation |
|---|---|
| **Auth** | `web/security.py`: API-key (`X-API-Key`), HTTP-Basic login → HMAC-signed session cookie. Secrets from env; random + logged once if unset. |
| **CORS / rate-limit / headers** | CORS allowlist (never `*` by default), slowapi token-bucket rate limiting, strict security headers (HSTS, CSP, X-Frame-Options…). |
| **Async runtime** | `runtime.py`: bounded queue + drop-oldest backpressure, sensor-read retries, watchdog restart, periodic GC, health/ready/status. |
| **Event store** | `persistence/`: SQLAlchemy models (events, alerts, tracks, config history) + **Alembic** migrations + retention (delete rows older than N days). |
| **Alerting** | `alerts/`: console, JSONL, webhook, SMTP email, MQTT, Telegram — **debounced**, per-severity routing, bounded retries with backoff, **dead-letter** log, optional DB persistence. |
| **Observability** | `observability.py`: structlog JSON logging + Prometheus metrics (frame rate, latency, person count, queue depth, drops, sensor errors, watchdog restarts, alert counters) on `/metrics`. |
| **Packaging / deploy** | ARM-friendly Dockerfile, docker-compose, hardened `systemd` unit, Pi installer. |
| **Secrets** | env-only; `.env.example` holds placeholders; nothing secret is committed. |

### CI / quality gates

`pytest` + coverage (numpy-only path imports cleanly **without torch**), `ruff`,
`mypy`, `bandit`; Alembic migrations apply to a fresh DB. ML training/export
deps live in `requirements-train.txt` and are **not** part of the runtime/CI
install.
