"""Configuration for thermal-sentry.

All settings are environment-driven via ``pydantic-settings``. Prefix every env
var with ``TS_`` (e.g. ``TS_SOURCE=simulate``, ``TS_WEB_PORT=8000``). Nested
settings use a double-underscore delimiter (e.g. ``TS_ALERTS__WEBHOOK_URL=...``).

See ``.env.example`` for a copy-pasteable template.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceType(str, Enum):
    """Where thermal frames come from."""

    SIMULATE = "simulate"
    MLX90640 = "mlx90640"
    FILE = "file"


class MLXRefreshRate(str, Enum):
    """Supported MLX90640 refresh rates (Hz)."""

    HZ_0_5 = "0.5"
    HZ_1 = "1"
    HZ_2 = "2"
    HZ_4 = "4"
    HZ_8 = "8"
    HZ_16 = "16"
    HZ_32 = "32"
    HZ_64 = "64"


class DetectionSettings(BaseSettings):
    """Parameters for the classical thermal blob detector."""

    model_config = SettingsConfigDict(env_prefix="TS_DET_", extra="ignore")

    # Absolute temperature (deg C) above which a pixel is considered "hot".
    hot_threshold_c: float = Field(
        default=29.0, description="Min temperature for a hot pixel (deg C)."
    )
    # If True, use an adaptive threshold (background mean + k * std) and take the
    # larger of that and ``hot_threshold_c``.
    adaptive: bool = Field(default=True, description="Use adaptive thresholding.")
    adaptive_k: float = Field(
        default=3.0, description="Std-dev multiplier for adaptive threshold."
    )
    # Minimum blob area (in upscaled pixels) to be reported as a detection.
    # NB: areas are measured on the UPSCALED frame, so they scale with the
    # square of TS_UPSCALE. These defaults are tuned for the default upscale=20
    # (a person occupies ~3.5k-11k px). Re-tune if you change TS_UPSCALE.
    min_area: int = Field(default=400, description="Minimum blob area in pixels.")
    # Area band (in upscaled pixels) classifying a blob as a person.
    person_min_area: int = Field(default=800, description="Min area to be a person.")
    person_max_area: int = Field(default=20000, description="Max area to be a person.")
    # A person's peak temperature band (deg C). Human skin reads ~30-37 deg C.
    person_min_temp_c: float = Field(default=30.0)
    person_max_temp_c: float = Field(default=40.0)
    # Anything hotter than this peak is a "hotspot" regardless of size.
    hotspot_temp_c: float = Field(default=45.0)


class TrackerSettings(BaseSettings):
    """Parameters for the centroid/IOU tracker."""

    model_config = SettingsConfigDict(env_prefix="TS_TRACK_", extra="ignore")

    # Max centroid distance (upscaled pixels) to associate a detection with a track.
    max_distance: float = Field(default=40.0)
    # Frames a track may go unmatched before it is dropped.
    max_missed: int = Field(default=8)
    # Minimum IOU to allow association (0 disables the IOU gate).
    min_iou: float = Field(default=0.0)


class AnomalySettings(BaseSettings):
    """Parameters for anomaly / alert rules."""

    model_config = SettingsConfigDict(env_prefix="TS_ANOM_", extra="ignore")

    # Overheat: any pixel/blob peak above this triggers a fire/overheat alert.
    overheat_temp_c: float = Field(default=50.0)
    # Rapid rise: deg C increase in scene max over the rise window.
    rapid_rise_delta_c: float = Field(default=8.0)
    rapid_rise_window_s: float = Field(default=5.0)
    # Loitering: dwell seconds before a tracked person is "loitering".
    loiter_seconds: float = Field(default=20.0)
    # Person count threshold (crowding).
    max_person_count: int = Field(default=5)
    # Restricted zones: list of polygons in NORMALISED coords (0..1, x then y),
    # e.g. [[[0.1,0.1],[0.4,0.1],[0.4,0.5],[0.1,0.5]]]. A person centroid inside
    # any polygon raises an intrusion alert.
    restricted_zones: List[List[Tuple[float, float]]] = Field(default_factory=list)


class AlertSettings(BaseSettings):
    """Alert channel configuration. No secrets are hardcoded -- env only."""

    model_config = SettingsConfigDict(env_prefix="TS_ALERTS_", extra="ignore")

    console: bool = Field(default=True, description="Print alerts to stdout.")
    jsonl_path: Optional[str] = Field(
        default="captures/alerts.jsonl",
        description="Append alerts as JSON lines here (None disables).",
    )
    dead_letter_path: Optional[str] = Field(
        default="captures/alerts.deadletter.jsonl",
        description="Append permanently-failed deliveries here (None disables).",
    )

    # --- webhook ---
    webhook_url: Optional[str] = Field(
        default=None, description="POST alerts as JSON to this URL (e.g. Slack)."
    )

    # --- email (SMTP). Credentials come from env, never hardcoded. ---
    email_enabled: bool = Field(default=False)
    email_to: Optional[str] = Field(default=None)
    email_from: Optional[str] = Field(default=None)
    smtp_host: Optional[str] = Field(default=None)
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[str] = Field(default=None)
    smtp_starttls: bool = Field(default=True)

    # --- MQTT (IoT). Credentials from env. ---
    mqtt_enabled: bool = Field(default=False)
    mqtt_host: Optional[str] = Field(default=None)
    mqtt_port: int = Field(default=1883)
    mqtt_topic: str = Field(default="thermal-sentry/alerts")
    mqtt_username: Optional[str] = Field(default=None)
    mqtt_password: Optional[str] = Field(default=None)
    mqtt_tls: bool = Field(default=False)
    mqtt_qos: int = Field(default=1)

    # --- Telegram. Token/chat from env. ---
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)

    # --- delivery policy ---
    # Debounce: minimum seconds between two alerts of the same (rule, key).
    debounce_seconds: float = Field(default=15.0)
    max_retries: int = Field(default=3, description="Delivery retry attempts.")
    retry_backoff_s: float = Field(default=1.0, description="Base backoff seconds.")
    # Per-severity routing: which channels receive each severity. Empty list
    # means "all configured channels".
    route_info: List[str] = Field(default_factory=list)
    route_warning: List[str] = Field(default_factory=list)
    route_critical: List[str] = Field(default_factory=list)


class SecuritySettings(BaseSettings):
    """Authentication, CORS and rate-limiting for the dashboard + API."""

    model_config = SettingsConfigDict(env_prefix="TS_SEC_", extra="ignore")

    # When True, all non-public endpoints require auth (API key or basic auth).
    auth_enabled: bool = Field(default=True)
    # API key checked against the ``X-API-Key`` header or ``?api_key=``.
    # Empty -> a random key is generated at startup and logged (dev convenience).
    api_key: Optional[str] = Field(default=None)
    # HTTP Basic credentials for the browser dashboard login.
    basic_auth_user: str = Field(default="admin")
    basic_auth_password: Optional[str] = Field(default=None)
    # Signed session cookie secret. Empty -> random per-process (sessions reset
    # on restart). Set TS_SEC_SESSION_SECRET in production.
    session_secret: Optional[str] = Field(default=None)
    session_ttl_seconds: int = Field(default=86_400)

    # CORS allowlist (exact origins). "*" is intentionally NOT a default.
    cors_origins: List[str] = Field(default_factory=list)

    # Rate limiting (token bucket via slowapi). e.g. "120/minute".
    rate_limit: str = Field(default="120/minute")
    rate_limit_enabled: bool = Field(default=True)

    # Send strict security headers (HSTS, X-Frame-Options, CSP, ...).
    security_headers: bool = Field(default=True)


class MLSettings(BaseSettings):
    """Learned thermal model backend (two-head CNN).

    ``backend`` selects how the model is used:

    * ``classical`` -- no model; the classical heuristic detector/label is kept.
    * ``onnx`` / ``tflite`` -- the model *refines* a classical blob's label from
      its thermal crop (classification head only).
    * ``ml`` -- the full-frame ML detector localises *and* classifies warm bodies
      from the whole frame (center-heatmap + classification heads). Falls back to
      the classical detector if the model can't be loaded.
    """

    model_config = SettingsConfigDict(env_prefix="TS_ML_", extra="ignore")

    # classical | onnx | tflite | ml. Falls back to classical if a model is missing.
    backend: str = Field(default="classical")
    onnx_model_path: str = Field(default="models/thermal_cnn.onnx")
    int8_onnx_model_path: str = Field(default="models/thermal_cnn_int8.onnx")
    tflite_model_path: str = Field(default="models/thermal_cnn_int8.tflite")
    # Input crop size the crop-classifier path resizes to (legacy crop backend).
    input_size: int = Field(default=24)
    # Minimum class probability to accept the ML label; below this the classical
    # label is kept.
    min_confidence: float = Field(default=0.30)
    # Heatmap activation threshold for the full-frame ML detector's peak picking.
    heatmap_peak_threshold: float = Field(default=0.30)
    # If True, the ML classifier refines labels of classical blobs; if the model
    # is unavailable the classical label is used unchanged.
    refine_only: bool = Field(default=True)


class DatabaseSettings(BaseSettings):
    """Event store (SQLAlchemy) + retention + clip recording."""

    model_config = SettingsConfigDict(env_prefix="TS_DB_", extra="ignore")

    enabled: bool = Field(default=True)
    url: str = Field(default="sqlite:///captures/thermal_sentry.db")
    # Retention: delete events/alerts older than this many days (0 = keep all).
    retention_days: int = Field(default=30)
    retention_interval_s: float = Field(
        default=3600.0, description="How often to run retention cleanup."
    )
    # Clip recording of frames around critical alerts.
    record_clips: bool = Field(default=False)
    clips_dir: str = Field(default="recordings")
    clip_max_total_mb: int = Field(
        default=512, description="Disk-usage cap for recordings; oldest rotated."
    )


class ObservabilitySettings(BaseSettings):
    """Structured logging + Prometheus metrics."""

    model_config = SettingsConfigDict(env_prefix="TS_OBS_", extra="ignore")

    # json (production) | console (dev, pretty).
    log_format: str = Field(default="json")
    log_level: str = Field(default="INFO")
    metrics_enabled: bool = Field(default=True)


class RuntimeSettings(BaseSettings):
    """Async runtime: queue sizing, watchdog, recovery, GC."""

    model_config = SettingsConfigDict(env_prefix="TS_RT_", extra="ignore")

    queue_maxsize: int = Field(default=4, description="Bounded frame queue depth.")
    # Sensor read retries before the watchdog restarts the capture loop.
    sensor_max_retries: int = Field(default=3)
    sensor_retry_backoff_s: float = Field(default=0.5)
    # Watchdog: if no frame is captured within this many seconds, restart the
    # capture task.
    watchdog_timeout_s: float = Field(default=10.0)
    watchdog_interval_s: float = Field(default=2.0)
    # Periodic GC of in-memory history to keep memory bounded over days.
    history_max_items: int = Field(default=512)
    gc_interval_s: float = Field(default=300.0)
    # Config hot-reload: poll the config file for changes.
    config_watch_enabled: bool = Field(default=False)
    config_path: Optional[str] = Field(default=None)
    config_watch_interval_s: float = Field(default=5.0)


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="TS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- source ---
    source: SourceType = Field(default=SourceType.SIMULATE)
    fps: float = Field(default=8.0, description="Target processing frames per sec.")

    # --- MLX90640 hardware ---
    mlx_refresh_rate: MLXRefreshRate = Field(default=MLXRefreshRate.HZ_8)
    mlx_i2c_frequency: int = Field(default=800_000, description="I2C bus freq (Hz).")

    # --- simulator ---
    sim_seed: int = Field(default=42)
    sim_num_bodies: int = Field(default=2)
    sim_ambient_c: float = Field(default=22.0)
    sim_body_temp_c: float = Field(default=34.0)
    sim_noise_std: float = Field(default=0.4)

    # --- file source ---
    file_path: Optional[str] = Field(default=None, description="Path to .npy sequence.")
    file_loop: bool = Field(default=True)

    # --- preprocessing ---
    upscale: int = Field(default=20, description="Bilinear upscale factor per axis.")
    colormap: str = Field(default="ironbow", description="ironbow | inferno | grayscale")
    temp_display_min_c: float = Field(default=18.0)
    temp_display_max_c: float = Field(default=40.0)

    # --- web dashboard ---
    # Binds all interfaces by design: the dashboard must be reachable on the LAN
    # from the headless Pi / inside a container. Access is gated by the auth layer
    # (API key / session) + rate limiting; override TS_WEB_HOST=127.0.0.1 to
    # restrict to loopback.
    web_host: str = Field(default="0.0.0.0")  # nosec B104
    web_port: int = Field(default=8000)

    # --- profile ---
    profile: str = Field(default="default", description="Named config profile.")

    # --- nested groups ---
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("fps must be > 0")
        return v

    @field_validator("upscale")
    @classmethod
    def _upscale_range(cls, v: int) -> int:
        if not 1 <= v <= 64:
            raise ValueError("upscale must be in [1, 64]")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        if self.temp_display_min_c >= self.temp_display_max_c:
            raise ValueError("temp_display_min_c must be < temp_display_max_c")
        if self.runtime.queue_maxsize < 1:
            raise ValueError("runtime.queue_maxsize must be >= 1")
        if self.ml.backend not in ("classical", "onnx", "tflite", "ml"):
            raise ValueError("ml.backend must be one of classical|onnx|tflite|ml")
        return self


def get_settings(**overrides) -> Settings:
    """Build a :class:`Settings`, applying explicit keyword overrides last.

    Settings are validated on construction; invalid config raises pydantic's
    ``ValidationError`` so the service fails fast on startup.
    """

    return Settings(**overrides)


def load_settings_from_yaml(path: str, **overrides) -> Settings:
    """Load settings from a YAML file, then apply keyword overrides.

    Environment variables still take precedence for individual fields not present
    in the YAML (pydantic-settings resolves env first for unset keys). This lets
    you keep a ``config.yaml`` profile while injecting secrets via env.
    """
    import yaml  # type: ignore[import-untyped]

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping, got {type(data).__name__}")
    data.update(overrides)
    return Settings(**data)
