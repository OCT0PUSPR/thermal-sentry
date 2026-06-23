"""Configuration for thermal-sentry.

All settings are environment-driven via ``pydantic-settings``. Prefix every env
var with ``TS_`` (e.g. ``TS_SOURCE=simulate``, ``TS_WEB_PORT=8000``). Nested
settings use a double-underscore delimiter (e.g. ``TS_ALERTS__WEBHOOK_URL=...``).

See ``.env.example`` for a copy-pasteable template.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import Field
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
    webhook_url: Optional[str] = Field(
        default=None, description="POST alerts as JSON to this URL (e.g. Slack)."
    )
    # Email is a stub: credentials come from env, never hardcoded.
    email_enabled: bool = Field(default=False)
    email_to: Optional[str] = Field(default=None)
    email_from: Optional[str] = Field(default=None)
    smtp_host: Optional[str] = Field(default=None)
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[str] = Field(default=None)
    # Debounce: minimum seconds between two alerts of the same (rule, key).
    debounce_seconds: float = Field(default=15.0)


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
    web_host: str = Field(default="0.0.0.0")
    web_port: int = Field(default=8000)

    # --- nested groups ---
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)


def get_settings(**overrides) -> Settings:
    """Build a :class:`Settings`, applying explicit keyword overrides last."""

    return Settings(**overrides)
