"""Import-safety and config tests.

These assert the package imports cleanly with NO hardware libraries present
(``board`` / ``busio`` / ``adafruit_mlx90640`` absent), which is the whole point
of the import guards.
"""

from __future__ import annotations

import importlib

import pytest


def test_top_level_import():
    import thermalsentry

    assert thermalsentry.FRAME_ROWS == 24
    assert thermalsentry.FRAME_COLS == 32
    assert thermalsentry.FRAME_PIXELS == 768


def test_mlx_module_imports_without_hardware():
    # Importing the module must NOT fail even though hardware libs are missing.
    mod = importlib.import_module("thermalsentry.sensors.mlx90640")
    # The import error is captured, not raised at import time.
    assert hasattr(mod, "MLX90640Source")


def test_mlx_instantiation_errors_clearly_without_hardware():
    from thermalsentry.sensors.mlx90640 import _HW_IMPORT_ERROR, MLX90640Source

    if _HW_IMPORT_ERROR is None:
        pytest.skip("hardware libraries are installed in this environment")
    with pytest.raises(RuntimeError) as exc:
        MLX90640Source()
    assert "simulate" in str(exc.value).lower()


def test_settings_defaults():
    from thermalsentry.config import Settings, SourceType

    s = Settings()
    assert s.source == SourceType.SIMULATE
    assert s.web_port == 8000
    assert s.detection.person_min_temp_c < s.detection.person_max_temp_c


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("TS_WEB_PORT", "9100")
    monkeypatch.setenv("TS_SOURCE", "simulate")
    from thermalsentry.config import Settings

    s = Settings()
    assert s.web_port == 9100


def test_build_source_file_requires_path():
    from thermalsentry.config import SourceType, get_settings
    from thermalsentry.sensors.base import build_source

    s = get_settings(source=SourceType.FILE, file_path=None)
    with pytest.raises(ValueError):
        build_source(s)
