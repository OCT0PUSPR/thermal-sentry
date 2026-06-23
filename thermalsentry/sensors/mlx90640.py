"""Adafruit MLX90640 driver wrapper (Raspberry Pi hardware path).

The MLX90640 is a 32x24 far-infrared thermal array. This module wraps Adafruit's
CircuitPython driver. The hardware imports (``board``, ``busio``,
``adafruit_mlx90640``) are guarded so that simply importing this module never
fails on a laptop -- the import error is only raised when you actually try to
instantiate :class:`MLX90640Source`.

Wiring (I2C):
    MLX90640 VIN -> Pi 3V3 (pin 1)
    MLX90640 GND -> Pi GND (pin 6)
    MLX90640 SDA -> Pi SDA / GPIO2 (pin 3)
    MLX90640 SCL -> Pi SCL / GPIO3 (pin 5)

Enable I2C with ``sudo raspi-config`` (Interface Options -> I2C) and bump the bus
clock (see ``deploy/install_pi.sh``) for stable 16/32 Hz refresh rates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import FRAME_COLS, FRAME_PIXELS, FRAME_ROWS

# --- import guard --------------------------------------------------------------
# These succeed only on a Pi with the hardware libraries installed.
try:  # pragma: no cover - hardware-only path
    import adafruit_mlx90640  # type: ignore
    import board  # type: ignore
    import busio  # type: ignore

    _HW_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised on laptops
    board = None  # type: ignore
    busio = None  # type: ignore
    adafruit_mlx90640 = None  # type: ignore
    _HW_IMPORT_ERROR = exc


_REFRESH_MAP = {}


def _build_refresh_map() -> dict:
    """Map Hz strings to the driver's RefreshRate enum (only when libs present)."""
    if adafruit_mlx90640 is None:  # pragma: no cover
        return {}
    rr = adafruit_mlx90640.RefreshRate
    return {
        "0.5": rr.REFRESH_0_5_HZ,
        "1": rr.REFRESH_1_HZ,
        "2": rr.REFRESH_2_HZ,
        "4": rr.REFRESH_4_HZ,
        "8": rr.REFRESH_8_HZ,
        "16": rr.REFRESH_16_HZ,
        "32": rr.REFRESH_32_HZ,
        "64": rr.REFRESH_64_HZ,
    }


class MLX90640Source:
    """Live thermal source backed by an Adafruit MLX90640 over I2C."""

    def __init__(
        self,
        refresh_rate: str = "8",
        i2c_frequency: int = 800_000,
    ) -> None:
        if _HW_IMPORT_ERROR is not None:
            raise RuntimeError(
                "MLX90640 hardware libraries are not available. Install the Pi "
                "extras with `pip install -r requirements-pi.txt` and run on a "
                "Raspberry Pi with I2C enabled. To develop on a laptop, use "
                "`--source simulate` instead.\n"
                f"Original import error: {_HW_IMPORT_ERROR!r}"
            )

        global _REFRESH_MAP
        if not _REFRESH_MAP:
            _REFRESH_MAP = _build_refresh_map()

        # 800 kHz I2C is required for stable high refresh rates.
        self._i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_frequency)
        self._mlx = adafruit_mlx90640.MLX90640(self._i2c)
        self._mlx.refresh_rate = _REFRESH_MAP.get(
            str(refresh_rate), adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        )
        # Reusable flat buffer the driver fills in place (768 floats).
        self._buf = [0.0] * FRAME_PIXELS

    def read(self) -> np.ndarray:
        """Read one frame and return it as a ``(24, 32)`` deg-C array.

        The driver fills a flat 768-element buffer in row-major order. A
        transient ``ValueError`` (CRC / I2C glitch) is retried a few times.
        """
        attempts = 0
        while True:
            try:
                self._mlx.getFrame(self._buf)
                break
            except ValueError:  # pragma: no cover - hardware glitch path
                attempts += 1
                if attempts >= 3:
                    raise
        frame = np.asarray(self._buf, dtype=np.float32).reshape(FRAME_ROWS, FRAME_COLS)
        return frame

    def close(self) -> None:
        try:  # pragma: no cover - hardware-only path
            self._i2c.deinit()
        except Exception:
            pass
