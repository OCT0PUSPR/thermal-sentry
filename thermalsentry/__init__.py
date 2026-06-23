"""thermal-sentry: edge thermal AI for Raspberry Pi.

An on-device AI system that detects people, heat-sources, and thermal anomalies
from a thermal camera (Melexis MLX90640) on a Raspberry Pi, serves a live local
web dashboard, and raises alerts.

The package is import-safe on any machine: hardware drivers are import-guarded so
``import thermalsentry`` works without ``RPi.GPIO`` / ``board`` / ``adafruit-blinka``.
Use the built-in ``--simulate`` source to run the entire pipeline on a laptop.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "OCT0PUSPR"
__license__ = "MIT"

# Thermal frame geometry for the MLX90640 (24 rows x 32 cols).
FRAME_ROWS = 24
FRAME_COLS = 32
FRAME_PIXELS = FRAME_ROWS * FRAME_COLS

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "FRAME_ROWS",
    "FRAME_COLS",
    "FRAME_PIXELS",
]
