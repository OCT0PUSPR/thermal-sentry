"""Thermal frame sources.

Every source yields a ``numpy.ndarray`` of shape ``(24, 32)`` holding
temperatures in degrees Celsius.
"""

from __future__ import annotations

from .base import ThermalSource, build_source
from .file_source import FileThermalSource
from .simulator import SyntheticThermalSource

__all__ = [
    "ThermalSource",
    "build_source",
    "SyntheticThermalSource",
    "FileThermalSource",
]
