"""Thermal source protocol and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings


@runtime_checkable
class ThermalSource(Protocol):
    """A source of thermal frames.

    Implementations must return a ``numpy.ndarray`` of shape ``(24, 32)`` with
    ``float`` temperatures in degrees Celsius.
    """

    def read(self) -> np.ndarray:
        """Return the next thermal frame, shape ``(24, 32)``, dtype float, deg C."""
        ...

    def close(self) -> None:
        """Release any underlying resources (bus handles, file handles)."""
        ...


def build_source(settings: "Settings") -> ThermalSource:
    """Construct a :class:`ThermalSource` from settings.

    The MLX90640 driver is imported lazily so that the simulate/file paths never
    require ``board`` / ``busio`` / ``adafruit_mlx90640`` to be installed.
    """

    from ..config import SourceType  # local import avoids cycle at module load

    if settings.source == SourceType.SIMULATE:
        from .simulator import SyntheticThermalSource

        return SyntheticThermalSource(
            num_bodies=settings.sim_num_bodies,
            ambient_c=settings.sim_ambient_c,
            body_temp_c=settings.sim_body_temp_c,
            noise_std=settings.sim_noise_std,
            seed=settings.sim_seed,
        )

    if settings.source == SourceType.FILE:
        from .file_source import FileThermalSource

        if not settings.file_path:
            raise ValueError("source=file requires TS_FILE_PATH to be set")
        return FileThermalSource(settings.file_path, loop=settings.file_loop)

    if settings.source == SourceType.MLX90640:
        # Imported lazily; raises a friendly error if hardware libs are absent.
        from .mlx90640 import MLX90640Source

        return MLX90640Source(
            refresh_rate=settings.mlx_refresh_rate.value,
            i2c_frequency=settings.mlx_i2c_frequency,
        )

    raise ValueError(f"Unknown source type: {settings.source!r}")
