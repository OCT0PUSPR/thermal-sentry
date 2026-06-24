"""Thermal sensor calibration.

Two corrections are supported and applied in :func:`apply_calibration`:

1. Ambient offset -- a per-pixel (or scalar) additive correction learned by
   pointing the sensor at a uniform surface of known temperature. Removes fixed
   pattern offset and self-heating bias.
2. Emissivity correction -- a simple radiometric rescale toward a reference
   ambient, approximating ``T_corrected = T_amb + (T_meas - T_amb) / emissivity``.

Calibration data is JSON-serialisable so it can be saved/loaded and shipped per
device. Pure numpy -- unit-testable, no hardware required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import FRAME_COLS, FRAME_ROWS


@dataclass
class Calibration:
    """Per-device thermal calibration."""

    # Scalar OR per-pixel (24x32) additive offset in deg C.
    offset: float = 0.0
    offset_map: Optional[List[List[float]]] = None
    emissivity: float = 0.95
    reference_ambient_c: float = 22.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a calibrated copy of ``frame``."""
        out = frame.astype(np.float32)
        # Emissivity correction toward the reference ambient.
        eps = float(np.clip(self.emissivity, 0.1, 1.0))
        if eps < 0.999:
            out = self.reference_ambient_c + (out - self.reference_ambient_c) / eps
        # Additive offset (scalar or per-pixel).
        if self.offset_map is not None:
            omap = np.asarray(self.offset_map, dtype=np.float32)
            if omap.shape == out.shape:
                out = out + omap
        out = out + float(self.offset)
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "offset_map": self.offset_map,
            "emissivity": self.emissivity,
            "reference_ambient_c": self.reference_ambient_c,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(**data)


def calibrate_offset(
    frames: np.ndarray, known_temp_c: float, per_pixel: bool = True
) -> Calibration:
    """Learn an ambient offset from frames of a uniform known-temperature target.

    Parameters
    ----------
    frames:
        Array of shape ``(N, 24, 32)`` captured while viewing a uniform surface.
    known_temp_c:
        The true temperature of that surface.
    per_pixel:
        If True learn a per-pixel offset map (corrects fixed-pattern noise);
        otherwise a single scalar offset.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.shape[1:] != (FRAME_ROWS, FRAME_COLS):
        raise ValueError(f"frames must be (N, {FRAME_ROWS}, {FRAME_COLS})")
    mean_frame = frames.mean(axis=0)
    if per_pixel:
        offset_map = (known_temp_c - mean_frame).tolist()
        return Calibration(offset=0.0, offset_map=offset_map)
    scalar = float(known_temp_c - float(mean_frame.mean()))
    return Calibration(offset=scalar, offset_map=None)


def apply_calibration(frame: np.ndarray, calibration: Optional[Calibration]) -> np.ndarray:
    """Apply a calibration (or return the frame unchanged if None)."""
    if calibration is None:
        return frame
    return calibration.apply(frame)
