"""Optional Pi Camera (visible-light) fusion source.

The MLX90640 sees heat; the Pi Camera sees light. Fusing them lets the dashboard
overlay thermal detections on a visible image. This module is import-guarded:
``picamera2`` is only present on a Raspberry Pi with a camera, so instantiating
:class:`PiCameraSource` raises a clear error elsewhere.

This source returns a visible RGB frame (not temperatures); it is consumed
alongside a thermal source, not in place of one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - hardware-only
    from picamera2 import Picamera2  # type: ignore

    _CAM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised on laptops
    Picamera2 = None  # type: ignore
    _CAM_IMPORT_ERROR = exc


class PiCameraSource:
    """Visible-light frames from the Raspberry Pi Camera."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)) -> None:
        if _CAM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "picamera2 is not available. Install it on a Raspberry Pi "
                "(`sudo apt install -y python3-picamera2`) with a camera attached. "
                "Camera fusion is optional; the pipeline runs without it."
            )
        self._cam = Picamera2()  # pragma: no cover - hardware-only
        config = self._cam.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()

    def read(self) -> np.ndarray:  # pragma: no cover - hardware-only
        """Return an (H, W, 3) uint8 RGB visible frame."""
        return self._cam.capture_array()

    def close(self) -> None:  # pragma: no cover - hardware-only
        try:
            self._cam.stop()
        except Exception:
            pass


def overlay_thermal_on_visible(
    visible_rgb: np.ndarray, thermal_rgb: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Alpha-blend a (resized) thermal RGB image over a visible RGB frame.

    Pure numpy so it is testable without hardware. ``thermal_rgb`` is nearest-
    resized to the visible frame's resolution before blending.
    """
    from ..processing.preprocess import _bilinear_numpy

    vh, vw = visible_rgb.shape[:2]
    # Resize each thermal channel to the visible resolution.
    resized = np.stack(
        [_bilinear_numpy(thermal_rgb[:, :, c].astype(np.float32), vh, vw) for c in range(3)],
        axis=2,
    )
    blended = (1.0 - alpha) * visible_rgb.astype(np.float32) + alpha * resized
    return np.clip(blended, 0, 255).astype(np.uint8)
