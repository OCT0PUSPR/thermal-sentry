"""Replay thermal frames from a saved ``.npy`` sequence.

The file may hold either a single frame ``(24, 32)`` or a sequence
``(N, 24, 32)``. Frames are yielded in order, optionally looping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import FRAME_COLS, FRAME_ROWS


class FileThermalSource:
    """Replay a recorded thermal sequence stored as a NumPy array."""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        if not self.path.exists():
            raise FileNotFoundError(f"Thermal recording not found: {self.path}")

        data = np.load(self.path)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3 or data.shape[1:] != (FRAME_ROWS, FRAME_COLS):
            raise ValueError(
                f"Expected array of shape (N, {FRAME_ROWS}, {FRAME_COLS}); "
                f"got {data.shape}"
            )
        self._frames = data.astype(np.float32)
        self._index = 0

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def read(self) -> np.ndarray:
        """Return the next frame; raises ``StopIteration`` at the end if not looping."""
        if self._index >= len(self):
            if not self.loop:
                raise StopIteration("End of thermal recording")
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return frame.copy()

    def close(self) -> None:
        self._frames = np.empty((0, FRAME_ROWS, FRAME_COLS), dtype=np.float32)
