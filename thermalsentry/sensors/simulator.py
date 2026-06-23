"""Synthetic thermal source.

This is the workhorse for laptop development: it synthesises realistic 24x32
thermal frames with an ambient gradient, ``N`` moving gaussian "warm bodies"
(~30-37 deg C) over a ~22 deg C background, plus per-pixel sensor noise.

It is deterministic given a seed, so unit tests can assert exact behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .. import FRAME_COLS, FRAME_ROWS


@dataclass
class _Body:
    """A single moving warm blob in (row, col) image space."""

    row: float
    col: float
    v_row: float
    v_col: float
    peak_c: float
    sigma_row: float
    sigma_col: float

    def step(self, rng: np.random.Generator) -> None:
        """Advance the body one frame, bouncing off the frame edges."""
        # Small random walk on velocity keeps motion organic but bounded.
        self.v_row += rng.normal(0.0, 0.05)
        self.v_col += rng.normal(0.0, 0.05)
        self.v_row = float(np.clip(self.v_row, -1.5, 1.5))
        self.v_col = float(np.clip(self.v_col, -1.5, 1.5))

        self.row += self.v_row
        self.col += self.v_col

        # Reflect at the borders so bodies stay on-screen.
        if self.row < 0:
            self.row = -self.row
            self.v_row = abs(self.v_row)
        elif self.row > FRAME_ROWS - 1:
            self.row = 2 * (FRAME_ROWS - 1) - self.row
            self.v_row = -abs(self.v_row)
        if self.col < 0:
            self.col = -self.col
            self.v_col = abs(self.v_col)
        elif self.col > FRAME_COLS - 1:
            self.col = 2 * (FRAME_COLS - 1) - self.col
            self.v_col = -abs(self.v_col)


@dataclass
class SyntheticThermalSource:
    """Generate deterministic synthetic 24x32 thermal frames.

    Parameters
    ----------
    num_bodies:
        Number of moving warm bodies to render.
    ambient_c:
        Mean background temperature (deg C).
    body_temp_c:
        Mean peak temperature of a warm body (deg C).
    noise_std:
        Per-pixel gaussian noise standard deviation (deg C).
    gradient_c:
        Magnitude of the static ambient gradient across the frame (deg C).
    seed:
        RNG seed for full determinism.
    """

    num_bodies: int = 2
    ambient_c: float = 22.0
    body_temp_c: float = 34.0
    noise_std: float = 0.4
    gradient_c: float = 1.5
    seed: int = 42

    _rng: np.random.Generator = field(init=False, repr=False)
    _bodies: List[_Body] = field(init=False, repr=False, default_factory=list)
    _gradient: np.ndarray = field(init=False, repr=False)
    _frame_index: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._bodies = [self._spawn_body() for _ in range(self.num_bodies)]
        self._gradient = self._build_gradient()

    # -- construction helpers -------------------------------------------------

    def _spawn_body(self) -> _Body:
        rng = self._rng
        return _Body(
            row=float(rng.uniform(2, FRAME_ROWS - 3)),
            col=float(rng.uniform(2, FRAME_COLS - 3)),
            v_row=float(rng.uniform(-0.8, 0.8)),
            v_col=float(rng.uniform(-0.8, 0.8)),
            peak_c=float(self.body_temp_c + rng.uniform(-1.5, 3.0)),
            sigma_row=float(rng.uniform(2.0, 3.2)),
            sigma_col=float(rng.uniform(1.6, 2.6)),
        )

    def _build_gradient(self) -> np.ndarray:
        """A smooth static gradient mimicking uneven room/wall temperatures."""
        rr = np.linspace(-1.0, 1.0, FRAME_ROWS)[:, None]
        cc = np.linspace(-1.0, 1.0, FRAME_COLS)[None, :]
        # Diagonal-ish gradient, scaled to +/- gradient_c/2.
        g = (0.6 * rr + 0.4 * cc)
        return (self.gradient_c / 2.0) * g

    # -- public API -----------------------------------------------------------

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def read(self) -> np.ndarray:
        """Return the next synthetic frame, shape ``(24, 32)``, deg C."""
        frame = np.full((FRAME_ROWS, FRAME_COLS), self.ambient_c, dtype=np.float64)
        frame += self._gradient

        rows = np.arange(FRAME_ROWS)[:, None]
        cols = np.arange(FRAME_COLS)[None, :]

        # Composite bodies with a per-pixel MAX (not a sum). Overlapping bodies
        # therefore never stack into physically absurd readings, while a single
        # genuinely hot source (e.g. a 60 C overheat) keeps its true peak.
        for body in self._bodies:
            body.step(self._rng)
            dr = (rows - body.row) / body.sigma_row
            dc = (cols - body.col) / body.sigma_col
            blob = self.ambient_c + (body.peak_c - self.ambient_c) * np.exp(
                -0.5 * (dr * dr + dc * dc)
            )
            np.maximum(frame, blob, out=frame)

        # Per-pixel sensor noise.
        if self.noise_std > 0:
            frame += self._rng.normal(0.0, self.noise_std, size=frame.shape)

        self._frame_index += 1
        return frame.astype(np.float32)

    def body_positions(self) -> List[tuple]:
        """Return current ground-truth body centroids as (row, col) tuples."""
        return [(b.row, b.col) for b in self._bodies]

    def close(self) -> None:  # noqa: D401 - nothing to release
        """No-op: the simulator holds no external resources."""
        return None
