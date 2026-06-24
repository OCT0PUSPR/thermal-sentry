"""Tests for sensor sources: build_source factory, file source, mlx guard."""

from __future__ import annotations

import numpy as np
import pytest

from thermalsentry import FRAME_COLS, FRAME_ROWS
from thermalsentry.config import SourceType, get_settings
from thermalsentry.sensors.base import build_source
from thermalsentry.sensors.file_source import FileThermalSource
from thermalsentry.sensors.mlx90640 import MLX90640Source


def _write_clip(tmp_path, frames):
    path = tmp_path / "clip.npy"
    np.save(path, np.asarray(frames, dtype=np.float32))
    return str(path)


# -- build_source -------------------------------------------------------------


def test_build_source_simulate():
    src = build_source(get_settings(source=SourceType.SIMULATE))
    frame = src.read()
    assert frame.shape == (FRAME_ROWS, FRAME_COLS)
    src.close()


def test_build_source_file(tmp_path):
    clip = np.zeros((2, FRAME_ROWS, FRAME_COLS), dtype=np.float32)
    path = _write_clip(tmp_path, clip)
    src = build_source(get_settings(source=SourceType.FILE, file_path=path, file_loop=False))
    assert isinstance(src, FileThermalSource)
    src.close()


def test_build_source_file_missing_path_raises():
    with pytest.raises(ValueError, match="TS_FILE_PATH"):
        build_source(get_settings(source=SourceType.FILE, file_path=None))


def test_build_source_mlx_raises_without_hardware():
    with pytest.raises(RuntimeError, match="MLX90640 hardware libraries"):
        build_source(get_settings(source=SourceType.MLX90640))


# -- file source --------------------------------------------------------------


def test_file_source_single_frame_promoted(tmp_path):
    single = np.full((FRAME_ROWS, FRAME_COLS), 25.0, dtype=np.float32)
    path = tmp_path / "single.npy"
    np.save(path, single)
    src = FileThermalSource(str(path), loop=False)
    assert len(src) == 1
    np.testing.assert_allclose(src.read(), single)


def test_file_source_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        FileThermalSource("/no/such/file.npy")


def test_file_source_bad_shape_raises(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((3, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        FileThermalSource(str(path))


def test_file_source_loop_and_stopiteration(tmp_path):
    clip = np.stack(
        [np.zeros((FRAME_ROWS, FRAME_COLS), np.float32),
         np.ones((FRAME_ROWS, FRAME_COLS), np.float32)],
        axis=0,
    )
    path = _write_clip(tmp_path, clip)

    # Non-looping: exhausts then raises StopIteration.
    src = FileThermalSource(path, loop=False)
    src.read()
    src.read()
    with pytest.raises(StopIteration):
        src.read()
    src.close()
    assert len(src) == 0  # close empties the buffer

    # Looping: wraps around forever.
    looped = FileThermalSource(path, loop=True)
    seen = [float(looped.read()[0, 0]) for _ in range(5)]
    assert seen == [0.0, 1.0, 0.0, 1.0, 0.0]
    looped.close()


# -- mlx90640 guard -----------------------------------------------------------


def test_mlx_source_construct_raises_without_hardware():
    with pytest.raises(RuntimeError, match="hardware libraries are not available"):
        MLX90640Source()
