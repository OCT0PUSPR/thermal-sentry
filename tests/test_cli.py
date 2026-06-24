"""Tests for the command-line interface."""

from __future__ import annotations

import numpy as np
import pytest

from thermalsentry import cli
from thermalsentry.config import SourceType


def test_build_parser_requires_subcommand():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_settings_from_args_simulate_shorthand():
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--simulate", "--fps", "12"])
    settings = cli._settings_from_args(args)
    assert settings.source == SourceType.SIMULATE
    assert settings.fps == 12.0


def test_settings_from_args_explicit_source_and_overrides():
    parser = cli._build_parser()
    args = parser.parse_args(
        ["run", "--source", "simulate", "--bodies", "3", "--seed", "7",
         "--host", "127.0.0.1", "--port", "9000"]
    )
    settings = cli._settings_from_args(args)
    assert settings.sim_num_bodies == 3
    assert settings.sim_seed == 7
    assert settings.web_host == "127.0.0.1"
    assert settings.web_port == 9000


def test_settings_from_args_file_source(tmp_path):
    clip = tmp_path / "clip.npy"
    np.save(clip, np.zeros((2, 24, 32), dtype=np.float32))
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--source", "file", "--file", str(clip)])
    settings = cli._settings_from_args(args)
    assert settings.source == SourceType.FILE
    assert settings.file_path == str(clip)


def test_main_run_headless_processes_frames(capsys):
    rc = cli.main(["run", "--simulate", "--frames", "3", "--fps", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Processed 3 frames" in out


def test_main_run_headless_file_exhaustion(tmp_path, capsys):
    clip = tmp_path / "clip.npy"
    np.save(clip, np.zeros((2, 24, 32), dtype=np.float32))
    # No --frames cap; the file source exhausts (loop default True in settings,
    # but file_loop defaults True so it would loop -> cap with --frames instead).
    rc = cli.main(["run", "--source", "file", "--file", str(clip), "--frames", "4", "--fps", "50"])
    assert rc == 0
    assert "Processed 4 frames" in capsys.readouterr().out


def test_main_serve_dispatch_monkeypatched(monkeypatch):
    called = {}

    def fake_run_web(settings):
        called["settings"] = settings
        return 0

    monkeypatch.setattr(cli, "_run_web", fake_run_web)
    rc = cli.main(["serve", "--source", "simulate", "--port", "8123"])
    assert rc == 0
    assert called["settings"].web_port == 8123


def test_main_run_web_dispatch_monkeypatched(monkeypatch):
    called = {}

    def fake_run_web(settings):
        called["hit"] = True
        return 0

    monkeypatch.setattr(cli, "_run_web", fake_run_web)
    rc = cli.main(["run", "--simulate", "--web"])
    assert rc == 0
    assert called["hit"] is True


def test_run_headless_stopiteration_breaks(tmp_path, capsys):
    # A short non-looping file source exhausts and the headless loop stops.
    clip = tmp_path / "clip.npy"
    np.save(clip, np.zeros((2, 24, 32), dtype=np.float32))
    from thermalsentry.config import get_settings

    settings = get_settings(source=SourceType.FILE, file_path=str(clip), file_loop=False, fps=50.0)
    rc = cli._run_headless(settings, record_path=None, max_frames=0)
    assert rc == 0
    assert "Processed 2 frames" in capsys.readouterr().out
