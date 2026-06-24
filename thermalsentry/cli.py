"""Command-line interface for thermal-sentry.

Examples
--------
Run the full pipeline against the simulator with the live dashboard::

    thermal-sentry run --source simulate --web

Run headless against the MLX90640 and POST alerts to a webhook::

    TS_ALERTS_WEBHOOK_URL=https://hooks... thermal-sentry run --source mlx90640

Record 200 simulated frames to a .npy for later replay::

    thermal-sentry run --source simulate --record captures/clip.npy --frames 200

Replay a recording::

    thermal-sentry run --source file --file captures/clip.npy --web

Serve the dashboard (alias for ``run --web``)::

    thermal-sentry serve
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Optional

from .config import SourceType, get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-sentry",
        description="Edge thermal AI: detect people / heat / anomalies on a Pi.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the detection pipeline.")
    run.add_argument(
        "--source",
        choices=[s.value for s in SourceType],
        default=None,
        help="Frame source (default from env, else simulate).",
    )
    run.add_argument("--simulate", action="store_true", help="Shorthand for --source simulate.")
    run.add_argument("--web", action="store_true", help="Serve the live web dashboard.")
    run.add_argument("--host", default=None, help="Dashboard host (default 0.0.0.0).")
    run.add_argument("--port", type=int, default=None, help="Dashboard port (default 8000).")
    run.add_argument("--fps", type=float, default=None, help="Target FPS.")
    run.add_argument("--file", default=None, help="Path to .npy recording (source=file).")
    run.add_argument("--record", default=None, help="Record raw frames to this .npy.")
    run.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Headless: stop after N frames (0 = run forever).",
    )
    run.add_argument("--bodies", type=int, default=None, help="Simulator: number of bodies.")
    run.add_argument("--seed", type=int, default=None, help="Simulator: RNG seed.")

    serve = sub.add_parser("serve", help="Serve the dashboard (alias for run --web).")
    serve.add_argument("--source", choices=[s.value for s in SourceType], default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    return parser


def _settings_from_args(args: argparse.Namespace):
    overrides = {}
    source = getattr(args, "source", None)
    if getattr(args, "simulate", False):
        source = SourceType.SIMULATE.value
    if source:
        overrides["source"] = source
    if getattr(args, "fps", None) is not None:
        overrides["fps"] = args.fps
    if getattr(args, "file", None):
        overrides["file_path"] = args.file
    if getattr(args, "host", None):
        overrides["web_host"] = args.host
    if getattr(args, "port", None) is not None:
        overrides["web_port"] = args.port
    if getattr(args, "bodies", None) is not None:
        overrides["sim_num_bodies"] = args.bodies
    if getattr(args, "seed", None) is not None:
        overrides["sim_seed"] = args.seed
    return get_settings(**overrides)


def _run_web(settings) -> int:
    import uvicorn

    from .observability import configure_logging
    from .web.server import create_app

    configure_logging(settings.observability)
    app = create_app(settings=settings, autostart=True)
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level=settings.observability.log_level.lower(),
    )
    return 0


def _run_headless(settings, record_path: Optional[str], max_frames: int) -> int:
    from .app import ThermalSentryApp

    runtime = ThermalSentryApp(settings=settings, record_path=record_path)
    logging.info("Running headless (source=%s, frames=%s)", settings.source.value, max_frames or "inf")
    target_dt = 1.0 / max(0.1, settings.fps)
    count = 0
    try:
        while True:
            t0 = time.monotonic()
            try:
                runtime.process_once()
            except StopIteration:
                break
            count += 1
            if max_frames and count >= max_frames:
                break
            sleep = target_dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.")
    finally:
        runtime.stop()
    print(f"Processed {count} frames.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        settings = _settings_from_args(args)
        return _run_web(settings)

    if args.command == "run":
        settings = _settings_from_args(args)
        record_path = getattr(args, "record", None)
        if args.web:
            return _run_web(settings)
        return _run_headless(settings, record_path=record_path, max_frames=args.frames)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
