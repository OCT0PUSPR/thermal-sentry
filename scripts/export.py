#!/usr/bin/env python3
"""Export + INT8-quantise the trained thermal CNN for edge deployment.

Produces, from ``models/thermal_cnn.pt``:

* ``models/thermal_cnn.onnx``       -- FP32 ONNX (laptop/CI ML backend).
* ``models/thermal_cnn_int8.onnx``  -- INT8 (static QDQ) ONNX (runs on the CPU
  execution provider; the default edge artefact when TFLite isn't built).
* ``models/thermal_cnn_int8.tflite``-- INT8 TFLite, *only if* the heavy TF
  tooling (tensorflow + onnx2tf) imports on this machine. On macOS this usually
  does not run; the script says so plainly and the INT8 ONNX is used instead.

Usage:
    python scripts/export.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalsentry.ml.export import export_all  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Export ONNX + INT8 (+ TFLite if possible).")
    p.add_argument("--checkpoint", default="models/thermal_cnn.pt")
    p.add_argument("--onnx", default="models/thermal_cnn.onnx")
    p.add_argument("--int8-onnx", default="models/thermal_cnn_int8.onnx")
    p.add_argument("--tflite", default="models/thermal_cnn_int8.tflite")
    args = p.parse_args()
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}. Run scripts/train.py first.")
        return 1
    export_all(
        checkpoint=args.checkpoint,
        onnx_out=args.onnx,
        int8_onnx_out=args.int8_onnx,
        tflite_out=args.tflite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
