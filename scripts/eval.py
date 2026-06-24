#!/usr/bin/env python3
"""Evaluate the exported FP32 + INT8 thermal models on a held-out split.

Reports classification accuracy / macro-F1 / per-class accuracy and localisation
recall/precision for each model, plus the INT8-vs-FP32 accuracy delta and the
size reduction. Uses onnxruntime only (no torch), so it runs in CI too.

Usage:
    python scripts/eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalsentry.ml.eval import evaluate_all  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate FP32 + INT8 thermal models.")
    p.add_argument("--onnx", default="models/thermal_cnn.onnx")
    p.add_argument("--int8-onnx", default="models/thermal_cnn_int8.onnx")
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", default="models/thermal_cnn.eval.json")
    args = p.parse_args()
    if not Path(args.onnx).exists():
        print(f"FP32 ONNX not found: {args.onnx}. Run scripts/train.py + scripts/export.py first.")
        return 1
    res = evaluate_all(
        fp32_onnx=args.onnx,
        int8_onnx=args.int8_onnx,
        n_eval=args.n_eval,
        seed=args.seed,
        out_path=args.out,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
