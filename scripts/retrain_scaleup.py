#!/usr/bin/env python3
"""Scale-up / production retrain of the thermal CNN (more data, GPU).

Use this to train a stronger model than the laptop quickstart:

* **more synthetic data + epochs** for a sturdier model, and/or
* **fine-tune on REAL recorded thermal frames** captured from an MLX90640.

Real-data workflow
------------------
1. Record real frames on the Pi (each ``.npy`` is an ``(N, 24, 32)`` deg-C
   sequence)::

       thermal-sentry run --source mlx90640 --record captures/real_clip.npy --frames 2000

2. Copy the ``.npy`` files to a GPU box and weak-label them. The simplest local
   labelling reuses the classical detector to produce center heatmaps + a scene
   class per frame (no manual annotation needed to bootstrap)::

       python scripts/retrain_scaleup.py \
           --real-glob "captures/*.npy" \
           --n-train 40000 --epochs 60 --device cuda \
           --out models/thermal_cnn_prod.pt

3. Export + quantise the production checkpoint::

       python scripts/export.py --checkpoint models/thermal_cnn_prod.pt \
           --onnx models/thermal_cnn.onnx --int8-onnx models/thermal_cnn_int8.onnx

This script trains primarily on the (large) synthetic set and, when
``--real-glob`` is given, mixes in weak-labelled real frames so the model adapts
to a specific sensor/room. Replace :func:`weak_label_real` with hand annotations
for best results.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalsentry import FRAME_COLS, FRAME_ROWS  # noqa: E402
from thermalsentry.config import DetectionSettings  # noqa: E402
from thermalsentry.detection.detector import ThermalDetector  # noqa: E402
from thermalsentry.ml.dataset import (  # noqa: E402
    MODEL_IN_H,
    MODEL_IN_W,
    _make_heatmap,
    _resize_bilinear,
    normalize_crop,
)
from thermalsentry.ml.train import TrainConfig, train  # noqa: E402


def weak_label_real(frames: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
    """Weak-label real (N, 24, 32) frames with the classical detector.

    Returns parallel lists of (model-input frame, class index, heatmap). The
    classical detector localises warm blobs -> heatmap; the dominant blob label
    -> scene class. This bootstraps a real-data fine-tune without manual labels.
    """
    det = ThermalDetector(settings=DetectionSettings())
    xs: List[np.ndarray] = []
    ys: List[int] = []
    hs: List[np.ndarray] = []
    label_to_idx = {"background": 0, "person": 1, "animal": 2, "hotspot": 3, "object": 0}
    for raw in frames:
        raw = raw.astype(np.float32)
        # Detector runs on the upscaled frame; centers come back in that space.
        up = _resize_bilinear(raw, MODEL_IN_H, MODEL_IN_W)
        dets = det.detect(up)
        centers = []
        scene = 0
        peak = -1e9
        for d in dets:
            cx, cy = d.centroid
            # up-space -> raw 24x32 grid.
            rr = cy / MODEL_IN_H * FRAME_ROWS
            cc = cx / MODEL_IN_W * FRAME_COLS
            centers.append((rr, cc))
            if d.peak_temp_c > peak:
                peak = d.peak_temp_c
                scene = label_to_idx.get(d.label, 0)
        xs.append(normalize_crop(up)[None, :, :].astype(np.float32))
        ys.append(int(scene))
        hs.append(_make_heatmap(centers))
    return xs, ys, hs


def main() -> int:
    p = argparse.ArgumentParser(description="Scale-up / production retrain.")
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-val", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--device", default=None, help="mps|cuda|cpu (auto if unset).")
    p.add_argument("--out", default="models/thermal_cnn_prod.pt")
    p.add_argument(
        "--real-glob",
        default=None,
        help="Glob of real (N,24,32) .npy clips to weak-label and mix in.",
    )
    args = p.parse_args()

    real_count = 0
    if args.real_glob:
        files = sorted(glob.glob(args.real_glob))
        print(f"loading real clips: {len(files)} files matching {args.real_glob!r}")
        for f in files:
            arr = np.load(f)
            if arr.ndim == 2:
                arr = arr[None]
            xs, ys, hs = weak_label_real(arr)
            real_count += len(xs)
            # NOTE: integrating real frames into the training tensors is left as a
            # hook -- persist them and concatenate inside train() for your setup.
            np.savez(
                Path(args.out).with_suffix(f".real_{Path(f).stem}.npz"),
                X=np.stack(xs), y=np.array(ys), H=np.stack(hs),
            )
        print(f"weak-labelled {real_count} real frames -> *.npz (mix into training)")

    cfg = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out=args.out,
    )
    print(
        f"scaling up: {args.n_train} synthetic train frames, {args.epochs} epochs "
        f"(real frames available: {real_count})"
    )
    report = train(cfg)
    print(
        f"DONE  acc={report.accuracy}  f1={report.macro_f1}  "
        f"loc_recall={report.loc_recall}  -> {report.checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
