#!/usr/bin/env python3
"""Train the from-scratch two-head thermal CNN on synthetic data.

Trains :class:`thermalsentry.ml.model.ThermalNet` (classification +
center-heatmap detection heads) on the local synthetic+augmented dataset, then
saves a checkpoint + a JSON report with REAL held-out metrics (classification
accuracy / macro-F1 / per-class accuracy and localisation recall/precision).

Device is auto-selected MPS > CUDA > CPU. A default run is ~30-40 min-safe.

Examples
--------
    python scripts/train.py                       # full default run
    python scripts/train.py --epochs 12 --n-train 3000
    python scripts/train.py --device cpu

Then export + quantise for the edge:
    python scripts/export.py
    python scripts/eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermalsentry.ml.train import TrainConfig, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train the two-head thermal CNN.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-val", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--heatmap-weight", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/thermal_cnn.pt")
    p.add_argument("--device", default=None, help="Force device (mps|cuda|cpu).")
    args = p.parse_args()

    cfg = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        heatmap_weight=args.heatmap_weight,
        seed=args.seed,
        out=args.out,
        device=args.device,
    )
    report = train(cfg)
    # Non-zero exit if the model clearly failed to learn (guards automation).
    ok = report.macro_f1 >= 0.80 and report.loc_recall >= 0.70
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
