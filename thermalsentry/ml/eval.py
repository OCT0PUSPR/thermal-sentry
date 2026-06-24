"""Evaluate exported thermal models on a held-out synthetic split.

Runs the **FP32 ONNX** and (if present) the **INT8 ONNX** models over a fresh
held-out dataset and reports, for each:

* classification accuracy + macro-F1 (+ per-class accuracy), and
* localisation hit-rate (recall) + precision from the center-heatmap head.

It also reports the **accuracy delta** introduced by INT8 quantisation and the
on-disk size of each artefact -- the numbers quoted in the README/ARCHITECTURE.

Uses onnxruntime only (no torch needed), so it runs anywhere the ONNX backend
runs, including CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .dataset import generate_frame_dataset
from .labels import CLASS_NAMES
from .train import localization_hit_rate, macro_f1, per_class_accuracy


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_onnx(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    loc_tolerance_px: float = 3.0,
    peak_threshold: float = 0.40,
) -> Dict[str, object]:
    """Evaluate one ONNX model on (X, y, H). Returns a metrics dict."""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    logits = np.empty((len(X), len(CLASS_NAMES)), dtype=np.float32)
    heat = np.empty_like(H)
    batch = 256
    for i in range(0, len(X), batch):
        xb = X[i : i + batch].astype(np.float32)
        out_logits, out_heat = sess.run(None, {in_name: xb})
        logits[i : i + batch] = np.asarray(out_logits, dtype=np.float32)
        heat[i : i + batch] = _sigmoid(np.asarray(out_heat, dtype=np.float32))

    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    f1 = macro_f1(y, pred)
    pca = per_class_accuracy(y, pred)
    recall, prec = localization_hit_rate(
        heat, H, tolerance_px=loc_tolerance_px, peak_threshold=peak_threshold
    )
    size_kb = round(Path(model_path).stat().st_size / 1024.0, 1)
    return {
        "model": model_path,
        "size_kb": size_kb,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "per_class_acc": {k: round(v, 4) for k, v in pca.items()},
        "loc_recall": round(recall, 4),
        "loc_precision": round(prec, 4),
    }


def evaluate_all(
    fp32_onnx: str = "models/thermal_cnn.onnx",
    int8_onnx: str = "models/thermal_cnn_int8.onnx",
    n_eval: int = 2000,
    seed: int = 777,
    out_path: Optional[str] = "models/thermal_cnn.eval.json",
    log=print,
) -> Dict[str, object]:
    """Evaluate FP32 (+ INT8 if present) on a fresh held-out split."""
    log(f"generating held-out eval set: {n_eval} frames (seed={seed}) ...")
    X, y, H = generate_frame_dataset(n=n_eval, seed=seed)

    results: Dict[str, object] = {}
    if not Path(fp32_onnx).exists():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32_onnx}. Train + export first.")

    log(f"evaluating FP32  -> {fp32_onnx}")
    fp32 = evaluate_onnx(fp32_onnx, X, y, H)
    results["fp32"] = fp32
    log(
        f"  acc={fp32['accuracy']}  f1={fp32['macro_f1']}  "
        f"loc_recall={fp32['loc_recall']}  loc_prec={fp32['loc_precision']}  "
        f"size={fp32['size_kb']}KB"
    )

    if Path(int8_onnx).exists():
        log(f"evaluating INT8  -> {int8_onnx}")
        int8 = evaluate_onnx(int8_onnx, X, y, H)
        results["int8"] = int8
        log(
            f"  acc={int8['accuracy']}  f1={int8['macro_f1']}  "
            f"loc_recall={int8['loc_recall']}  loc_prec={int8['loc_precision']}  "
            f"size={int8['size_kb']}KB"
        )
        results["accuracy_delta_int8"] = round(int8["accuracy"] - fp32["accuracy"], 4)
        results["f1_delta_int8"] = round(int8["macro_f1"] - fp32["macro_f1"], 4)
        results["size_reduction_pct"] = round(
            100.0 * (1.0 - int8["size_kb"] / fp32["size_kb"]), 1
        )
        log(
            f"INT8 vs FP32: dAcc={results['accuracy_delta_int8']}  "
            f"dF1={results['f1_delta_int8']}  size -{results['size_reduction_pct']}%"
        )
    else:
        log(f"INT8 ONNX not present ({int8_onnx}); skipping INT8 eval.")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2))
        log(f"wrote eval report -> {out_path}")
    return results
