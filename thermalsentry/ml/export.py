"""Export + quantise the trained thermal CNN for edge deployment.

Pipeline:

1. Load the trained checkpoint (``models/thermal_cnn.pt``).
2. Export FP32 **ONNX** (cross-platform; the laptop/CI ML backend uses this).
3. Produce an **INT8** model. We try, in order:
     a. ONNX Runtime **dynamic** INT8 quantisation (always available here) --
        ``models/thermal_cnn_int8.onnx``.
     b. PyTorch **static** INT8 quantisation as a size/sanity reference.
     c. **TFLite** INT8 via tensorflow (only if the heavy TF tooling imports on
        this machine) -- ``models/thermal_cnn_int8.tflite``.

The function reports exactly which artefacts were produced + their sizes so the
README/ARCHITECTURE numbers are never fabricated.

``torch`` / ``onnx`` are imported lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .dataset import generate_frame_dataset
from .labels import MODEL_IN_H, MODEL_IN_W, NUM_CLASSES


def load_model(checkpoint: str):
    """Load :class:`ThermalNet` weights from a checkpoint, in eval mode."""
    import torch

    from .model import build_model

    model = build_model(NUM_CLASSES)
    # weights_only=True: our checkpoints contain only tensors + primitive metadata
    # (no pickled objects), so this is both safe and sufficient.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(model, out_path: str, opset: int = 13) -> str:
    """Export the two-head model to FP32 ONNX. Returns the path."""
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, MODEL_IN_H, MODEL_IN_W)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits", "heatmap"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "heatmap": {0: "batch"},
        },
        opset_version=opset,
    )
    return str(out)


def quantize_onnx_static(fp32_onnx: str, out_path: str, n_calib: int = 256) -> Optional[str]:
    """Static INT8 (QDQ) quantise via onnxruntime with a calibration reader.

    Static quantisation produces ``QLinearConv``/``QuantizeLinear`` ops that the
    CPU execution provider can actually run (unlike dynamic quant, which emits
    ``ConvInteger`` ops with no CPU kernel). Calibration uses the same synthetic
    distribution the model trained on, so the INT8 activation ranges are correct.
    Returns the output path, or None if the tooling is unavailable.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception:
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    Xc, _, _ = generate_frame_dataset(n=n_calib, seed=4242)

    class _Reader(CalibrationDataReader):
        def __init__(self, data, input_name):
            self._data = data
            self._name = input_name
            self._i = 0

        def get_next(self):
            if self._i >= len(self._data):
                return None
            x = self._data[self._i : self._i + 1].astype(np.float32)
            self._i += 1
            return {self._name: x}

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        reader = _Reader(Xc, input_name)
        # Pre-process the float model (shape inference + folding) for cleaner quant.
        prepped = str(out.with_name(out.stem + ".prep.onnx"))
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            quant_pre_process(fp32_onnx, prepped)
            src_model = prepped
        except Exception:
            src_model = fp32_onnx
        quantize_static(
            model_input=src_model,
            model_output=str(out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        if src_model == prepped and Path(prepped).exists():
            Path(prepped).unlink()
    except Exception:
        return None
    return str(out)


def quantize_torch_static(model) -> Optional[float]:
    """PyTorch static INT8 quant as a size reference. Returns size KB or None."""
    try:
        import tempfile

        import torch
        import torch.ao.quantization as tq

        m = model
        m.eval()
        m.qconfig = tq.get_default_qconfig("fbgemm")
        prepared = tq.prepare(m, inplace=False)
        # Calibrate on a small representative batch.
        X, _, _ = generate_frame_dataset(n=128, seed=123)
        with torch.no_grad():
            prepared(torch.from_numpy(X))
        quantized = tq.convert(prepared, inplace=False)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as fh:
            torch.save(quantized.state_dict(), fh.name)
            return Path(fh.name).stat().st_size / 1024.0
    except Exception:
        return None


def try_export_tflite(fp32_onnx: str, out_path: str, n_calib: int = 128) -> Optional[str]:
    """Best-effort INT8 TFLite export (heavy TF tooling). Returns path or None."""
    try:
        import onnx2tf  # noqa: F401
        import tensorflow as tf  # type: ignore
    except Exception:
        return None
    try:
        saved_dir = str(Path(out_path).with_suffix("")) + "_savedmodel"
        import onnx2tf

        onnx2tf.convert(
            input_onnx_file_path=fp32_onnx,
            output_folder_path=saved_dir,
            non_verbose=True,
            output_signaturedefs=True,
        )
        X, _, _ = generate_frame_dataset(n=n_calib, seed=7)

        def rep():
            for i in range(len(X)):
                yield [X[i : i + 1].astype(np.float32)]

        conv = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
        out = Path(out_path)
        out.write_bytes(blob)
        return str(out)
    except Exception:
        return None


def _kb(path: Optional[str]) -> Optional[float]:
    if path and Path(path).exists():
        return round(Path(path).stat().st_size / 1024.0, 1)
    return None


def export_all(
    checkpoint: str = "models/thermal_cnn.pt",
    onnx_out: str = "models/thermal_cnn.onnx",
    int8_onnx_out: str = "models/thermal_cnn_int8.onnx",
    tflite_out: str = "models/thermal_cnn_int8.tflite",
    log=print,
) -> Dict[str, object]:
    """Export FP32 ONNX + INT8 ONNX (+ optional TFLite). Returns a summary dict."""
    model = load_model(checkpoint)

    log("exporting FP32 ONNX ...")
    fp32 = export_onnx(model, onnx_out)
    log(f"  -> {fp32} ({_kb(fp32)} KB)")

    log("quantising INT8 ONNX (static QDQ) ...")
    int8 = quantize_onnx_static(fp32, int8_onnx_out)
    if int8:
        log(f"  -> {int8} ({_kb(int8)} KB)")
    else:
        log("  INT8-ONNX quantisation unavailable on this machine.")

    log("PyTorch static INT8 (size reference) ...")
    torch_int8_kb = quantize_torch_static(model)
    if torch_int8_kb is not None:
        log(f"  torch INT8 state_dict ~ {round(torch_int8_kb, 1)} KB")

    log("attempting TFLite INT8 export (heavy; optional) ...")
    tflite = try_export_tflite(fp32, tflite_out)
    if tflite:
        log(f"  -> {tflite} ({_kb(tflite)} KB)")
    else:
        log("  TFLite tooling did not run on this machine (expected on macOS).")

    summary = {
        "fp32_onnx": fp32,
        "fp32_onnx_kb": _kb(fp32),
        "int8_onnx": int8,
        "int8_onnx_kb": _kb(int8),
        "torch_static_int8_kb": round(torch_int8_kb, 1) if torch_int8_kb else None,
        "tflite_int8": tflite,
        "tflite_int8_kb": _kb(tflite),
    }
    summary_path = Path(onnx_out).with_suffix(".export.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote export summary -> {summary_path}")
    return summary
