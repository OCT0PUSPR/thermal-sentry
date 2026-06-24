"""Tests for the ML export + evaluation pipeline and ONNX-backed backends.

These exercise the FP32 ONNX export, INT8 (static QDQ) ONNX quantisation, the
held-out evaluation, and the ONNX detector/crop backends end-to-end. They train
a *tiny* model on CPU once (module-scoped fixture) so they are self-contained and
do not depend on the committed model artefacts -- but they are skipped cleanly if
torch/onnxruntime are unavailable (the numpy-only CI path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermalsentry.config import MLSettings
from thermalsentry.ml.backends import (
    MLDetectorBackend,
    OnnxBackend,
    build_backend,
    build_detector_backend,
)

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


@pytest.fixture(scope="module")
def trained_onnx(tmp_path_factory):
    """Train a tiny model, export FP32 + INT8 ONNX. Returns (fp32, int8) paths."""
    from thermalsentry.ml.export import export_all
    from thermalsentry.ml.train import TrainConfig, train

    d = tmp_path_factory.mktemp("ml")
    ckpt = str(d / "tiny.pt")
    cfg = TrainConfig(n_train=200, n_val=80, epochs=3, batch_size=32, out=ckpt, device="cpu")
    train(cfg, log=lambda *a, **k: None)

    fp32 = str(d / "tiny.onnx")
    int8 = str(d / "tiny_int8.onnx")
    summary = export_all(
        checkpoint=ckpt,
        onnx_out=fp32,
        int8_onnx_out=int8,
        tflite_out=str(d / "tiny_int8.tflite"),
        log=lambda *a, **k: None,
    )
    assert Path(fp32).exists()
    assert summary["fp32_onnx_kb"] is not None
    return fp32, int8


def test_export_produces_fp32_and_int8(trained_onnx):
    fp32, int8 = trained_onnx
    assert Path(fp32).exists()
    # INT8 static quant should produce a smaller file when available.
    if Path(int8).exists():
        assert Path(int8).stat().st_size < Path(fp32).stat().st_size


def test_evaluate_onnx_fp32(trained_onnx):
    from thermalsentry.ml.dataset import generate_frame_dataset
    from thermalsentry.ml.eval import evaluate_onnx

    fp32, _ = trained_onnx
    X, y, H = generate_frame_dataset(n=120, seed=321)
    res = evaluate_onnx(fp32, X, y, H)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert 0.0 <= res["loc_recall"] <= 1.0
    assert res["size_kb"] > 0
    assert set(res["per_class_acc"]) == {"background", "person", "animal", "hotspot"}


def test_evaluate_all_writes_report(trained_onnx, tmp_path):
    from thermalsentry.ml.eval import evaluate_all

    fp32, int8 = trained_onnx
    out = tmp_path / "eval.json"
    res = evaluate_all(
        fp32_onnx=fp32,
        int8_onnx=int8,
        n_eval=120,
        seed=99,
        out_path=str(out),
        log=lambda *a, **k: None,
    )
    assert out.exists()
    assert "fp32" in res
    if Path(int8).exists():
        assert "int8" in res
        assert "accuracy_delta_int8" in res
        assert "size_reduction_pct" in res


def test_evaluate_all_missing_model_raises(tmp_path):
    from thermalsentry.ml.eval import evaluate_all

    with pytest.raises(FileNotFoundError):
        evaluate_all(fp32_onnx=str(tmp_path / "nope.onnx"), log=lambda *a, **k: None)


def test_ml_detector_backend_on_trained_model(trained_onnx):
    fp32, _ = trained_onnx
    be = MLDetectorBackend(fp32)
    assert be.available()
    # A frame with one clear hot body in the middle.
    frame = np.full((24, 32), 22.0, dtype=np.float32)
    frame[10:14, 14:18] = 34.0
    dets, scene_label, conf = be.detect_frame(frame)
    assert isinstance(dets, list)
    assert scene_label in ("background", "person", "animal", "hotspot")
    assert 0.0 <= conf <= 1.0
    probs, heat = be.infer(frame)
    assert probs.shape == (4,)
    assert heat.ndim == 2


def test_build_detector_backend_prefers_int8(trained_onnx):
    fp32, int8 = trained_onnx
    s = MLSettings(backend="ml", onnx_model_path=fp32, int8_onnx_model_path=int8)
    be = build_detector_backend(s)
    assert be is not None
    # Prefers INT8 when it exists, else FP32.
    expected = int8 if Path(int8).exists() else fp32
    assert be.model_path == expected


def test_onnx_crop_backend_classifies(trained_onnx):
    fp32, _ = trained_onnx
    be = OnnxBackend(fp32, min_confidence=0.0)
    assert be.available()
    # A person-like warm crop.
    crop = np.full((24, 32), 22.0, dtype=np.float32)
    crop[8:18, 12:20] = 34.0
    label, conf = be.classify(crop, fallback_label="object")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0


def test_build_backend_onnx_returns_onnx_when_present(trained_onnx):
    fp32, _ = trained_onnx
    be = build_backend(MLSettings(backend="onnx", onnx_model_path=fp32))
    assert be.name == "onnx"
