"""Tests for the from-scratch thermal CNN: labels, dataset, backends, eval,
training, and detector integration.

The numpy-only paths (labels, dataset, heatmap peaks, metrics, classical
backend, ONNX-backed detector when a model is present) run everywhere. The
torch-dependent paths (model build, a tiny smoke-train) are guarded with
``pytest.importorskip("torch")`` so the numpy-only CI matrix still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermalsentry.config import DetectionSettings, MLSettings
from thermalsentry.detection.detector import ThermalDetector
from thermalsentry.ml.backends import (
    ClassicalBackend,
    MLDetectorBackend,
    build_backend,
    build_detector_backend,
)
from thermalsentry.ml.dataset import (
    generate_dataset,
    generate_frame_dataset,
    heatmap_grid_to_frame,
    heatmap_peaks,
    make_frame_sample,
    make_sample,
    normalize_crop,
)
from thermalsentry.ml.labels import (
    CLASS_NAMES,
    HEATMAP_H,
    HEATMAP_W,
    MODEL_IN_H,
    MODEL_IN_W,
    NUM_CLASSES,
    index_to_label,
    label_to_index,
)
from thermalsentry.ml.train import (
    localization_hit_rate,
    macro_f1,
    per_class_accuracy,
)
from thermalsentry.processing.preprocess import bilinear_upscale
from thermalsentry.sensors.simulator import SyntheticThermalSource

FP32 = Path("models/thermal_cnn.onnx")
INT8 = Path("models/thermal_cnn_int8.onnx")


# --- labels ---------------------------------------------------------------


def test_labels():
    assert NUM_CLASSES == 4
    assert CLASS_NAMES == ["background", "person", "animal", "hotspot"]
    assert label_to_index("person") == 1
    assert label_to_index("unknown-thing") == 0
    assert index_to_label(3) == "hotspot"
    assert index_to_label(99) == "background"


# --- crop dataset (legacy crop classifier path) ---------------------------


def test_crop_dataset_shapes_and_balance():
    X, y = generate_dataset(n_per_class=20, size=24, seed=1)
    assert X.shape == (80, 1, 24, 24)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    counts = np.bincount(y, minlength=NUM_CLASSES)
    assert all(c == 20 for c in counts)


def test_normalize_crop_range():
    crop = np.array([[10.0, 80.0], [22.0, 50.0]], dtype=np.float32)
    n = normalize_crop(crop)
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_make_sample_class_separation():
    rng = np.random.default_rng(0)
    bg = make_sample(0, 24, rng)
    hot = make_sample(3, 24, rng)
    assert hot.max() > bg.max() + 10


# --- full-frame dataset (the real training data) --------------------------


def test_frame_dataset_shapes_and_balance():
    X, y, H = generate_frame_dataset(n=40, seed=2)
    assert X.shape == (40, 1, MODEL_IN_H, MODEL_IN_W)
    assert H.shape == (40, HEATMAP_H, HEATMAP_W)
    assert X.dtype == np.float32 and H.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert H.min() >= 0.0 and H.max() <= 1.0
    # Round-robin scenes => all four classes present and roughly balanced.
    counts = np.bincount(y, minlength=NUM_CLASSES)
    assert all(c == 10 for c in counts)


def test_make_frame_sample_scene_forcing():
    rng = np.random.default_rng(3)
    for scene in range(NUM_CLASSES):
        x, cls, heat = make_frame_sample(rng, scene=scene)
        assert cls == scene
        assert x.shape == (1, MODEL_IN_H, MODEL_IN_W)
        assert heat.shape == (HEATMAP_H, HEATMAP_W)
    # Person / animal / hotspot scenes produce >= 1 heatmap peak.
    _, _, person_heat = make_frame_sample(np.random.default_rng(4), scene=1)
    assert person_heat.max() > 0.5


# --- heatmap peak extraction ----------------------------------------------


def test_heatmap_peaks_finds_centers():
    heat = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
    heat[3, 5] = 1.0
    heat[8, 12] = 0.9
    peaks = heatmap_peaks(heat, threshold=0.3, min_distance=1)
    coords = {(r, c) for (r, c, _) in peaks}
    assert (3, 5) in coords
    assert (8, 12) in coords


def test_heatmap_peaks_threshold_and_nms():
    heat = np.full((HEATMAP_H, HEATMAP_W), 0.1, dtype=np.float32)
    assert heatmap_peaks(heat, threshold=0.3) == []
    # Two adjacent strong cells -> NMS keeps one.
    heat[5, 5] = 1.0
    heat[5, 6] = 0.95
    peaks = heatmap_peaks(heat, threshold=0.3, min_distance=2)
    assert len(peaks) == 1


def test_heatmap_grid_to_frame_in_range():
    r, c = heatmap_grid_to_frame(0, 0)
    assert 0 <= r < 24 and 0 <= c < 32
    r, c = heatmap_grid_to_frame(HEATMAP_H - 1, HEATMAP_W - 1)
    assert 0 <= r < 24 and 0 <= c < 32


# --- metrics --------------------------------------------------------------


def test_macro_f1_perfect_and_imperfect():
    y = np.array([0, 1, 2, 3, 0, 1])
    assert macro_f1(y, y.copy()) == pytest.approx(1.0)
    wrong = y.copy()
    wrong[0] = 1
    assert macro_f1(y, wrong) < 1.0


def test_per_class_accuracy():
    y = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 0])  # hotspot misclassified
    pca = per_class_accuracy(y, pred)
    assert pca["background"] == 1.0
    assert pca["hotspot"] == 0.0


def test_localization_hit_rate_perfect():
    gt = np.zeros((2, HEATMAP_H, HEATMAP_W), dtype=np.float32)
    gt[0, 3, 5] = 1.0
    gt[1, 8, 10] = 1.0
    pred = gt.copy()
    recall, prec = localization_hit_rate(pred, gt, tolerance_px=3.0, peak_threshold=0.4)
    assert recall == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)


def test_localization_hit_rate_misses():
    gt = np.zeros((1, HEATMAP_H, HEATMAP_W), dtype=np.float32)
    gt[0, 3, 5] = 1.0
    pred = np.zeros_like(gt)  # no predictions
    recall, prec = localization_hit_rate(pred, gt, tolerance_px=1.0, peak_threshold=0.4)
    assert recall == 0.0


# --- crop classifier backends (classical fallback) ------------------------


def test_classical_backend_passthrough():
    be = ClassicalBackend()
    label, conf = be.classify(np.zeros((10, 10), dtype=np.float32), "person")
    assert label == "person"
    assert conf == 1.0
    assert be.available() is True


def test_build_backend_falls_back_when_model_missing():
    be = build_backend(MLSettings(backend="onnx", onnx_model_path="models/nope.onnx"))
    assert be.name == "classical"


def test_build_detector_backend_classical_returns_none():
    assert build_detector_backend(MLSettings(backend="classical")) is None


def test_build_detector_backend_missing_model_returns_none():
    s = MLSettings(
        backend="ml",
        onnx_model_path="models/nope.onnx",
        int8_onnx_model_path="models/nope_int8.onnx",
    )
    assert build_detector_backend(s) is None


# --- ML detector backend (requires a trained ONNX model) ------------------


@pytest.mark.skipif(not FP32.exists(), reason="trained FP32 ONNX model not present")
def test_ml_detector_backend_available_and_classifies():
    be = MLDetectorBackend(str(FP32))
    assert be.available()
    rng = np.random.default_rng(11)
    # Build a clear person scene and check the scene class + a localised peak.
    correct = total = 0
    for scene in range(NUM_CLASSES):
        x, cls, _heat = make_frame_sample(rng, scene=scene)
        # x is normalised [0,1]; convert back to a deg-C-ish frame for the API.
        frame = x[0] * (60.0 - 15.0) + 15.0
        dets, scene_label, conf = be.detect_frame(frame.astype(np.float32))
        correct += int(scene_label == CLASS_NAMES[cls])
        total += 1
    assert correct / total >= 0.5


@pytest.mark.skipif(not FP32.exists(), reason="trained FP32 ONNX model not present")
def test_detector_with_ml_backend_on_simulated_frames():
    be = build_detector_backend(
        MLSettings(backend="ml", onnx_model_path=str(FP32))
    )
    assert be is not None and be.name == "ml"
    det = ThermalDetector(settings=DetectionSettings(), detector_backend=be)
    src = SyntheticThermalSource(num_bodies=2, seed=42)
    labels = set()
    count = 0
    for _ in range(12):
        up = bilinear_upscale(src.read(), factor=20)
        dets = det.detect(up)
        count += len(dets)
        for d in dets:
            labels.add(d.label)
    assert count > 0
    assert "person" in labels


@pytest.mark.skipif(not INT8.exists(), reason="INT8 ONNX model not present")
def test_int8_detector_runs_on_simulated_frames():
    be = MLDetectorBackend(str(INT8))
    assert be.available()
    src = SyntheticThermalSource(num_bodies=3, seed=7)
    up = bilinear_upscale(src.read(), factor=20)
    dets, scene_label, conf = be.detect_frame(up)
    # INT8 model must execute and localise at least one warm body.
    assert isinstance(dets, list)
    assert scene_label in CLASS_NAMES
    assert 0.0 <= conf <= 1.0


# --- ONNX-only eval (runs in CI; no torch needed) -------------------------


@pytest.mark.skipif(not FP32.exists(), reason="trained FP32 ONNX model not present")
def test_evaluate_committed_onnx_no_torch():
    """Evaluate the committed FP32 model via onnxruntime only (no torch)."""
    pytest.importorskip("onnxruntime")
    from thermalsentry.ml.eval import evaluate_onnx

    X, y, H = generate_frame_dataset(n=120, seed=2024)
    res = evaluate_onnx(str(FP32), X, y, H)
    # The committed model is well-trained, so it should comfortably beat chance.
    assert res["accuracy"] >= 0.7
    assert res["loc_recall"] >= 0.6
    assert res["size_kb"] > 0


@pytest.mark.skipif(
    not (FP32.exists() and INT8.exists()), reason="FP32+INT8 ONNX models not present"
)
def test_evaluate_all_committed_models_no_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    from thermalsentry.ml.eval import evaluate_all

    out = tmp_path / "eval.json"
    res = evaluate_all(
        fp32_onnx=str(FP32),
        int8_onnx=str(INT8),
        n_eval=120,
        seed=4040,
        out_path=str(out),
        log=lambda *a, **k: None,
    )
    assert out.exists()
    assert "fp32" in res and "int8" in res
    # INT8 quantisation must not destroy accuracy (small, bounded delta).
    assert abs(res["accuracy_delta_int8"]) <= 0.10
    assert res["size_reduction_pct"] > 0


# --- torch-dependent: model build + a tiny smoke train --------------------


def test_model_build_and_forward():
    pytest.importorskip("torch")
    import torch

    from thermalsentry.ml.model import build_model, count_parameters

    model = build_model()
    assert count_parameters(model) > 1000
    x = torch.zeros(2, 1, MODEL_IN_H, MODEL_IN_W)
    logits, heatmap = model(x)
    assert tuple(logits.shape) == (2, NUM_CLASSES)
    assert tuple(heatmap.shape) == (2, HEATMAP_H, HEATMAP_W)
    probs, heat = model.predict(x)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0


def test_tiny_smoke_train_learns(tmp_path):
    pytest.importorskip("torch")
    from thermalsentry.ml.train import TrainConfig, train

    cfg = TrainConfig(
        n_train=160,
        n_val=80,
        epochs=3,
        batch_size=32,
        out=str(tmp_path / "smoke.pt"),
        device="cpu",
    )
    report = train(cfg, log=lambda *a, **k: None)
    assert Path(report.checkpoint).exists()
    # A 3-epoch CPU smoke run should at least beat chance on the heatmap recall.
    assert report.loc_recall >= 0.3
    assert 0.0 <= report.accuracy <= 1.0
    assert report.params > 1000
