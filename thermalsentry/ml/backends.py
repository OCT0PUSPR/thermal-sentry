"""Inference backends for the learned thermal model.

Two families of backend:

**Crop classifiers** (refine a classical blob's label from its thermal crop):

* :class:`ClassicalBackend` -- pass-through; keeps the heuristic label.
* :class:`OnnxBackend`      -- onnxruntime; cross-platform (laptop, CI, Pi).
* :class:`TFLiteBackend`    -- INT8 TFLite via tflite-runtime (Pi).

**Full-frame detector** (localises *and* classifies from the whole frame using
the two-head CNN's center-heatmap + classification heads):

* :class:`MLDetectorBackend` -- runs the exported ONNX model (FP32 or INT8) and
  returns detections directly. Selectable via ``TS_ML_BACKEND=ml``; the classical
  detector remains the fallback whenever a model can't be loaded.

All third-party imports (torch, onnxruntime, tflite_runtime, tensorflow) are
lazy/guarded so the package imports on any machine. ``build_backend`` /
``build_detector_backend`` return the classical fallback if the requested model
is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .dataset import (
    heatmap_grid_to_frame,
    heatmap_peaks,
    normalize_crop,
)
from .labels import (
    CLASS_NAMES,
    MODEL_IN_H,
    MODEL_IN_W,
    index_to_label,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..config import MLSettings


# ---------------------------------------------------------------------------
# Crop-classifier backends
# ---------------------------------------------------------------------------


@runtime_checkable
class ClassifierBackend(Protocol):
    name: str

    def classify(self, crop: np.ndarray, fallback_label: str) -> Tuple[str, float]:
        ...

    def available(self) -> bool:
        ...


def _prep_crop(crop: np.ndarray, input_size: int) -> np.ndarray:
    """Resize a deg-C crop to (1, 1, S, S) normalised float32."""
    from ..processing.preprocess import _bilinear_numpy

    if crop.shape[0] != input_size or crop.shape[1] != input_size:
        crop = _bilinear_numpy(crop.astype(np.float32), input_size, input_size)
    norm = normalize_crop(crop)
    return norm[None, None, :, :].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class ClassicalBackend:
    """No-op backend: returns the classical heuristic label unchanged."""

    name = "classical"

    def classify(self, crop: np.ndarray, fallback_label: str) -> Tuple[str, float]:
        return fallback_label, 1.0

    def available(self) -> bool:
        return True


class OnnxBackend:
    """onnxruntime crop-classifier backend (uses the classification head)."""

    name = "onnx"

    def __init__(self, model_path: str, input_size: int = 24, min_confidence: float = 0.55):
        self.model_path = model_path
        self.input_size = input_size
        self.min_confidence = min_confidence
        self._session = None
        self._input_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._session = None

    def available(self) -> bool:
        return self._session is not None

    def classify(self, crop: np.ndarray, fallback_label: str) -> Tuple[str, float]:
        if self._session is None:
            return fallback_label, 1.0
        # The full-frame model's crop-path: resize the crop to the model input
        # and read the classification head.
        x = _prep_full(crop, MODEL_IN_H, MODEL_IN_W)
        outs = self._session.run(None, {self._input_name: x})
        logits = np.asarray(outs[0], dtype=np.float32)[0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = index_to_label(idx)
        if conf < self.min_confidence or label == "background":
            return fallback_label, conf
        return label, conf


class TFLiteBackend:
    """INT8 TFLite crop-classifier backend (tflite-runtime on the Pi)."""

    name = "tflite"

    def __init__(self, model_path: str, input_size: int = 24, min_confidence: float = 0.55):
        self.model_path = model_path
        self.input_size = input_size
        self.min_confidence = min_confidence
        self._interp = None
        self._in = None
        self._out = None
        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            return
        try:  # pragma: no cover - tflite usually only on the Pi
            try:
                from tflite_runtime.interpreter import Interpreter
            except Exception:
                from tensorflow.lite import Interpreter  # type: ignore
            self._interp = Interpreter(model_path=self.model_path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()[0]
        except Exception:
            self._interp = None

    def available(self) -> bool:
        return self._interp is not None

    def classify(self, crop: np.ndarray, fallback_label: str) -> Tuple[str, float]:
        if self._interp is None:  # pragma: no cover
            return fallback_label, 1.0
        x = _prep_full(crop, MODEL_IN_H, MODEL_IN_W)
        in_dtype = self._in["dtype"]
        if in_dtype == np.int8:
            scale, zero = self._in["quantization"]
            xq = np.round(x / (scale or 1.0) + zero).astype(np.int8)
            self._interp.set_tensor(self._in["index"], xq)
        else:
            self._interp.set_tensor(self._in["index"], x.astype(in_dtype))
        self._interp.invoke()
        out = self._interp.get_tensor(self._out["index"])[0]
        if self._out["dtype"] == np.int8:
            scale, zero = self._out["quantization"]
            out = (out.astype(np.float32) - zero) * (scale or 1.0)
        probs = _softmax(np.asarray(out, dtype=np.float32))
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = index_to_label(idx)
        if conf < self.min_confidence or label == "background":
            return fallback_label, conf
        return label, conf


# ---------------------------------------------------------------------------
# Full-frame ML detector backend (two-head model)
# ---------------------------------------------------------------------------


def _prep_full(frame: np.ndarray, in_h: int, in_w: int) -> np.ndarray:
    """Resize a deg-C frame to (1, 1, in_h, in_w) normalised float32."""
    from ..processing.preprocess import _bilinear_numpy

    if frame.shape[0] != in_h or frame.shape[1] != in_w:
        frame = _bilinear_numpy(frame.astype(np.float32), in_h, in_w)
    norm = normalize_crop(frame)
    return norm[None, None, :, :].astype(np.float32)


class MLDetectorBackend:
    """Full-frame ONNX detector using the two-head CNN.

    ``detect_frame`` takes a deg-C frame (any HxW; resized to the model input)
    and returns ``(detections, scene_label, scene_conf)`` where each detection is
    a dict with center coordinates in the *passed-in frame's* pixel space.
    """

    name = "ml"

    def __init__(
        self,
        model_path: str,
        min_confidence: float = 0.30,
        peak_threshold: float = 0.30,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.peak_threshold = peak_threshold
        self._session = None
        self._input_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._session = None

    def available(self) -> bool:
        return self._session is not None

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(class_probs, heatmap_prob)`` for a deg-C frame."""
        x = _prep_full(frame, MODEL_IN_H, MODEL_IN_W)
        logits, heatmap = self._session.run(None, {self._input_name: x})
        probs = _softmax(np.asarray(logits, dtype=np.float32)[0])
        heat = 1.0 / (1.0 + np.exp(-np.asarray(heatmap, dtype=np.float32)[0]))
        return probs, heat

    def detect_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[dict], str, float]:
        """Localise + classify warm bodies in ``frame`` (deg C)."""
        if self._session is None:  # pragma: no cover - guarded by available()
            return [], "background", 1.0
        probs, heat = self.infer(frame)
        scene_idx = int(np.argmax(probs))
        scene_label = index_to_label(scene_idx)
        scene_conf = float(probs[scene_idx])

        fh, fw = frame.shape
        peaks = heatmap_peaks(heat, threshold=self.peak_threshold, min_distance=1)
        dets: List[dict] = []
        for (r, c, score) in peaks:
            # Heatmap grid -> raw 24x32 -> the passed-in frame's pixel space.
            raw_r, raw_c = heatmap_grid_to_frame(r, c)
            from .. import FRAME_COLS, FRAME_ROWS

            cy = raw_r / FRAME_ROWS * fh
            cx = raw_c / FRAME_COLS * fw
            peak_temp = float(frame[int(min(fh - 1, max(0, cy))), int(min(fw - 1, max(0, cx)))])
            label = "hotspot" if (scene_label == "hotspot" and score >= 0.5) else "person"
            dets.append(
                {
                    "centroid": (float(cx), float(cy)),
                    "score": float(score),
                    "peak_temp_c": peak_temp,
                    "label": label,
                }
            )
        return dets, scene_label, scene_conf


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_backend(settings: "MLSettings") -> ClassifierBackend:
    """Build the configured *crop classifier* backend (classical fallback)."""
    backend = settings.backend
    if backend == "onnx":
        be = OnnxBackend(settings.onnx_model_path, settings.input_size, settings.min_confidence)
        if be.available():
            return be
        return ClassicalBackend()
    if backend == "tflite":
        be = TFLiteBackend(
            settings.tflite_model_path, settings.input_size, settings.min_confidence
        )
        if be.available():
            return be
        return ClassicalBackend()
    return ClassicalBackend()


def build_detector_backend(settings: "MLSettings"):
    """Build the full-frame ML detector backend, or None if unavailable.

    Returns an :class:`MLDetectorBackend` when ``settings.backend == "ml"`` and a
    model is loadable; otherwise ``None`` (the caller keeps the classical
    detector). Prefers the INT8 ONNX model, falling back to the FP32 ONNX.
    """
    if settings.backend != "ml":
        return None
    candidates = [settings.int8_onnx_model_path, settings.onnx_model_path]
    for path in candidates:
        if path and Path(path).exists():
            be = MLDetectorBackend(
                path,
                min_confidence=settings.min_confidence,
                peak_threshold=settings.heatmap_peak_threshold,
            )
            if be.available():
                return be
    return None


__all__ = [
    "ClassifierBackend",
    "ClassicalBackend",
    "OnnxBackend",
    "TFLiteBackend",
    "MLDetectorBackend",
    "build_backend",
    "build_detector_backend",
    "CLASS_NAMES",
]
