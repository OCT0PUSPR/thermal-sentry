"""From-scratch thermal CNN: model, datasets, training, export, and backends.

The package ships a small two-head convolutional network (built from primitive
conv/BN/ReLU/pool blocks -- no pretrained backbones) that, from an up-scaled
24x32 thermal frame, predicts:

* a **classification** head (background / person / animal / hotspot), and
* a **center-heatmap** detection head localising warm-body centers.

It is trained on a fully-local synthetic+augmented dataset, exported to ONNX and
INT8-quantised, and exposed as a *selectable* detector backend. The classical
threshold+connected-components detector remains the fallback.

All heavy imports (torch, onnxruntime, tflite_runtime, tensorflow) are lazy /
guarded so ``import thermalsentry.ml`` works on any machine, including the
numpy-only CI path.
"""

from __future__ import annotations

from .backends import (
    ClassicalBackend,
    MLDetectorBackend,
    build_backend,
    build_detector_backend,
)
from .labels import CLASS_NAMES, NUM_CLASSES, index_to_label, label_to_index

__all__ = [
    "build_backend",
    "build_detector_backend",
    "ClassicalBackend",
    "MLDetectorBackend",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "label_to_index",
    "index_to_label",
]
