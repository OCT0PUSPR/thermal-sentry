"""Shared class taxonomy + geometry for the thermal CNN.

The learned model has two heads:

* a **classification** head over :data:`CLASS_NAMES` describing the dominant
  scene content (``background`` / ``person`` / ``animal`` / ``hotspot``), and an
  explicit ``person-present`` / ``anomaly-present`` reading derived from it, and
* a **center-heatmap** head that localises warm-body centers on a down-sampled
  grid (used by the ML detector backend to place detections).

The model input is the (lightly) up-scaled thermal frame at
:data:`MODEL_IN_H` x :data:`MODEL_IN_W`; the heatmap is produced at a stride of
:data:`HEATMAP_STRIDE` (so :data:`HEATMAP_H` x :data:`HEATMAP_W`).
"""

from __future__ import annotations

from typing import List

# Classification ordering is the model's output index order. Do NOT reorder
# without retraining (it is baked into the exported ONNX/TFLite artefacts).
CLASS_NAMES: List[str] = ["background", "person", "animal", "hotspot"]
NUM_CLASSES = len(CLASS_NAMES)
_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# "Anomaly" classes for the binary anomaly-present readout (a hotspot/overheat
# is the anomaly we care about most). ``person`` is tracked separately.
ANOMALY_CLASSES = {"hotspot"}

# Model input geometry. The raw sensor is 24x32; we up-scale x2 so the tiny CNN
# has enough spatial resolution for the heatmap while staying Pi-cheap.
MODEL_IN_H = 48
MODEL_IN_W = 64
HEATMAP_STRIDE = 4
HEATMAP_H = MODEL_IN_H // HEATMAP_STRIDE  # 12
HEATMAP_W = MODEL_IN_W // HEATMAP_STRIDE  # 16


def label_to_index(label: str) -> int:
    """Map a string label to its class index (unknown -> background)."""
    return _INDEX.get(label, 0)


def index_to_label(index: int) -> str:
    if 0 <= index < NUM_CLASSES:
        return CLASS_NAMES[index]
    return "background"
