"""From-scratch thermal CNN (PyTorch) with two heads.

Architecture (all built from primitive conv / BN / ReLU / pool blocks -- no
pretrained backbones, no model-zoo wrappers; ``torch`` + ``numpy`` only):

    input  (B, 1, 48, 64)   up-scaled, normalised thermal frame
      |
    ConvBlock 1 -> 16ch, /2  (24x32)
    ConvBlock 2 -> 32ch, /2  (12x16)     <-- heatmap resolution (stride 4)
    ConvBlock 3 -> 48ch       (12x16)    shared feature map
      |-------------------------------|
      |                               |
    classification head            detection head
    GAP -> FC(48->48) -> FC(48->4) 1x1 conv(48->1) -> center heatmap (12x16)

The classification head predicts the dominant scene class; the detection head
predicts a per-cell "is there a warm-body center here" heatmap. The two share
the backbone so the whole thing is a few-tens-of-thousands of parameters: it
trains in minutes on a laptop, exports to a sub-megabyte ONNX/TFLite model, and
runs in real time on a Raspberry Pi.

``torch`` is imported lazily inside :func:`build_model` so the package imports
on machines without torch (the numpy-only CI path never calls this).
"""

from __future__ import annotations

from .labels import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES):
    """Construct :class:`ThermalNet`. Requires torch (clear error otherwise)."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - torch is a train-only dep
        raise RuntimeError(
            "PyTorch is required to build/train the model. Install training deps "
            "with `pip install -r requirements-train.txt` (CPU build is fine)."
        ) from exc

    class ConvBlock(nn.Module):
        """conv -> BN -> ReLU, optionally followed by a 2x2 max-pool.

        Hand-rolled (not a library block) so the architecture is fully ours.
        """

        def __init__(self, c_in: int, c_out: int, pool: bool = True) -> None:
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
            self.act = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(2) if pool else None

        def forward(self, x):
            x = self.act(self.bn(self.conv(x)))
            if self.pool is not None:
                x = self.pool(x)
            return x

    class ThermalNet(nn.Module):
        """Two-head thermal CNN: classification + center-heatmap detection."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            # Backbone: 48x64 -> /2 -> /2 = 12x16 at the heatmap stride (4).
            self.block1 = ConvBlock(1, 16, pool=True)   # 48x64 -> 24x32
            self.block2 = ConvBlock(16, 32, pool=True)  # 24x32 -> 12x16
            self.block3 = ConvBlock(32, 48, pool=False)  # 12x16 (shared feats)

            # Classification head: global average pool -> small MLP.
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.10),
                nn.Linear(48, 48),
                nn.ReLU(inplace=True),
                nn.Linear(48, num_classes),
            )

            # Detection head: 1x1 conv -> single-channel center heatmap logits.
            self.det_head = nn.Sequential(
                nn.Conv2d(48, 24, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 1, kernel_size=1),
            )

        def forward(self, x):
            f = self.block3(self.block2(self.block1(x)))
            logits = self.cls_head(self.gap(f))            # (B, num_classes)
            heatmap = self.det_head(f).squeeze(1)          # (B, Hh, Hw) logits
            return logits, heatmap

        @torch.no_grad()
        def predict(self, x):
            """Return ``(class_probs, heatmap_prob)`` with activations applied."""
            logits, heatmap = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            heat = torch.sigmoid(heatmap)
            return probs, heat

    return ThermalNet(num_classes)


def count_parameters(model) -> int:
    """Total number of trainable parameters in ``model``."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
