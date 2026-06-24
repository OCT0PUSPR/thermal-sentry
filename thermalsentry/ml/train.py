"""Training pipeline for the two-head thermal CNN (PyTorch).

Trains :class:`ThermalNet` on the synthetic full-frame dataset, jointly
optimising:

* the **classification** head with cross-entropy, and
* the **center-heatmap** detection head with a penalty-reduced focal loss
  (CornerNet-style) so sparse positive cells are not swamped by background.

Device is auto-selected ``MPS > CUDA > CPU``. Uses AdamW + a cosine LR schedule
with warmup, checkpoints the best validation model, and reports real metrics on
a held-out split: classification **accuracy** + **macro-F1**, and a localisation
**hit-rate** (fraction of ground-truth body centers matched by a predicted
heatmap peak within a tolerance).

``torch`` is imported lazily so importing this module never requires torch; only
calling :func:`train` does.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import (
    generate_frame_dataset,
    heatmap_grid_to_frame,
    heatmap_peaks,
)
from .labels import CLASS_NAMES, NUM_CLASSES


def select_device(prefer: Optional[str] = None):
    """Return the best available torch device: MPS > CUDA > CPU."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    n_train: int = 4000
    n_val: int = 1000
    epochs: int = 24
    batch_size: int = 64
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 2
    heatmap_weight: float = 3.0
    seed: int = 0
    out: str = "models/thermal_cnn.pt"
    device: Optional[str] = None
    # Localisation match tolerance, in raw 24x32 grid pixels.
    loc_tolerance_px: float = 3.0
    heatmap_peak_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _heatmap_loss(pred_logits, target, pos_weight: float = 12.0):
    """Weighted BCE on the soft gaussian center heatmap.

    ``pred_logits`` are raw logits (B, H, W); ``target`` is a gaussian heatmap in
    [0, 1] with peaks==1 at object centers. Center cells are sparse (a few of
    ~192), so positives are up-weighted by ``pos_weight`` to keep recall high.
    Treating the gaussian as a *soft* label gives smooth, stable gradients for
    this tiny problem (CornerNet focal collapses to all-background here).
    """
    import torch.nn.functional as F

    # Per-cell weight: emphasise cells that are (near) a center.
    weight = 1.0 + (pos_weight - 1.0) * target
    return F.binary_cross_entropy_with_logits(
        pred_logits, target, weight=weight, reduction="mean"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    """Unweighted (macro) F1 over ``num_classes``."""
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ci, name in enumerate(CLASS_NAMES):
        mask = y_true == ci
        out[name] = float((y_pred[mask] == ci).mean()) if mask.any() else 0.0
    return out


def localization_hit_rate(
    pred_heatmaps: np.ndarray,
    gt_heatmaps: np.ndarray,
    tolerance_px: float = 3.0,
    peak_threshold: float = 0.30,
) -> Tuple[float, float]:
    """Compute localisation recall + precision over a batch of heatmaps.

    A ground-truth center (a peak in ``gt_heatmaps``) counts as a *hit* if some
    predicted peak lies within ``tolerance_px`` (in raw 24x32 pixels). Returns
    ``(recall, precision)``.
    """
    total_gt = 0
    total_hits = 0
    total_pred = 0
    matched_pred = 0
    for pred, gt in zip(pred_heatmaps, gt_heatmaps):
        gt_peaks = heatmap_peaks(gt, threshold=0.5, min_distance=1)
        pred_peaks = heatmap_peaks(pred, threshold=peak_threshold, min_distance=1)
        gt_xy = [heatmap_grid_to_frame(r, c) for (r, c, _) in gt_peaks]
        pred_xy = [heatmap_grid_to_frame(r, c) for (r, c, _) in pred_peaks]
        total_gt += len(gt_xy)
        total_pred += len(pred_xy)
        used = set()
        for (gr, gc) in gt_xy:
            best = None
            best_d = tolerance_px
            for pi, (pr, pc) in enumerate(pred_xy):
                if pi in used:
                    continue
                d = ((gr - pr) ** 2 + (gc - pc) ** 2) ** 0.5
                if d <= best_d:
                    best_d = d
                    best = pi
            if best is not None:
                used.add(best)
                total_hits += 1
                matched_pred += 1
    recall = (total_hits / total_gt) if total_gt else 1.0
    precision = (matched_pred / total_pred) if total_pred else 1.0
    return float(recall), float(precision)


@dataclass
class TrainReport:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_acc: Dict[str, float] = field(default_factory=dict)
    loc_recall: float = 0.0
    loc_precision: float = 0.0
    params: int = 0
    device: str = "cpu"
    epochs: int = 0
    n_train: int = 0
    n_val: int = 0
    train_seconds: float = 0.0
    classes: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    checkpoint: str = ""
    history: List[dict] = field(default_factory=list)


def train(cfg: Optional[TrainConfig] = None, log=print) -> TrainReport:
    """Run the full training loop and return a :class:`TrainReport`."""
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset

    from .model import build_model, count_parameters

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = select_device(cfg.device)
    log(f"device: {device}")

    log(f"generating dataset: {cfg.n_train} train + {cfg.n_val} val frames ...")
    Xtr, ytr, Htr = generate_frame_dataset(n=cfg.n_train, seed=cfg.seed)
    Xva, yva, Hva = generate_frame_dataset(n=cfg.n_val, seed=cfg.seed + 10_000)

    train_ds = TensorDataset(
        torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Htr)
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = build_model(NUM_CLASSES).to(device)
    n_params = count_parameters(model)
    log(f"model parameters: {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_dl))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    ce = torch.nn.CrossEntropyLoss()
    Xva_t = torch.from_numpy(Xva).to(device)

    best_score = -1.0
    best_state = None
    history: List[dict] = []
    t0 = time.time()
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = ep_cls = ep_hm = 0.0
        ep_correct = ep_total = 0
        for xb, yb, hb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            hb = hb.to(device)

            scale = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * scale

            opt.zero_grad()
            logits, heat = model(xb)
            loss_cls = ce(logits, yb)
            loss_hm = _heatmap_loss(heat, hb)
            loss = loss_cls + cfg.heatmap_weight * loss_hm
            loss.backward()
            opt.step()

            ep_loss += loss.item() * len(yb)
            ep_cls += loss_cls.item() * len(yb)
            ep_hm += loss_hm.item() * len(yb)
            ep_correct += int((logits.argmax(1) == yb).sum())
            ep_total += len(yb)
            step += 1

        # --- validation ---
        model.eval()
        with torch.no_grad():
            v_logits, v_heat = model(Xva_t)
            v_pred = v_logits.argmax(1).cpu().numpy()
            v_heat_np = torch.sigmoid(v_heat).cpu().numpy()
        v_acc = float((v_pred == yva).mean())
        v_f1 = macro_f1(yva, v_pred)
        # Localisation on a subset for speed.
        sub = min(256, len(yva))
        v_recall, v_prec = localization_hit_rate(
            v_heat_np[:sub], Hva[:sub],
            tolerance_px=cfg.loc_tolerance_px,
            peak_threshold=cfg.heatmap_peak_threshold,
        )
        # Combined model-selection score.
        score = 0.5 * v_f1 + 0.3 * v_acc + 0.2 * v_recall

        rec = {
            "epoch": epoch + 1,
            "lr": round(cfg.lr * lr_at(step), 6),
            "train_loss": round(ep_loss / ep_total, 4),
            "train_cls": round(ep_cls / ep_total, 4),
            "train_hm": round(ep_hm / ep_total, 4),
            "train_acc": round(ep_correct / ep_total, 4),
            "val_acc": round(v_acc, 4),
            "val_f1": round(v_f1, 4),
            "val_loc_recall": round(v_recall, 4),
            "val_loc_prec": round(v_prec, 4),
        }
        history.append(rec)
        log(
            f"epoch {epoch + 1:2d}/{cfg.epochs}  loss={rec['train_loss']:.4f} "
            f"(cls={rec['train_cls']:.4f} hm={rec['train_hm']:.4f})  "
            f"train_acc={rec['train_acc']:.3f}  val_acc={rec['val_acc']:.3f}  "
            f"val_f1={rec['val_f1']:.3f}  loc_recall={rec['val_loc_recall']:.3f}  "
            f"loc_prec={rec['val_loc_prec']:.3f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.time() - t0

    # Restore best weights + final eval.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        f_logits, f_heat = model(Xva_t)
        f_pred = f_logits.argmax(1).cpu().numpy()
        f_heat_np = torch.sigmoid(f_heat).cpu().numpy()
    acc = float((f_pred == yva).mean())
    f1 = macro_f1(yva, f_pred)
    pca = per_class_accuracy(yva, f_pred)
    recall, prec = localization_hit_rate(
        f_heat_np, Hva,
        tolerance_px=cfg.loc_tolerance_px,
        peak_threshold=cfg.heatmap_peak_threshold,
    )

    # Checkpoint (weights only; tiny).
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from .labels import MODEL_IN_H, MODEL_IN_W

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(CLASS_NAMES),
            "input_hw": [MODEL_IN_H, MODEL_IN_W],
        },
        str(out_path),
    )

    report = TrainReport(
        accuracy=round(acc, 4),
        macro_f1=round(f1, 4),
        per_class_acc={k: round(v, 4) for k, v in pca.items()},
        loc_recall=round(recall, 4),
        loc_precision=round(prec, 4),
        params=n_params,
        device=str(device),
        epochs=cfg.epochs,
        n_train=cfg.n_train,
        n_val=cfg.n_val,
        train_seconds=round(train_seconds, 1),
        checkpoint=str(out_path),
        history=history,
    )
    log(
        f"\nFINAL  acc={acc:.4f}  macro_f1={f1:.4f}  "
        f"loc_recall={recall:.4f}  loc_precision={prec:.4f}  "
        f"({train_seconds:.1f}s on {device})"
    )

    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2))
    log(f"wrote checkpoint -> {out_path}")
    log(f"wrote report     -> {report_path}")
    return report
