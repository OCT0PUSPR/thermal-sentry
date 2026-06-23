"""Tests for the thermal blob detector and connected-components labelling."""

from __future__ import annotations

import numpy as np

from thermalsentry.config import DetectionSettings
from thermalsentry.detection.detector import (
    ThermalDetector,
    label_connected_components,
)
from thermalsentry.processing.preprocess import bilinear_upscale

from .conftest import make_frame


def _detector():
    # Tune areas for upscaled (factor 10) frames.
    s = DetectionSettings(
        hot_threshold_c=28.0,
        adaptive=True,
        adaptive_k=2.5,
        min_area=20,
        person_min_area=40,
        person_max_area=40000,
        person_min_temp_c=30.0,
        person_max_temp_c=42.0,
        hotspot_temp_c=55.0,
    )
    return ThermalDetector(settings=s)


def test_labelling_numpy_and_scipy_agree_on_count():
    mask = np.zeros((24, 32), dtype=bool)
    mask[2:5, 2:5] = True
    mask[15:18, 20:24] = True
    _, n_np = label_connected_components(mask, prefer_scipy=False)
    labels_pref, n_pref = label_connected_components(mask, prefer_scipy=True)
    assert n_np == 2
    assert n_pref == 2


def test_detect_two_bodies():
    frame = make_frame([(6, 8, 34.0), (16, 22, 35.0)], noise=0.2, seed=1)
    up = bilinear_upscale(frame, factor=10)
    dets = _detector().detect(up)
    # Exactly two well-separated warm blobs.
    assert len(dets) == 2
    for d in dets:
        assert d.area >= 20
        assert 30.0 <= d.peak_temp_c <= 40.0


def test_detect_count_matches_k_bodies():
    for k in (1, 2, 3):
        positions = [(4 + i * 6, 4 + i * 8, 34.0) for i in range(k)]
        frame = make_frame(positions, noise=0.1, seed=10 + k)
        up = bilinear_upscale(frame, factor=10)
        dets = _detector().detect(up)
        assert len(dets) == k, f"expected {k} blobs, got {len(dets)}"


def test_person_classification():
    frame = make_frame([(12, 16, 35.0)], noise=0.1, seed=2)
    up = bilinear_upscale(frame, factor=10)
    dets = _detector().detect(up)
    assert len(dets) == 1
    assert dets[0].label == "person"


def test_hotspot_classification():
    frame = make_frame([(12, 16, 70.0)], noise=0.0, seed=2)
    up = bilinear_upscale(frame, factor=10)
    dets = _detector().detect(up)
    assert len(dets) == 1
    assert dets[0].label == "hotspot"
    assert dets[0].peak_temp_c >= 55.0


def test_empty_scene_no_detections():
    frame = make_frame([], ambient=22.0, noise=0.05, seed=4)
    up = bilinear_upscale(frame, factor=10)
    dets = _detector().detect(up)
    assert dets == []


def test_detection_as_dict_has_norm_coords():
    frame = make_frame([(12, 16, 35.0)], seed=2)
    up = bilinear_upscale(frame, factor=10)
    dets = _detector().detect(up)
    d = dets[0].as_dict(up.shape[1], up.shape[0])
    assert 0.0 <= d["centroid_norm"][0] <= 1.0
    assert 0.0 <= d["centroid_norm"][1] <= 1.0
    assert len(d["bbox_norm"]) == 4
