"""Unit tests for the metric, optimizer, and signal helpers (no plugins required)."""

from __future__ import annotations

import numpy as np


def test_synthetic_di_shape():
    from tonematcher.data import synthetic_di

    di = synthetic_di(sample_rate=16000, seconds=0.5)
    assert di.shape == (1, 8000)
    assert di.dtype == np.float32
    assert np.isfinite(di).all()


def test_mrstft_zero_on_identical():
    from tonematcher.metrics import MRSTFTMetric

    t = np.arange(8000) / 16000.0
    a = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None, :]
    metric = MRSTFTMetric()
    assert metric.distance(a, a) == 0.0
    assert metric.distance(a, a * 0.25) > 0.0  # quieter signal differs


def test_fingerprint_level_invariance():
    from tonematcher.metrics import ToneFingerprintMetric

    t = np.arange(48000) / 48000.0
    a = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)).astype(
        np.float32
    )[None, :]
    m = ToneFingerprintMetric()
    assert m.distance(a, a) == 0.0
    assert m.distance(a, 0.25 * a) < 1e-6  # loudness must not read as a tone change
    distorted = np.tanh(6.0 * a).astype(np.float32)  # saturation must read as a tone change
    assert m.distance(a, distorted) > 0.05


def test_openamp_metric_if_vendored():
    import pytest

    from pathlib import Path

    ckpt = (
        Path(__file__).resolve().parents[1]
        / "vendor/OpenAmp/Checkpoints/FxEncoder-splendidbreeze23-ep45.pt"
    )
    if not ckpt.is_file():
        pytest.skip("Open-Amp not vendored (vendor/ is local-only)")

    from tonematcher.metrics import OpenAmpToneMetric

    t = np.arange(48000) / 48000.0
    a = (0.4 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)).astype(
        np.float32
    )[None, :]
    m = OpenAmpToneMetric()
    e = m.embed(a)
    assert e.shape == (64,)
    assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-5
    assert m.distance(a, a) < 1e-6
    assert m.distance(a, 0.25 * a) < 1e-4  # loudness must not read as tone
    assert m.distance(a, np.tanh(8.0 * a).astype(np.float32)) > 0.005  # saturation must


def test_richer_di_regimes():
    from tonematcher.data import REGIMES

    for name, fn in REGIMES.items():
        di = fn(sr=48000)
        assert di.ndim == 2 and di.shape[0] == 1, name
        assert di.shape[1] > 0 and np.isfinite(di).all(), name
        assert 0.0 < float(np.max(np.abs(di))) <= 1.0, name


def test_minimize_recovers_quadratic():
    from tonematcher.optimize import minimize

    target = np.array([0.6, 0.3])
    res = minimize(lambda x: float(np.sum((x - target) ** 2)), dim=2, budget=80, seed=1)
    assert res.n_evals > 0
    assert np.linalg.norm(res.x - target) < 0.15
