"""Known-answer tests for time-based-effect detection (synthetic, no plugins)."""

from __future__ import annotations

import numpy as np


def _riff(sr: int, seconds: float = 8.0, seed: int = 0) -> np.ndarray:
    """Staccato pluck pattern (chug-like: fast decays, many onsets)."""
    from tonematcher.data.guitar import karplus_strong

    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr))
    t = 0.1
    while t < seconds - 0.3:
        note = karplus_strong(float(rng.choice([82.41, 110.0, 146.83])), 0.22, sr,
                              decay=0.94, brightness=0.35, seed=int(rng.integers(1e6)))
        # palm-mute damping: an undamped KS string rings with RT60 near 1 s, which is
        # genuinely reverberant; a dry chug must actually decay fast
        note = note * np.exp(-np.arange(len(note)) / sr / 0.015)
        s = int(t * sr)
        out[s : s + len(note)] += note
        t += float(rng.choice([0.18, 0.18, 0.36]))
    return (0.5 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def _add_echo(x: np.ndarray, sr: int, ms: float, gain: float) -> np.ndarray:
    d = int(ms / 1000.0 * sr)
    y = x.copy()
    y[d:] += gain * x[:-d]
    return y


def _add_reverb(x: np.ndarray, sr: int, rt60: float, wet: float, seed: int = 1) -> np.ndarray:
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    ir = np.random.default_rng(seed).standard_normal(n) * 10 ** (-3.0 * t / rt60)
    tail = np.convolve(x, ir.astype(np.float32))[: len(x)]
    tail = tail / (np.max(np.abs(tail)) + 1e-9) * np.max(np.abs(x))
    return (x + wet * tail).astype(np.float32)


SR = 48000


def test_dry_riff_reports_nothing():
    from tonematcher.analysis import detect_time_fx

    r = detect_time_fx(_riff(SR), SR)
    assert not r["delay"].present
    assert not r["reverb"].present


def test_echo_detected_at_known_time():
    from tonematcher.analysis import detect_delay

    d = detect_delay(_add_echo(_riff(SR), SR, ms=320.0, gain=0.5), SR)
    assert d.present
    assert abs(d.time_ms - 320.0) < 15.0


def test_reverb_detected_with_plausible_rt60():
    from tonematcher.analysis import detect_reverb

    r = detect_reverb(_add_reverb(_riff(SR), SR, rt60=0.9, wet=0.6), SR)
    assert r.present
    assert 0.3 < r.rt60_s < 2.0
