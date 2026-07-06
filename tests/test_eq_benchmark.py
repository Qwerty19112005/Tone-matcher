"""Tests for the pre/post EQ and the benchmark scorecard (synthetic, no plugins)."""

from __future__ import annotations

import numpy as np

SR = 48000


def _noise(seconds=2.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(seconds * SR)).astype(np.float32)


def test_graphic_eq_boosts_target_band():
    from tonematcher.dsp import graphic_eq

    x = _noise()

    def band_energy(y, lo, hi):
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return s[(f >= lo) & (f < hi)].sum()

    # +12 dB at 80 Hz, flat elsewhere
    boosted = graphic_eq(x, SR, [12, 0, 0, 0, 0, 0])
    assert band_energy(boosted, 60, 120) > 3 * band_energy(x, 60, 120)
    assert abs(band_energy(boosted, 2000, 4000) - band_energy(x, 2000, 4000)) < 0.2 * band_energy(x, 2000, 4000)


def test_fit_post_eq_reduces_envelope_gap():
    from tonematcher.dsp import fit_post_eq, graphic_eq, smoothed_env_db

    src = _noise(seed=1)
    target = graphic_eq(src, SR, [10, 4, -3, 2, -6, 5])  # target = src through some EQ
    corrected, (fw, curve) = fit_post_eq(src, target, SR)

    def env_gap(a, b):
        _, ea = smoothed_env_db(a, SR)
        _, eb = smoothed_env_db(b, SR)
        return float(np.mean(np.abs(ea - eb)))

    assert env_gap(corrected, target) < 0.4 * env_gap(src, target)


def test_scorecard_keys():
    from tonematcher.analysis import scorecard

    a = _noise(seed=2)
    b = _noise(seed=3)
    sc = scorecard(a, b, SR, target_r=_noise(seed=4))
    for k in ("mrstft_raw", "mrstft_posteq", "floor_mrstft", "floor_ratio_posteq",
              "post_eq_curve_db", "band_energy_pct"):
        assert k in sc
