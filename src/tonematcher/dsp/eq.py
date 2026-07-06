"""Zero-phase FFT EQ: a searchable pre-amp graphic EQ and an analytic post-amp envelope fit.

Pre-EQ shapes the DI BEFORE the amp, so it is nonlinear-relevant (boosting lows pushes more
low end into the amp's saturation) and must live in the render loop, driven by the search.
Post-EQ is linear and applied AFTER the cab, so its OPTIMAL setting is solved analytically:
the smoothed log-spectrum difference (target minus render) is the EQ curve. That decouples
amp/character search (scored after post-EQ compensation) from spectral balance (handled by
post-EQ) - the two-stage matching insight made concrete, and it prevents the amp knobs from
railing to chase a tilt that an EQ should own.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

_EPS = 1e-10
PRE_EQ_FREQS = (80.0, 200.0, 500.0, 1200.0, 3000.0, 7000.0)


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0) if x.ndim == 2 else x


def graphic_eq(x: np.ndarray, sr: int, gains_db, freqs=PRE_EQ_FREQS,
               max_db: float = 15.0) -> np.ndarray:
    """Apply a zero-phase graphic EQ (gains at ``freqs``, log-frequency interpolated)."""
    m = _mono(x)
    n = len(m)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.clip(np.asarray(gains_db, dtype=float), -max_db, max_db)
    g_db = np.interp(np.log2(np.maximum(f, 1.0)), np.log2(freqs), g, left=g[0], right=g[-1])
    y = np.fft.irfft(np.fft.rfft(m) * 10 ** (g_db / 20.0), n=n)
    return y.astype(np.float32)


def smoothed_env_db(x: np.ndarray, sr: int, smooth_oct: float = 1 / 3.0,
                    nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Octave-smoothed log power spectrum (dB). Returns (freqs, env_db)."""
    m = _mono(x)
    fw, pxx = welch(m, fs=sr, nperseg=min(nperseg, len(m)))
    env = 10.0 * np.log10(pxx + _EPS)
    lo = max(fw[1], 20.0)
    grid = np.linspace(np.log2(lo), np.log2(fw[-1]), 256)
    on = np.interp(grid, np.log2(np.maximum(fw, lo)), env)
    step = (grid[-1] - grid[0]) / (len(grid) - 1)
    half = max(1, int(round(smooth_oct / step / 2)))
    k = np.ones(2 * half + 1) / (2 * half + 1)
    sm = np.convolve(np.pad(on, half, mode="edge"), k, mode="valid")
    return fw, np.interp(np.log2(np.maximum(fw, lo)), grid, sm)


def fit_post_eq(render: np.ndarray, target: np.ndarray, sr: int,
                smooth_oct: float = 1 / 3.0, max_db: float = 18.0):
    """Analytic post-EQ: morph the render's envelope toward the target's.

    Returns (corrected_render, (freqs, curve_db)). The curve is what a real post-amp EQ
    (e.g. FabFilter Pro-Q) would be dialed to; it cannot create energy in bands the render
    left empty (that is the honest limit the benchmark then measures).
    """
    fw, er = smoothed_env_db(render, sr, smooth_oct)
    _, et = smoothed_env_db(target, sr, smooth_oct)
    curve = np.clip(et - er, -max_db, max_db)
    m = _mono(render)
    n = len(m)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g_db = np.interp(f, fw, curve)
    y = np.fft.irfft(np.fft.rfft(m) * 10 ** (g_db / 20.0), n=n)
    return y.astype(np.float32), (fw, curve)
