"""EQ-invariant tone distance: spectral whitening + MRSTFT.

Motivation: a post-amp EQ / cab / mic / mix chain is (to good approximation) a linear,
time-invariant filter. It tilts the spectral envelope but creates no new harmonics, no
intermodulation, and no dynamics changes. The amp/preamp nonlinearity is what defines a
tone's identity (distortion density, fizz texture, compression), and a matcher that
selects amps should see THAT, not the tilt.

Whitening removes each signal's own smoothed long-term spectral envelope (a zero-phase
FFT-domain correction, limited to +/-24 dB inside [60 Hz, 12 kHz]). Any linear filter
applied to a signal lands in its envelope and is cancelled; the residual fine structure
(harmonic vs noise energy, temporal behaviour) survives and is compared with MRSTFT.

Intended use (two-stage matching):
  1. amp/channel selection and core-knob fine-tuning scored with WhitenedMRSTFT
  2. spectral balance then matched separately with the plugin's own EQ section (or cab
     choice), scored with plain level-normalized MRSTFT
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=0)
    return a


def _smooth_log_envelope(freqs: np.ndarray, mag_db: np.ndarray, octaves: float) -> np.ndarray:
    """Smooth a magnitude curve (dB) over a fixed bandwidth in OCTAVES (log-frequency)."""
    lo = max(freqs[1], 20.0)
    log_grid = np.linspace(np.log2(lo), np.log2(freqs[-1]), 384)
    on_log = np.interp(log_grid, np.log2(np.maximum(freqs, lo)), mag_db)
    step = (log_grid[-1] - log_grid[0]) / (len(log_grid) - 1)
    half = max(1, int(round(octaves / step / 2)))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    sm = np.convolve(np.pad(on_log, half, mode="edge"), kernel, mode="valid")
    return np.interp(np.log2(np.maximum(freqs, lo)), log_grid, sm)


def whiten(audio: np.ndarray, sample_rate: int, smooth_octaves: float = 1.0 / 3.0,
           max_db: float = 24.0, f_lo: float = 30.0, f_hi: float | None = None) -> np.ndarray:
    """Flatten a signal's own long-term spectral envelope (zero-phase, offline).

    Returns mono float32 audio whose smoothed spectrum is ~flat. Applying any linear EQ
    to the input changes the envelope, which is removed here, so the output is invariant
    to such EQ (up to the +/-max_db correction limit).
    """
    from scipy.signal import welch

    x = _mono(audio)
    n = len(x)
    freqs_w, pxx = welch(x, fs=sample_rate, nperseg=4096, noverlap=2048)
    env_db = _smooth_log_envelope(freqs_w, 10.0 * np.log10(pxx + _EPS) / 2.0, smooth_octaves)

    if f_hi is None:
        f_hi = 0.95 * sample_rate / 2.0  # correct nearly full band by default
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    corr_db = -np.interp(f, freqs_w, env_db)
    corr_db -= np.median(corr_db[(f > f_lo) & (f < f_hi)])  # unity overall level
    corr_db = np.clip(corr_db, -max_db, max_db)
    # smooth band edges (half-octave raised-cosine ramps): a hard step would ring
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(f, 1e-3))
    ramp_in = np.clip((lf - np.log2(f_lo / 1.5)) / (np.log2(f_lo) - np.log2(f_lo / 1.5)), 0, 1)
    ramp_out = np.clip((np.log2(f_hi * 1.2) - lf) / (np.log2(f_hi * 1.2) - np.log2(f_hi)), 0, 1)
    band = 0.5 * (1 - np.cos(np.pi * ramp_in)) * 0.5 * (1 - np.cos(np.pi * ramp_out))
    corr = 10.0 ** (corr_db * band / 20.0)
    y = np.fft.irfft(spec * corr, n=n)
    return (y / (np.sqrt(np.mean(y**2)) + _EPS) * 0.1).astype(np.float32)


class WhitenedMRSTFT:
    """MRSTFT on spectrally whitened signals: EQ/cab/tilt-invariant tone distance."""

    def __init__(self, sample_rate: int = 48000, **whiten_kwargs):
        from .mrstft import MRSTFTMetric

        self.sr = sample_rate
        self.kw = whiten_kwargs
        self._mrstft = MRSTFTMetric()

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        wa = whiten(a, self.sr, **self.kw)
        wb = whiten(b, self.sr, **self.kw)
        return self._mrstft.distance(wa, wb)

    __call__ = distance
