"""Richer synthetic guitar DI: a regime-spanning probe signal.

The plain ``synthetic_di`` (a single sustained note) is fine for controlled self-recovery
tests, but real amp/effect tone is program-dependent: distortion behaves differently on a
chord (intermodulation) than a single note, compression reacts to dynamics, and different
registers excite different parts of the cab/EQ curve. This module builds DIs that exercise
those regimes, using Karplus-Strong plucked-string synthesis (much more guitar-like than
summed sines).

Each generator returns mono float32 audio shaped (1, samples), peak-normalized to ~0.5.
Use the individual regimes as training/probe material, or ``richer_di`` for one probe that
spans them all.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

# Standard tuning open-string fundamentals (Hz) and the 12-TET ratio.
OPEN_STRINGS = {"E2": 82.41, "A2": 110.00, "D3": 146.83, "G3": 196.00, "B3": 246.94, "E4": 329.63}
SEMITONE = 2 ** (1 / 12)


def karplus_strong(f0: float, seconds: float, sr: int = 48000,
                   decay: float = 0.997, brightness: float = 0.5, seed: int = 0) -> np.ndarray:
    """One plucked-string note via Karplus-Strong (a noise burst through a tuned comb filter)."""
    period = max(2, int(round(sr / f0)))
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float64)
    burst = np.random.default_rng(seed).uniform(-1.0, 1.0, period)
    x[:period] = burst
    # Loop filter: y[k] = x[k] + decay*((1-b)*y[k-period] + b*y[k-period-1]); loop gain = decay < 1.
    a = np.zeros(period + 2)
    a[0] = 1.0
    a[period] = -decay * (1.0 - brightness)
    a[period + 1] = -decay * brightness
    y = lfilter([1.0], a, x)
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def _place(timeline: np.ndarray, start: float, sig: np.ndarray, sr: int, amp: float = 1.0) -> None:
    s = int(start * sr)
    if s < 0 or s >= len(timeline):
        return  # note starts past the end of this (possibly shortened) timeline
    length = min(len(timeline) - s, len(sig))
    timeline[s : s + length] += amp * sig[:length]


def _finalize(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    x = x / (np.max(np.abs(x)) + 1e-9) * peak
    return x.astype(np.float32)[np.newaxis, :]


def single_notes(sr: int = 48000, seconds: float = 4.2, seed: int = 0) -> np.ndarray:
    """A line of sustained single notes across the guitar's register (low to high)."""
    freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # E2..E4 open strings
    out = np.zeros(int(seconds * sr))
    for i, f in enumerate(freqs):
        _place(out, i * 0.7, karplus_strong(f, 0.75, sr, decay=0.9975, seed=seed + i), sr)
    return _finalize(out)


def power_chords(sr: int = 48000, seconds: float = 3.6, seed: int = 0) -> np.ndarray:
    """Root + fifth (+ octave) power chords at several roots."""
    roots = [82.41, 110.00, 146.83]  # E5, A5, D5 shapes
    out = np.zeros(int(seconds * sr))
    for i, r in enumerate(roots):
        for j, ratio in enumerate((1.0, 1.4983, 2.0)):  # root, perfect 5th, octave
            _place(out, i * 1.1, karplus_strong(r * ratio, 1.0, sr, decay=0.9965, seed=seed + i * 4 + j), sr)
    return _finalize(out)


def chords(sr: int = 48000, seconds: float = 3.6, seed: int = 0) -> np.ndarray:
    """Strummed major triads (root, major third, fifth) with a small strum spread."""
    roots = [82.41, 110.00, 146.83]
    out = np.zeros(int(seconds * sr))
    for i, r in enumerate(roots):
        for j, ratio in enumerate((1.0, SEMITONE ** 4, 1.4983, 2.0, 2 * SEMITONE ** 4)):
            strum = j * 0.012  # slight downstroke spread
            _place(out, i * 1.1 + strum, karplus_strong(r * ratio, 1.0, sr, decay=0.996, seed=seed + i * 8 + j), sr)
    return _finalize(out)


def palm_mutes(sr: int = 48000, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Rhythmic palm-muted low chugs: short, heavily damped root+fifth hits."""
    out = np.zeros(int(seconds * sr))
    hits = 16
    for i in range(hits):
        for j, ratio in enumerate((1.0, 1.4983)):
            note = karplus_strong(82.41 * ratio, 0.16, sr, decay=0.945, brightness=0.35, seed=seed + i * 2 + j)
            _place(out, i * 0.18, note, sr, amp=0.9)
    return _finalize(out)


def dynamic_phrase(sr: int = 48000, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """The same note struck at increasing then varied dynamics (to exercise compression/sag)."""
    amps = [0.15, 0.3, 0.55, 1.0, 0.25, 0.8]
    out = np.zeros(int(seconds * sr))
    for i, a in enumerate(amps):
        _place(out, i * 0.5, karplus_strong(110.0, 0.55, sr, decay=0.997, seed=seed + i), sr, amp=a)
    return _finalize(out)


def chromatic(sr: int = 48000, seconds: float = 2.6, seed: int = 0) -> np.ndarray:
    """An ascending chromatic run for broadband spectral coverage."""
    out = np.zeros(int(seconds * sr))
    f = 110.0
    for i in range(10):
        _place(out, i * 0.25, karplus_strong(f, 0.3, sr, decay=0.992, seed=seed + i), sr)
        f *= SEMITONE
    return _finalize(out)


def richer_di(sr: int = 48000, seed: int = 0) -> np.ndarray:
    """One regime-spanning probe: single notes, a power chord, a chord, palm mutes, dynamics."""
    parts = [
        single_notes(sr, 2.2, seed)[0, : int(2.2 * sr)],
        power_chords(sr, 1.3, seed + 1)[0, : int(1.3 * sr)],
        chords(sr, 1.3, seed + 2)[0, : int(1.3 * sr)],
        palm_mutes(sr, 1.6, seed + 3)[0, : int(1.6 * sr)],
        dynamic_phrase(sr, 1.6, seed + 4)[0, : int(1.6 * sr)],
    ]
    return _finalize(np.concatenate(parts))


REGIMES = {
    "single_notes": single_notes,
    "power_chords": power_chords,
    "chords": chords,
    "palm_mutes": palm_mutes,
    "dynamic_phrase": dynamic_phrase,
    "chromatic": chromatic,
    "richer_di": richer_di,
}
