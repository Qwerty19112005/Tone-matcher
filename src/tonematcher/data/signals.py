"""DI loading and synthetic test-signal generation.

Audio is represented as float32 NumPy arrays shaped (channels, samples), matching the
hosting layer's render convention.
"""

from __future__ import annotations

import numpy as np


def synthetic_di(sample_rate: int = 48000, seconds: float = 1.5, seed: int = 0) -> np.ndarray:
    """A short, deterministic DI-like signal: decaying harmonics plus light noise.

    Returns mono audio shaped (1, samples). Useful as a stand-in DI when no real
    recording is available.
    """
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    env = np.exp(-3.0 * t).astype(np.float32)
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(110, 0.6), (220, 0.3), (330, 0.15)])
    noise = 0.01 * np.random.default_rng(seed).standard_normal(n)
    return (env * (tone + noise)).astype(np.float32)[np.newaxis, :]


def load_di(path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file to (channels, samples) float32 and its sample rate.

    If ``sample_rate`` is given and differs from the file, the audio is resampled.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    x = np.ascontiguousarray(audio.T)  # (channels, frames)
    if sample_rate and sr != sample_rate:
        import librosa

        x = np.stack(
            [librosa.resample(ch, orig_sr=sr, target_sr=sample_rate) for ch in x]
        ).astype(np.float32)
        sr = sample_rate
    return x, sr
