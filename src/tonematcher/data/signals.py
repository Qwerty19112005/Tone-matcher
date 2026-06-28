"""DI loading and synthetic test-signal generation.

Audio is represented as float32 NumPy arrays shaped (channels, samples), matching the
hosting layer's render convention.
"""

from __future__ import annotations

import numpy as np


def synthetic_di(sample_rate: int = 48000, seconds: float = 1.5, seed: int = 0) -> np.ndarray:
    """A deterministic, SUSTAINED DI-like signal: a plucked note that rings on.

    Fast attack and slow decay so the note sustains. The sustained (steady-state) portion
    is what an amp's tone stack / EQ acts on, so a sustained probe lets the optimizer
    resolve EQ knobs far better than a fast-decaying transient would. Returns mono audio
    shaped (1, samples).
    """
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    env = (1.0 - np.exp(-t / 0.01)) * (0.4 + 0.6 * np.exp(-t / 2.0))  # ~10ms attack, slow decay
    f0 = 110.0  # A2, a typical low guitar note
    harmonics = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    harmonics = harmonics / np.max(np.abs(harmonics))
    noise = 0.005 * np.random.default_rng(seed).standard_normal(n)
    return (0.5 * env * (harmonics + noise)).astype(np.float32)[np.newaxis, :]


def trim_seconds(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Drop the first ``seconds`` from the time axis.

    Used to discard a plugin's startup transient before scoring: some neural amps are not
    bit-reproducible for the first ~100 ms, which would otherwise put a noise floor under
    the similarity metric.
    """
    n = int(max(0.0, seconds) * sample_rate)
    a = np.asarray(audio)
    return a[:, n:] if a.ndim == 2 else a[n:]


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
