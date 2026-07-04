"""Content-robust tone fingerprint: a handcrafted, level-invariant tone descriptor.

Motivation: MRSTFT compares spectrograms of the literal audio, so it is dominated by WHAT
is played (notes, chords) rather than the tone applied to it. That makes it unusable
against a real target whose performance differs from our DI. This fingerprint aggregates
away the note-level detail and keeps production/tone attributes:

* cepstrally-smoothed long-term average spectrum in mel bands, level-normalized: the
  smooth spectral ENVELOPE (amp EQ / cab / voicing), with fine harmonic structure (which
  notes were played) discarded via DCT truncation - classic source/filter separation
* spectral flatness (saturation density: clean -> overdrive -> fuzz)
* spectral centroid (brightness)
* crest factor and short-term level variation (dynamics / compression)

Measured status (validated on Emissary, 2026-06-29): perfectly LEVEL-invariant (scaled
signal -> distance 0.0), but NOT yet content-invariant. Even with cepstral envelope
smoothing, transposing a phrase by 3 semitones moves the fingerprint (~0.28) more than a
large audible gain change does (~0.03-0.17), because a harmonic source's time-averaged
envelope tracks note register; source/filter separation from a time-average alone is not
identifiable. This reproduces, on our own rig, the failure mode that motivates LEARNED
content-invariant embeddings (Open-Amp-style contrastive training with different-content
positives), which is the Phase 2 plan. Keep this metric as: (a) the honest baseline a
learned embedding must beat, (b) level-invariant feature vectors for per-plugin point
clouds in profile/nominate, and (c) auxiliary interpretable axes (flatness ~ saturation).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


class ToneFingerprintMetric:
    """Callable tone distance built on a fixed-length fingerprint vector.

    ``distance(a, b)`` accepts (channels, samples) or (samples,) arrays and returns a
    scalar >= 0 (0 for identical tone). Level-invariant by construction: scaling a signal
    does not change its fingerprint. ``fingerprint(audio)`` exposes the raw vector.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop: int = 512,
        n_mels: int = 48,
        n_cepstra: int = 13,
        fmin: float = 40.0,
        fmax: float = 12000.0,
        silence_floor_db: float = 40.0,
        ltas_weight: float = 1.0,
        flatness_weight: float = 1.0,
        centroid_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ):
        import librosa

        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop
        self.n_cepstra = n_cepstra
        self.silence_floor_db = silence_floor_db
        self.w = dict(
            ltas=ltas_weight, flat=flatness_weight, cent=centroid_weight, dyn=dynamics_weight
        )
        self._mel = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

    # -- feature extraction -------------------------------------------------------
    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        a = np.asarray(audio, dtype=np.float64)
        if a.ndim == 2:
            a = a.mean(axis=0)
        return a

    def fingerprint(self, audio: np.ndarray) -> np.ndarray:
        """Compute the fingerprint: [cepstral envelope (n_cepstra), flatness, centroid, crest, level_var]."""
        import librosa

        x = self._to_mono(audio)
        S = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop)) ** 2  # (freq, t)

        # Keep only frames with signal (drop silence/decay tails so they do not bias stats).
        frame_energy = S.sum(axis=0)
        energy_db = 10.0 * np.log10(frame_energy + _EPS)
        keep = energy_db > (energy_db.max() - self.silence_floor_db)
        if not keep.any():
            keep = np.ones_like(keep, dtype=bool)
        S = S[:, keep]

        mel = self._mel @ S  # (n_mels, t)

        # 1) Cepstrally-smoothed, level-normalized long-term average spectrum.
        #    DCT the mel LTAS and keep only low-quefrency coefficients: the smooth
        #    spectral envelope (tone coloration), discarding per-note harmonic peaks.
        from scipy.fft import dct

        ltas_db = 10.0 * np.log10(mel.mean(axis=1) + _EPS)
        cep = dct(ltas_db, type=2, norm="ortho")
        envelope = cep[1 : self.n_cepstra + 1]  # drop c0 (overall level) -> level invariance

        # 2) Spectral flatness (saturation density), mean over frames, in dB.
        geo = np.exp(np.mean(np.log(S + _EPS), axis=0))
        arith = S.mean(axis=0) + _EPS
        flatness_db = float(np.mean(10.0 * np.log10(geo / arith + _EPS)))

        # 3) Spectral centroid (brightness), in octaves re 1 kHz.
        freqs = np.linspace(0, self.sr / 2, S.shape[0])
        centroid_hz = float((freqs[:, None] * S).sum() / (S.sum() + _EPS))
        centroid_oct = np.log2(max(centroid_hz, 20.0) / 1000.0)

        # 4) Dynamics: crest factor (peak/RMS, dB) and frame-level RMS spread (dB std).
        rms = float(np.sqrt(np.mean(x**2))) + _EPS
        crest_db = 20.0 * np.log10(np.max(np.abs(x)) / rms + _EPS)
        frame_rms_db = 10.0 * np.log10(mel.sum(axis=0) + _EPS)
        level_var_db = float(np.std(frame_rms_db))

        return np.concatenate(
            [
                self.w["ltas"] * envelope / 10.0,  # ~0.1 per dB of envelope shape
                [
                    self.w["flat"] * flatness_db / 10.0,
                    self.w["cent"] * centroid_oct,
                    self.w["dyn"] * crest_db / 10.0,
                    self.w["dyn"] * level_var_db / 10.0,
                ],
            ]
        )

    # -- distance ------------------------------------------------------------------
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        fa, fb = self.fingerprint(a), self.fingerprint(b)
        return float(np.linalg.norm(fa - fb) / np.sqrt(len(fa)))

    __call__ = distance
