"""Multi-Resolution STFT distance (auraloss) - the baseline tone metric.

This is the cheap, no-training baseline used before the learned embeddings (ST-ITO /
Open-Amp). It compares two audio buffers in mono by default, which keeps it robust to
channel-count differences between renders.
"""

from __future__ import annotations

import numpy as np
import torch


class MRSTFTMetric:
    """Callable wrapper around ``auraloss.freq.MultiResolutionSTFTLoss``.

    ``distance(a, b)`` accepts (channels, samples) or (samples,) float arrays and returns a
    scalar distance (0.0 for identical signals). Lower is more similar.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        hop_sizes: tuple[int, ...] = (120, 240, 50),
        win_lengths: tuple[int, ...] = (600, 1200, 240),
        device: str = "cpu",
    ):
        import auraloss

        self.device = device
        self.loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=list(fft_sizes),
            hop_sizes=list(hop_sizes),
            win_lengths=list(win_lengths),
        ).to(device)

    @staticmethod
    def _to_mono_batch(a: np.ndarray, device: str) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(a, dtype=np.float32))
        if x.ndim == 1:
            x = x[None, :]
        x = x.mean(dim=0)  # downmix channels to mono
        return x.reshape(1, 1, -1).to(device)  # (batch, channel, time)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = self._to_mono_batch(a, self.device)
        tb = self._to_mono_batch(b, self.device)
        n = min(ta.shape[-1], tb.shape[-1])
        with torch.no_grad():
            return float(self.loss(ta[..., :n], tb[..., :n]))

    __call__ = distance
