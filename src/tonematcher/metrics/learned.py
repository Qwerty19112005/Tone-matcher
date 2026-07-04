"""Learned tone-embedding metrics (Phase 2).

OpenAmpToneMetric wraps the pretrained guitar effects encoder from Open-Amp
(Wright et al., arXiv 2411.14972; vendored at vendor/OpenAmp, checkpoint
FxEncoder-splendidbreeze23-ep45.pt, config_set=1). The encoder was trained with a
SimCLR objective whose positives are two DIFFERENT clean clips processed by the SAME
effect model, so the representation is content-invariant by construction: exactly the
property MRSTFT and the handcrafted fingerprint lack (both measured content-dominated,
see metrics/fingerprint.py).

Encoder facts (verified against the vendored code, not assumed):
* architecture: 6 residual 1-D conv blocks (channels 16-64, kernel 5), batch norm,
  ReLU, global average pool -> 64-dim embedding
* input: mono ``[N, 1, T]`` at 44.1 kHz; trained on 0.75 s segments
* loading: ``FXencoder(conf['encoder_conf'])`` + ``load_state_dict(torch.load(ckpt))``

This wrapper resamples from the project rate, chunks audio into training-length windows,
embeds each chunk, and averages L2-normalized chunk embeddings. Distance = cosine
distance in [0, 2].
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

# Encoder config for the shipped checkpoint (config_set=1 in Checkpoints/conf_fxenc.py).
# Inlined so we do not import the vendored config module (it globs unrelated paths).
_ENCODER_CONF = {
    "channels": [16, 32, 32, 64, 64, 64],
    "kernels": [5, 5, 5, 5, 5, 5],
    "strides": [1, 1, 1, 1, 1, 1],
    "dilation": [1, 1, 1, 1, 1, 1],
    "bias": True,
    "norm": "batch",
    "conv_block": "res",
    "activation": "relu",
}
_TRAIN_SR = 44100
_TRAIN_SEG = int(0.75 * _TRAIN_SR)  # 33075 samples, the training segment length


def _default_vendor_dir() -> Path:
    # src/tonematcher/metrics/learned.py -> repo root is three parents up from src/
    return Path(__file__).resolve().parents[3] / "vendor" / "OpenAmp"


def _load_fxenc_module(vendor_dir: Path):
    """Import the vendored fxenc_models.py by file path (no sys.path pollution)."""
    path = vendor_dir / "Utils" / "fxenc_models.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Vendored Open-Amp encoder not found at {path}. "
            "Clone https://github.com/Alec-Wright/OpenAmp into vendor/OpenAmp."
        )
    spec = importlib.util.spec_from_file_location("openamp_fxenc_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenAmpToneMetric:
    """Content-invariant tone distance using the pretrained Open-Amp guitar FX encoder.

    ``distance(a, b)`` accepts (channels, samples) or (samples,) arrays at ``sample_rate``
    and returns cosine distance in [0, 2] (0 = same tone). ``embed(audio)`` returns the
    L2-normalized 64-dim embedding, reusable for profiling point clouds.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        vendor_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        hop_ratio: float = 0.5,
    ):
        vendor_dir = Path(vendor_dir) if vendor_dir else _default_vendor_dir()
        checkpoint = (
            Path(checkpoint)
            if checkpoint
            else vendor_dir / "Checkpoints" / "FxEncoder-splendidbreeze23-ep45.pt"
        )
        module = _load_fxenc_module(vendor_dir)
        conf = {k: (list(v) if isinstance(v, list) else v) for k, v in _ENCODER_CONF.items()}
        self.model = module.FXencoder(conf)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.device = device
        self.sample_rate = sample_rate
        self.hop = max(1, int(_TRAIN_SEG * hop_ratio))

    def _prepare(self, audio: np.ndarray) -> torch.Tensor:
        """Mono, resample to 44.1 kHz, chunk into training-length windows -> [N, 1, T]."""
        import torchaudio.functional as AF

        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=0)
        # RMS-normalize: loudness is a confound, not a tone attribute (measured: without
        # this, a 0.3x gain scales the embedding distance to 0.23 while a real amp-gain
        # change only moves it 0.01).
        a = a / (np.sqrt(np.mean(a**2)) + 1e-10) * 0.1
        x = torch.from_numpy(a)
        if self.sample_rate != _TRAIN_SR:
            x = AF.resample(x, self.sample_rate, _TRAIN_SR)
        if x.shape[-1] < _TRAIN_SEG:  # pad short audio to one full segment
            x = torch.nn.functional.pad(x, (0, _TRAIN_SEG - x.shape[-1]))
        starts = list(range(0, x.shape[-1] - _TRAIN_SEG + 1, self.hop))
        chunks = torch.stack([x[s : s + _TRAIN_SEG] for s in starts])
        return chunks.unsqueeze(1).to(self.device)  # [N, 1, T]

    @torch.no_grad()
    def embed(self, audio: np.ndarray) -> np.ndarray:
        chunks = self._prepare(audio)
        embs = self.model(chunks)  # [N, 64]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        mean = torch.nn.functional.normalize(embs.mean(dim=0), dim=-1)
        return mean.cpu().numpy()

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        ea, eb = self.embed(a), self.embed(b)
        return float(1.0 - np.dot(ea, eb))

    __call__ = distance
