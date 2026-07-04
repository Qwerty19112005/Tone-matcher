"""metrics - audio tone-similarity distances and embeddings.

The metric is the single biggest lever in this project: a weak tone embedding silently
breaks nomination and optimization alike.

* Baseline: auraloss Multi-Resolution STFT (and mel-STFT / ESR) - a plain distance.
* Upgrade path: learned style embeddings (ST-ITO "AFx-Rep", Open-Amp's guitar-tuned
  encoder), exposed behind a common ``embed(audio) -> vector`` / ``distance(a, b)``
  interface so the optimizer and nominator are metric-agnostic.

Caveat (ST-ITO, ISMIR 2024): published encoders do NOT yet work well for guitar tone -
expect to adapt/retrain. Treat any learned metric as a NOMINATOR, not ground truth.
"""

from __future__ import annotations

from .fingerprint import ToneFingerprintMetric
from .learned import OpenAmpToneMetric
from .mrstft import MRSTFTMetric

__all__ = ["MRSTFTMetric", "ToneFingerprintMetric", "OpenAmpToneMetric"]
