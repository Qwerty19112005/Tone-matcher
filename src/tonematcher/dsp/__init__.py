"""dsp - lightweight signal processing used around the plugin chain (EQ, etc.)."""

from __future__ import annotations

from .eq import PRE_EQ_FREQS, fit_post_eq, graphic_eq, smoothed_env_db

__all__ = ["graphic_eq", "fit_post_eq", "smoothed_env_db", "PRE_EQ_FREQS"]
