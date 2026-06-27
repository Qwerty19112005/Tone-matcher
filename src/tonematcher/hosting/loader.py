"""Headless plugin loading and rendering, hardened against quirks seen in the wild.

This wraps pedalboard's ``load_plugin`` and the loaded plugin object. Behaviours baked in
from load-testing the local library (see Phase 0 / the plugin inventory):

* inner-binary fallback: some bundles only scan from ``Contents/<arch>/*.vst3``
* load retries: a few plugins fail on first load (slow model init / transient) then succeed
* stereo render fallback: some effects reject a 1-channel buffer, so a mono DI is
  duplicated to stereo on demand
* a uniform [0, 1] ``raw_value`` parameter space that respects each plugin's own mapping

Note: a hard process crash on load (rare, e.g. heap corruption) cannot be caught in-process;
isolate such a plugin in a subprocess if you hit it.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass

import numpy as np

try:
    import pedalboard
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(f"pedalboard is required for hosting ({exc}). Run `uv sync`.") from exc

META_PARAMS = {"bypass_vst3", "reserved"}  # not musically useful knobs


class PluginLoadError(RuntimeError):
    """Raised when a plugin cannot be loaded headless after fallbacks and retries."""


def _inner_binary(bundle_path: str) -> str | None:
    """Return the inner ``Contents/<arch>/*.vst3`` binary of a bundle, if present."""
    matches = glob.glob(os.path.join(bundle_path, "Contents", "*-win", "*.vst3"))
    matches += glob.glob(os.path.join(bundle_path, "Contents", "*", "*.vst3"))
    return matches[0] if matches else None


def load_plugin(
    path: str,
    *,
    plugin_name: str | None = None,
    initialization_timeout: float = 30.0,
    retries: int = 2,
    retry_timeout: float = 120.0,
):
    """Load a VST3 plugin headless, trying the inner binary and retrying on failure.

    Returns the raw pedalboard plugin object. Raises ``PluginLoadError`` on giving up.
    """
    candidates = [path]
    inner = _inner_binary(path)
    if inner:
        candidates.append(inner)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        timeout = initialization_timeout if attempt == 0 else retry_timeout
        for candidate in candidates:
            try:
                kwargs = {"initialization_timeout": timeout}
                if plugin_name:
                    kwargs["plugin_name"] = plugin_name
                return pedalboard.load_plugin(candidate, **kwargs)
            except Exception as exc:  # noqa: BLE001 - we want to retry on anything
                last_err = exc
        if attempt < retries:
            time.sleep(0.5)

    first_line = str(last_err).splitlines()[0] if last_err else "unknown error"
    raise PluginLoadError(f"Failed to load {path}: {first_line}")


@dataclass(frozen=True)
class ParamSpec:
    """Lightweight description of one plugin parameter."""

    name: str
    type: str  # "float" | "bool" | "str"
    min_value: object
    max_value: object
    units: str | None
    discrete: bool


def _param_type(p) -> str:
    return getattr(getattr(p, "type", None), "__name__", "?")


def _to_channels_first(audio: np.ndarray) -> np.ndarray:
    """Coerce audio to contiguous float32 shaped (channels, samples).

    Accepts mono (N,) or 2-D. For 2-D, the longer axis is treated as time, since audio
    always has far more samples than channels.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        if a.shape[0] > a.shape[1]:  # (samples, channels) -> (channels, samples)
            a = a.T
    else:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {a.shape}")
    return np.ascontiguousarray(a)


class PluginHost:
    """A loaded plugin plus a small, optimizer-friendly parameter/render interface."""

    def __init__(self, plugin, path: str | None = None):
        self._plugin = plugin
        self.path = path

    @classmethod
    def load(cls, path: str, **kwargs) -> "PluginHost":
        return cls(load_plugin(path, **kwargs), path=str(path))

    # -- identity ---------------------------------------------------------------
    @property
    def plugin(self):
        return self._plugin

    @property
    def name(self) -> str:
        return getattr(self._plugin, "name", "")

    @property
    def is_instrument(self) -> bool:
        return bool(getattr(self._plugin, "is_instrument", False))

    # -- parameters -------------------------------------------------------------
    @property
    def parameters(self):
        return self._plugin.parameters

    def param_specs(self) -> list[ParamSpec]:
        specs = []
        for name, p in self._plugin.parameters.items():
            t = _param_type(p)
            specs.append(
                ParamSpec(
                    name=name,
                    type=t,
                    min_value=getattr(p, "min_value", None),
                    max_value=getattr(p, "max_value", None),
                    units=getattr(p, "units", None),
                    discrete=t != "float",
                )
            )
        return specs

    def continuous_params(self) -> list[str]:
        """Names of continuous float knobs, excluding bypass/meta params."""
        return [
            name
            for name, p in self._plugin.parameters.items()
            if _param_type(p) == "float" and name not in META_PARAMS
        ]

    def get_raw(self, name: str) -> float:
        return float(self._plugin.parameters[name].raw_value)

    def set_raw(self, name: str, value: float) -> None:
        """Set a parameter in normalized [0, 1] space (respects the plugin's own mapping)."""
        self._plugin.parameters[name].raw_value = float(np.clip(value, 0.0, 1.0))

    def get_raw_vector(self, names: list[str]) -> np.ndarray:
        return np.array([self.get_raw(n) for n in names], dtype=np.float64)

    def set_raw_vector(self, names: list[str], values) -> None:
        for name, value in zip(names, values):
            self.set_raw(name, float(value))

    def get_value(self, name: str):
        """Current parameter value in real units (e.g. dB, Hz, ratio string)."""
        return getattr(self._plugin, name)

    def set_value(self, name: str, value) -> None:
        setattr(self._plugin, name, value)

    # -- rendering --------------------------------------------------------------
    def render(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Render audio through the plugin, returning (channels, samples) float32.

        Falls back to stereo if the plugin rejects a 1-channel buffer.
        """
        x = _to_channels_first(audio)
        try:
            return self._plugin(x, sample_rate)
        except Exception as exc:  # noqa: BLE001
            if x.shape[0] == 1 and "1-channel" in str(exc):
                return self._plugin(np.repeat(x, 2, axis=0), sample_rate)
            raise

    def reset(self) -> None:
        """Reset internal plugin state between renders, if the plugin supports it."""
        reset = getattr(self._plugin, "reset", None)
        if callable(reset):
            reset()
