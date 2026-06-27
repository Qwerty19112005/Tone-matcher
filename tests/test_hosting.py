"""Hosting tests against a real plugin. Skipped automatically if it is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

EMISSARY = r"C:\Program Files\Common Files\VST3\Emissary.vst3"
pytestmark = pytest.mark.skipif(not Path(EMISSARY).exists(), reason="Emissary not installed")


def test_load_params_and_render():
    from tonematcher.data import synthetic_di
    from tonematcher.hosting import PluginHost

    host = PluginHost.load(EMISSARY)
    assert not host.is_instrument
    assert len(host.continuous_params()) > 0

    di = synthetic_di(seconds=0.25)
    out = host.render(di, 48000)
    assert out.shape[1] == di.shape[1]
    assert np.isfinite(out).all()


def test_set_raw_roundtrip():
    from tonematcher.hosting import PluginHost

    host = PluginHost.load(EMISSARY)
    knob = host.continuous_params()[0]
    host.set_raw(knob, 0.42)
    assert abs(host.get_raw(knob) - 0.42) < 1e-3
