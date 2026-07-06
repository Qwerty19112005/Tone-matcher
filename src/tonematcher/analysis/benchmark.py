"""Formal scorecard for replicating a produced target (an album stem) with our chain.

Turns the ad-hoc diagnostics into one repeatable report so every future change is judged
by the same numbers. The headline is FLOOR-RELATIVE: our post-EQ-compensated distance
divided by the double-track floor (the album's own L-vs-R distance), i.e. how close we get
relative to how close the record's own two guitar takes are to each other. A single mono
render cannot beat that floor, so ~1.0 is "as good as physically possible without doubling".
"""

from __future__ import annotations

import numpy as np

_BANDS = [(0, 100), (100, 500), (500, 2000), (2000, 6000), (6000, 24000)]
_REPORT_FREQS = [60, 120, 250, 500, 1000, 2000, 4000, 8000]


def _rn(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    return x / (np.sqrt(np.mean(x**2)) + 1e-10) * 0.1


def _crest_db(x: np.ndarray) -> float:
    x = _rn(x)
    return 20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))


def _band_fracs(x: np.ndarray, sr: int) -> dict:
    x = _rn(x)
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = s.sum() + 1e-12
    return {f"{lo}-{hi}": float(s[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in _BANDS}


def scorecard(render: np.ndarray, target_l: np.ndarray, sr: int = 48000,
              target_r: np.ndarray | None = None) -> dict:
    """Full benchmark scorecard for a mono render vs a target channel."""
    from ..dsp.eq import fit_post_eq
    from ..metrics import MRSTFTMetric, WhitenedMRSTFT

    mr = MRSTFTMetric()
    wm = WhitenedMRSTFT(sample_rate=sr)
    r, t = _rn(render), _rn(target_l)
    eqd, (fw, curve) = fit_post_eq(r, t, sr)

    out = {
        "mrstft_raw": round(mr.distance(r, t), 3),
        "whitened": round(wm.distance(r, t), 3),
        "mrstft_posteq": round(mr.distance(_rn(eqd), t), 3),
        "crest_render_db": round(_crest_db(render), 1),
        "crest_target_db": round(_crest_db(target_l), 1),
        "post_eq_curve_db": {hz: round(float(np.interp(hz, fw, curve)), 1) for hz in _REPORT_FREQS},
    }
    br, bt = _band_fracs(render, sr), _band_fracs(target_l, sr)
    out["band_energy_pct"] = {k: (round(br[k] * 100, 1), round(bt[k] * 100, 1)) for k in br}

    if target_r is not None:
        floor = mr.distance(_rn(target_l), _rn(target_r))
        out["floor_mrstft"] = round(floor, 3)
        out["floor_ratio_posteq"] = round(out["mrstft_posteq"] / (floor + 1e-9), 2)
        tl, tr = _rn(target_l), _rn(target_r)
        mid, side = (tl + tr) / 2, (tl - tr) / 2
        out["target_side_mid"] = round(float(np.mean(side**2) / (np.mean(mid**2) + 1e-12)), 2)
    return out
