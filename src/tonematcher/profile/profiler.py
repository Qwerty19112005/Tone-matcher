"""Profiler: render a library of (plugin, mode, settings, clip) tone samples to disk.

The "profile once" data engine. For each configured (plugin, amp mode) it Latin-Hypercube
samples the verified-active tone knobs plus an input-drive axis, renders each setting on
several different DI clips, and writes wavs plus a JSONL manifest. One dataset serves three
consumers:

* contrastive fine-tuning of the tone embedding (positives = same setting, different clip)
* per-plugin point clouds for the static nominate stage
* metric evaluation material

Design notes baked in from the audit and multi-amp gate experiments:
* fresh plugin instance per mode (mode switching can latch state, e.g. Archetype Gojira)
* knobs must be VERIFIED active for the mode (an inert knob poisons the sampling); the
  caller provides the verified knob map (from the gate run) rather than guessing here
* input drive is part of tone for nonlinear amps: sampled as a DI pre-gain in dB
* silence guard on every render; silent renders are logged and skipped, not written
* incremental manifest writes so a crash loses at most one render
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ModeSpec:
    """One (plugin, amp mode) to profile, with its verified-active knobs."""

    plugin_name: str
    plugin_path: str
    mode_param: str | None  # None = leave plugin at defaults (e.g. Gojira CLN)
    mode_value: object | None  # str selector value, or float for raw params
    mode_kind: str  # "str" | "float" | "none"
    label: str  # short mode label used in ids/paths
    knobs: list[str]  # verified-active continuous knobs to sample (raw [0,1])
    slow_load: bool = False
    split: str = "train"  # "train" | "heldout"


@dataclass
class ProfileConfig:
    out_dir: str
    sample_rate: int = 48000
    n_settings: int = 40  # LHS samples per mode
    clips_per_setting: int = 3  # different DI clips rendered per setting
    input_db_range: tuple[float, float] = (-9.0, 9.0)  # DI pre-gain axis
    warmup_s: float = 0.15  # discarded from analysis, but kept in the wav
    seed: int = 0
    knob_range: tuple[float, float] = (0.05, 0.95)


def lhs_settings(n: int, dims: int, seed: int) -> np.ndarray:
    """Latin Hypercube sample in [0,1]^dims (scipy qmc; verified installed)."""
    from scipy.stats import qmc

    if dims == 0:
        return np.zeros((n, 0))
    return qmc.LatinHypercube(d=dims, seed=seed).random(n)


def profile_mode(
    spec: ModeSpec,
    clips: dict[str, np.ndarray],
    cfg: ProfileConfig,
    manifest_path: Path,
) -> dict:
    """Render the full sample grid for one mode. Returns a small summary dict."""
    from tonematcher.hosting import PluginHost

    rng = np.random.default_rng(cfg.seed + hash((spec.plugin_name, spec.label)) % (2**16))
    out_root = Path(cfg.out_dir) / spec.plugin_name / spec.label
    out_root.mkdir(parents=True, exist_ok=True)

    timeout = 120.0 if spec.slow_load else 30.0
    host = PluginHost.load(spec.plugin_path, initialization_timeout=timeout, retry_timeout=180.0)
    if spec.mode_param is not None:
        if spec.mode_kind == "float":
            host.set_raw(spec.mode_param, float(spec.mode_value))
        else:
            host.set_value(spec.mode_param, spec.mode_value)

    lo, hi = cfg.knob_range
    grid = lhs_settings(cfg.n_settings, len(spec.knobs), cfg.seed)
    db_lo, db_hi = cfg.input_db_range
    drive_col = lhs_settings(cfg.n_settings, 1, cfg.seed + 1)[:, 0]
    clip_names = list(clips.keys())

    import soundfile as sf

    written, silent = 0, 0
    t0 = time.time()
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for i in range(cfg.n_settings):
            knob_vals = {k: float(lo + (hi - lo) * grid[i, j]) for j, k in enumerate(spec.knobs)}
            input_db = float(db_lo + (db_hi - db_lo) * drive_col[i])
            # clip 0 always included (canonical, for point clouds); rest sampled
            chosen = [clip_names[0]] + list(
                rng.choice(clip_names[1:], size=cfg.clips_per_setting - 1, replace=False)
            )
            for clip_name in chosen:
                for k, v in knob_vals.items():
                    host.set_raw(k, v)
                host.reset()
                y = host.render(clips[clip_name] * (10 ** (input_db / 20.0)), cfg.sample_rate)
                peak = float(np.max(np.abs(y)))
                if peak < 1e-4:
                    silent += 1
                    continue
                wav_name = f"s{i:03d}_{clip_name}.wav"
                sf.write(out_root / wav_name, np.asarray(y, dtype=np.float32).T, cfg.sample_rate)
                row = {
                    "plugin": spec.plugin_name,
                    "mode": spec.label,
                    "split": spec.split,
                    "setting_id": f"{spec.plugin_name}/{spec.label}/s{i:03d}",
                    "knobs": knob_vals,
                    "input_db": round(input_db, 2),
                    "clip": clip_name,
                    "wav": str((out_root / wav_name).relative_to(cfg.out_dir)),
                    "peak": round(peak, 4),
                    "sr": cfg.sample_rate,
                }
                mf.write(json.dumps(row) + "\n")
                written += 1
    return {
        "plugin": spec.plugin_name,
        "mode": spec.label,
        "written": written,
        "silent_skipped": silent,
        "seconds": round(time.time() - t0, 1),
    }
