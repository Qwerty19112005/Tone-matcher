"""Profiler v0: render the training dataset for the tone-embedding viability experiment.

Renders LHS-sampled settings (verified-active knobs + input drive) x diverse DI clips for
the TRAIN-split modes only. Held-out plugins (Abasi, MishaX, PetrucciX, Gojira, NTS,
Nameless, CoryWong) are never profiled; they are evaluated on the frozen 31-mode benchmark
(data/profiles/gate_multi.json) to measure generalization.

Run:  uv run python scripts/profile_v0.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tonematcher.data.guitar import (  # noqa: E402
    _finalize,
    _place,
    dynamic_phrase,
    karplus_strong,
    palm_mutes,
    power_chords,
)
from tonematcher.profile.profiler import ModeSpec, ProfileConfig, profile_mode  # noqa: E402

SR = 48000
GATE_JSON = REPO_ROOT / "data" / "profiles" / "gate_multi.json"
OUT_DIR = REPO_ROOT / "data" / "profiles" / "v0"

ROOT = r"C:\Program Files\Common Files\VST3"
ND = ROOT + r"\Neural DSP"

# Train split: 5 plugins, 13 modes. Everything else in the benchmark stays held out.
TRAIN_PLUGINS = [
    dict(name="Emissary", path=ROOT + r"\Emissary.vst3", mode_param="channel",
         modes=[("clean", 0.0), ("lead", 1.0)], mode_kind="float"),
    dict(name="Nolly", path=ND + r"\Archetype Nolly.vst3", mode_param="amp_type",
         modes=["Clean", "Crunch", "Rhythm", "Lead"], mode_kind="str"),
    dict(name="Plini", path=ND + r"\Archetype Plini.vst3", mode_param="amp_type",
         modes=["Clean", "Crunch", "Lead"], mode_kind="str"),
    dict(name="FortinCali", path=ND + r"\Fortin Cali Suite.vst3", mode_param="amp_type",
         modes=["Clean", "OD2", "OD1"], mode_kind="str"),
    dict(name="Granophyre", path=ND + r"\OMEGA Ampworks Granophyre.vst3", mode_param=None,
         modes=[None], mode_kind="none"),
]


def note_line(freqs, seed, peak=0.25):
    out = np.zeros(int(0.7 * len(freqs) * SR))
    for i, f in enumerate(freqs):
        _place(out, i * 0.7, karplus_strong(f, 0.75, SR, decay=0.9975, seed=seed + i), SR)
    return _finalize(out, peak=peak)


def clip_bank() -> dict[str, np.ndarray]:
    """Six diverse clips; 0.25 peak leaves +12 dB drive headroom."""
    return {
        "line_low": note_line([82.41, 110.0, 146.83, 196.0], seed=0),
        "line_mid": note_line([98.0, 130.81, 174.61, 233.08], seed=50),
        "line_high": note_line([146.83, 196.0, 246.94, 329.63], seed=90),
        "chords": (power_chords(SR, 2.6, seed=1) * 0.5).astype(np.float32),
        "mutes": (palm_mutes(SR, 2.4, seed=2) * 0.5).astype(np.float32),
        "dyn": (dynamic_phrase(SR, 2.6, seed=3) * 0.5).astype(np.float32),
    }


def build_specs() -> list[ModeSpec]:
    """Mode specs with knob lists taken from the gate run's VERIFIED knob map."""
    gate = json.load(open(GATE_JSON))
    knob_map = {(g["plugin"], g["mode"]): (g.get("knobs") or {}) for g in gate}

    specs = []
    for spec in TRAIN_PLUGINS:
        for mode in spec["modes"]:
            label = mode[0] if isinstance(mode, tuple) else (mode or "default")
            knobs_rec = knob_map.get((spec["name"], label), {})
            knobs = [knobs_rec.get(x) for x in ("gain", "bass", "treble")]
            knobs = [k for k in knobs if k]
            if not knobs:
                print(f"  ! {spec['name']}/{label}: no verified knobs, skipping mode")
                continue
            specs.append(
                ModeSpec(
                    plugin_name=spec["name"],
                    plugin_path=spec["path"],
                    mode_param=spec["mode_param"],
                    mode_value=(mode[1] if isinstance(mode, tuple) else mode),
                    mode_kind=spec["mode_kind"],
                    label=label,
                    knobs=knobs,
                    split="train",
                )
            )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run: 1 plugin, 4 settings")
    parser.add_argument("--settings", type=int, default=40)
    parser.add_argument("--clips-per-setting", type=int, default=3)
    args = parser.parse_args()

    cfg = ProfileConfig(
        out_dir=str(OUT_DIR),
        sample_rate=SR,
        n_settings=4 if args.smoke else args.settings,
        clips_per_setting=2 if args.smoke else args.clips_per_setting,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.jsonl"

    clips = clip_bank()
    specs = build_specs()
    if args.smoke:
        specs = [s for s in specs if s.plugin_name == "Emissary"][:1]

    # resume: skip modes already fully present in the manifest
    done = set()
    if manifest.is_file():
        for line in open(manifest, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["plugin"], r["mode"]))

    print(f"{len(specs)} modes to profile, {cfg.n_settings} settings x "
          f"{cfg.clips_per_setting} clips each -> ~{len(specs)*cfg.n_settings*cfg.clips_per_setting} renders")
    for s in specs:
        if (s.plugin_name, s.label) in done:
            print(f"skip {s.plugin_name}/{s.label} (in manifest)")
            continue
        print(f"[{s.plugin_name} / {s.label}] knobs={s.knobs} ...", end=" ", flush=True)
        try:
            summary = profile_mode(s, clips, cfg, manifest)
            print(f"ok: {summary['written']} wavs, {summary['silent_skipped']} silent, "
                  f"{summary['seconds']}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"FAILED: {str(exc).splitlines()[0][:120]}", flush=True)
    print("profiling complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
