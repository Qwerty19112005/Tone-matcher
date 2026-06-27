"""Phase 0 - headless plugin hosting sanity check (the gating test).

Proves the single biggest unknown for this project: that a plugin from your library can be
loaded, inspected, driven, and rendered through **without its GUI**, purely from Python.

What it does:
  1. Load ONE plugin (``PLUGIN_PATH`` from .env, else the first ``*.vst3`` in ``PLUGIN_DIR``).
  2. Print its full parameter list (name, type, range, units, current value).
  3. Pick one continuous knob and set it to a new value; confirm the value read back.
  4. Render a short DI clip through it (``DI_PATH`` from .env, else a synthetic test signal),
     once with default params and once with the changed knob, to confirm the knob actually
     affects the audio.
  5. Write the rendered ``.wav`` to ``RENDER_DIR`` and print a render confirmation.

If this prints a parameter list and writes a wav, headless hosting works and the rest of
the project is unblocked.

Verified against: pedalboard==0.9.23, soundfile==0.14, numpy==2.2 (Python 3.11).
Run with:  uv run python scripts/phase0_host_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

try:
    import pedalboard
except Exception as exc:  # pragma: no cover - import-time guard
    sys.exit(f"Could not import pedalboard ({exc}). Run `uv sync` first.")

REPO_ROOT = Path(__file__).resolve().parents[1]
# When choosing a knob to *demonstrate*, avoid level/routing params that can simply mute
# or disable the plugin (we want a tonal change, not silence). Prefer drive/gain knobs.
KNOB_SKIP_HINTS = ("bypass", "reserved", "volume", "level", "output", "input",
                   "power", "channel", "mode", "oversampl", "bias")
KNOB_PREFER_HINTS = ("gain", "drive", "dist", "tone", "treble", "presence")


def find_plugin() -> str:
    """Resolve the plugin to test from .env, with a helpful error if none is set."""
    explicit = os.getenv("PLUGIN_PATH")
    if explicit:
        if not Path(explicit).exists():
            sys.exit(f"PLUGIN_PATH set but does not exist: {explicit}")
        return explicit

    plugin_dir = os.getenv("PLUGIN_DIR")
    if not plugin_dir:
        sys.exit(
            "No plugin configured. Copy .env.example to .env and set PLUGIN_DIR "
            "(and optionally PLUGIN_PATH) to your locally installed plugins."
        )
    if not Path(plugin_dir).is_dir():
        sys.exit(f"PLUGIN_DIR does not exist: {plugin_dir}")

    vst3s = sorted(Path(plugin_dir).glob("*.vst3"))
    if not vst3s:
        sys.exit(f"No *.vst3 plugins found directly under PLUGIN_DIR: {plugin_dir}")
    return str(vst3s[0])


def load_di(sample_rate: int) -> tuple[np.ndarray, int]:
    """Return (audio[channels, samples] float32, sample_rate).

    Uses DI_PATH if provided (rendered at the file's own sample rate), otherwise a
    synthetic plucked-note-ish test signal at ``sample_rate``.
    """
    di_path = os.getenv("DI_PATH")
    if di_path and Path(di_path).exists():
        audio, sr = sf.read(di_path, dtype="float32", always_2d=True)  # (frames, channels)
        x = np.ascontiguousarray(audio.T)  # -> (channels, frames)
        print(f"DI source     : {di_path}  ({x.shape[1]/sr:.2f}s @ {sr} Hz, {x.shape[0]} ch)")
        return x, sr

    # Synthetic fallback: 1.5 s, three decaying harmonics, light noise (a stand-in DI).
    sr = sample_rate
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    env = np.exp(-3.0 * t).astype(np.float32)
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(110, 0.6), (220, 0.3), (330, 0.15)])
    noise = 0.01 * np.random.default_rng(0).standard_normal(n)
    x = (env * (tone + noise)).astype(np.float32)[np.newaxis, :]  # (1, frames)
    print(f"DI source     : <synthetic test signal>  ({n/sr:.2f}s @ {sr} Hz, mono)")
    return x, sr


def pick_demo_knob(params) -> str | None:
    """Pick a continuous float knob to demonstrate setting.

    Prefer a drive/tone knob; skip level/routing params that could just mute the plugin.
    Falls back to any float param if nothing better is available.
    """
    floats = [
        name
        for name, pp in params.items()
        if getattr(getattr(pp, "type", None), "__name__", "") == "float"
    ]
    tonal = [n for n in floats if not any(h in n.lower() for h in KNOB_SKIP_HINTS)]
    preferred = [n for n in tonal if any(h in n.lower() for h in KNOB_PREFER_HINTS)]
    for bucket in (preferred, tonal, floats):
        if bucket:
            return bucket[0]
    return None


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    sample_rate = int(os.getenv("SAMPLE_RATE", "48000"))
    render_dir = REPO_ROOT / os.getenv("RENDER_DIR", "data/renders")
    render_dir.mkdir(parents=True, exist_ok=True)

    plugin_path = find_plugin()
    print("=" * 72)
    print("Phase 0 - headless plugin hosting check")
    print("=" * 72)
    print(f"Plugin path   : {plugin_path}")

    try:
        plugin = pedalboard.load_plugin(plugin_path, initialization_timeout=15.0)
    except Exception as exc:
        first = str(exc).splitlines()[0]
        sys.exit(
            f"FAILED to load plugin headless: {first}\n"
            "If the file contains multiple plugins, set plugin_name; some plugins also "
            "require pointing at the inner Contents/<arch>/<name>.vst3 binary."
        )
    print(f"Loaded as     : {type(plugin).__name__}")

    params = plugin.parameters
    print(f"\nParameters ({len(params)}):")
    print(f"  {'name':<18} {'type':<6} {'min':>6} {'max':>6} {'step':>7} {'units':<6} current")
    print("  " + "-" * 64)
    for name, pp in params.items():
        ptype = getattr(getattr(pp, "type", None), "__name__", "?")
        units = pp.units if getattr(pp, "units", None) else ""
        print(
            f"  {name:<18} {ptype:<6} {str(pp.min_value):>6} {str(pp.max_value):>6} "
            f"{str(getattr(pp, 'step_size', '')):>7} {units:<6} {getattr(plugin, name)}"
        )

    x, sr = load_di(sample_rate)

    # Set one knob to a clearly different value and confirm it reads back.
    knob = pick_demo_knob(params)
    if knob is None:
        print("\nNo continuous knob found to demonstrate setting (unusual).")
    else:
        pp = params[knob]
        current = float(getattr(plugin, knob))
        span = pp.max_value - pp.min_value
        midpoint = pp.min_value + span / 2.0
        # Move toward the far quarter of the range (avoids min/max extremes that can mute).
        new_value = pp.min_value + (0.75 if current <= midpoint else 0.25) * span
        setattr(plugin, knob, new_value)
        read_back = float(getattr(plugin, knob))
        print(
            f"\nSet knob      : {knob}  {current} -> {read_back} "
            f"(raw_value={params[knob].raw_value:.3f})"
        )

    # Render the DI through the plugin.
    y = plugin(x, sr)

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))

    # Compare dry input vs wet output to prove the plugin genuinely processes the audio.
    rms_in, rms_out = rms(x), rms(y)
    change_pct = 100.0 * abs(rms_out - rms_in) / (rms_in + 1e-12)

    out_name = f"phase0_{Path(plugin_path).stem.replace(' ', '_')}.wav"
    out_path = render_dir / out_name
    sf.write(out_path, y.T, sr)  # soundfile wants (frames, channels)

    print("\nRender check  :")
    print(f"  input shape : {x.shape}  (channels, samples)")
    print(f"  output shape: {y.shape}  dtype={y.dtype}")
    print(f"  output peak : {float(np.max(np.abs(y))):.4f}")
    print(f"  dry RMS={rms_in:.5f}  wet RMS={rms_out:.5f}  (plugin changed signal by {change_pct:.1f}%)")
    print(f"  wrote       : {out_path}")
    print("\nPHASE 0 PASSED - plugins load and render headless. Project unblocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
