"""Phase 1 - self-recovery of a single knob.

Proves the render-in-the-loop dynamic stage works end to end:

  1. Load a plugin and pick one continuous knob (auto-pick the most tonally sensitive one,
     or pass --knob).
  2. Render the DI through it at a KNOWN target value -> the "target" audio.
  3. Hand the optimizer ONLY the audio (not the value). A gradient-free optimizer searches
     that one knob in [0, 1] to minimize an auraloss MRSTFT distance to the target.
  4. Report the recovered value vs the known target. Small error == the loop works.

Run:  uv run python scripts/phase1_recover_one_knob.py
      uv run python scripts/phase1_recover_one_knob.py --knob lead_gain --true-raw 0.8 --budget 80
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tonematcher.data import load_di, synthetic_di  # noqa: E402
from tonematcher.hosting import PluginHost  # noqa: E402
from tonematcher.metrics import MRSTFTMetric  # noqa: E402
from tonematcher.optimize import minimize  # noqa: E402


def resolve_plugin() -> str:
    explicit = os.getenv("PLUGIN_PATH")
    if explicit and Path(explicit).exists():
        return explicit
    default = r"C:\Program Files\Common Files\VST3\Emissary.vst3"
    if Path(default).exists():
        return default
    sys.exit("No plugin found. Set PLUGIN_PATH in .env.")


# Routing / level / on-off style params: recovering these is trivial (silence vs sound or
# pure gain), so the auto-picker skips them in favour of a genuine tone-shaping knob.
SKIP_HINTS = ("power", "channel", "mode", "oversampl", "bypass", "reserved",
              "volume", "level", "input", "output")


def pick_sensitive_knob(host: PluginHost, di: np.ndarray, sr: int, metric: MRSTFTMetric):
    """Pick the tone knob whose output changes most between raw 0.25 and 0.75."""
    knobs = [k for k in host.continuous_params()
             if not any(h in k.lower() for h in SKIP_HINTS)]
    if not knobs:  # fall back to all continuous params if everything was filtered
        knobs = host.continuous_params()
    original = {k: host.get_raw(k) for k in knobs}
    scored = []
    for k in knobs:
        host.set_raw(k, 0.25); host.reset(); lo = host.render(di, sr)
        host.set_raw(k, 0.75); host.reset(); hi = host.render(di, sr)
        scored.append((metric.distance(lo, hi), k))
        host.set_raw(k, original[k])  # restore
    for k, v in original.items():  # full restore
        host.set_raw(k, v)
    scored.sort(reverse=True)
    return scored[0][1], scored[:5]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: recover one knob from audio.")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--knob", default=None, help="parameter name; default = auto-pick")
    parser.add_argument("--true-raw", type=float, default=0.8, help="target knob value in [0,1]")
    parser.add_argument("--budget", type=int, default=80, help="render evaluations")
    parser.add_argument("--backend", default="nevergrad", choices=["nevergrad", "cma"])
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--di", default=None, help="path to a DI wav; default = synthetic")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    plugin_path = args.plugin or resolve_plugin()

    print("=" * 72)
    print("Phase 1 - single-knob self-recovery")
    print("=" * 72)
    host = PluginHost.load(plugin_path)
    print(f"Plugin   : {host.name}  ({Path(plugin_path).name})")
    if host.is_instrument:
        sys.exit("Plugin is an instrument; cannot process a DI.")

    if args.di:
        di, sr = load_di(args.di, args.sr)
        print(f"DI       : {args.di}")
    else:
        di, sr = synthetic_di(args.sr, args.seconds), args.sr
        print(f"DI       : synthetic ({args.seconds}s @ {sr} Hz)")

    metric = MRSTFTMetric()

    if args.knob:
        knob = args.knob
        print(f"Knob     : {knob} (user-specified)")
    else:
        knob, top = pick_sensitive_knob(host, di, sr, metric)
        print(f"Knob     : {knob} (auto-picked; most tonally sensitive)")
        print("           top sensitivities: "
              + ", ".join(f"{k}={s:.2f}" for s, k in top))

    # Build the target by rendering at a KNOWN value.
    host.set_raw(knob, args.true_raw)
    host.reset()
    target = host.render(di, sr)
    true_value = host.get_value(knob)
    print(f"Target   : {knob} raw={args.true_raw}  (real value: {true_value})")

    # The optimizer sees only `target`; it searches the knob to match it.
    evals = {"n": 0}

    def objective(x: np.ndarray) -> float:
        host.set_raw(knob, x[0])
        host.reset()
        rendered = host.render(di, sr)
        evals["n"] += 1
        return metric.distance(rendered, target)

    print(f"\nOptimizing ({args.backend}, budget {args.budget}) ...")
    t0 = time.time()
    result = minimize(objective, dim=1, budget=args.budget, backend=args.backend, seed=args.seed)
    elapsed = time.time() - t0

    recovered_raw = float(result.x[0])
    host.set_raw(knob, recovered_raw)
    recovered_value = host.get_value(knob)
    err = abs(recovered_raw - args.true_raw)

    print("\nResult   :")
    print(f"  true raw      = {args.true_raw:.4f}   (real: {true_value})")
    print(f"  recovered raw = {recovered_raw:.4f}   (real: {recovered_value})")
    print(f"  abs error     = {err:.4f}   final MRSTFT = {result.loss:.5f}")
    print(f"  evaluations   = {result.n_evals}   time = {elapsed:.1f}s")
    verdict = "PASS" if err < 0.05 else ("CLOSE" if err < 0.12 else "OFF")
    print(f"\nPHASE 1 {verdict} - recovered the knob to within {err:.3f} of truth.")
    return 0 if err < 0.12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
