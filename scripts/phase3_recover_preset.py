"""Phase 3 - full multi-knob preset recovery on a single plugin.

Generalizes Phase 1 from one knob to a whole preset:

  1. Load a plugin and screen its continuous knobs by tonal sensitivity (a knob that does
     not change the sound is unidentifiable from audio, so we skip it honestly).
  2. Set a RANDOM target preset on the selected knobs and render the DI through it -> target.
  3. Hand the optimizer ONLY the audio. CMA-ES searches all selected knobs jointly in
     [0, 1] to minimize an auraloss MRSTFT distance to the target.
  4. Report per-knob recovered-vs-true error AND the final audio match. The audio match is
     the real objective; per-knob error measures how identifiable each knob is.

Run:  uv run python scripts/phase3_recover_preset.py
      uv run python scripts/phase3_recover_preset.py --plugin "C:\\...\\Archetype Gojira.vst3" --max-knobs 8
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

from tonematcher.data import load_di, synthetic_di, trim_seconds  # noqa: E402
from tonematcher.hosting import PluginHost  # noqa: E402
from tonematcher.metrics import MRSTFTMetric  # noqa: E402
from tonematcher.optimize import minimize  # noqa: E402

# Routing / level / switch params: not part of a "tone preset" in the usual sense.
SKIP_HINTS = ("power", "channel", "mode", "oversampl", "bypass", "reserved",
              "volume", "level", "input", "output")


def resolve_plugin() -> str:
    explicit = os.getenv("PLUGIN_PATH")
    if explicit and Path(explicit).exists():
        return explicit
    default = r"C:\Program Files\Common Files\VST3\Emissary.vst3"
    if Path(default).exists():
        return default
    sys.exit("No plugin found. Set PLUGIN_PATH in .env or pass --plugin.")


def rank_knobs_by_sensitivity(host: PluginHost, di, sr, metric, warmup):
    """Return [(sensitivity, knob)] sorted desc, for tone knobs only."""
    knobs = [k for k in host.continuous_params()
             if not any(h in k.lower() for h in SKIP_HINTS)]
    original = {k: host.get_raw(k) for k in knobs}
    scored = []
    for k in knobs:
        host.set_raw(k, 0.25); host.reset(); lo = trim_seconds(host.render(di, sr), sr, warmup)
        host.set_raw(k, 0.75); host.reset(); hi = trim_seconds(host.render(di, sr), sr, warmup)
        scored.append((metric.distance(lo, hi), k))
        host.set_raw(k, original[k])
    for k, v in original.items():
        host.set_raw(k, v)
    scored.sort(reverse=True)
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: recover a full preset from audio.")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--max-knobs", type=int, default=8, help="cap on knobs to recover")
    parser.add_argument("--min-sensitivity", type=float, default=0.15,
                        help="skip knobs whose MRSTFT change is below this")
    parser.add_argument("--budget", type=int, default=None, help="evals; default scales with dim")
    parser.add_argument("--backend", default="cma", choices=["cma", "nevergrad"])
    parser.add_argument("--warmup", type=float, default=0.15,
                        help="seconds of startup transient to trim before scoring")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--di", default=None)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    plugin_path = args.plugin or resolve_plugin()

    print("=" * 74)
    print("Phase 3 - full multi-knob preset recovery")
    print("=" * 74)
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

    print("\nScreening knobs by tonal sensitivity ...")
    ranked = rank_knobs_by_sensitivity(host, di, sr, metric, args.warmup)
    selected = [(s, k) for s, k in ranked if s >= args.min_sensitivity][: args.max_knobs]
    if not selected:
        sys.exit("No sufficiently sensitive knobs found; lower --min-sensitivity.")
    knobs = [k for _, k in selected]
    sens = {k: s for s, k in selected}
    dim = len(knobs)
    print(f"Selected {dim} recoverable knobs (sensitivity >= {args.min_sensitivity}):")
    for s, k in selected:
        print(f"    {k:16} sensitivity {s:.2f}")
    skipped = [k for s, k in ranked if s < args.min_sensitivity]
    if skipped:
        print(f"  (skipped {len(skipped)} low-sensitivity knobs: unidentifiable from audio)")

    # Random target preset on the selected knobs.
    rng = np.random.default_rng(args.seed)
    true_raw = rng.uniform(0.1, 0.9, size=dim)
    host.set_raw_vector(knobs, true_raw)
    host.reset()
    target = trim_seconds(host.render(di, sr), sr, args.warmup)

    # Objective: optimizer sees only the target audio.
    def objective(x: np.ndarray) -> float:
        host.set_raw_vector(knobs, x)
        host.reset()
        return metric.distance(trim_seconds(host.render(di, sr), sr, args.warmup), target)

    budget = args.budget or max(500, 100 * dim)
    print(f"\nOptimizing {dim} knobs jointly ({args.backend}, budget {budget}) ...")
    t0 = time.time()
    result = minimize(objective, dim=dim, budget=budget, backend=args.backend, seed=args.seed)
    elapsed = time.time() - t0

    rec_raw = np.asarray(result.x, dtype=np.float64)
    host.set_raw_vector(knobs, rec_raw)
    abs_err = np.abs(rec_raw - true_raw)

    print("\nPer-knob recovery:")
    print(f"  {'knob':16} {'true':>7} {'recovered':>10} {'err':>7} {'sens':>6}")
    print("  " + "-" * 50)
    for i, k in enumerate(knobs):
        print(f"  {k:16} {true_raw[i]:7.3f} {rec_raw[i]:10.3f} {abs_err[i]:7.3f} {sens[k]:6.2f}")

    print("\nSummary:")
    print(f"  mean abs param error = {abs_err.mean():.4f}   max = {abs_err.max():.4f}")
    print(f"  final MRSTFT (audio match) = {result.loss:.5f}")
    print(f"  evaluations = {result.n_evals}   time = {elapsed:.1f}s")
    verdict = "PASS" if result.loss < 0.05 else ("CLOSE" if result.loss < 0.2 else "OFF")
    print(f"\nPHASE 3 {verdict} - audio matched to MRSTFT {result.loss:.4f} "
          f"({dim} knobs, mean param error {abs_err.mean():.3f}).")
    return 0 if result.loss < 0.2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
