"""Full-preset matcher: jointly optimize ALL tone-path knobs of one plugin mode.

Unlike match_target.py (staged: scan -> core knobs -> EQ), this optimizes everything at
once with CMA-ES: amp knobs, boost/overdrive, compressor, gate, EQ bands, cab mic types
and positions, doubler, chorus2. Booleans become thresholded dimensions; mic selectors
become index-mapped dimensions. CMA-ES proposes new values for EVERY knob at EVERY
iteration (it revisits gain constantly), and stops when improvement stagnates
(convergence) rather than at a fixed budget; --budget is a safety cap.

The target is a SINGLE channel (default L) of the stem: double-tracked stems are panned
takes, so one channel is one performance, cleaner than a mono sum. The render is compared
on the same channel.

Run:  uv run python scripts/full_match.py --plugin petrucci
      uv run python scripts/full_match.py --plugin plini --channel R --budget 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tonematcher.data import trim_seconds  # noqa: E402
from tonematcher.hosting import PluginHost  # noqa: E402
from tonematcher.metrics import MRSTFTMetric  # noqa: E402
from tonematcher.optimize import minimize  # noqa: E402

SR = 48000
WU = 0.15
OUT_DIR = REPO_ROOT / "data" / "renders"
ROOT = r"C:\Program Files\Common Files\VST3"
ND = ROOT + r"\Neural DSP"

PLUGINS = {
    "petrucci": dict(path=ROOT + r"\Archetype Petrucci X.vst3", mode_param="amp_type",
                     mode="Rhythm", own_prefixes=("rhythm",),
                     other_prefixes=("clean", "lead", "piezo")),
    "plini": dict(path=ND + r"\Archetype Plini.vst3", mode_param="amp_type",
                  mode="Lead", own_prefixes=("lead",),
                  other_prefixes=("clean", "crunch")),
}

# Only true utility params are excluded; ALL tone/FX sections (gate, wah, compressor,
# pedals, chorus/flanger/phaser, delay, reverb, doubler, cab + room mics, section
# toggles) are searchable per user request.
EXCLUDE = ("bypass", "reserved", "output_gain", "metronome", "tuner", "transpose",
           "volume")
BOOL_INCLUDE = ("active", "bite", "boost", "bright", "tight", "mode", "phase", "shimmer",
                "stereo", "linked", "sync", "ping", "crystal", "tape", "air", "soar")


def rms_norm(a: np.ndarray) -> np.ndarray:
    return (a / (np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-10) * 0.1).astype(np.float32)


def load_channel(path: str, channel: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    x = audio[:, min(channel, audio.shape[1] - 1)]
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def ptype(p) -> str:
    return getattr(getattr(p, "type", None), "__name__", "?")


def build_dims(host: PluginHost, spec) -> list[dict]:
    """Every searchable parameter as a [0,1] dimension with a decoder."""
    dims = []
    params = host.parameters
    for name, p in params.items():
        nl = name.lower()
        if any(e in nl for e in EXCLUDE):
            continue
        if any(o + "_" in nl or nl.startswith(o) for o in spec["other_prefixes"]):
            continue  # another channel's knobs
        t = ptype(p)
        if t == "float":
            dims.append({"name": name, "kind": "float"})
        elif t == "bool" and any(b in nl for b in BOOL_INCLUDE):
            dims.append({"name": name, "kind": "bool"})
        elif t == "str" and "mic" in nl and "type" in nl:
            vals = list(getattr(p, "valid_values", []))
            if len(vals) >= 2:
                dims.append({"name": name, "kind": "sel", "values": vals})
        elif t == "str" and "active" in nl:
            vals = list(getattr(p, "valid_values", []))
            if vals == ["Inactive", "Active"]:
                dims.append({"name": name, "kind": "onoff"})
    return dims


def apply_dims(host: PluginHost, dims, v: np.ndarray):
    for d, x in zip(dims, v):
        x = float(np.clip(x, 0.0, 1.0))
        if d["kind"] == "float":
            host.set_raw(d["name"], x)
        elif d["kind"] == "bool":
            host.set_value(d["name"], x > 0.5)
        elif d["kind"] == "onoff":
            host.set_value(d["name"], "Active" if x > 0.5 else "Inactive")
        else:  # sel
            idx = min(int(x * len(d["values"])), len(d["values"]) - 1)
            host.set_value(d["name"], d["values"][idx])


def current_x0(host: PluginHost, dims) -> np.ndarray:
    x0 = []
    for d in dims:
        p = host.parameters[d["name"]]
        raw = float(getattr(p, "raw_value", 0.5))
        x0.append(np.clip(raw, 0.02, 0.98))
    return np.array(x0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True, choices=list(PLUGINS))
    ap.add_argument("--target", default=str(REPO_ROOT / "data/targets/Untethered Angel Guitar stems/Rythm.wav"))
    ap.add_argument("--probe-file", default=str(REPO_ROOT / "data/targets/Untethered Angel Guitar stems/Rythm gtr DI recorded (intro).wav"))
    ap.add_argument("--segment", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--probe-at", type=float, default=14.0)
    ap.add_argument("--channel", default="L", choices=["L", "R"])
    ap.add_argument("--budget", type=int, default=3500, help="safety cap; stops earlier on convergence")
    ap.add_argument("--tolfun", type=float, default=5e-3)
    ap.add_argument("--list", action="store_true", help="print searchable dims and exit")
    args = ap.parse_args()
    ch = 0 if args.channel == "L" else 1
    spec = PLUGINS[args.plugin]
    metric = MRSTFTMetric()

    # target: ONE channel of the stem
    tgt_full = load_channel(args.target, ch)
    seg = rms_norm(tgt_full[int(args.segment * SR): int((args.segment + args.seconds) * SR)])
    # how different are L and R? (double-tracking check, informational)
    other = load_channel(args.target, 1 - ch)
    o = rms_norm(other[int(args.segment * SR): int((args.segment + args.seconds) * SR)])
    corr = float(np.corrcoef(seg, o)[0, 1])
    print(f"target channel {args.channel}; L/R correlation {corr:.3f} "
          f"(low = double-tracked takes)", flush=True)

    di = load_channel(args.probe_file, 0)
    n = int((args.seconds + WU) * SR)
    pseg = di[int(args.probe_at * SR): int(args.probe_at * SR) + n]
    probe = (pseg / (np.max(np.abs(pseg)) + 1e-9) * 0.25)[None, :].astype(np.float32)

    host = PluginHost.load(spec["path"], initialization_timeout=120, retry_timeout=180)
    host.set_value(spec["mode_param"], spec["mode"])
    if "transpose" in host.parameters:
        host.set_value("transpose", "0 st")

    dims = build_dims(host, spec)
    print(f"{args.plugin}/{spec['mode']}: {len(dims)} searchable dimensions", flush=True)
    for d in dims:
        print(f"  {d['kind']:5} {d['name']}", flush=True)
    if args.list:
        return 0

    def render_channel(y: np.ndarray) -> np.ndarray:
        y = trim_seconds(y, SR, WU)
        return y[min(ch, y.shape[0] - 1)] if y.ndim == 2 else y

    evals = {"n": 0}

    def objective(v: np.ndarray) -> float:
        apply_dims(host, dims, v)
        host.reset()
        y = host.render(probe, SR)
        evals["n"] += 1
        return metric.distance(rms_norm(render_channel(y)), seg)

    x0 = current_x0(host, dims)
    print(f"start score: {objective(x0):.3f} | joint CMA-ES, cap {args.budget}, "
          f"tolfun {args.tolfun}", flush=True)
    t0 = time.time()
    res = minimize(objective, dim=len(dims), budget=args.budget, backend="cma", seed=0,
                   x0=x0, cma_options={"tolfun": args.tolfun, "tolx": 1e-3})
    print(f"final: {res.loss:.3f} after {res.n_evals} evals ({time.time()-t0:.0f}s)")
    print(f"stopped because: {res.stop_reason}", flush=True)

    # report + save
    apply_dims(host, dims, res.x)
    host.reset()
    y = host.render(probe, SR)
    import soundfile as sf

    sf.write(OUT_DIR / f"full_match_{args.plugin}.wav", rms_norm(render_channel(y)), SR)
    sf.write(OUT_DIR / f"full_match_target_{args.channel}.wav", seg, SR)

    print(f"\n=== FULL PRESET: {args.plugin}/{spec['mode']} (channel {args.channel}) ===")
    out = {}
    for d, v in zip(dims, res.x):
        x = float(np.clip(v, 0, 1))
        if d["kind"] == "float":
            host.set_raw(d["name"], x)
        disp = host.get_value(d["name"])
        out[d["name"]] = {"raw": round(x, 4), "display": str(disp)}
        print(f"  {d['name']:28} {str(disp):>14}   (raw {x:.3f})")
    json.dump({"plugin": args.plugin, "mode": spec["mode"], "channel": args.channel,
               "final_score": res.loss, "stop": res.stop_reason, "params": out},
              open(OUT_DIR / f"full_match_{args.plugin}.json", "w"), indent=1)
    print(f"\nA/B: full_match_target_{args.channel}.wav vs full_match_{args.plugin}.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
