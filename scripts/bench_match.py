"""Benchmark matcher: fixed amp mode, searchable pre-EQ + amp knobs, analytic post-EQ.

Signal chain: DI -> pre-EQ (searched, 6 bands) -> plugin(mode, amp knobs, drive) -> cab
-> post-EQ (analytically fit to the target envelope at scoring). The search is scored on
the post-EQ-COMPENSATED MRSTFT, so amp knobs and pre-EQ optimize CHARACTER while post-EQ
owns spectral balance. Reports the full benchmark scorecard and both EQ curves, and
compares against the same amp with NO pre/post EQ to isolate the EQ's contribution.

Run:  uv run python scripts/bench_match.py --plugin plini --mode Lead
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

from tonematcher.analysis import scorecard  # noqa: E402
from tonematcher.data import trim_seconds  # noqa: E402
from tonematcher.dsp.eq import PRE_EQ_FREQS, fit_post_eq, graphic_eq  # noqa: E402
from tonematcher.hosting import PluginHost  # noqa: E402
from tonematcher.metrics import MRSTFTMetric  # noqa: E402
from tonematcher.optimize import minimize  # noqa: E402

SR = 48000
WU = 0.15
OUT = REPO_ROOT / "data" / "renders" / "untethered_angel" / "benchmark"
ROOT = r"C:\Program Files\Common Files\VST3"
ND = ROOT + r"\Neural DSP"
PLUGINS = {
    "plini": dict(path=ND + r"\Archetype Plini.vst3", mode_param="amp_type", own="lead"),
    "petrucci": dict(path=ROOT + r"\Archetype Petrucci X.vst3", mode_param="amp_type", own="rhythm"),
}
AMP_KNOB_HINTS = ("gain", "bass", "mid", "treble", "presence", "master", "tight", "depth", "low", "high")


def load_ch(path, ch):
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    x = a[:, min(ch, a.shape[1] - 1)]
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def rms_norm(a):
    return (a / (np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-10) * 0.1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="plini", choices=list(PLUGINS))
    ap.add_argument("--mode", default="Lead")
    ap.add_argument("--target", default=str(REPO_ROOT / "data/targets/Untethered Angel Guitar stems/Rythm.wav"))
    ap.add_argument("--probe-file", default=str(REPO_ROOT / "data/targets/Untethered Angel Guitar stems/Rythm gtr DI recorded (intro).wav"))
    ap.add_argument("--segment", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--probe-at", type=float, default=14.0)
    ap.add_argument("--budget", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = PLUGINS[args.plugin]
    metric = MRSTFTMetric()

    tL = load_ch(args.target, 0)
    tR = load_ch(args.target, 1)
    seg = rms_norm(tL[int(args.segment * SR): int((args.segment + args.seconds) * SR)])
    segR = rms_norm(tR[int(args.segment * SR): int((args.segment + args.seconds) * SR)])

    di = load_ch(args.probe_file, 0)
    n = int((args.seconds + WU) * SR)
    pseg = di[int(args.probe_at * SR): int(args.probe_at * SR) + n]
    probe = (pseg / (np.max(np.abs(pseg)) + 1e-9) * 0.2).astype(np.float32)  # headroom for pre-EQ

    host = PluginHost.load(spec["path"], initialization_timeout=120, retry_timeout=180)
    host.set_value(spec["mode_param"], args.mode)
    if "transpose" in host.parameters:
        host.set_value("transpose", "0 st")
    amp_knobs = [k for k in host.continuous_params()
                 if spec["own"] in k.lower() and any(h in k.lower() for h in AMP_KNOB_HINTS)]
    print(f"{args.plugin}/{args.mode}: {len(amp_knobs)} amp knobs + 6 pre-EQ + drive", flush=True)
    n_pre = len(PRE_EQ_FREQS)
    dim = len(amp_knobs) + n_pre + 1

    def render(v):
        for k, val in zip(amp_knobs, v[: len(amp_knobs)]):
            host.set_raw(k, float(np.clip(val, 0, 1)))
        pre_db = (v[len(amp_knobs): len(amp_knobs) + n_pre] - 0.5) * 24.0  # +/-12 dB
        drive_db = -6.0 + 15.0 * float(np.clip(v[-1], 0, 1))
        shaped = graphic_eq(probe, SR, pre_db) * (10 ** (drive_db / 20.0))
        host.reset()
        y = host.render(shaped[None, :], SR)
        return trim_seconds(y, SR, WU)

    def score(v):
        r = render(v)
        y = rms_norm(r[0] if r.ndim == 2 else r)
        eqd, _ = fit_post_eq(y, seg, SR)   # analytic post-EQ
        return metric.distance(rms_norm(eqd), seg)

    x0 = np.concatenate([[0.5] * len(amp_knobs), [0.5] * n_pre, [0.5]])
    print(f"start (post-eq compensated): {score(x0):.3f} | CMA cap {args.budget}", flush=True)
    t0 = time.time()
    res = minimize(score, dim=dim, budget=args.budget, backend="cma", seed=0, x0=x0,
                   cma_options={"tolfun": 3e-3, "tolx": 1e-3})
    print(f"final: {res.loss:.3f} ({res.n_evals} evals, {time.time()-t0:.0f}s) "
          f"stop={res.stop_reason}", flush=True)

    # final render + post-EQ, and a NO-EQ baseline (amp knobs only, pre/post flat)
    final = render(res.x)
    final = final[0] if final.ndim == 2 else final
    eqd_final, (fw, post_curve) = fit_post_eq(rms_norm(final), seg, SR)

    import soundfile as sf

    sf.write(OUT / f"bench_{args.plugin}_amponly.wav", rms_norm(final), SR)
    sf.write(OUT / f"bench_{args.plugin}_eqmatched.wav", rms_norm(eqd_final), SR)
    sf.write(OUT / "bench_target_L.wav", seg, SR)

    sc_ampeq = scorecard(final, seg, SR, target_r=segR)          # amp+pre-EQ, no post
    sc_full = scorecard(eqd_final, seg, SR, target_r=segR)       # amp+pre-EQ+post-EQ
    pre_db = (res.x[len(amp_knobs): len(amp_knobs) + n_pre] - 0.5) * 24.0

    print("\n=== SCORECARD (lower = closer; floor_ratio ~1 = as close as a mono render can) ===")
    print(f"  double-track floor (album L vs R): {sc_full['floor_mrstft']}")
    print(f"  amp + pre-EQ (no post-EQ):  mrstft {sc_ampeq['mrstft_raw']}  "
          f"floor-ratio {sc_ampeq['floor_ratio_posteq']}")
    print(f"  amp + pre-EQ + post-EQ:     mrstft {sc_full['mrstft_raw']}  "
          f"floor-ratio {sc_full['floor_ratio_posteq']}")
    print("\n  band energy % (render, target):")
    for k, (rv, tv) in sc_full["band_energy_pct"].items():
        print(f"    {k+' Hz':14} render {rv:5.1f}   target {tv:5.1f}")
    print(f"\n  pre-EQ (into amp) dB @ {list(map(int, PRE_EQ_FREQS))}: "
          f"{[round(x,1) for x in pre_db]}")
    print(f"  post-EQ (after cab) dB: {sc_full['post_eq_curve_db']}")
    print(f"\n  target width side/mid: {sc_full['target_side_mid']} "
          f"(a mono render has 0; this is the doubling gap)")

    json.dump({"plugin": args.plugin, "mode": args.mode,
               "amp_knobs": {k: round(float(np.clip(v, 0, 1)), 3) for k, v in zip(amp_knobs, res.x)},
               "pre_eq_db": {int(f): round(float(g), 1) for f, g in zip(PRE_EQ_FREQS, pre_db)},
               "post_eq_db": sc_full["post_eq_curve_db"],
               "scorecard_full": sc_full, "scorecard_ampeq_only": sc_ampeq},
              open(OUT / f"bench_{args.plugin}.json", "w"), indent=1)
    print(f"\nwavs + json in {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
