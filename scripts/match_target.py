"""Match a real-world target tone with the local plugin library, end to end.

Pipeline:
  1. Load the target recording; auto-select a steady high-energy segment (or --segment).
  2. Render a regime-matched probe clip (palm mutes + power chords) through EVERY amp mode
     at LHS-sampled settings; rank all renders by level-normalized MRSTFT distance to the
     target segment (ranking against a fixed target is the validated deployment shape).
  3. Fine-tune the winning mode's knobs (plus input drive) with CMA-ES.
  4. Report the final settings (raw and display values), save the matched render and the
     target segment for A/B listening, and dump everything to JSON.

Run:  uv run python scripts/match_target.py data/targets/song.wav
      ... --segment 43.5 --root-hz 73.42 --settings 16 --budget 700 --top-modes 3
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
from tonematcher.data.guitar import _finalize, _place, karplus_strong  # noqa: E402
from tonematcher.hosting import PluginHost  # noqa: E402
from tonematcher.metrics import MRSTFTMetric  # noqa: E402
from tonematcher.optimize import minimize  # noqa: E402

SR = 48000
WU = 0.15
GATE_JSON = REPO_ROOT / "data" / "profiles" / "gate_multi.json"
OUT_DIR = REPO_ROOT / "data" / "renders"

ROOT = r"C:\Program Files\Common Files\VST3"
ND = ROOT + r"\Neural DSP"
CANDIDATES = [
    dict(name="Emissary", path=ROOT + r"\Emissary.vst3", mode_param="channel",
         modes=[("clean", 0.0), ("lead", 1.0)], mode_kind="float"),
    dict(name="Nolly", path=ND + r"\Archetype Nolly.vst3", mode_param="amp_type",
         modes=["Clean", "Crunch", "Rhythm", "Lead"], mode_kind="str"),
    dict(name="PetrucciX", path=ROOT + r"\Archetype Petrucci X.vst3", mode_param="amp_type",
         modes=["Piezo", "Clean", "Rhythm", "Lead"], mode_kind="str"),
    dict(name="Plini", path=ND + r"\Archetype Plini.vst3", mode_param="amp_type",
         modes=["Clean", "Crunch", "Lead"], mode_kind="str"),
    dict(name="Abasi", path=ND + r"\Archetype Abasi.vst3", mode_param="amp_type",
         modes=["Clean", "Rhythm", "Lead"], mode_kind="str"),
    dict(name="CoryWong", path=ND + r"\Archetype Cory Wong.vst3", mode_param="amp_type",
         modes=["D.I. Funk Console", "The Clean Machine", "The Amp Snob"], mode_kind="str"),
    dict(name="MishaX", path=ROOT + r"\Archetype Misha Mansoor X.vst3", mode_param="amp_type",
         modes=["Clean", "Rhythm", "Lead"], mode_kind="str"),
    dict(name="Gojira", path=ND + r"\Archetype Gojira.vst3", mode_param=None,
         modes=[None], mode_kind="none"),
    dict(name="FortinCali", path=ND + r"\Fortin Cali Suite.vst3", mode_param="amp_type",
         modes=["Clean", "OD2", "OD1"], mode_kind="str"),
    dict(name="FortinNameless", path=ND + r"\Fortin Nameless Suite.vst3", mode_param="channel",
         modes=["Inactive", "Active"], mode_kind="str"),
    dict(name="FortinNTS", path=ND + r"\Fortin NTS Suite.vst3", mode_param="channel",
         modes=["Drive", "Clean"], mode_kind="str", slow=True),
    dict(name="Granophyre", path=ND + r"\OMEGA Ampworks Granophyre.vst3", mode_param=None,
         modes=[None], mode_kind="none"),
]

# Extra fine-tune knob name hints beyond the verified gain/bass/treble.
EXTRA_HINTS = ("mid", "presence", "treble", "high", "resonance", "master", "tight",
               "bright", "depth")
SKIP_HINTS = ("input", "output", "volume", "level", "mix", "bypass", "reserved", "power",
              "oversampl", "gate", "wow", "rev_", "dly", "delay", "reverb", "phsr", "chr_",
              "oct_", "od_", "drt_", "pedal", "boost", "room", "pan", "sync", "note",
              "chorus", "flanger", "phaser", "wah", "trem", "glitch", "pitch", "mod")


def rms_norm(a: np.ndarray) -> np.ndarray:
    return (a / (np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-10) * 0.1).astype(np.float32)


def load_target(path: str) -> np.ndarray:
    """Load any audio file to mono float32 at SR."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        x = audio.T.mean(axis=0)
    except Exception:
        import librosa

        x, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        import librosa

        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def pick_segment(x: np.ndarray, seconds: float, start_override: float | None):
    """Pick a steady, high-energy, LOW-REGISTER window (rhythm chugging), or override.

    Score = p10 of frame RMS (continuous playing, no gaps) x low-band energy fraction
    (80-400 Hz; palm-muted rhythm lives there, leads sit higher). Prints the top 5
    candidate windows and saves them as wavs so the choice can be audited by ear.
    """
    n = int(seconds * SR)
    if start_override is not None:
        s = int(start_override * SR)
        return x[s : s + n], start_override

    from scipy.signal import butter, sosfilt

    sos = butter(4, [80, 400], btype="bandpass", fs=SR, output="sos")
    x_low = sosfilt(sos, x.astype(np.float64))
    hop, frame = SR // 2, 2048
    cands = []
    for s in range(0, max(1, len(x) - n), hop):
        w = x[s : s + n]
        frames = w[: (len(w) // frame) * frame].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        p10 = float(np.percentile(rms, 10))
        full = float(np.mean(w.astype(np.float64) ** 2)) + 1e-12
        low_frac = float(np.mean(x_low[s : s + n] ** 2)) / full
        cands.append((p10 * low_frac, s))
    cands.sort(reverse=True)
    # top 5 non-overlapping candidates, saved for audit
    import soundfile as sf

    picked, top = set(), []
    for sc, s in cands:
        if any(abs(s - p) < n for p in picked):
            continue
        picked.add(s)
        top.append((sc, s))
        if len(top) == 5:
            break
    print("segment candidates (listen and override with --segment if wrong):")
    for i, (sc, s) in enumerate(top):
        sf.write(OUT_DIR / f"match_segment_cand{i}.wav", x[s : s + n], SR)
        print(f"  #{i}: {s/SR:7.1f}s  score {sc:.5f}  -> match_segment_cand{i}.wav")
    best = top[0][1]
    return x[best : best + n], best / SR


def make_probe(seconds: float, root_hz: float, seed: int = 5) -> np.ndarray:
    """Regime-matched probe: palm-muted chugs + power chords rooted at root_hz."""
    n_total = int(seconds * SR)
    out = np.zeros(n_total)
    t = 0.0
    for i in range(10):  # chug bar
        for j, ratio in enumerate((1.0, 1.4983)):
            note = karplus_strong(root_hz * ratio, 0.16, SR, decay=0.945,
                                  brightness=0.35, seed=seed + i * 2 + j)
            _place(out, t, note, SR, amp=0.9)
        t += 0.17
    for i in range(2):  # ringing power chords
        for j, ratio in enumerate((1.0, 1.4983, 2.0)):
            _place(out, t + i * 0.9, karplus_strong(root_hz * ratio, 1.0, SR, decay=0.9965,
                                                    seed=seed + 40 + i * 4 + j), SR)
    return _finalize(out[:n_total], peak=0.25)


def knob_map():
    gate = json.load(open(GATE_JSON))
    return {(g["plugin"], g["mode"]): (g.get("knobs") or {}) for g in gate}


def load_mode(spec, mode):
    timeout = 120.0 if spec.get("slow") else 30.0
    h = PluginHost.load(spec["path"], initialization_timeout=timeout, retry_timeout=180.0)
    if spec["mode_param"] is not None:
        if spec["mode_kind"] == "float":
            h.set_raw(spec["mode_param"], mode[1])
        else:
            h.set_value(spec["mode_param"], mode)
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="Match a target tone with local plugins.")
    ap.add_argument("target", help="path to the target recording")
    ap.add_argument("--segment", type=float, default=None, help="segment start in seconds")
    ap.add_argument("--seconds", type=float, default=3.0, help="segment length")
    ap.add_argument("--root-hz", type=float, default=73.42, help="probe root note (D2)")
    ap.add_argument("--probe-file", default=None,
                    help="use a real DI recording as the probe instead of the synthetic one")
    ap.add_argument("--settings", type=int, default=16, help="LHS settings per mode")
    ap.add_argument("--budget", type=int, default=700, help="fine-tune render budget")
    ap.add_argument("--top-modes", type=int, default=3, help="modes shown in the report")
    ap.add_argument("--mode-filter", default=None, help="only scan modes containing this")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric = MRSTFTMetric()

    # ---- stage 1: target segment ----
    x = load_target(args.target)
    seg, seg_start = pick_segment(x, args.seconds, args.segment)
    seg = rms_norm(seg)
    print(f"target: {args.target} ({len(x)/SR:.1f}s) | segment @ {seg_start:.1f}s "
          f"({args.seconds:.1f}s)", flush=True)
    import soundfile as sf

    sf.write(OUT_DIR / "match_target_segment.wav", seg, SR)

    if args.probe_file:
        # Real DI as probe: take its steadiest high-energy window, leave drive headroom.
        di = load_target(args.probe_file)
        n = int((args.seconds + WU) * SR)
        best, best_score = 0, -1.0
        for s in range(0, max(1, len(di) - n), SR // 2):
            w = di[s : s + n]
            frames = w[: (len(w) // 2048) * 2048].reshape(-1, 2048)
            sc = float(np.percentile(np.sqrt(np.mean(frames**2, axis=1) + 1e-12), 10))
            if sc > best_score:
                best, best_score = s, sc
        pseg = di[best : best + n]
        probe = (pseg / (np.max(np.abs(pseg)) + 1e-9) * 0.25)[None, :].astype(np.float32)
        print(f"probe: {args.probe_file} @ {best/SR:.1f}s (real DI)", flush=True)
    else:
        probe = make_probe(args.seconds + WU, args.root_hz)
    km = knob_map()

    def score(render):
        return metric.distance(rms_norm(trim_seconds(render, SR, WU)), seg)

    # ---- stage 2: scan all modes ----
    from scipy.stats import qmc

    results = []
    for spec in CANDIDATES:
        for mode in spec["modes"]:
            label = mode[0] if isinstance(mode, tuple) else (mode or "default")
            full = f"{spec['name']}/{label}"
            if args.mode_filter and args.mode_filter.lower() not in full.lower():
                continue
            knobs = [v for v in (km.get((spec["name"], label), {}).get(k)
                                 for k in ("gain", "bass", "treble")) if v]
            t0 = time.time()
            try:
                h = load_mode(spec, mode)
                sampler = qmc.LatinHypercube(d=len(knobs) + 1, seed=3)
                grid = sampler.random(args.settings)
                best = None
                for i in range(args.settings):
                    for j, k in enumerate(knobs):
                        h.set_raw(k, 0.05 + 0.9 * grid[i, j])
                    drive_db = -6.0 + 15.0 * grid[i, -1]
                    h.reset()
                    y = h.render(probe * (10 ** (drive_db / 20.0)), SR)
                    if float(np.max(np.abs(y))) < 1e-4:
                        continue
                    s = score(y)
                    if best is None or s < best["score"]:
                        best = {"score": s,
                                "knobs": {k: float(h.get_raw(k)) for k in knobs},
                                "drive_db": round(drive_db, 2)}
                if best:
                    results.append({"mode": full, "spec": spec["name"], "label": label,
                                    **best})
                    print(f"[{full:28}] best {best['score']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                else:
                    print(f"[{full:28}] all renders silent, skipped", flush=True)
                del h
            except Exception as exc:  # noqa: BLE001
                print(f"[{full:28}] FAILED {str(exc).splitlines()[0][:90]}", flush=True)

    results.sort(key=lambda r: r["score"])
    print("\n=== mode ranking (lower = closer tone) ===")
    for r in results[: max(args.top_modes, 5)]:
        print(f"  {r['mode']:30} {r['score']:.3f}")
    winner = results[0]
    print(f"\nWINNER: {winner['mode']}", flush=True)

    # ---- stage 3: fine-tune the winner ----
    w_spec = next(s for s in CANDIDATES if s["name"] == winner["spec"])
    w_mode = next(m for m in w_spec["modes"]
                  if (m[0] if isinstance(m, tuple) else (m or "default")) == winner["label"])
    h = load_mode(w_spec, w_mode)

    base_knobs = list(winner["knobs"].keys())
    mode_token = winner["label"].lower().replace(" ", "_")
    extras = [k for k in h.continuous_params()
              if k not in base_knobs
              and not any(s in k.lower() for s in SKIP_HINTS)
              and any(hint in k.lower() for hint in EXTRA_HINTS)
              and (mode_token in k.lower() or not any(
                  (mm[0] if isinstance(mm, tuple) else str(mm or "")).lower().replace(" ", "_")
                  in k.lower() for mm in w_spec["modes"]
                  if mm is not w_mode))][:4]
    knobs = base_knobs + extras
    dim = len(knobs) + 1  # + input drive
    print(f"fine-tuning {dim - 1} knobs + drive: {knobs}", flush=True)

    x0 = np.array([winner["knobs"][k] for k in base_knobs]
                  + [0.5] * len(extras) + [(winner["drive_db"] + 6.0) / 15.0])

    def objective(v: np.ndarray) -> float:
        for k, val in zip(knobs, v[:-1]):
            h.set_raw(k, float(np.clip(val, 0.0, 1.0)))
        drive_db = -6.0 + 15.0 * float(np.clip(v[-1], 0.0, 1.0))
        h.reset()
        return score(h.render(probe * (10 ** (drive_db / 20.0)), SR))

    print(f"start score: {objective(x0):.3f} | optimizing (cma, budget {args.budget})",
          flush=True)
    t0 = time.time()
    res = minimize(objective, dim=dim, budget=args.budget, backend="cma", seed=0)
    print(f"final score: {res.loss:.3f} ({res.n_evals} evals, {time.time()-t0:.0f}s)",
          flush=True)

    # ---- stage 4: report ----
    final_vals = {}
    for k, val in zip(knobs, res.x[:-1]):
        h.set_raw(k, float(np.clip(val, 0.0, 1.0)))
    drive_db = -6.0 + 15.0 * float(np.clip(res.x[-1], 0.0, 1.0))
    h.reset()
    matched = h.render(probe * (10 ** (drive_db / 20.0)), SR)
    sf.write(OUT_DIR / "match_result.wav", rms_norm(trim_seconds(matched, SR, WU)).T, SR)

    print("\n=== DIAL THIS IN ===")
    print(f"plugin: {winner['mode']}")
    print(f"input drive relative to a 0.25-peak DI: {drive_db:+.1f} dB")
    for k, val in zip(knobs, res.x[:-1]):
        v = float(np.clip(val, 0.0, 1.0))
        h.set_raw(k, v)
        print(f"  {k:26} raw {v:.3f}   display: {h.get_value(k)}")
    report = {
        "target": str(args.target), "segment_start_s": seg_start,
        "winner": winner["mode"], "final_score": res.loss,
        "knobs_raw": {k: float(np.clip(v, 0, 1)) for k, v in zip(knobs, res.x[:-1])},
        "drive_db": drive_db,
        "mode_ranking": [{"mode": r["mode"], "score": r["score"]} for r in results[:10]],
    }
    json.dump(report, open(OUT_DIR / "match_report.json", "w"), indent=1)
    print(f"\nA/B files: {OUT_DIR}\\match_target_segment.wav vs match_result.wav")
    print(f"report: {OUT_DIR}\\match_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
