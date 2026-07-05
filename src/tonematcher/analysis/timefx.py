"""Blind detection of time-based effects (delay, reverb) in a recording.

Methods are classical and well-proven:

* Delay: averaged real CEPSTRUM (Bogert/Healy/Tukey 1963, invented for echo delay
  estimation): a delayed copy adds a log-spectral ripple with period 1/tau, i.e. a peak
  at quefrency tau. Averaging cepstra across frames reinforces the echo peak against
  musical content. An onset-strength autocorrelation provides a cross-check, and the
  detected time is snapped to musical divisions of the estimated tempo (production
  delays are usually tempo-synced). Echo repeats (feedback) appear as rahmonics at
  multiples of tau.
* Reverb: free-decay-region analysis (the standard blind-RT60 family: Ratnam 2003 ML
  decay estimation, Spectral Decay Distributions, ACE Challenge literature): find the
  gaps after note offsets where only the tail rings, fit the decay slope of the dB
  envelope per segment, and read RT60 from the slowest consistent decays. Works best on
  staccato-rich material, which palm-muted riffs are.

Band conventions (to keep double-tracking out of the delay detector):
  < 40 ms      comb/doubling territory, reported separately, never "delay"
  40-160 ms    slapback
  160-1200 ms  musical delay (tempo-snapped)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-10


# ---------------------------------------------------------------------------- helpers
def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=0)
    return a


def _active_region(x: np.ndarray, sr: int, max_seconds: float = 60.0) -> np.ndarray:
    """Trim leading/trailing quiet and cap the analysis length."""
    frame = 2048
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(-1, frame) ** 2, axis=1) + _EPS)
    thresh = np.max(rms) * 10 ** (-40 / 20)
    idx = np.where(rms > thresh)[0]
    if len(idx) == 0:
        return x
    x = x[idx[0] * frame : (idx[-1] + 1) * frame]
    return x[: int(max_seconds * sr)]


# ---------------------------------------------------------------------------- delay
@dataclass
class DelayEstimate:
    present: bool
    time_ms: float | None = None
    confidence: float = 0.0  # z-score of the cepstral peak
    division: str | None = None  # nearest musical division at the estimated tempo
    bpm: float | None = None
    feedback: bool = False  # rahmonics found at 2x/3x the delay time
    doubling_ms: float | None = None  # strongest sub-40ms peak (double-tracking/comb)
    candidates: list = field(default_factory=list)  # (ms, z) for transparency


_DIVISIONS = {
    "1/1": 4.0, "1/2.": 3.0, "1/2": 2.0, "1/4.": 1.5, "1/2T": 4 / 3, "1/4": 1.0,
    "1/8.": 0.75, "1/4T": 2 / 3, "1/8": 0.5, "1/16.": 0.375, "1/8T": 1 / 3,
    "1/16": 0.25, "1/16T": 1 / 6,
}


def _avg_cepstrum(x: np.ndarray, sr: int, n_fft: int = 65536, hop: int = 16384):
    """Average real cepstrum over energetic frames. Returns (quefrency_s, cepstrum)."""
    frames = []
    for s in range(0, max(1, len(x) - n_fft), hop):
        w = x[s : s + n_fft]
        if np.sqrt(np.mean(w**2)) < 1e-4:
            continue
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) + _EPS
        frames.append(np.fft.irfft(np.log(spec)))
    if not frames:
        return None, None
    cep = np.mean(frames, axis=0)
    q = np.arange(len(cep)) / sr
    return q, cep


def _zscore_peaks(values: np.ndarray, lo: int, hi: int, min_z: float, distance: int):
    """Peaks in values[lo:hi] scored as z-scores against that region's statistics."""
    from scipy.signal import find_peaks

    region = values[lo:hi]
    mu, sigma = np.mean(region), np.std(region) + _EPS
    z = (region - mu) / sigma
    peaks, props = find_peaks(z, height=min_z, distance=distance)
    return [(lo + int(p), float(z[p])) for p in peaks]


def _pitch_bases(cep: np.ndarray, sr: int) -> list[float]:
    """Strong low-quefrency peaks (2-15 ms) = pitch periods; their rahmonics pollute
    the delay band (cepstral pitch detection is the same phenomenon)."""
    ms = lambda t: int(t * sr / 1000.0)  # noqa: E731
    peaks = _zscore_peaks(cep, ms(2), ms(15), min_z=5.0, distance=ms(1))
    return [idx / sr * 1000.0 for idx, _ in sorted(peaks, key=lambda p: p[1], reverse=True)[:3]]


def _is_pitch_rahmonic(t_ms: float, bases: list[float]) -> bool:
    # pitch rahmonics fade with multiple; beyond ~12x they are negligible, and an
    # unbounded comb would blanket the delay band with false rejections
    for b in bases:
        k = round(t_ms / b)
        if 1 <= k <= 12 and abs(t_ms - k * b) < max(0.6, 0.12 * b):
            return True
    return False


def _gap_evidence(x: np.ndarray, sr: int, t_ms: float) -> tuple[float, float]:
    """Echo-in-the-rests evidence for a delay at lag tau.

    Anchors are INITIAL events: loud frames not preceded by a loud frame tau earlier.
    (Without that exclusion, echo frames themselves anchor the test, and a single-repeat
    echo is followed by silence, poisoning the statistics.) For a true delay, even the
    quietest quartile of frames tau after an anchor carries echo energy; in dry playing
    those frames are silent whenever the player rests. The off-lag ratio vetoes smooth
    reverb tails, which leak at every lag (ratio ~ 1 or below) unlike a tau-specific
    echo. Returns (absolute_leak, off_lag_ratio); (-1, -1) if unverifiable.
    """
    frame, hop = 1024, 256
    n = (len(x) - frame) // hop
    if n < 50:
        return (-1.0, -1.0)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    e = np.sqrt(np.mean(x[idx] ** 2, axis=1) + _EPS)
    loud = e > 0.3 * float(np.percentile(e, 90))
    loud_median = float(np.median(e[loud])) + _EPS

    def leak(lag_frames: int) -> float:
        vals = [e[f + lag_frames] for f in range(lag_frames, n - lag_frames)
                if loud[f] and not loud[f - lag_frames]]
        if len(vals) < 12:
            return -1.0
        # p10, not p25: a true delay echoes after EVERY anchor (100% coverage), while
        # rhythmic playing can put notes at some IOI multiple ~80% of the time
        return float(np.percentile(vals, 10)) / loud_median

    lag = int(round(t_ms / 1000.0 * sr / hop))
    l_tau = leak(lag)
    if l_tau < 0:
        return (-1.0, -1.0)
    l_ctrl = max(leak(int(lag * 0.71)), leak(int(lag * 1.37)), 0.02)
    return (l_tau, l_tau / l_ctrl)


def detect_delay(audio: np.ndarray, sr: int, min_z: float = 6.0) -> DelayEstimate:
    """Detect a delay/echo effect and estimate its time.

    Evidence chain: an averaged-cepstrum peak (z >= min_z) that is not a pitch
    rahmonic AND that shows tau-specific echo energy after initial events (gap
    evidence). Unverifiable material (no rests) requires an overwhelming peak.
    """
    x = _active_region(_mono(audio), sr)
    q, cep = _avg_cepstrum(x, sr)
    if q is None:
        return DelayEstimate(present=False)

    ms = lambda t: int(t * sr / 1000.0)  # noqa: E731
    bases = _pitch_bases(cep, sr)

    # doubling band (< 40 ms), pitch-filtered, reported but never classified as delay
    dbl = _zscore_peaks(cep, ms(8), ms(40), min_z=3.0, distance=ms(2))
    dbl = [(i, z) for i, z in dbl if not _is_pitch_rahmonic(i / sr * 1000.0, bases)]
    doubling = (dbl[0][0] / sr * 1000.0) if dbl else None

    # delay band (40 - 1200 ms), pitch-filtered
    cands = _zscore_peaks(cep, ms(40), ms(1200), min_z=min_z, distance=ms(8))
    cands_ms = sorted(((idx / sr * 1000.0, z) for idx, z in cands
                       if not _is_pitch_rahmonic(idx / sr * 1000.0, bases)),
                      key=lambda c: c[1], reverse=True)[:6]

    est = DelayEstimate(present=False, doubling_ms=doubling,
                        candidates=[(round(m, 1), round(z, 1)) for m, z in cands_ms])
    if not cands_ms:
        return est


    # onset-strength autocorrelation cross-check (playing periodicity also peaks here,
    # so this only RAISES confidence; the cepstral peak is the required evidence)
    import librosa

    onset = librosa.onset.onset_strength(y=x.astype(np.float32), sr=sr, hop_length=512)
    onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    ac = ac / (ac[0] + _EPS)
    lag_of = lambda m: int(round(m / 1000.0 * sr / 512))  # noqa: E731

    tempo, _ = librosa.beat.beat_track(y=x.astype(np.float32), sr=sr)
    bpm = float(np.atleast_1d(tempo)[0]) or None

    # gap-evidence gate: tau-specific echo after initial events, reverb-tail vetoed
    best = None
    for m, z in cands_ms:
        l_abs, ratio = _gap_evidence(x, sr, m)
        if l_abs < 0:  # no rests to verify against: demand an overwhelming peak
            if z >= 25.0:
                best = (m, z)
                break
        elif l_abs >= 0.08 and ratio >= 1.5:
            best = (m, z)
            break
    if best is None:
        return est
    best_ms, best_z = best
    est.present = True
    est.time_ms = round(best_ms, 1)
    est.confidence = round(best_z, 2)
    lag = lag_of(best_ms)
    if 0 < lag < len(ac) and ac[lag] > 0.15:
        est.confidence = round(est.confidence * 1.25, 2)  # corroborated

    # rahmonics -> feedback/repeats
    for mult in (2.0, 3.0):
        m2 = best_ms * mult
        if m2 < 1200 and any(abs(c - m2) < max(4.0, 0.04 * m2) for c, _ in cands_ms[1:]):
            est.feedback = True

    # tempo-division snap
    if bpm:
        est.bpm = round(bpm, 1)
        beat_ms = 60000.0 / bpm
        best_div, best_err = None, 1.0
        for name, frac in _DIVISIONS.items():
            err = abs(best_ms - beat_ms * frac) / (beat_ms * frac)
            if err < best_err:
                best_div, best_err = name, err
        if best_err < 0.06:
            est.division = best_div
    return est


# ---------------------------------------------------------------------------- reverb
@dataclass
class ReverbEstimate:
    present: bool
    rt60_s: float | None = None
    klass: str = "dry"  # dry | room | hall
    wet_rough: float | None = None  # tail energy relative to note energy (rough)
    n_segments: int = 0
    confidence: float = 0.0


def _decay_segments(env_db: np.ndarray, hop_s: float, floor_db: float):
    """Free-decay regions: from a local peak, monotonic-ish fall until rise or floor."""
    segs = []
    i, n = 1, len(env_db)
    min_len = int(0.12 / hop_s)
    while i < n - min_len:
        if env_db[i] >= env_db[i - 1] and (i + 1 >= n or env_db[i] > env_db[i + 1]):
            j = i + 1
            run_up = 0
            while j < n:
                if env_db[j] > env_db[j - 1] + 0.5:
                    run_up += 1
                    if run_up >= 3:  # a real new onset, not jitter
                        j -= 2
                        break
                else:
                    run_up = 0
                if env_db[j] < floor_db:
                    break
                j += 1
            if j - i >= min_len and env_db[i] - env_db[min(j, n - 1)] > 12.0:
                segs.append((i, min(j, n - 1)))
            i = j + 1
        else:
            i += 1
    return segs


def detect_reverb(audio: np.ndarray, sr: int) -> ReverbEstimate:
    """Estimate reverb presence and RT60 from free-decay regions of the envelope."""
    x = _active_region(_mono(audio), sr)
    frame, hop = 1024, 256
    n = (len(x) - frame) // hop
    if n < 20:
        return ReverbEstimate(present=False)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    env = np.sqrt(np.mean(x[idx] ** 2, axis=1) + _EPS)
    env_db = 20 * np.log10(env + _EPS)
    env_db -= np.max(env_db)
    hop_s = hop / sr
    floor_db = float(np.percentile(env_db, 5)) + 6.0

    segs = _decay_segments(env_db, hop_s, floor_db)
    if not segs:
        return ReverbEstimate(present=False)

    rts = []
    for a, b in segs:
        seg = env_db[a : b + 1]
        # fit the -5 dB .. -25 dB portion below the segment peak (standard T20-style)
        peak = seg[0]
        lo_i = np.argmax(seg < peak - 5.0) if np.any(seg < peak - 5.0) else 0
        hi_i = np.argmax(seg < peak - 25.0) if np.any(seg < peak - 25.0) else len(seg) - 1
        if hi_i - lo_i < int(0.05 / hop_s):
            continue
        t = np.arange(lo_i, hi_i + 1) * hop_s
        y = seg[lo_i : hi_i + 1]
        slope = np.polyfit(t, y, 1)[0]  # dB per second (negative)
        if slope < -1.0:
            rts.append(-60.0 / slope)
    if not rts:
        return ReverbEstimate(present=False, n_segments=len(segs))

    rts = np.array(rts)
    # the reverb tail sets the SLOWEST consistent decay; notes themselves decay faster
    rt60 = float(np.percentile(rts, 75))
    klass = "dry" if rt60 < 0.15 else ("room" if rt60 < 0.6 else "hall")
    spread = float(np.std(rts) / (np.mean(rts) + _EPS))
    conf = max(0.0, 1.0 - spread) * min(1.0, len(rts) / 6.0)

    # rough wetness: energy in decay regions vs total
    tail_e = sum(float(np.sum(env[a : b + 1] ** 2)) for a, b in segs)
    wet = tail_e / (float(np.sum(env**2)) + _EPS)
    return ReverbEstimate(present=rt60 >= 0.15, rt60_s=round(rt60, 3), klass=klass,
                          wet_rough=round(wet, 3), n_segments=len(rts),
                          confidence=round(conf, 2))


def detect_time_fx(audio: np.ndarray, sr: int) -> dict:
    """Run both detectors; the result is what the matcher LOCKS before optimizing."""
    d = detect_delay(audio, sr)
    r = detect_reverb(audio, sr)
    return {"delay": d, "reverb": r}
