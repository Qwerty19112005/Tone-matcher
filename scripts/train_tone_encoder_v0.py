"""Fine-tune the Open-Amp guitar FX encoder on our own plugin renders (P2.b viability).

Objective: settings-contrastive NT-Xent.
  * positive  = two DIFFERENT DI clips rendered through the SAME (plugin, mode, setting)
  * negatives = other settings in the batch
  * curation  = within a mode, setting pairs whose SAME-CLIP MRSTFT distance is below a
    threshold are masked out of the denominator (saturation-ceiling duplicates would
    otherwise be false negatives; same-content MRSTFT is a trustworthy audibility check)

The trained backbone is saved in the SAME state-dict format as the vendored checkpoint, so
``OpenAmpToneMetric(checkpoint=...)`` loads it directly.

Run:  uv run python scripts/train_tone_encoder_v0.py [--steps 1500] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tonematcher.metrics.learned import (  # noqa: E402
    _ENCODER_CONF,
    _TRAIN_SEG,
    _TRAIN_SR,
    _default_vendor_dir,
    _load_fxenc_module,
)

V0_DIR = REPO_ROOT / "data" / "profiles" / "v0"
MANIFEST = V0_DIR / "manifest.jsonl"
CURATION_CACHE = V0_DIR / "curation_mrstft.json"
CKPT_OUT = REPO_ROOT / "checkpoints" / "tone_encoder_v0.pt"
SR = 48000
WARMUP_S = 0.15
CROP_48K = 36000  # 0.75 s at 48 kHz; resampled to _TRAIN_SEG at 44.1 kHz
SIM_THRESHOLD = 0.15  # same-clip MRSTFT below this -> settings treated as same tone


def load_manifest():
    settings = defaultdict(dict)  # setting_id -> {clip_name: wav_path}
    meta = {}
    for line in open(MANIFEST, encoding="utf-8"):
        r = json.loads(line)
        settings[r["setting_id"]][r["clip"]] = str(V0_DIR / r["wav"])
        meta[r["setting_id"]] = {"mode": f"{r['plugin']}/{r['mode']}"}
    # keep settings with at least 2 clips (need a positive pair)
    settings = {k: v for k, v in settings.items() if len(v) >= 2}
    return settings, meta


def build_curation(settings, meta):
    """Same-clip MRSTFT distance between all setting pairs within each mode (cached)."""
    if CURATION_CACHE.is_file():
        return {tuple(k.split("|")): v for k, v in json.load(open(CURATION_CACHE)).items()}
    import soundfile as sf

    from tonematcher.metrics import MRSTFTMetric

    mrstft = MRSTFTMetric()
    by_mode = defaultdict(list)
    for sid in settings:
        by_mode[meta[sid]["mode"]].append(sid)

    def read(path):
        a, _ = sf.read(path, dtype="float32", always_2d=True)
        return a.T[:, int(WARMUP_S * SR):]

    out = {}
    t0 = time.time()
    for mode, sids in by_mode.items():
        sids = sorted(sids)
        # use the canonical clip (present for every setting)
        canon = {}
        for sid in sids:
            clip0 = sorted(settings[sid].keys())[0]
            canon[sid] = read(settings[sid][clip0])
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                d = mrstft.distance(canon[sids[i]], canon[sids[j]])
                out[(sids[i], sids[j])] = round(float(d), 4)
        print(f"  curation {mode}: {len(sids)} settings ({time.time()-t0:.0f}s)", flush=True)
    json.dump({f"{a}|{b}": v for (a, b), v in out.items()}, open(CURATION_CACHE, "w"))
    return out


class Batcher:
    def __init__(self, settings, meta, sids, batch_settings, seed=0):
        self.settings, self.meta, self.sids = settings, meta, list(sids)
        self.bs = batch_settings
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def _read(self, path):
        if path not in self._cache:
            import soundfile as sf

            a, _ = sf.read(path, dtype="float32", always_2d=True)
            self._cache[path] = np.ascontiguousarray(a.T.mean(axis=0))  # mono
        return self._cache[path]

    def _crop(self, x):
        start_min = int(WARMUP_S * SR)
        hi = max(start_min + 1, len(x) - CROP_48K)
        s = int(self.rng.integers(start_min, hi))
        c = x[s : s + CROP_48K]
        if len(c) < CROP_48K:
            c = np.pad(c, (0, CROP_48K - len(c)))
        return c

    def batch(self):
        chosen = self.rng.choice(self.sids, size=self.bs, replace=False)
        views, ids = [], []
        for sid in chosen:
            clips = list(self.settings[sid].values())
            p1, p2 = self.rng.choice(clips, size=2, replace=False)
            for p in (p1, p2):
                c = self._crop(self._read(p))
                c = c / (np.sqrt(np.mean(c**2)) + 1e-10) * 0.1  # RMS norm, matches wrapper
                views.append(torch.from_numpy(c.astype(np.float32)))
                ids.append(sid)
        return torch.stack(views), ids  # [2B, 36000] at 48k; resample after .to(device)


def to_model_input(x48: torch.Tensor, device: str) -> torch.Tensor:
    """Move a 48 kHz batch to the device, resample there, add channel dim."""
    import torchaudio.functional as AF

    x = AF.resample(x48.to(device), SR, _TRAIN_SR)[:, :_TRAIN_SEG]
    return x.unsqueeze(1)  # [2B, 1, T]


def ntxent_masked(z, ids, meta, curation, temp=0.1):
    """NT-Xent where same-mode near-duplicate settings are masked from the denominator."""
    z = torch.nn.functional.normalize(z, dim=-1)
    n = z.shape[0]
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)

    pos_idx = torch.arange(n) ^ 1  # views are interleaved pairs: 0<->1, 2<->3, ...
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i == j or j == int(pos_idx[i]):
                continue
            a, b = ids[i], ids[j]
            if a == b:
                mask[i, j] = True  # same setting appearing twice: never a negative
                continue
            key = (a, b) if (a, b) in curation else (b, a)
            if key in curation and curation[key] < SIM_THRESHOLD:
                mask[i, j] = True  # audibly-identical tone: not a valid negative
    sim = sim.masked_fill(mask.to(z.device), -1e9)
    return torch.nn.functional.cross_entropy(sim, pos_idx.to(z.device))


@torch.no_grad()
def quick_separation(model, batcher, n_batches=6, device="cpu"):
    """Proxy metric: mean diff-setting distance / mean same-setting distance on val data."""
    model.eval()
    same, diff = [], []
    for _ in range(n_batches):
        x, ids = batcher.batch()
        z = torch.nn.functional.normalize(model(to_model_input(x, device)), dim=-1)
        d = 1.0 - (z @ z.T)
        n = z.shape[0]
        for i in range(0, n, 2):
            same.append(float(d[i, i + 1]))
        for i in range(n):
            for j in range(i + 1, n):
                if ids[i] != ids[j]:
                    diff.append(float(d[i, j]))
    model.train()
    return float(np.mean(diff) / (np.mean(same) + 1e-12)), float(np.mean(same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-settings", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 20
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

    torch.manual_seed(0)
    settings, meta = load_manifest()
    print(f"{len(settings)} settings loaded", flush=True)
    print("building/loading curation matrix (same-clip MRSTFT) ...", flush=True)
    curation = build_curation(settings, meta)
    n_dupe = sum(1 for v in curation.values() if v < SIM_THRESHOLD)
    print(f"curation: {len(curation)} pairs, {n_dupe} masked as audibly-identical "
          f"({100*n_dupe/max(1,len(curation)):.1f}%)", flush=True)

    # split: last 4 settings of each mode -> validation
    by_mode = defaultdict(list)
    for sid in sorted(settings):
        by_mode[meta[sid]["mode"]].append(sid)
    val_sids = [s for sids in by_mode.values() for s in sids[-4:]]
    train_sids = [s for s in settings if s not in set(val_sids)]
    print(f"train {len(train_sids)} / val {len(val_sids)} settings", flush=True)

    train_b = Batcher(settings, meta, train_sids, args.batch_settings, seed=1)
    val_b = Batcher(settings, meta, val_sids, min(12, len(val_sids)), seed=2)

    module = _load_fxenc_module(_default_vendor_dir())
    conf = {k: (list(v) if isinstance(v, list) else v) for k, v in _ENCODER_CONF.items()}
    model = module.FXencoder(conf)
    state = torch.load(_default_vendor_dir() / "Checkpoints" / "FxEncoder-splendidbreeze23-ep45.pt",
                       map_location="cpu")
    model.load_state_dict(state)
    model.to(device).train()
    head = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
    head.to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    sep0, same0 = quick_separation(model, val_b, device=device)
    print(f"step 0 (pretrained): val separation={sep0:.2f} same-dist={same0:.4f}", flush=True)

    CKPT_OUT.parent.mkdir(exist_ok=True)
    best_sep = sep0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, ids = train_b.batch()
        loss = ntxent_masked(head(model(to_model_input(x, device))), ids, meta, curation,
                             temp=args.temp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:5d} loss {float(loss):.4f} ({(time.time()-t0)/step:.2f}s/step)",
                  flush=True)
        if step % 250 == 0 or step == args.steps:
            sep, same_d = quick_separation(model, val_b, device=device)
            tag = ""
            if sep > best_sep:
                best_sep = sep
                torch.save(model.state_dict(), CKPT_OUT)
                tag = " -> saved"
            print(f"step {step:5d} VAL separation={sep:.2f} same-dist={same_d:.4f}{tag}",
                  flush=True)
    if not CKPT_OUT.is_file():  # never improved: save final anyway for inspection
        torch.save(model.state_dict(), CKPT_OUT)
    print(f"done. best val separation {best_sep:.2f} (pretrained was {sep0:.2f}). "
          f"checkpoint: {CKPT_OUT}", flush=True)


if __name__ == "__main__":
    main()
