# Guitar Tone Matcher

Match the guitar tone of a target recording using **your own locally-installed,
licensed plugin library**. Given a clean DI guitar recording and a target guitar tone,
the system searches for a short chain of your plugins and their knob settings that
reproduces the target.

This is an **offline / research** system: no real-time constraint, pure Python. The whole
design leans on being able to afford slow, exhaustive, render-in-the-loop search.

## Two-stage architecture

1. **Static stage - *nominate* (cheap):** prune the whole plugin library down to a few
   candidate plugins/chains whose capability could plausibly reach the target. This is an
   embedding/metadata query - **no audio is rendered at query time.** Approximate is fine;
   it only prunes.
2. **Dynamic stage - *render + optimize* (expensive, exact):** take the few nominated
   chains, **host and render the real plugins** (turn their knobs), and optimize the knob
   settings with a gradient-free optimizer to minimize a learned audio-similarity distance
   to the target. Reality handles nonlinearity and composition exactly because we actually
   run the plugins.

> **Guiding principle:** surrogates/embeddings only need to **nominate**; the real render
> **disposes**. The metric never has to predict a chain accurately - only propose
> promising candidates that the real render then validates.

It builds on prior work in audio-production style transfer and effect-parameter
estimation - notably ST-ITO, Open-Amp, DeepAFx-ST, and blind audio-graph estimation.

## Status

**Phase 0 - headless plugin hosting sanity check.** Everything else is gated on confirming
your plugins load and render *without their GUI* via [Pedalboard](https://github.com/spotify/pedalboard).

## Setup

Requires **Python 3.11** and [uv](https://github.com/astral-sh/uv) (used to provision
3.11 and lock dependencies).

```bash
# 1. Create the venv and install pinned dependencies (uses uv.lock)
uv sync --extra dev

# 2. Point the project at your locally-installed plugins
cp .env.example .env
#   then edit .env: set PLUGIN_DIR and (optionally) a specific PLUGIN_PATH + DI_PATH

# 3. Gating test - load ONE plugin, list its params, render a DI clip through it
uv run python scripts/phase0_host_check.py
```

If Phase 0 prints a parameter list and writes an output `.wav`, headless hosting works and
the rest of the project is unblocked.

## What is **not** in this repo (by design)

Kept out of git (see `.gitignore`): plugin binaries, all audio (DI / targets / renders /
profiles), model checkpoints, your `.env`, and the virtualenv. **Commercial plugins are
never downloaded, cracked, or sourced by this project** - you install and license them
yourself, and only point the code at their paths via `.env`.

## Layout

```
src/tonematcher/
  hosting/    pedalboard wrappers: load plugins, set params, render audio
  metrics/    MRSTFT baseline + learned-embedding (ST-ITO / Open-Amp) interface
  optimize/   gradient-free knob search (the dynamic stage)
  profile/    "profile once": LHS sampling -> render -> embed -> per-plugin point clouds
  nominate/   static stage: embedding query / blind graph estimation
  data/       DI loading, reference/test-signal generation
  cli.py
scripts/      phase0_host_check.py, phase1_recover_one_knob.py, phase3_recover_preset.py,
              profile_v0.py, train_tone_encoder_v0.py, match_target.py, full_match.py
```

## License

No license yet - to be decided.
