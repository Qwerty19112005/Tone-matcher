"""data — DI loading and reference / test-signal generation.

* Load clean DI guitar recordings (and target tracks) to mono/stereo float arrays at a
  fixed sample rate.
* Generate canonical reference signals (e.g. the fixed DI used to profile every plugin)
  and synthetic test signals (sine sweeps, noise bursts, plucks) for sanity checks when no
  real DI is available.

All audio lives under ``data/`` and is gitignored — only the directory skeleton is tracked.
"""
