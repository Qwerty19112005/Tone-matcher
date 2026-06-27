"""Phase 1 - self-recovery of a single knob (STUB, not yet implemented).

Goal (do NOT build until Phase 0 passes):
  1. Pick one plugin and one continuous knob.
  2. Render the DI through it at a KNOWN target value -> the "target" audio.
  3. Hand the optimizer only the audio (not the value) and let a gradient-free optimizer
     (CMA-ES / nevergrad) search that one knob to minimize an auraloss MRSTFT distance.
  4. Check the recovered value matches the known target -> proves the render+optimize loop.

This is intentionally left as a stub; implement it after Phase 0 (host check) succeeds.
"""

from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "Phase 1 is not implemented yet. Run scripts/phase0_host_check.py first; "
        "implement this only after headless hosting is confirmed."
    )


if __name__ == "__main__":
    main()
