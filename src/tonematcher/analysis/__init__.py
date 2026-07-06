"""analysis - blind analysis of the TARGET recording before matching.

First member: time-based effect detection (delay presence/time/tempo-division, reverb
presence/decay). Per the matching pipeline's ordering rule, time-based effects are
detected here first and then LOCKED during preset optimization: chain placement may be
searched, but on/off state and timing are fixed by measurement, not by the optimizer.
"""

from __future__ import annotations

from .benchmark import scorecard
from .timefx import DelayEstimate, ReverbEstimate, detect_delay, detect_reverb, detect_time_fx

__all__ = ["detect_delay", "detect_reverb", "detect_time_fx", "DelayEstimate",
           "ReverbEstimate", "scorecard"]
