"""hosting - Pedalboard wrappers for headless plugin hosting.

Responsibilities (the engine under the *dynamic* stage):

* Load a VST3/AU plugin from a path (no GUI).
* Enumerate its parameters and their ranges/types.
* Set parameters from a normalized float vector (the optimizer's search space).
* Render a DI buffer through the plugin (or a chain) at a fixed sample rate.

API note: Pedalboard's plugin-parameter surface has shifted across versions. Verify the
installed version's parameter access (``plugin.parameters`` mapping vs attribute access)
before relying on it - do not assume. See ``scripts/phase0_host_check.py``.
"""
