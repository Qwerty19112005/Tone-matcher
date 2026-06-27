"""tonematcher - match a target guitar tone using a local plugin library.

Two-stage design:

* ``nominate`` (static): cheaply prune the plugin library to a few candidate chains
  via an embedding/metadata query - no audio rendered at query time.
* ``optimize`` (dynamic): host the real plugins and search their knobs with a
  gradient-free optimizer to minimize a learned audio-similarity distance.

Guiding principle: surrogates/embeddings only NOMINATE; the real render DISPOSES.
"""

__version__ = "0.0.1"
