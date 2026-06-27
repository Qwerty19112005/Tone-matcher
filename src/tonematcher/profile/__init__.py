"""profile — "profile once": characterize each plugin's reachable tone region offline.

For every owned plugin, sample its parameter space (Latin Hypercube Sampling), render each
sample on a canonical reference DI, embed the renders with the tone metric, and store the
resulting **per-plugin point cloud**. These clouds describe the region of tone-space a
plugin can reach and are the index the static (nominate) stage queries.

Done offline / amortized — never on the query path.
"""
