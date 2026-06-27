"""nominate - the static stage: propose candidate plugins/chains for a target.

Embed the target tone and query the per-plugin point clouds (from ``profile``) for the
nearest reachable regions, returning a short list of candidate plugins/chains for the
dynamic stage to render and dispose of. May grow into blind audio-processing-graph
estimation (which blocks, what order) per Lee et al. (arXiv 2303.08610).

No audio is rendered here - this is a cheap embedding/metadata query that only PRUNES.
"""
