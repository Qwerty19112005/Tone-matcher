"""optimize — gradient-free knob search (the dynamic stage).

Given a hosted plugin/chain, a target, and a metric, search the (normalized) parameter
vector that minimizes ``metric.distance(render(params), target)``. The plugin is treated
as an arbitrary, non-differentiable black box, so we use gradient-free optimizers:

* CMA-ES (``cma``)
* nevergrad (NGOpt / TBPSA / ...)

This mirrors ST-ITO's optimizer role. The render is exact, so this stage DISPOSES of the
candidates the static stage nominated.
"""
