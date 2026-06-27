"""Gradient-free minimization over a bounded box - the core of the dynamic stage.

The objective treats the plugin/chain as an arbitrary black box: it takes a normalized
parameter vector in [0, 1]^dim, renders, and returns a scalar distance to the target. We
optimize it with a derivative-free optimizer (nevergrad by default, CMA-ES optionally).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptResult:
    x: np.ndarray  # best parameter vector found, shape (dim,)
    loss: float  # best objective value
    n_evals: int  # number of objective evaluations
    history: list[float] = field(default_factory=list)  # loss per evaluation


def minimize(
    fn,
    dim: int,
    *,
    budget: int = 200,
    backend: str = "nevergrad",
    seed: int = 0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> OptResult:
    """Minimize ``fn`` over ``[bounds]^dim`` and return the best point seen.

    ``fn`` takes a 1-D ``np.ndarray`` of length ``dim`` and returns a float.
    """
    lo, hi = bounds
    state = {"n": 0, "best_x": None, "best": float("inf")}
    history: list[float] = []

    def wrapped(x) -> float:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        value = float(fn(vec))
        state["n"] += 1
        history.append(value)
        if value < state["best"]:
            state["best"] = value
            state["best_x"] = vec.copy()
        return value

    if backend == "nevergrad":
        import nevergrad as ng

        param = ng.p.Array(shape=(dim,)).set_bounds(lo, hi)
        param.random_state.seed(seed)
        optimizer = ng.optimizers.NGOpt(parametrization=param, budget=budget)
        optimizer.minimize(wrapped)
    elif backend == "cma":
        import cma

        x0 = np.full(dim, (lo + hi) / 2.0)
        es = cma.CMAEvolutionStrategy(
            x0,
            0.25 * (hi - lo),
            {"bounds": [lo, hi], "maxfevals": budget, "seed": seed, "verbose": -9},
        )
        es.optimize(wrapped)
    else:
        raise ValueError(f"unknown backend: {backend!r} (use 'nevergrad' or 'cma')")

    best_x = state["best_x"] if state["best_x"] is not None else np.full(dim, (lo + hi) / 2.0)
    return OptResult(x=best_x, loss=state["best"], n_evals=state["n"], history=history)
