
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from cellpilot.model import FeedEvent, InitialState, ModelParams, integral_vcd, simulate
from cellpilot.schema import Variable


_DIMS = ["glc0_mM", "gln0_mM", "feed_glc_mM", "feed_gln_mM", "feed_vol_ml"]
_BOUNDS = np.array([[20.0, 60.0], [3.0, 10.0], [100.0, 600.0], [20.0, 150.0], [0.0, 12.0]])


@dataclass
class DesignSpace:
    feed_times_h: tuple[float, ...] = (72, 120, 168, 216)
    t_end: float = 264.0
    max_lactate_mM: float = 90.0
    max_ammonia_mM: float = 12.0
    penalty_weight: float = 50.0


def design_to_setup(d: np.ndarray, space: DesignSpace) -> tuple[InitialState, list[FeedEvent]]:
    """Map a design vector to an initial state + feed schedule."""
    glc0, gln0, feed_glc, feed_gln, feed_vol = d
    initial = InitialState(Glc=float(glc0), Gln=float(gln0))
    feeds = [
        FeedEvent(time_h=t, volume_ml=float(feed_vol), glucose_mM=float(feed_glc), glutamine_mM=float(feed_gln))
        for t in space.feed_times_h
    ] if feed_vol > 0.1 else []
    return initial, feeds


def evaluate_design(d: np.ndarray, params: ModelParams, space: DesignSpace) -> float:
    """Objective: integral viable cell density, penalized for breaching inhibitor caps."""
    initial, feeds = design_to_setup(d, space)
    traj = simulate(initial, t_end=space.t_end, params=params, feeds=feeds)
    ivcd = integral_vcd(traj)
    pen = space.penalty_weight * (
        max(0.0, float(traj[Variable.LACTATE.value].max()) - space.max_lactate_mM)
        + max(0.0, float(traj[Variable.AMMONIA.value].max()) - space.max_ammonia_mM)
    )
    return ivcd - pen


def _normalize(X: np.ndarray) -> np.ndarray:
    return (X - _BOUNDS[:, 0]) / (_BOUNDS[:, 1] - _BOUNDS[:, 0])


def _sample(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.random((n, len(_DIMS)))
    return _BOUNDS[:, 0] + u * (_BOUNDS[:, 1] - _BOUNDS[:, 0])


def _expected_improvement(Xc: np.ndarray, gp: GaussianProcessRegressor, best: float, xi: float = 0.01) -> np.ndarray:
    mu, sigma = gp.predict(Xc, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class DesignResult:
    best_design: dict
    best_objective: float
    history_bo: list[float] = field(default_factory=list)      
    history_random: list[float] = field(default_factory=list)  


def bayesian_optimize(
    params: ModelParams,
    space: DesignSpace | None = None,
    n_init: int = 5,
    n_iter: int = 20,
    pool: int = 600,
    seed: int = 0,
) -> DesignResult:
    """Run BO over the design space; also run matched random search for comparison."""
    space = space or DesignSpace()
    rng = np.random.default_rng(seed)

    
    X = _sample(rng, n_init)
    y = np.array([evaluate_design(d, params, space) for d in X])
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)

    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=0)
        gp.fit(_normalize(X), y)
        cand = _sample(rng, pool)
        ei = _expected_improvement(_normalize(cand), gp, best=float(y.max()))
        nxt = cand[int(np.argmax(ei))]
        X = np.vstack([X, nxt])
        y = np.append(y, evaluate_design(nxt, params, space))

    
    hist_bo = [float(v) for v in np.maximum.accumulate(y)]

    
    rng2 = np.random.default_rng(seed + 1)
    Xr = _sample(rng2, n_init + n_iter)
    yr = np.array([evaluate_design(d, params, space) for d in Xr])
    hist_rand = list(np.maximum.accumulate(yr))

    best_d = X[int(np.argmax(y))]
    return DesignResult(
        best_design={dim: round(float(v), 1) for dim, v in zip(_DIMS, best_d)},
        best_objective=float(y.max()),
        history_bo=hist_bo,
        history_random=[float(v) for v in hist_rand],
    )
