"""Calibrate the virtual cell model to an observed run.


"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from cellpilot.model import InitialState, ModelParams, simulate
from cellpilot.schema import CultureRun, Variable





_FIT_VARS: dict[Variable, float] = {
    Variable.VCD: 3.0,
    Variable.GLUCOSE: 1.0,
    Variable.GLUTAMINE: 1.0,
    Variable.LACTATE: 1.0,
    Variable.AMMONIA: 1.0,
}



_PARAMS = [
    ("mu_max", 0.005, 0.08),
    ("Y_x_glc", 1.0, 6.0),
    ("Y_x_gln", 3.0, 14.0),
    ("Y_lac_glc", 0.6, 2.2),
    ("kd_amm", 0.0005, 0.006),
    ("kd_lac", 0.00005, 0.0015),
]


@dataclass
class FitResult:
    params: ModelParams
    initial: InitialState
    rmse: float
    success: bool
    feeds: list = field(default_factory=list)  


def _initial_from_run(run: CultureRun) -> InitialState:
    """Seed inoculation state from the earliest measurement of each variable."""
    init = InitialState()
    for var, attr in [
        (Variable.VCD, "Xv"),
        (Variable.GLUCOSE, "Glc"),
        (Variable.GLUTAMINE, "Gln"),
        (Variable.LACTATE, "Lac"),
        (Variable.AMMONIA, "Amm"),
    ]:
        t, y = run.series(var)
        if y.size:
            setattr(init, attr, float(y[0]))
    vol = run.series(Variable.VOLUME)[1]
    if vol.size:
        init.V = float(vol[0])
    return init


def _residuals(x: np.ndarray, run: CultureRun, initial: InitialState, t_end: float, feeds: list) -> np.ndarray:
    params = ModelParams(**{name: val for (name, _, _), val in zip(_PARAMS, x)})
    try:
        traj = simulate(initial, t_end=t_end, params=params, feeds=feeds)
    except RuntimeError:
        return np.full(_n_obs(run), 1e3)

    res: list[float] = []
    for var, weight in _FIT_VARS.items():
        t_obs, y_obs = run.series(var)
        if not y_obs.size:
            continue
        scale = max(float(np.abs(y_obs).max()), 1e-6)  
        y_hat = np.interp(t_obs, traj.index.to_numpy(), traj[var.value].to_numpy())
        res.extend((weight * (y_hat - y_obs) / scale).tolist())
    return np.asarray(res)


def _n_obs(run: CultureRun) -> int:
    return sum(run.series(v)[1].size for v in _FIT_VARS)


def fit_run(run: CultureRun, feeds: list | None = None) -> FitResult:
    """Estimate run-specific parameters + initial state by least-squares.

    ``feeds`` are known feed boluses (process inputs); pass them for fed-batch runs
    so calibration explains the data given the real feeding, not despite it.
    """
    feeds = feeds or []
    initial = _initial_from_run(run)
    t_end = float(run.times().max()) if run.times().size else 240.0

    x0 = np.array([getattr(ModelParams(), name) for name, _, _ in _PARAMS])
    lo = np.array([b[1] for b in _PARAMS])
    hi = np.array([b[2] for b in _PARAMS])

    sol = least_squares(
        _residuals, x0, bounds=(lo, hi), args=(run, initial, t_end, feeds),
        method="trf", max_nfev=200,
    )
    params = ModelParams(**{name: float(val) for (name, _, _), val in zip(_PARAMS, sol.x)})
    rmse = float(np.sqrt(np.mean(sol.fun**2)))
    return FitResult(
        params=params, initial=initial, rmse=rmse, success=bool(sol.success), feeds=feeds
    )
