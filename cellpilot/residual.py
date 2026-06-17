


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cellpilot.fit import FitResult, fit_run
from cellpilot.model import simulate
from cellpilot.schema import CultureRun, Variable

TARGET = Variable.VCD
_STATE_VARS = [Variable.VCD, Variable.GLUCOSE, Variable.GLUTAMINE, Variable.LACTATE, Variable.AMMONIA]


def _mech_states(fit: FitResult, times: np.ndarray) -> dict[Variable, np.ndarray]:
    """Mechanistic state predictions interpolated onto ``times``."""
    t_end = float(times.max()) if times.size else 240.0
    mech = simulate(fit.initial, t_end=t_end, params=fit.params, feeds=fit.feeds)
    grid = mech.index.to_numpy()
    return {v: np.interp(times, grid, mech[v.value].to_numpy()) for v in _STATE_VARS}


def _hybrid_features(fit: FitResult, times: np.ndarray) -> np.ndarray:
    """Features for the residual model: time + mechanistic state context."""
    states = _mech_states(fit, times)
    t_norm = times / (times.max() if times.size and times.max() > 0 else 1.0)
    return np.column_stack(
        [
            times,
            t_norm,
            states[Variable.VCD],
            states[Variable.GLUCOSE],
            states[Variable.GLUTAMINE],
            states[Variable.LACTATE],
            states[Variable.AMMONIA],
            (states[Variable.GLUCOSE] < 3.0).astype(float),  
        ]
    )


def _pure_ml_features(run: CultureRun, fit: FitResult, times: np.ndarray) -> np.ndarray:
    
    init = fit.initial
    t_norm = times / (times.max() if times.size and times.max() > 0 else 1.0)
    n = times.size
    return np.column_stack(
        [
            times,
            t_norm,
            np.full(n, init.Xv),
            np.full(n, init.Glc),
            np.full(n, init.Gln),
            np.full(n, init.Amm),
        ]
    )


def _new_gbm() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
    )


@dataclass
class HybridModel:
    

    residual_gbm: GradientBoostingRegressor

    @classmethod
    def train(cls, runs: list[CultureRun], fits: list[FitResult]) -> "HybridModel":
        X, resid = [], []
        for run, fit in zip(runs, fits):
            t, y = run.series(TARGET)
            if not y.size:
                continue
            mech_vcd = _mech_states(fit, t)[TARGET]
            X.append(_hybrid_features(fit, t))
            resid.append(y - mech_vcd)
        gbm = _new_gbm()
        gbm.fit(np.vstack(X), np.concatenate(resid))
        return cls(residual_gbm=gbm)

    def predict(self, fit: FitResult, times: np.ndarray) -> np.ndarray:
        mech_vcd = _mech_states(fit, times)[TARGET]
        return mech_vcd + self.residual_gbm.predict(_hybrid_features(fit, times))


@dataclass
class PureMLModel:
    gbm: GradientBoostingRegressor

    @classmethod
    def train(cls, runs: list[CultureRun], fits: list[FitResult]) -> "PureMLModel":
        X, y_all = [], []
        for run, fit in zip(runs, fits):
            t, y = run.series(TARGET)
            if not y.size:
                continue
            X.append(_pure_ml_features(run, fit, t))
            y_all.append(y)
        gbm = _new_gbm()
        gbm.fit(np.vstack(X), np.concatenate(y_all))
        return cls(gbm=gbm)

    def predict(self, run: CultureRun, fit: FitResult, times: np.ndarray) -> np.ndarray:
        return self.gbm.predict(_pure_ml_features(run, fit, times))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass
class CVResult:
    rmse_mechanistic: float
    rmse_pure_ml: float
    rmse_hybrid: float
    n_runs: int
    n_folds: int


def cross_validate(
    runs: list[CultureRun], n_folds: int = 4, feeds_list: list | None = None
) -> CVResult:

    feeds_list = feeds_list or [[] for _ in runs]
    fits = [fit_run(r, feeds=f) for r, f in zip(runs, feeds_list)]
    idx = np.arange(len(runs))
    folds = np.array_split(idx, n_folds)

    e_mech, e_ml, e_hyb = [], [], []
    for fold in folds:
        test = set(fold.tolist())
        tr = [i for i in idx if i not in test]
        tr_runs, tr_fits = [runs[i] for i in tr], [fits[i] for i in tr]
        hybrid = HybridModel.train(tr_runs, tr_fits)
        pure = PureMLModel.train(tr_runs, tr_fits)

        for i in fold:
            run, fit = runs[i], fits[i]
            t, y = run.series(TARGET)
            if not y.size:
                continue
            mech = _mech_states(fit, t)[TARGET]
            e_mech.append(_rmse(y, mech))
            e_ml.append(_rmse(y, pure.predict(run, fit, t)))
            e_hyb.append(_rmse(y, hybrid.predict(fit, t)))

    return CVResult(
        rmse_mechanistic=float(np.mean(e_mech)),
        rmse_pure_ml=float(np.mean(e_ml)),
        rmse_hybrid=float(np.mean(e_hyb)),
        n_runs=len(runs),
        n_folds=n_folds,
    )
