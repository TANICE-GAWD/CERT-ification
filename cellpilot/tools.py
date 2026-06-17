

from __future__ import annotations

import numpy as np

from cellpilot.design import DesignSpace, bayesian_optimize
from cellpilot.fit import FitResult, fit_run
from cellpilot.model import FeedEvent, ModelParams, integral_vcd, simulate
from cellpilot.optimize import OptimizerConfig, optimize_feeds
from cellpilot.schema import CultureRun, Variable


LOW_GLUCOSE = 2.0
LOW_GLUTAMINE = 0.5
HIGH_LACTATE = 80.0
HIGH_AMMONIA = 10.0


def calibrate(run: CultureRun) -> FitResult:
    
    return fit_run(run)


def query_run(run: CultureRun) -> dict:
    
    out: dict = {"run_id": run.run_id, "cell_line": run.cell_line}
    times = run.times()
    out["duration_h"] = float(times.max()) if times.size else 0.0
    t, vcd = run.series(Variable.VCD)
    out["peak_vcd"] = float(vcd.max()) if vcd.size else None
    out["peak_vcd_time_h"] = float(t[int(np.argmax(vcd))]) if vcd.size else None
    _, viab = run.series(Variable.VIABILITY)
    out["final_viability"] = float(viab[-1]) if viab.size else None
    for var in (Variable.GLUCOSE, Variable.GLUTAMINE, Variable.LACTATE, Variable.AMMONIA):
        _, y = run.series(var)
        out[f"final_{var.value}"] = float(y[-1]) if y.size else None
    return out


def diagnose_state(run: CultureRun) -> dict:
    
    flags: list[dict] = []

    def _first_cross(var: Variable, threshold: float, direction: str) -> float | None:
        t, y = run.series(var)
        if not y.size:
            return None
        mask = y < threshold if direction == "below" else y > threshold
        idx = np.argmax(mask) if mask.any() else None
        return float(t[idx]) if idx is not None and mask.any() else None

    checks = [
        (Variable.GLUCOSE, LOW_GLUCOSE, "below", "glucose depletion"),
        (Variable.GLUTAMINE, LOW_GLUTAMINE, "below", "glutamine depletion"),
        (Variable.LACTATE, HIGH_LACTATE, "above", "lactate accumulation"),
        (Variable.AMMONIA, HIGH_AMMONIA, "above", "ammonia accumulation"),
    ]
    for var, thr, direction, label in checks:
        when = _first_cross(var, thr, direction)
        if when is not None:
            flags.append({"issue": label, "variable": var.value, "threshold": thr, "first_time_h": when})

    _, viab = run.series(Variable.VIABILITY)
    healthy = bool(viab.size and viab[-1] > 0.7)
    return {"healthy": healthy and not flags, "issues": flags}


def predict_trajectory(run: CultureRun, extend_h: float = 0.0, fit: FitResult | None = None) -> dict:
    
    fit = fit or calibrate(run)
    t_end = (run.times().max() if run.times().size else 240.0) + extend_h
    traj = simulate(fit.initial, t_end=float(t_end), params=fit.params, feeds=fit.feeds)
    return {
        "fit_rmse": fit.rmse,
        "predicted_peak_vcd": float(traj[Variable.VCD.value].max()),
        "predicted_ivcd": integral_vcd(traj),
        "times_h": traj.index.tolist(),
        "vcd": traj[Variable.VCD.value].tolist(),
    }


def recommend_feed(run: CultureRun, fit: FitResult | None = None, seed: int = 0) -> dict:

    fit = fit or calibrate(run)
    t_end = float(run.times().max()) if run.times().size else 240.0
    best, baseline = optimize_feeds(fit.initial, t_end=t_end, params=fit.params, seed=seed)
    improvement = (best.ivcd - baseline.ivcd) / baseline.ivcd * 100.0 if baseline.ivcd else 0.0
    return {
        "recommended_feeds": [
            {
                "time_h": f.time_h,
                "volume_ml": round(f.volume_ml, 2),
                "glucose_mM": f.glucose_mM,
                "glutamine_mM": f.glutamine_mM,
            }
            for f in best.feeds
        ],
        "projected_ivcd": round(best.ivcd, 1),
        "baseline_ivcd": round(baseline.ivcd, 1),
        "projected_improvement_pct": round(improvement, 1),
        "projected_peak_lactate_mM": round(best.peak_lactate, 1),
        "projected_peak_ammonia_mM": round(best.peak_ammonia, 1),
    }


def propose_next_experiment(run: CultureRun, fit: FitResult | None = None, n_iter: int = 12, seed: int = 0) -> dict:

    fit = fit or calibrate(run)
    res = bayesian_optimize(fit.params, DesignSpace(), n_iter=n_iter, seed=seed)
    return {
        "proposed_design": res.best_design,
        "predicted_ivcd": round(res.best_objective, 1),
        "experiments_simulated": len(res.history_bo),
        "note": "Design proposed by Bayesian optimization over the calibrated virtual cell model.",
    }
