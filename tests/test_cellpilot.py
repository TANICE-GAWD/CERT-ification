

from __future__ import annotations

import numpy as np
import pytest

from cellpilot.data import make_dataset, simulate_run, to_messy_csv
from cellpilot.ingest import ingest_dataframe, map_columns
from cellpilot.model import FeedEvent, InitialState, integral_vcd, simulate
from cellpilot.optimize import OptimizerConfig, optimize_feeds
from cellpilot.residual import cross_validate
from cellpilot.schema import CultureRun, Variable




def test_simulate_returns_all_observables():
    traj = simulate(InitialState(), t_end=120)
    for var in (Variable.VCD, Variable.GLUCOSE, Variable.LACTATE, Variable.VIABILITY):
        assert var.value in traj.columns
    assert traj.index.min() == 0.0
    assert traj.index.max() == pytest.approx(120.0)


def test_growth_then_decline_and_substrate_depletes():
    traj = simulate(InitialState(Xv=0.3, Glc=25, Gln=5), t_end=240)
    vcd = traj[Variable.VCD.value]
    
    assert vcd.max() > 5 * 0.3
    assert traj[Variable.GLUCOSE.value].iloc[-1] < traj[Variable.GLUCOSE.value].iloc[0]
    
    viab = traj[Variable.VIABILITY.value]
    assert viab.between(0.0, 1.0).all()


def test_feed_dilutes_then_raises_glucose():
    traj = simulate(
        InitialState(Glc=25), t_end=120,
        feeds=[FeedEvent(time_h=60, volume_ml=5, glucose_mM=400)],
    )
    pre = traj.loc[traj.index < 60, Variable.GLUCOSE.value].iloc[-1]
    post = traj.loc[traj.index >= 60, Variable.GLUCOSE.value].iloc[0]
    assert post > pre  


def test_fed_batch_beats_batch_ivcd():
    init = InitialState(Xv=0.3, Glc=25, Gln=5)
    batch = integral_vcd(simulate(init, t_end=240))
    fed = integral_vcd(
        simulate(init, t_end=240, feeds=[FeedEvent(72, 5, 300, 50), FeedEvent(144, 5, 300, 50)])
    )
    assert fed > batch




def test_messy_headers_map_to_canonical_variables():
    run, _ = simulate_run("t", seed=3)
    messy = to_messy_csv(run)
    time_col, mapping, unresolved = map_columns(list(messy.columns))
    assert time_col is not None
    mapped_vars = set(mapping.values())
    assert {Variable.VCD, Variable.GLUCOSE, Variable.LACTATE, Variable.AMMONIA} <= mapped_vars
    assert unresolved == []


def test_viability_percent_normalized_to_fraction():
    run, _ = simulate_run("t", seed=4)
    messy = to_messy_csv(run)  
    reingested = ingest_dataframe(messy, run_id="rt")
    _, viab = reingested.series(Variable.VIABILITY)
    assert viab.size > 0
    assert viab.max() <= 1.0


def test_ingestion_roundtrip_preserves_measurement_count():
    run, _ = simulate_run("t", seed=5)
    messy = to_messy_csv(run)
    reingested = ingest_dataframe(messy, run_id="rt")
    assert len(reingested.measurements) == len(run.measurements)




def test_optimizer_improves_on_baseline_within_constraints():
    init = InitialState(Xv=0.3, Glc=25, Gln=5)
    config = OptimizerConfig(n_random=80, n_refine=20)
    best, baseline = optimize_feeds(init, t_end=240, config=config, seed=1)
    assert best.score >= baseline.score
    assert best.ivcd > baseline.ivcd
    
    assert best.penalty == pytest.approx(0.0)




def test_hybrid_cross_validation_invariants():
    cv = cross_validate(make_dataset(n_runs=8, seed=0), n_folds=2)
    
    for v in (cv.rmse_mechanistic, cv.rmse_pure_ml, cv.rmse_hybrid):
        assert np.isfinite(v) and v > 0
    
    assert cv.rmse_hybrid < cv.rmse_pure_ml




def test_culturerun_pivot_and_series():
    run = CultureRun(run_id="x")
    run, _ = simulate_run("t", seed=6)
    wide = run.pivot()
    assert Variable.VCD.value in wide.columns
    t, y = run.series(Variable.GLUCOSE)
    assert t.size == y.size > 0
    assert np.all(np.diff(t) >= 0)  
