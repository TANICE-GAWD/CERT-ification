
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cellpilot.model import FeedEvent, InitialState, ModelParams, integral_vcd, simulate
from cellpilot.schema import Variable


@dataclass
class FeedPolicy:
    """A candidate feeding strategy: a list of boluses and its evaluated score."""

    feeds: list[FeedEvent]
    ivcd: float = 0.0
    peak_lactate: float = 0.0
    peak_ammonia: float = 0.0
    penalty: float = 0.0

    @property
    def score(self) -> float:
        return self.ivcd - self.penalty


@dataclass
class OptimizerConfig:
    

    feed_times_h: tuple[float, ...] = (48, 72, 96, 120, 144, 168)
    glucose_feed_mM: float = 300.0       
    glutamine_feed_mM: float = 50.0
    volume_per_feed_ml: tuple[float, ...] = (0.0, 3.0, 5.0, 7.0)  
    max_lactate_mM: float = 90.0         
    max_ammonia_mM: float = 12.0
    penalty_weight: float = 50.0         
    n_random: int = 200                  
    n_refine: int = 60                   


def evaluate_policy(
    initial: InitialState,
    feeds: list[FeedEvent],
    config: OptimizerConfig,
    t_end: float,
    params: ModelParams | None = None,
) -> FeedPolicy:
    """Simulate a feed schedule and score it (IVCD minus constraint penalties)."""
    traj = simulate(initial, t_end=t_end, params=params, feeds=feeds)
    peak_lac = float(traj[Variable.LACTATE.value].max())
    peak_amm = float(traj[Variable.AMMONIA.value].max())
    penalty = config.penalty_weight * (
        max(0.0, peak_lac - config.max_lactate_mM) + max(0.0, peak_amm - config.max_ammonia_mM)
    )
    return FeedPolicy(
        feeds=feeds,
        ivcd=integral_vcd(traj),
        peak_lactate=peak_lac,
        peak_ammonia=peak_amm,
        penalty=penalty,
    )


def _random_schedule(rng: np.random.Generator, config: OptimizerConfig) -> list[FeedEvent]:
    feeds = []
    for t in config.feed_times_h:
        vol = float(rng.choice(config.volume_per_feed_ml))
        if vol > 0:
            feeds.append(
                FeedEvent(
                    time_h=t,
                    volume_ml=vol,
                    glucose_mM=config.glucose_feed_mM,
                    glutamine_mM=config.glutamine_feed_mM,
                )
            )
    return feeds


def _neighbors(feeds: list[FeedEvent], config: OptimizerConfig, rng: np.random.Generator) -> list[FeedEvent]:
    
    by_time = {f.time_h: f for f in feeds}
    t = float(rng.choice(config.feed_times_h))
    vol = float(rng.choice(config.volume_per_feed_ml))
    if vol == 0:
        by_time.pop(t, None)
    else:
        by_time[t] = FeedEvent(t, vol, config.glucose_feed_mM, config.glutamine_feed_mM)
    return [by_time[k] for k in sorted(by_time)]


def optimize_feeds(
    initial: InitialState,
    t_end: float = 240.0,
    params: ModelParams | None = None,
    config: OptimizerConfig | None = None,
    seed: int = 0,
) -> tuple[FeedPolicy, FeedPolicy]:

    config = config or OptimizerConfig()
    rng = np.random.default_rng(seed)

    baseline = evaluate_policy(initial, [], config, t_end, params)
    best = baseline

    
    for _ in range(config.n_random):
        cand = evaluate_policy(initial, _random_schedule(rng, config), config, t_end, params)
        if cand.score > best.score:
            best = cand

    
    for _ in range(config.n_refine):
        cand = evaluate_policy(initial, _neighbors(best.feeds, config, rng), config, t_end, params)
        if cand.score > best.score:
            best = cand

    return best, baseline
