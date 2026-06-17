

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cellpilot.design import DesignSpace, bayesian_optimize
from cellpilot.fit import fit_run
from cellpilot.realdata import load_ieks_batches

ART = Path("artifacts")
N_INIT, N_ITER, SEEDS = 4, 16, [0, 1, 2, 3, 4]


def _experiments_to_threshold(curve: list[float], target: float) -> int:
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1
    return len(curve)


def main() -> None:
    ART.mkdir(exist_ok=True)
    run, feeds = load_ieks_batches()[0]
    params = fit_run(run, feeds=feeds).params
    print(f"Oracle = virtual cell model calibrated to {run.run_id}.")

    bo_curves, rand_curves = [], []
    best_designs = []
    for s in SEEDS:
        res = bayesian_optimize(params, DesignSpace(), n_init=N_INIT, n_iter=N_ITER, seed=s)
        bo_curves.append(res.history_bo)
        rand_curves.append(res.history_random)
        best_designs.append((res.best_objective, res.best_design))

    bo = np.array(bo_curves)
    rand = np.array(rand_curves)
    bo_mean, rand_mean = bo.mean(0), rand.mean(0)

    
    target = 0.95 * max(bo.max(), rand.max())
    bo_n = np.mean([_experiments_to_threshold(c, target) for c in bo_curves])
    rand_n = np.mean([_experiments_to_threshold(c, target) for c in rand_curves])

    best_obj, best_design = max(best_designs, key=lambda x: x[0])
    print(f"\nBest design found (IVCD {best_obj:.0f}): {best_design}")
    print(f"\nExperiments to reach 95% of best (averaged over {len(SEEDS)} seeds):")
    print(f"  Bayesian optimization : {bo_n:.1f}")
    print(f"  Random search         : {rand_n:.1f}")
    if rand_n > 0:
        print(f"  -> BO is ~{rand_n / max(bo_n, 1e-9):.1f}x more sample-efficient")

    
    xs = np.arange(1, bo_mean.size + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, bo_mean, "-o", color="seagreen", lw=2, ms=4, label="Bayesian optimization")
    ax.plot(xs, rand_mean, "--s", color="indianred", lw=2, ms=4, label="Random search")
    ax.fill_between(xs, bo.min(0), bo.max(0), color="seagreen", alpha=0.15)
    ax.axhline(target, color="gray", ls=":", label="95% of best")
    ax.set_xlabel("experiment number")
    ax.set_ylabel("best IVCD found so far")
    ax.set_title("Active learning: experiments proposed by CellPilot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "active_learning.png", dpi=130)
    print(f"\nsaved {ART / 'active_learning.png'}")


if __name__ == "__main__":
    main()
