
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cellpilot.fit import fit_run
from cellpilot.model import simulate
from cellpilot.realdata import load_ieks_batches
from cellpilot.residual import HybridModel, cross_validate
from cellpilot.schema import Variable

ART = Path("artifacts")


def main() -> None:
    ART.mkdir(exist_ok=True)
    batches = load_ieks_batches()
    runs = [r for r, _ in batches]
    feeds_list = [f for _, f in batches]
    print(f"Loaded {len(runs)} REAL third-party fed-batch runs (ieks-9batches).")

    cv = cross_validate(runs, n_folds=len(runs), feeds_list=feeds_list)  
    print("\nHeld-out VCD RMSE (10^6 cells/mL), leave-one-out CV on REAL data:")
    print(f"  mechanistic only : {cv.rmse_mechanistic:.3f}")
    print(f"  pure ML          : {cv.rmse_pure_ml:.3f}")
    print(f"  HYBRID           : {cv.rmse_hybrid:.3f}")
    best_base = min(cv.rmse_mechanistic, cv.rmse_pure_ml)
    delta = (1 - cv.rmse_hybrid / best_base) * 100
    verdict = "better than" if delta > 0 else "no better than"
    print(f"  -> hybrid is {abs(delta):.0f}% {verdict} the next-best baseline")

    
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [cv.rmse_mechanistic, cv.rmse_pure_ml, cv.rmse_hybrid]
    bars = ax.bar(["Mechanistic", "Pure ML", "Hybrid"], vals,
                  color=["cornflowerblue", "indianred", "seagreen"])
    ax.set_ylabel("Held-out VCD RMSE (10^6 cells/mL)")
    ax.set_title(f"REAL fed-batch data (leave-one-out, n={cv.n_runs})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(ART / "model_comparison.png", dpi=130)
    print(f"\nsaved {ART / 'model_comparison.png'}")

    
    hold_run, hold_feeds = batches[-1]
    train = batches[:-1]
    fits = [fit_run(r, feeds=f) for r, f in train]
    hybrid = HybridModel.train([r for r, _ in train], fits)
    hold_fit = fit_run(hold_run, feeds=hold_feeds)

    t_obs, y_obs = hold_run.series(Variable.VCD)
    grid = np.linspace(0, float(t_obs.max()), 160)
    mech = simulate(hold_fit.initial, t_end=float(t_obs.max()), params=hold_fit.params, feeds=hold_feeds)
    mech_vcd = np.interp(grid, mech.index.to_numpy(), mech[Variable.VCD.value].to_numpy())
    hyb_vcd = hybrid.predict(hold_fit, grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_obs, y_obs, color="black", zorder=3, label=f"observed ({hold_run.run_id}, held out)")
    ax.plot(grid, mech_vcd, "--", color="cornflowerblue", label="mechanistic")
    ax.plot(grid, hyb_vcd, "-", color="seagreen", lw=2, label="hybrid")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("VCD (10^6 cells/mL)")
    ax.set_title("Real held-out run: mechanistic vs hybrid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "fit_example.png", dpi=130)
    print(f"saved {ART / 'fit_example.png'}")


if __name__ == "__main__":
    main()
