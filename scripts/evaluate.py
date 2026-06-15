

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cellpilot.data import make_dataset
from cellpilot.fit import fit_run
from cellpilot.residual import HybridModel, PureMLModel, cross_validate
from cellpilot.model import simulate
from cellpilot.schema import Variable

ART = Path("artifacts")


def main() -> None:
    ART.mkdir(exist_ok=True)
    runs = make_dataset(n_runs=20, seed=0)
    print(f"Generated {len(runs)} synthetic CHO fed-batch runs (with lactate shift).")

    cv = cross_validate(runs, n_folds=4)
    print("\nHeld-out VCD RMSE (1e6 cells/mL), 4-fold CV:")
    print(f"  mechanistic only : {cv.rmse_mechanistic:.3f}")
    print(f"  pure ML          : {cv.rmse_pure_ml:.3f}")
    print(f"  HYBRID           : {cv.rmse_hybrid:.3f}")
    best = min(cv.rmse_mechanistic, cv.rmse_pure_ml)
    print(f"  -> hybrid is {(1 - cv.rmse_hybrid / best) * 100:.0f}% better than the next best")

    
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Mechanistic", "Pure ML", "Hybrid"]
    vals = [cv.rmse_mechanistic, cv.rmse_pure_ml, cv.rmse_hybrid]
    bars = ax.bar(labels, vals, color=["
    ax.set_ylabel("Held-out VCD RMSE (1e6 cells/mL)")
    ax.set_title(f"Virtual cell model accuracy ({cv.n_folds}-fold CV, n={cv.n_runs})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(ART / "model_comparison.png", dpi=130)
    print(f"\nsaved {ART / 'model_comparison.png'}")

    
    hold = runs[-1]
    train = runs[:-1]
    fits = [fit_run(r) for r in train]
    hybrid = HybridModel.train(train, fits)
    hold_fit = fit_run(hold)

    t_obs, y_obs = hold.series(Variable.VCD)
    grid = np.linspace(0, float(t_obs.max()), 120)
    mech = simulate(hold_fit.initial, t_end=float(t_obs.max()), params=hold_fit.params)
    mech_vcd = np.interp(grid, mech.index.to_numpy(), mech[Variable.VCD.value].to_numpy())
    hyb_vcd = hybrid.predict(hold_fit, grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_obs, y_obs, color="black", zorder=3, label="observed (held out)")
    ax.plot(grid, mech_vcd, "--", color="
    ax.plot(grid, hyb_vcd, "-", color="
    ax.set_xlabel("time (h)")
    ax.set_ylabel("VCD (1e6 cells/mL)")
    ax.set_title("Held-out run: mechanistic vs hybrid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "fit_example.png", dpi=130)
    print(f"saved {ART / 'fit_example.png'}")


if __name__ == "__main__":
    main()
