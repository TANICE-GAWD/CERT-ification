"""Offline end-to-end demo
"""

from __future__ import annotations

from cellpilot import tools
from cellpilot.data import simulate_run, to_messy_csv
from cellpilot.ingest import ingest_dataframe


def main() -> None:
    
    synth_run, _ = simulate_run("DEMO", seed=7)
    messy = to_messy_csv(synth_run)
    print("=== 1. Raw instrument export (messy headers) ===")
    print("columns:", list(messy.columns))

    
    run = ingest_dataframe(messy, run_id="DEMO-001")
    print(f"\n=== 2. Ingested -> {len(run.measurements)} normalized measurements ===")

    
    print("\n=== 3. Run summary (query_run) ===")
    for k, v in tools.query_run(run).items():
        print(f"  {k}: {v}")

    
    print("\n=== 4. Diagnosis (diagnose_state) ===")
    diag = tools.diagnose_state(run)
    print(f"  healthy: {diag['healthy']}")
    for issue in diag["issues"]:
        print(f"  - {issue['issue']}: {issue['variable']} crosses {issue['threshold']} at t={issue['first_time_h']}h")

    
    fit = tools.calibrate(run)
    pred = tools.predict_trajectory(run, fit=fit)
    print("\n=== 5. Calibrated virtual cell model (predict_trajectory) ===")
    print(f"  fit RMSE (normalized): {pred['fit_rmse']:.3f}")
    print(f"  predicted peak VCD: {pred['predicted_peak_vcd']:.1f} (1e6 cells/mL)")

    
    print("\n=== 6. Recommendation (recommend_feed) ===")
    rec = tools.recommend_feed(run, fit=fit)
    for f in rec["recommended_feeds"]:
        print(f"  feed @ {f['time_h']}h: {f['volume_ml']} mL "
              f"(glucose {f['glucose_mM']} mM, glutamine {f['glutamine_mM']} mM)")
    print(f"  projected IVCD: {rec['projected_ivcd']} vs baseline {rec['baseline_ivcd']} "
          f"(+{rec['projected_improvement_pct']}%)")
    print(f"  projected peak lactate {rec['projected_peak_lactate_mM']} mM, "
          f"ammonia {rec['projected_peak_ammonia_mM']} mM")


if __name__ == "__main__":
    main()
