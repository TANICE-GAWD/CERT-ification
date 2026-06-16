

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from cellpilot import tools
from cellpilot.data import simulate_run, to_messy_csv
from cellpilot.ingest import ingest_dataframe
from cellpilot.model import simulate
from cellpilot.schema import Variable

st.set_page_config(page_title="CellPilot", page_icon="🛰️", layout="wide")
st.title(" CellPilot — media-optimization copilot")
st.caption("A v0 of a virtual-cell control panel for CHO fed-batch culture.")

with st.sidebar:
    st.header("Run input")
    source = st.radio("Source", ["Synthetic example", "Upload CSV"])
    if source == "Synthetic example":
        seed = st.number_input("Seed", 0, 999, 7)
        run_df = to_messy_csv(simulate_run(f"S{seed}", seed=int(seed))[0])
    else:
        up = st.file_uploader("Run CSV / Excel", type=["csv", "xlsx", "xls"])
        run_df = pd.read_csv(up) if up and up.name.endswith("csv") else (
            pd.read_excel(up) if up else None
        )

if run_df is None:
    st.info("Upload a run or pick the synthetic example in the sidebar.")
    st.stop()

run = ingest_dataframe(run_df, run_id="UI-run")
fit = tools.calibrate(run)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Virtual cell model fit")
    pred = tools.predict_trajectory(run, fit=fit)
    model_traj = simulate(fit.initial, t_end=float(run.times().max()), params=fit.params)
    obs = run.pivot()
    for var in (Variable.VCD, Variable.GLUCOSE, Variable.LACTATE):
        chart = pd.DataFrame({
            f"{var.value} (model)": model_traj[var.value],
        })
        if var.value in obs.columns:
            chart[f"{var.value} (observed)"] = obs[var.value]
        st.line_chart(chart)
    st.caption(f"Calibration RMSE (normalized): {pred['fit_rmse']:.3f}")

with col2:
    st.subheader("Diagnosis")
    diag = tools.diagnose_state(run)
    if diag["healthy"]:
        st.success("No metabolic issues flagged.")
    for issue in diag["issues"]:
        st.warning(f"**{issue['issue']}** — {issue['variable']} crosses "
                   f"{issue['threshold']} at t={issue['first_time_h']}h")

    st.subheader("Recommended feed")
    rec = tools.recommend_feed(run, fit=fit)
    st.metric("Projected IVCD improvement", f"+{rec['projected_improvement_pct']}%")
    st.dataframe(pd.DataFrame(rec["recommended_feeds"]), hide_index=True)
    st.caption(f"Projected peak lactate {rec['projected_peak_lactate_mM']} mM, "
               f"ammonia {rec['projected_peak_ammonia_mM']} mM (within limits).")

st.divider()
if st.button("Ask the agent to explain", disabled=not os.getenv("ANTHROPIC_API_KEY")):
    from cellpilot.agent import analyze_run
    with st.spinner("CellPilot agent analyzing..."):
        st.markdown(analyze_run(run))
elif not os.getenv("ANTHROPIC_API_KEY"):
    st.caption("Set ANTHROPIC_API_KEY and install the 'agent' extra to enable the LLM narration.")
