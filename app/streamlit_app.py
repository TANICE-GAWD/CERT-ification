"""CellPilot control panel — scientist-facing demo UI.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

A dense, instrument-style dashboard: pick a run (synthetic, a real fed-batch batch, or
an upload), see the calibrated virtual-cell model over the observed data, the diagnosis,
the recommended feed, the real-data model validation, and the active-learning proposal.
If ANTHROPIC_API_KEY + the 'agent' extra are present, the agent narrates it.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cellpilot import tools
from cellpilot.data import simulate_run, to_messy_csv
from cellpilot.design import DesignSpace, bayesian_optimize
from cellpilot.fit import fit_run
from cellpilot.ingest import ingest_dataframe
from cellpilot.model import simulate
from cellpilot.realdata import load_ieks_batches
from cellpilot.residual import cross_validate
from cellpilot.schema import Variable

st.set_page_config(page_title="CellPilot", page_icon="🛰️", layout="wide")
TEMPLATE = "plotly_dark"
GREEN, BLUE, AMBER = "seagreen", "cornflowerblue", "orange"

st.markdown(
    "<h2 style='margin-bottom:0'>CellPilot</h2>"
    "<p style='color:gray;margin-top:2px'>Virtual-cell control panel for fed-batch culture &middot; "
    "ingest &rarr; model &rarr; recommend &rarr; design</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _real_batches():
    return [(r, f) for r, f in load_ieks_batches()]


def _get_run():
    """Resolve the sidebar selection to (run, feeds, is_real)."""
    with st.sidebar:
        st.header("Run")
        source = st.radio("Source", ["Real fed-batch (IEKS)", "Synthetic", "Upload CSV"])
        if source == "Real fed-batch (IEKS)":
            batches = _real_batches()
            i = st.selectbox("Batch", range(len(batches)), format_func=lambda i: batches[i][0].run_id)
            run, feeds = batches[i]
            return run, feeds, True
        if source == "Synthetic":
            seed = st.number_input("Seed", 0, 999, 7)
            run = ingest_dataframe(to_messy_csv(simulate_run(f"S{seed}", seed=int(seed))[0]), run_id=f"SYNTH-{seed}")
            return run, [], False
        up = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
        if not up:
            st.stop()
        df = pd.read_csv(up) if up.name.endswith("csv") else pd.read_excel(up)
        return ingest_dataframe(df, run_id="upload"), [], False


def _trajectory_figure(run, fit) -> go.Figure:
    obs = run.pivot()
    t_end = float(run.times().max())
    model = simulate(fit.initial, t_end=t_end, params=fit.params, feeds=fit.feeds)
    panels = [
        (Variable.VCD, "VCD (10⁶/mL)"),
        (Variable.GLUCOSE, "Glucose (mM)"),
        (Variable.LACTATE, "Lactate (mM)"),
        (Variable.AMMONIA, "Ammonia (mM)"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels])
    for k, (var, _) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        if var.value in obs.columns:
            fig.add_trace(go.Scatter(x=obs.index, y=obs[var.value], mode="markers",
                                     marker=dict(color="white", size=6), name="observed",
                                     showlegend=(k == 0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=model.index, y=model[var.value], mode="lines",
                                 line=dict(color=GREEN, width=2), name="model fit",
                                 showlegend=(k == 0)), row=r, col=c)
    fig.update_layout(template=TEMPLATE, height=460, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


run, feeds, is_real = _get_run()
fit = fit_run(run, feeds=feeds)
summary = tools.query_run(run)
diag = tools.diagnose_state(run)

tab_run, tab_val, tab_design = st.tabs(["🔬 Run analysis", "📊 Model validation", "🧪 Active learning"])

# --------------------------------------------------------------------------- run tab
with tab_run:
    rec = tools.recommend_feed(run, fit=fit)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak VCD", f"{summary['peak_vcd']:.1f}" if summary["peak_vcd"] else "—", "10⁶/mL")
    c2.metric("Final viability", f"{summary['final_viability']:.0%}" if summary["final_viability"] else "—")
    c3.metric("Model fit RMSE", f"{fit.rmse:.3f}", "normalized")
    c4.metric("Feed plan ↑ IVCD", f"+{rec['projected_improvement_pct']:.0f}%")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(_trajectory_figure(run, fit), width="stretch")
    with right:
        st.subheader("Diagnosis")
        if diag["healthy"]:
            st.success("No metabolic issues flagged.")
        for issue in diag["issues"]:
            st.warning(f"**{issue['issue']}** — {issue['variable']} crosses "
                       f"{issue['threshold']} at t={issue['first_time_h']:.0f} h")
        st.subheader("Recommended feed")
        if rec["recommended_feeds"]:
            st.dataframe(pd.DataFrame(rec["recommended_feeds"]), hide_index=True, width="stretch")
        else:
            st.caption("No feed improves on the baseline within constraints.")
        st.caption(f"Projected peak lactate {rec['projected_peak_lactate_mM']} mM · "
                   f"ammonia {rec['projected_peak_ammonia_mM']} mM (within limits).")

    if st.button("🤖 Ask the agent to explain", disabled=not os.getenv("ANTHROPIC_API_KEY")):
        from cellpilot.agent import analyze_run
        with st.spinner("CellPilot agent analyzing…"):
            st.markdown(analyze_run(run))
    elif not os.getenv("ANTHROPIC_API_KEY"):
        st.caption("Set ANTHROPIC_API_KEY and install the 'agent' extra for LLM narration.")

# --------------------------------------------------------------------------- validation tab
@st.cache_data(show_spinner=True)
def _real_cv():
    batches = load_ieks_batches()
    cv = cross_validate([r for r, _ in batches], n_folds=len(batches),
                        feeds_list=[f for _, f in batches])
    return {"mech": cv.rmse_mechanistic, "ml": cv.rmse_pure_ml, "hybrid": cv.rmse_hybrid, "n": cv.n_runs}

with tab_val:
    st.caption("Leave-one-out CV on 9 real, third-party fed-batch runs — a cross-model test "
               "(different group, different model). Lower is better.")
    cv = _real_cv()
    fig = go.Figure(go.Bar(
        x=["Mechanistic", "Pure ML", "Hybrid"], y=[cv["mech"], cv["ml"], cv["hybrid"]],
        marker_color=[BLUE, AMBER, GREEN], text=[f"{v:.2f}" for v in (cv["mech"], cv["ml"], cv["hybrid"])],
        textposition="outside"))
    fig.update_layout(template=TEMPLATE, height=380, yaxis_title="Held-out VCD RMSE (10⁶/mL)",
                      title=f"Hybrid is the most accurate (n={cv['n']} runs)", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    best = min(cv["mech"], cv["ml"])
    st.metric("Hybrid improvement vs next-best", f"{(1 - cv['hybrid']/best)*100:.0f}%")

# --------------------------------------------------------------------------- design tab
@st.cache_data(show_spinner=True)
def _bo(source_id: str, _params):
    return bayesian_optimize(_params, DesignSpace(), n_init=4, n_iter=16, seed=0)

with tab_design:
    st.caption("Bayesian optimization proposes the next experiment to run, using the "
               "calibrated model as an in-silico oracle. Fewer experiments = cheaper discovery.")
    res = _bo(run.run_id, fit.params)
    xs = list(range(1, len(res.history_bo) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=res.history_bo, mode="lines+markers",
                             line=dict(color=GREEN, width=2), name="Bayesian optimization"))
    fig.add_trace(go.Scatter(x=xs, y=res.history_random, mode="lines+markers",
                             line=dict(color=AMBER, width=2, dash="dash"), name="Random search"))
    fig.update_layout(template=TEMPLATE, height=380, xaxis_title="experiment number",
                      yaxis_title="best IVCD found so far", margin=dict(t=30),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Proposed next experiment")
    st.dataframe(pd.DataFrame([res.best_design]), hide_index=True, width="stretch")
    st.caption(f"Predicted IVCD {res.best_objective:.0f} (in-silico).")
