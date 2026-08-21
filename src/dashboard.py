"""
dashboard.py — Streamlit dashboard for Driftboard.
Provides: experiment browser, drift alerts, KPI timeseries, and written summaries.

Run with:  streamlit run src/dashboard.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = "data"

st.set_page_config(
    page_title="Driftboard — Campaign Analytics",
    page_icon="📡",
    layout="wide",
)

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading experiment results…")
def load_ab_results() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ab_results.csv")
    if not os.path.exists(path):
        st.error("Run `python src/run_pipeline.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading KPI anomalies…")
def load_anomalies() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "kpi_anomalies.csv")
    if not os.path.exists(path):
        st.error("Run `python src/run_pipeline.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading drift report…")
def load_drift_report() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "drift_report.csv")
    if not os.path.exists(path):
        st.error("Run `python src/run_pipeline.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading summary…")
def load_summary() -> str:
    path = os.path.join(DATA_DIR, "ab_summary.txt")
    if not os.path.exists(path):
        return "_No summary found. Run `python src/run_pipeline.py` first._"
    return open(path, encoding="utf-8").read()


def sig_badge(val):
    return "✅ Yes" if val else "❌ No"


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Driftboard")
    st.caption("H&R Block · Campaign Analytics")
    st.divider()

    ab = load_ab_results()
    drift_rep = load_drift_report()

    campaign_types = ["All"] + sorted(ab["campaign_type"].unique().tolist())
    sel_campaign = st.selectbox("Filter by Campaign Type", campaign_types)

    kpis = ["All"] + sorted(ab["kpi"].unique().tolist())
    sel_kpi = st.selectbox("Filter by KPI", kpis)

    sig_only = st.checkbox("Significant results only (BH FDR)", value=False)

    st.divider()
    n_exps = len(ab)
    n_sig  = ab["significant_bh"].sum()
    st.metric("Total Experiments", n_exps)
    st.metric("Significant (BH FDR)", int(n_sig))
    st.metric("Sig Rate", f"{n_sig/n_exps:.0%}")

# ── filter data ───────────────────────────────────────────────────────────────
ab_f = ab.copy()
if sel_campaign != "All":
    ab_f = ab_f[ab_f["campaign_type"] == sel_campaign]
if sel_kpi != "All":
    ab_f = ab_f[ab_f["kpi"] == sel_kpi]
if sig_only:
    ab_f = ab_f[ab_f["significant_bh"]]

# ── header ───────────────────────────────────────────────────────────────────
st.title("Driftboard — Campaign Analytics & A/B Testing")
st.caption("Statistical hypothesis testing · BH FDR correction · Metric drift anomaly detection")
st.divider()

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Experiments Shown", len(ab_f))
c2.metric("Sig Results", int(ab_f["significant_bh"].sum()))
c3.metric("Avg Lift (sig.)",
          f"{ab_f.loc[ab_f['significant_bh'], 'relative_lift_pct'].mean():+.1f}%"
          if ab_f["significant_bh"].any() else "—")
c4.metric("Avg Cohen's d (sig.)",
          f"{ab_f.loc[ab_f['significant_bh'], 'cohens_d'].abs().mean():.3f}"
          if ab_f["significant_bh"].any() else "—")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🧪 Experiment Browser", "📉 Metric Drift Alerts", "📊 KPI Timeseries", "📝 Summary Report"])

# ─── Tab 1: Experiment Browser ────────────────────────────────────────────────
with tab1:
    st.subheader("All Experiments")

    # Significance volcano-style scatter
    fig = px.scatter(
        ab_f,
        x="relative_lift_pct",
        y=-np.log10(ab_f["p_value"].clip(lower=1e-10)),
        color=ab_f["significant_bh"].map({True: "Significant", False: "Not Significant"}),
        color_discrete_map={"Significant": "#10B981", "Not Significant": "#9CA3AF"},
        size=ab_f["cohens_d"].abs().clip(lower=0.01),
        hover_data=["experiment_id", "campaign_type", "kpi", "p_value_bh", "effect_size_label"],
        labels={"x": "Relative Lift (%)", "y": "-log10(p-value)", "color": ""},
        title="Volcano Plot — Experiment Lift vs. Significance",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=450, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart: sig rate by campaign type
    by_type = ab.groupby("campaign_type").agg(
        n=("experiment_id", "count"),
        n_sig=("significant_bh", "sum"),
    ).reset_index()
    by_type["sig_rate"] = by_type["n_sig"] / by_type["n"]

    fig2 = px.bar(
        by_type.sort_values("sig_rate", ascending=False),
        x="campaign_type", y="sig_rate",
        text=by_type.sort_values("sig_rate", ascending=False)["n_sig"].astype(str)
            + "/" + by_type.sort_values("sig_rate", ascending=False)["n"].astype(str),
        color="sig_rate",
        color_continuous_scale="Teal",
        labels={"sig_rate": "Sig. Rate", "campaign_type": "Campaign Type"},
        title="Significance Rate by Campaign Type",
    )
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(height=360, margin=dict(t=40, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Detailed table
    with st.expander("Full experiment table"):
        disp_cols = ["experiment_id", "campaign_type", "kpi", "control_mean",
                     "treatment_mean", "relative_lift_pct", "p_value", "p_value_bh",
                     "significant_bh", "cohens_d", "effect_size_label"]
        tbl = ab_f[[c for c in disp_cols if c in ab_f.columns]].copy()
        tbl["significant_bh"] = tbl["significant_bh"].map(sig_badge)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ─── Tab 2: Drift Alerts ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Metric Drift Alerts by KPI")

    # Severity table
    dr = load_drift_report()
    dr["drift_severity"] = dr["drift_severity"].fillna("low")
    severity_color = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}
    dr["severity_icon"] = dr["drift_severity"].map(severity_color)

    for _, row in dr.iterrows():
        exp = st.expander(
            f"{row['severity_icon']} {row['kpi']} — {row['anomalies_detected']} anomalies "
            f"({row['anomaly_rate']:.1%} rate)"
        )
        with exp:
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomalies", int(row["anomalies_detected"]))
            c2.metric("Anomaly Rate", f"{row['anomaly_rate']:.1%}")
            c3.metric("Severity", row["drift_severity"].upper())

            img_path = os.path.join(DATA_DIR, f"drift_{row['kpi']}.png")
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)

# ─── Tab 3: KPI Timeseries ────────────────────────────────────────────────────
with tab3:
    st.subheader("KPI Timeseries with Anomalies")

    anom_df = load_anomalies()
    anom_df["date"] = pd.to_datetime(anom_df["date"])

    kpi_choice = st.selectbox("Select KPI", sorted(anom_df["kpi"].unique()))
    g = anom_df[anom_df["kpi"] == kpi_choice].sort_values("date")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=g["date"], y=g["value"], mode="lines",
        name="Daily value", line=dict(color="#6B7280", width=1),
    ))
    fig3.add_trace(go.Scatter(
        x=g["date"], y=g["rolling_mean"], mode="lines",
        name="14d rolling mean", line=dict(color="#1D4ED8", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=g[g["anomaly_any"]]["date"], y=g[g["anomaly_any"]]["value"],
        mode="markers", name="Anomaly",
        marker=dict(color="#EF4444", size=8, symbol="x"),
    ))
    fig3.update_layout(
        title=f"{kpi_choice} — Daily Value & Anomalies",
        xaxis_title="Date", yaxis_title="Value",
        height=460, margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Anomaly table"):
        anom_rows = g[g["anomaly_any"]][["date", "value", "rolling_mean", "anomaly_iqr", "anomaly_zscore"]]
        anom_rows["date"] = anom_rows["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(anom_rows, use_container_width=True, hide_index=True)

# ─── Tab 4: Summary Report ────────────────────────────────────────────────────
with tab4:
    st.subheader("Executive Summary — A/B Experiment Batch")
    summary_md = load_summary()
    st.markdown(summary_md)
