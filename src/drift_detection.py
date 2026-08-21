"""
drift_detection.py — Metric drift / anomaly detection for business KPIs.
Uses IQR-based and Z-score methods; also computes rolling mean for trend context.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ZSCORE_THRESHOLD  = 3.0    # |z| > 3 → anomaly
IQR_MULTIPLIER    = 1.5    # IQR fence multiplier
ROLLING_WINDOW    = 14     # days for rolling stats


# ── per-KPI anomaly detection ─────────────────────────────────────────────────

def detect_iqr(series: pd.Series) -> pd.Series:
    """
    IQR fence method.
    Returns boolean mask: True where the value is an outlier.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    return (series < lower) | (series > upper)


def detect_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score method (rolling window for non-stationary series).
    Returns boolean mask: True where |z| > ZSCORE_THRESHOLD.
    """
    rolling_mean = series.rolling(ROLLING_WINDOW, min_periods=5).mean()
    rolling_std  = series.rolling(ROLLING_WINDOW, min_periods=5).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.abs() > ZSCORE_THRESHOLD


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run both IQR and Z-score anomaly detection on each KPI in the timeseries df.
    df columns: ['date', 'kpi', 'value', 'injected_anomaly']
    Returns df with added columns: 'anomaly_iqr', 'anomaly_zscore', 'anomaly_any'.
    """
    results = []

    for kpi, group in df.groupby("kpi"):
        group = group.sort_values("date").copy()
        group["anomaly_iqr"]    = detect_iqr(group["value"]).values
        group["anomaly_zscore"] = detect_zscore(group["value"]).values
        group["anomaly_any"]    = group["anomaly_iqr"] | group["anomaly_zscore"]

        # Rolling stats for trend display
        group["rolling_mean"] = group["value"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        group["rolling_std"]  = group["value"].rolling(ROLLING_WINDOW, min_periods=1).std()
        results.append(group)

    out = pd.concat(results, ignore_index=True)

    # Summary
    n_detected = out["anomaly_any"].sum()
    n_injected = out.get("injected_anomaly", pd.Series([False]*len(out))).sum()
    print(f"[drift_detection] Anomalies detected: {n_detected} / {len(out)} observations")
    if "injected_anomaly" in out.columns:
        true_positives = (out["anomaly_any"] & out["injected_anomaly"]).sum()
        print(f"  Recall on injected anomalies: {true_positives}/{n_injected} "
              f"({true_positives/max(n_injected,1):.1%})")

    return out


# ── drift report ──────────────────────────────────────────────────────────────

def drift_report(df_anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize anomaly counts and severity per KPI.
    """
    report = df_anomalies.groupby("kpi").agg(
        total_observations=("value", "count"),
        anomalies_detected=("anomaly_any", "sum"),
        iqr_anomalies=("anomaly_iqr", "sum"),
        zscore_anomalies=("anomaly_zscore", "sum"),
        mean_value=("value", "mean"),
        std_value=("value", "std"),
        latest_value=("value", "last"),
    ).reset_index()

    report["anomaly_rate"] = (report["anomalies_detected"] / report["total_observations"]).round(4)
    report["drift_severity"] = pd.cut(
        report["anomaly_rate"],
        bins=[0, 0.05, 0.12, 0.20, 1.0],
        labels=["low", "moderate", "high", "critical"],
    )
    report = report.sort_values("anomaly_rate", ascending=False)
    print("\n[drift_detection] Drift report:")
    print(report[["kpi", "anomalies_detected", "anomaly_rate", "drift_severity"]].to_string(index=False))
    return report


# ── diagnostic plots ──────────────────────────────────────────────────────────

def plot_kpi_drift(df_anomalies: pd.DataFrame, save_dir: str = "data") -> None:
    """
    Save one PNG per KPI showing: raw values, rolling mean ± 2σ, anomaly markers.
    """
    os.makedirs(save_dir, exist_ok=True)

    for kpi, group in df_anomalies.groupby("kpi"):
        group = group.sort_values("date")
        dates  = pd.to_datetime(group["date"])
        values = group["value"]
        rm     = group["rolling_mean"]
        rs     = group["rolling_std"].fillna(0)
        anom   = group["anomaly_any"]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(dates, values, color="#6B7280", linewidth=0.8, label="Daily value", alpha=0.8)
        ax.plot(dates, rm, color="#1D4ED8", linewidth=1.5, label=f"{ROLLING_WINDOW}d rolling mean")
        ax.fill_between(dates, rm - 2*rs, rm + 2*rs, color="#BFDBFE", alpha=0.4, label="±2σ band")
        ax.scatter(dates[anom], values[anom], color="#EF4444", zorder=5, s=25, label="Anomaly")

        if "injected_anomaly" in group.columns:
            inj = group["injected_anomaly"]
            ax.scatter(dates[inj], values[inj], color="#F59E0B", marker="x",
                       zorder=6, s=40, label="Injected anomaly (ground truth)")

        ax.set_title(f"KPI Drift — {kpi}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"drift_{kpi}.png"), dpi=150)
        plt.close(fig)

    print(f"[drift_detection] Drift plots saved to {save_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def run_drift_detection(df_ts: pd.DataFrame, save_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end drift detection pipeline.
    Returns (df_anomalies, report_df).
    """
    df_anomalies = detect_anomalies(df_ts)
    report       = drift_report(df_anomalies)
    plot_kpi_drift(df_anomalies, save_dir=save_dir)

    os.makedirs(save_dir, exist_ok=True)
    df_anomalies.to_csv(os.path.join(save_dir, "kpi_anomalies.csv"), index=False)
    report.to_csv(os.path.join(save_dir, "drift_report.csv"), index=False)

    return df_anomalies, report


if __name__ == "__main__":
    from data_loader import load_kpi_timeseries
    df_ts = load_kpi_timeseries()
    df_anom, report = run_drift_detection(df_ts, save_dir="data")
    print(report)
