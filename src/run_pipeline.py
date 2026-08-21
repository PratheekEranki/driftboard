"""
run_pipeline.py — End-to-end Driftboard pipeline.
Run:  python src/run_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_experiments, load_kpi_timeseries
from ab_testing import run_ab_analysis, generate_summary
from drift_detection import run_drift_detection

DATA_DIR = "data"

if __name__ == "__main__":
    print("=== Driftboard — Full Pipeline ===\n")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load data
    df_exps = load_experiments(path=os.path.join(DATA_DIR, "experiments.csv"))
    df_ts   = load_kpi_timeseries(path=os.path.join(DATA_DIR, "kpi_timeseries.csv"))

    # Save raw data for dashboard
    df_exps.to_csv(os.path.join(DATA_DIR, "experiments.csv"), index=False)
    df_ts.to_csv(os.path.join(DATA_DIR, "kpi_timeseries.csv"), index=False)

    # 2. A/B testing
    print("\n--- A/B Testing ---")
    results_df = run_ab_analysis(df_exps)
    results_df.to_csv(os.path.join(DATA_DIR, "ab_results.csv"), index=False)

    # 3. Written summary
    summary = generate_summary(results_df)
    with open(os.path.join(DATA_DIR, "ab_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)

    # 4. Drift detection
    print("\n--- Drift Detection ---")
    df_anom, report = run_drift_detection(df_ts, save_dir=DATA_DIR)

    print("\n=== Pipeline Complete ===")
    print(f"  A/B results:   {DATA_DIR}/ab_results.csv")
    print(f"  Summary:       {DATA_DIR}/ab_summary.txt")
    print(f"  KPI anomalies: {DATA_DIR}/kpi_anomalies.csv")
    print(f"  Drift report:  {DATA_DIR}/drift_report.csv")
    print(f"\nLaunch dashboard: streamlit run src/dashboard.py")
