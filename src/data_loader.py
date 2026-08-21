"""
data_loader.py — Simulates pulling experiment results from AWS Athena / SQL.
In production, replace the synthetic generator with a real Athena/DB connection.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 42
N_EXPERIMENTS = 60      # 50+ historical marketing experiments


def generate_synthetic_experiments(n: int = N_EXPERIMENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic dataset of A/B marketing experiment results.
    Each row = one experiment with control/treatment metrics.
    """
    rng = np.random.default_rng(seed)

    campaign_types = ["email", "sms", "push_notification", "display_ad", "direct_mail"]
    kpis = ["open_rate", "click_rate", "conversion_rate", "revenue_per_user", "retention_rate"]

    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(n):
        campaign_type = rng.choice(campaign_types)
        kpi = rng.choice(kpis)
        n_control   = int(rng.integers(1_000, 20_000))
        n_treatment = int(rng.integers(1_000, 20_000))

        # True effect: ~40% of experiments have a real positive effect
        has_effect = rng.random() < 0.40
        base_rate = float(rng.uniform(0.05, 0.35))

        if kpi == "revenue_per_user":
            base_rate = float(rng.uniform(5.0, 80.0))
            effect_size = float(rng.uniform(1.0, 10.0)) if has_effect else 0.0
            control_mean   = base_rate + rng.normal(0, base_rate * 0.05)
            treatment_mean = base_rate + effect_size + rng.normal(0, base_rate * 0.05)
            control_std    = base_rate * 0.30
            treatment_std  = base_rate * 0.30
        else:
            effect_size = float(rng.uniform(0.01, 0.08)) if has_effect else 0.0
            control_mean   = np.clip(base_rate + rng.normal(0, 0.01), 0.01, 0.99)
            treatment_mean = np.clip(base_rate + effect_size + rng.normal(0, 0.01), 0.01, 0.99)
            control_std    = float(rng.uniform(0.01, 0.05))
            treatment_std  = float(rng.uniform(0.01, 0.05))

        start_date = base_date + timedelta(days=int(rng.integers(0, 500)))
        duration_days = int(rng.integers(7, 42))

        records.append({
            "experiment_id": f"EXP_{i+1:04d}",
            "campaign_type": campaign_type,
            "kpi": kpi,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "duration_days": duration_days,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "control_mean": round(control_mean, 6),
            "treatment_mean": round(treatment_mean, 6),
            "control_std": round(control_std, 6),
            "treatment_std": round(treatment_std, 6),
            "ground_truth_effect": has_effect,   # for validation only
        })

    return pd.DataFrame(records)


def generate_kpi_timeseries(
    kpis: list[str] | None = None,
    n_days: int = 365,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate daily KPI timeseries (for metric drift detection).
    Injects random anomaly spikes in ~10% of days.
    """
    if kpis is None:
        kpis = ["open_rate", "click_rate", "conversion_rate", "revenue_per_user", "retention_rate"]

    rng = np.random.default_rng(seed)
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=d) for d in range(n_days)]

    rows = []
    for kpi in kpis:
        if kpi == "revenue_per_user":
            base = rng.uniform(30, 60)
            noise_scale = 5.0
        else:
            base = rng.uniform(0.08, 0.25)
            noise_scale = 0.015

        # Add trend + seasonality
        trend = np.linspace(0, noise_scale * 0.3, n_days)
        seasonality = noise_scale * 0.2 * np.sin(np.linspace(0, 4 * np.pi, n_days))
        noise = rng.normal(0, noise_scale * 0.5, n_days)
        values = base + trend + seasonality + noise

        # Inject anomalies
        anomaly_mask = rng.random(n_days) < 0.10
        anomaly_direction = rng.choice([-1, 1], size=n_days)
        values += anomaly_mask * anomaly_direction * noise_scale * rng.uniform(2, 5, n_days)
        values = np.clip(values, 0.001, None)

        for d, date, val, is_anomaly in zip(range(n_days), dates, values, anomaly_mask):
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "kpi": kpi,
                "value": round(float(val), 6),
                "injected_anomaly": bool(is_anomaly),
            })

    return pd.DataFrame(rows)


def load_experiments(path: str | None = None) -> pd.DataFrame:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    print("[data_loader] Generating synthetic experiment data...")
    df = generate_synthetic_experiments()
    print(f"[data_loader] {len(df)} experiments loaded.")
    return df


def load_kpi_timeseries(path: str | None = None) -> pd.DataFrame:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    print("[data_loader] Generating synthetic KPI timeseries...")
    df = generate_kpi_timeseries()
    print(f"[data_loader] {len(df)} KPI daily records generated.")
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    exps = load_experiments()
    ts = load_kpi_timeseries()
    exps.to_csv("data/experiments.csv", index=False)
    ts.to_csv("data/kpi_timeseries.csv", index=False)
    print(exps.head())
    print(ts.head())
