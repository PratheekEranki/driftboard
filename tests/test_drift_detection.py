"""
tests/test_drift_detection.py — Unit tests for drift detection module.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drift_detection import detect_iqr, detect_zscore, detect_anomalies, drift_report
from data_loader import generate_kpi_timeseries


@pytest.fixture(scope="module")
def ts_df():
    return generate_kpi_timeseries(n_days=180, seed=99)


def test_detect_iqr_flags_outlier():
    s = pd.Series([10.0] * 98 + [1000.0, -1000.0])
    mask = detect_iqr(s)
    assert mask.iloc[-2] and mask.iloc[-1]
    assert not mask.iloc[0]


def test_detect_zscore_flags_spike():
    rng = np.random.default_rng(0)
    base = rng.normal(5, 0.1, 100).tolist()
    base[50] = 50.0  # extreme spike
    s = pd.Series(base)
    mask = detect_zscore(s)
    assert mask.iloc[50]


def test_detect_anomalies_columns(ts_df):
    result = detect_anomalies(ts_df)
    for col in ["anomaly_iqr", "anomaly_zscore", "anomaly_any", "rolling_mean"]:
        assert col in result.columns


def test_detect_anomalies_recall(ts_df):
    result = detect_anomalies(ts_df)
    injected = result["injected_anomaly"].sum()
    detected = (result["anomaly_any"] & result["injected_anomaly"]).sum()
    recall = detected / injected if injected > 0 else 0
    # Recall should be reasonable (> 0.3) given injected anomalies are large
    assert recall >= 0.3, f"Recall too low: {recall:.2f}"


def test_drift_report_structure(ts_df):
    anom_df = detect_anomalies(ts_df)
    report = drift_report(anom_df)
    assert "kpi" in report.columns
    assert "anomaly_rate" in report.columns
    assert "drift_severity" in report.columns
    assert len(report) == ts_df["kpi"].nunique()
