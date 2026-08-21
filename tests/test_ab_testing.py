"""
tests/test_ab_testing.py — Unit tests for A/B testing framework.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ab_testing import (
    two_prop_ztest, welch_ttest, cohens_d, relative_lift,
    analyze_experiment, run_ab_analysis, generate_summary, ALPHA,
)
from data_loader import generate_synthetic_experiments


@pytest.fixture(scope="module")
def exp_df():
    return generate_synthetic_experiments(n=30, seed=7)


def test_two_prop_ztest_significant():
    # Large samples, big effect — should be significant
    z, p = two_prop_ztest(10_000, 0.10, 10_000, 0.15)
    assert p < ALPHA
    assert z < 0  # treatment > control → positive z in (treat - ctrl)


def test_two_prop_ztest_null():
    # Same rates → should NOT be significant
    _, p = two_prop_ztest(500, 0.12, 500, 0.12)
    assert p > ALPHA


def test_welch_ttest():
    t, p = welch_ttest(1_000, 50.0, 5.0, 1_000, 55.0, 5.0)
    assert p < ALPHA  # 5 unit difference with low noise should be significant


def test_cohens_d_sign():
    d = cohens_d(0.10, 0.15, 0.03, 0.03, 500, 500)
    assert d > 0  # treatment mean higher


def test_relative_lift():
    lift = relative_lift(0.10, 0.13)
    assert abs(lift - 30.0) < 0.01


def test_analyze_experiment_rate(exp_df):
    row = exp_df[exp_df["kpi"] != "revenue_per_user"].iloc[0]
    result = analyze_experiment(row)
    assert "p_value" in result
    assert "cohens_d" in result
    assert "relative_lift_pct" in result
    assert 0 <= result["p_value"] <= 1


def test_analyze_experiment_continuous(exp_df):
    rev_rows = exp_df[exp_df["kpi"] == "revenue_per_user"]
    if len(rev_rows) == 0:
        pytest.skip("No revenue_per_user rows in fixture")
    row = rev_rows.iloc[0]
    result = analyze_experiment(row)
    assert "t_stat" in result


def test_run_ab_analysis_shape(exp_df):
    results = run_ab_analysis(exp_df)
    assert len(results) == len(exp_df)
    assert "significant_bh" in results.columns
    assert "p_value_bh" in results.columns
    assert "effect_size_label" in results.columns


def test_summary_contains_key_sections(exp_df):
    results = run_ab_analysis(exp_df)
    summary = generate_summary(results)
    assert "Total experiments analyzed" in summary
    assert "Top Performing Experiment" in summary
    assert "Recommendations" in summary
