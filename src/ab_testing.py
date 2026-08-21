"""
ab_testing.py — A/B testing framework with:
  - Two-proportion z-test (for rate KPIs)
  - Welch's t-test (for continuous KPIs like revenue)
  - Effect size estimation (Cohen's d, relative lift)
  - Multiple-testing correction (Benjamini-Hochberg FDR)
  - Automated written summary generation
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Literal


ALPHA = 0.05        # significance level
MIN_DETECTABLE_EFFECT = 0.01   # MDE for sample size guidance

KPI_TYPE: dict[str, Literal["rate", "continuous"]] = {
    "open_rate":        "rate",
    "click_rate":       "rate",
    "conversion_rate":  "rate",
    "revenue_per_user": "continuous",
    "retention_rate":   "rate",
}


# ── statistical tests ─────────────────────────────────────────────────────────

def two_prop_ztest(
    n_ctrl: int, mean_ctrl: float,
    n_treat: int, mean_treat: float,
) -> tuple[float, float]:
    """
    Two-proportion z-test for rate KPIs.
    Returns (z_stat, p_value).
    """
    p_pool = (mean_ctrl * n_ctrl + mean_treat * n_treat) / (n_ctrl + n_treat)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_treat))
    if se == 0:
        return 0.0, 1.0
    z = (mean_treat - mean_ctrl) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def welch_ttest(
    n_ctrl: int, mean_ctrl: float, std_ctrl: float,
    n_treat: int, mean_treat: float, std_treat: float,
) -> tuple[float, float]:
    """
    Welch's t-test for continuous KPIs (unequal variances).
    Returns (t_stat, p_value).
    """
    se = np.sqrt(std_ctrl**2 / n_ctrl + std_treat**2 / n_treat)
    if se == 0:
        return 0.0, 1.0
    t = (mean_treat - mean_ctrl) / se
    # Welch-Satterthwaite degrees of freedom
    num = (std_ctrl**2 / n_ctrl + std_treat**2 / n_treat) ** 2
    denom = (std_ctrl**2 / n_ctrl) ** 2 / (n_ctrl - 1) + (std_treat**2 / n_treat) ** 2 / (n_treat - 1)
    df = num / denom if denom > 0 else n_ctrl + n_treat - 2
    p = 2 * (1 - stats.t.cdf(abs(t), df=df))
    return float(t), float(p)


def cohens_d(
    mean_ctrl: float, mean_treat: float,
    std_ctrl: float, std_treat: float,
    n_ctrl: int, n_treat: int,
) -> float:
    """Pooled Cohen's d effect size."""
    pooled_std = np.sqrt(
        ((n_ctrl - 1) * std_ctrl**2 + (n_treat - 1) * std_treat**2)
        / (n_ctrl + n_treat - 2)
    )
    return float((mean_treat - mean_ctrl) / pooled_std) if pooled_std > 0 else 0.0


def relative_lift(mean_ctrl: float, mean_treat: float) -> float:
    """Percentage lift of treatment over control."""
    return float((mean_treat - mean_ctrl) / mean_ctrl * 100) if mean_ctrl != 0 else 0.0


# ── per-experiment analysis ───────────────────────────────────────────────────

def analyze_experiment(row: pd.Series) -> dict:
    """Run stat test + effect size for a single experiment row."""
    kpi = row["kpi"]
    kpi_type = KPI_TYPE.get(kpi, "rate")

    n_ctrl,  mean_ctrl,  std_ctrl  = int(row["n_control"]),  row["control_mean"],  row["control_std"]
    n_treat, mean_treat, std_treat = int(row["n_treatment"]), row["treatment_mean"], row["treatment_std"]

    if kpi_type == "rate":
        stat, p_value = two_prop_ztest(n_ctrl, mean_ctrl, n_treat, mean_treat)
        stat_name = "z_stat"
    else:
        stat, p_value = welch_ttest(n_ctrl, mean_ctrl, std_ctrl, n_treat, mean_treat, std_treat)
        stat_name = "t_stat"

    d = cohens_d(mean_ctrl, mean_treat, std_ctrl, std_treat, n_ctrl, n_treat)
    lift = relative_lift(mean_ctrl, mean_treat)

    return {
        "experiment_id":  row["experiment_id"],
        "campaign_type":  row["campaign_type"],
        "kpi":            kpi,
        "kpi_type":       kpi_type,
        "n_control":      n_ctrl,
        "n_treatment":    n_treat,
        "control_mean":   mean_ctrl,
        "treatment_mean": mean_treat,
        stat_name:        round(stat, 4),
        "p_value":        round(p_value, 6),
        "cohens_d":       round(d, 4),
        "relative_lift_pct": round(lift, 2),
        "significant_raw": p_value < ALPHA,
    }


# ── full experiment batch analysis ────────────────────────────────────────────

def run_ab_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze all experiments in df.
    Applies Benjamini-Hochberg FDR correction across all tests.
    Returns enriched DataFrame with significance flags.
    """
    results = [analyze_experiment(row) for _, row in df.iterrows()]
    results_df = pd.DataFrame(results)

    # BH correction
    _, pvals_corrected, _, _ = multipletests(
        results_df["p_value"], alpha=ALPHA, method="fdr_bh"
    )
    results_df["p_value_bh"] = pvals_corrected.round(6)
    results_df["significant_bh"] = results_df["p_value_bh"] < ALPHA

    # Effect size label
    def label_effect(d):
        d = abs(d)
        if d < 0.2: return "negligible"
        if d < 0.5: return "small"
        if d < 0.8: return "medium"
        return "large"
    results_df["effect_size_label"] = results_df["cohens_d"].map(label_effect)

    print(f"[ab_testing] Analyzed {len(results_df)} experiments.")
    print(f"  Significant (raw):     {results_df['significant_raw'].sum()}")
    print(f"  Significant (BH FDR):  {results_df['significant_bh'].sum()}")
    print(f"  Avg relative lift:     {results_df['relative_lift_pct'].mean():.2f}%")

    return results_df


# ── written summary generation ────────────────────────────────────────────────

def generate_summary(results_df: pd.DataFrame) -> str:
    """
    Auto-generate a plain-English executive summary of all A/B experiment results.
    """
    n_total = len(results_df)
    n_sig   = results_df["significant_bh"].sum()
    n_pos   = ((results_df["significant_bh"]) & (results_df["relative_lift_pct"] > 0)).sum()
    n_neg   = n_sig - n_pos

    avg_lift_sig = results_df.loc[results_df["significant_bh"], "relative_lift_pct"].mean()
    best_exp = results_df.loc[results_df["relative_lift_pct"].idxmax()]
    worst_exp = results_df.loc[results_df["relative_lift_pct"].idxmin()]

    # By campaign type
    by_type = results_df.groupby("campaign_type")["significant_bh"].mean().sort_values(ascending=False)
    best_channel = by_type.index[0]

    # By KPI
    by_kpi = results_df.groupby("kpi")["relative_lift_pct"].mean().sort_values(ascending=False)
    best_kpi = by_kpi.index[0]

    summary = f"""
## A/B Experiment Batch Summary

**Total experiments analyzed:** {n_total}
**Statistically significant results (BH FDR q < {ALPHA}):** {n_sig} ({n_sig/n_total:.0%})
  - Positive effects: {n_pos}
  - Negative/null effects: {n_neg}
**Average lift among significant experiments:** {avg_lift_sig:+.1f}%

### Top Performing Experiment
- **{best_exp['experiment_id']}** ({best_exp['campaign_type']} · {best_exp['kpi']})
- Lift: {best_exp['relative_lift_pct']:+.1f}%, Cohen's d = {best_exp['cohens_d']:.3f} ({best_exp['effect_size_label']}), p = {best_exp['p_value']:.4f}

### Worst Performing Experiment
- **{worst_exp['experiment_id']}** ({worst_exp['campaign_type']} · {worst_exp['kpi']})
- Lift: {worst_exp['relative_lift_pct']:+.1f}%, Cohen's d = {worst_exp['cohens_d']:.3f} ({worst_exp['effect_size_label']}), p = {worst_exp['p_value']:.4f}

### Channel Performance
Best channel by significance rate: **{best_channel}** ({by_type[best_channel]:.0%} sig. rate)

### KPI Trends
Highest avg lift by KPI: **{best_kpi}** ({by_kpi[best_kpi]:+.1f}% avg lift)

### Recommendations
1. Prioritize **{best_channel}** campaigns — highest share of statistically significant results.
2. Scale winning experiments with Cohen's d ≥ 0.2 — ({results_df[results_df['cohens_d'].abs() >= 0.2]['significant_bh'].sum()} experiments qualify).
3. Re-evaluate {n_neg} experiments showing negative treatment effects before next campaign cycle.
""".strip()

    return summary


if __name__ == "__main__":
    from data_loader import load_experiments
    df = load_experiments()
    results = run_ab_analysis(df)
    print(results[["experiment_id", "kpi", "p_value_bh", "significant_bh", "relative_lift_pct"]].to_string())
    print("\n" + generate_summary(results))
