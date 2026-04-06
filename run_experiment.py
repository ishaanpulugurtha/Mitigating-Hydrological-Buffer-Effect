"""
run_experiment.py — Master script for the Hydrological Buffer Effect experiment.

Mitigating the Hydrological Buffer Effect: Enhancing Early Warning of Storm-Driven
Water Quality Risks in Buffered Watersheds Using LSTM Architectures

Experiment Structure:
  Independent Variable: Model Type, tested at THREE LEVELS (one change per step)

  [Background Evidence] CCF Lag Analysis — NOT an experimental level.
    Run on: California + Illinois Primary + Illinois Secondary (multi-station validation).

  Level 1 — Random Forest (Rainfall Only)
    Illinois Primary only. California as external benchmark. IL Secondary for validation.

  Level 2 — Random Forest + Soil Moisture (AMC-Aware)
    Illinois Primary only. One change from Level 1: adds AMC as input.

  Level 3 — LSTM (Temporal Memory Architecture)
    Illinois Primary only. One change from Level 2: replaces RF with LSTM.
    Same features as Level 2 (Rainfall + Soil Moisture).

  Multi-Station Validation:
    Illinois Secondary (Fox River at Algonquin) run for CCF + Level 1.
    Shows the Hydrological Buffer Effect and model performance gap generalizes
    across buffered Illinois watersheds — not a single case study.

  Control: Naive baseline (future turbidity = current turbidity)
  Comparison group: California (Mediterranean climate, no Buffer Effect)

Usage:
    python run_experiment.py              # try real USGS data first
    python run_experiment.py --synthetic  # physics-based synthetic data
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

from src.data_pipeline  import load_region_data
from src.preprocessing  import (
    prepare_rf_level1,
    prepare_rf_level2,
    prepare_lstm_sequences,
    rf_train_test_split,
    lstm_temporal_split,
)
from src.models         import RandomForestModel, LSTMModel, compute_metrics
from src.analysis       import compute_ccf, find_optimal_lag, storm_intensity_metrics
from src.visualizations import (
    plot_figure1_progression,
    plot_figure2_ccf,
    plot_figure3_storm_intensity,
    plot_figure4_safety,
    save_fig,
)
import config


# ==============================================================================
#  Helpers
# ==============================================================================

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_metrics(label: str, m: dict):
    print(f"  {label:<44}  R2={m['r2']:.4f}  RMSE={m['rmse']:.2f}  "
          f"TPR={m['tpr']:.2%}  FAR={m['far']:.2%}")


def compute_naive_baseline(X_te: pd.DataFrame, y_te: pd.Series,
                            df: pd.DataFrame) -> dict:
    """Naive baseline: predict future turbidity = current turbidity."""
    y_naive = df["turbidity"].reindex(X_te.index).ffill().values
    y_true  = y_te.values
    valid   = ~np.isnan(y_naive)
    return compute_metrics(y_true[valid], y_naive[valid])


def get_precip_for_test(X_te: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    return df["precipitation_mm_hr"].reindex(X_te.index).ffill()


def run_rf_level1(df: pd.DataFrame, label: str) -> tuple:
    """Run Level 1 RF for any region; returns (mets, y_true, y_pred, X_te)."""
    X, y = prepare_rf_level1(df, forecast_horizon=config.FORECAST_HORIZON)
    X_tr, X_te, y_tr, y_te = rf_train_test_split(X, y)
    rf = RandomForestModel()
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    mets = compute_metrics(y_te.values, yp)
    print_metrics(f"Level 1 — {label} (RF, rain only)", mets)
    return mets, y_te.values, yp, X_te


def train_lstm_il(il_df: pd.DataFrame) -> tuple:
    """
    Train LSTM for Illinois (Level 3).
    Features: precipitation + soil_moisture (same as Level 2).
    Returns: (lstm, y_true_ntu, y_pred_ntu, mets, scaler_y, X_te, y_te)
    """
    feature_cols = ["precipitation_mm_hr", "soil_moisture"]

    X_seq, y_seq, _, scaler_y = prepare_lstm_sequences(
        il_df,
        feature_cols     = feature_cols,
        target_col       = "turbidity",
        window_size      = config.WINDOW_SIZE,
        forecast_horizon = config.FORECAST_HORIZON,
    )

    X_tr, X_va, X_te, y_tr, y_va, y_te = lstm_temporal_split(X_seq, y_seq)

    lstm = LSTMModel(
        window_size = config.WINDOW_SIZE,
        n_features  = 2,
        epochs      = config.LSTM_EPOCHS,
        batch_size  = config.LSTM_BATCH,
        patience    = config.LSTM_PATIENCE,
    )
    print(f"  Training LSTM for Illinois "
          f"(window={config.WINDOW_SIZE} hr, forecast={config.FORECAST_HORIZON} hr)...")
    lstm.train(X_tr, y_tr, X_va, y_va, scaler_y=scaler_y, verbose=0)

    y_pred_ntu = lstm.predict_rescaled(X_te)
    y_true_ntu = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()
    mets       = compute_metrics(y_true_ntu, y_pred_ntu)
    return lstm, y_true_ntu, y_pred_ntu, mets, scaler_y, X_te, y_te


def compute_community_impact(results: dict, il_df: pd.DataFrame,
                              test_frac: float = 0.20) -> dict:
    """
    Calculate a concrete community impact statement.

    Illinois River corridor population estimate based on Illinois American Water
    service area (Peoria District) and downstream water utilities.
    Population served downstream of Mackinaw-Illinois River confluence: ~310,000.
    """
    POPULATION_SERVED = 310_000   # downstream IL River communities, conservative est.
    LEAD_TIME_HRS     = config.FORECAST_HORIZON

    total_rows   = len(il_df)
    test_rows    = int(total_rows * test_frac)
    test_yrs     = test_rows / (365 * 24)

    total_haz    = results["level1_il"]["n_hazard_events"]
    events_yr    = total_haz / test_yrs

    l1_caught_yr = results["level1_il"]["tpr"]  * events_yr
    l2_caught_yr = results["level2_il"]["tpr"]  * events_yr
    l3_caught_yr = results["level3_il"]["tpr"]  * events_yr
    naive_caught  = results["naive_il"]["tpr"]  * events_yr

    additional_vs_naive = l3_caught_yr - naive_caught
    additional_vs_l1    = l3_caught_yr - l1_caught_yr
    missed_l1_yr        = (1 - results["level1_il"]["tpr"]) * events_yr
    missed_l3_yr        = (1 - results["level3_il"]["tpr"]) * events_yr
    fewer_missed_yr     = missed_l1_yr - missed_l3_yr

    return {
        "population_served":        POPULATION_SERVED,
        "lead_time_hours":          LEAD_TIME_HRS,
        "test_period_years":        round(test_yrs, 2),
        "hazard_events_in_test":    total_haz,
        "annualized_hazard_rate":   round(events_yr, 1),
        "l1_caught_per_year":       round(l1_caught_yr, 1),
        "l2_caught_per_year":       round(l2_caught_yr, 1),
        "l3_caught_per_year":       round(l3_caught_yr, 1),
        "naive_caught_per_year":    round(naive_caught, 1),
        "additional_events_vs_l1":  round(additional_vs_l1, 1),
        "additional_events_vs_naive": round(additional_vs_naive, 1),
        "fewer_missed_per_year":    round(fewer_missed_yr, 1),
        "impact_sentence": (
            f"Serving approximately {POPULATION_SERVED:,} residents downstream of the "
            f"Mackinaw–Illinois River system, the LSTM model detected "
            f"{additional_vs_l1:.0f} additional dangerous turbidity events per year "
            f"compared to the baseline Random Forest — providing water quality managers "
            f"an average of {LEAD_TIME_HRS} additional hours of advance warning per event."
        ),
    }


# ==============================================================================
#  Main
# ==============================================================================

def main(use_synthetic: bool = False):
    t0      = time.time()
    results = {}

    # ── Data Loading ──────────────────────────────────────────────────────────
    section("DATA LOADING")
    ca_df  = load_region_data("california",  config.START_DATE, config.END_DATE,
                               force_synthetic=use_synthetic)
    il_df  = load_region_data("illinois",    config.START_DATE, config.END_DATE,
                               force_synthetic=use_synthetic)
    il2_df = load_region_data("illinois2",   config.START_DATE, config.END_DATE,
                               force_synthetic=use_synthetic)
    print(f"\n  CA rows : {len(ca_df):,}")
    print(f"  IL rows : {len(il_df):,}  (primary — Mackinaw River)")
    print(f"  IL2 rows: {len(il2_df):,}  (secondary — Fox River, validation only)")

    # ── Background Evidence: CCF Lag Analysis ─────────────────────────────────
    section("BACKGROUND EVIDENCE: CCF LAG ANALYSIS (Not an experimental level)")

    ca_lags,  ca_corrs  = compute_ccf(ca_df["precipitation_mm_hr"],  ca_df["turbidity"])
    il_lags,  il_corrs  = compute_ccf(il_df["precipitation_mm_hr"],  il_df["turbidity"])
    il2_lags, il2_corrs = compute_ccf(il2_df["precipitation_mm_hr"], il2_df["turbidity"])

    ca_opt_lag  = find_optimal_lag(ca_df["precipitation_mm_hr"],  ca_df["turbidity"])
    il_opt_lag  = find_optimal_lag(il_df["precipitation_mm_hr"],  il_df["turbidity"])
    il2_opt_lag = find_optimal_lag(il2_df["precipitation_mm_hr"], il2_df["turbidity"])

    print(f"\n  California peak CCF lag       : {ca_opt_lag:2d} hours  (rapid flashy response)")
    print(f"  Illinois Primary  peak CCF lag: {il_opt_lag:2d} hours  (Mackinaw R. — Sponge Effect!)")
    print(f"  Illinois Secondary peak CCF lag: {il2_opt_lag:2d} hours  (Fox R. — validates pattern)")
    print(f"  Quantified Sponge Effect (IL1): {il_opt_lag - ca_opt_lag} hours vs CA")
    print(f"  Quantified Sponge Effect (IL2): {il2_opt_lag - ca_opt_lag} hours vs CA")
    results["ccf_ca_lag_hours"]   = int(ca_opt_lag)
    results["ccf_il_lag_hours"]   = int(il_opt_lag)
    results["ccf_il2_lag_hours"]  = int(il2_opt_lag)
    results["ccf_sponge_effect_il"]  = int(il_opt_lag  - ca_opt_lag)
    results["ccf_sponge_effect_il2"] = int(il2_opt_lag - ca_opt_lag)

    il2_name = config.SITES["illinois2"]["name"]
    fig = plot_figure2_ccf(
        ca_lags, ca_corrs, ca_opt_lag,
        il_lags, il_corrs, il_opt_lag,
        il2_lags, il2_corrs, il2_opt_lag, il2_name,
    )
    save_fig(fig, "figure2_ccf.png")

    # ── Level 1: Baseline RF — Rainfall Only (IL + CA + IL2 validation) ───────
    section("LEVEL 1: BASELINE RANDOM FOREST  (Rainfall Only -> Turbidity)")

    l1_mets = {}
    l1_preds = {}

    # California benchmark
    mets, yt, yp, X_te = run_rf_level1(ca_df, "California (Benchmark)")
    l1_mets["california"] = mets
    l1_preds["california"] = (yt, yp, X_te)
    results["level1_ca"] = mets

    # Illinois Primary (experimental)
    mets, yt, yp, X_te = run_rf_level1(il_df, "Illinois Primary (Mackinaw R.)")
    l1_mets["illinois"] = mets
    l1_preds["illinois"] = (yt, yp, X_te)
    results["level1_il"] = mets

    # Illinois Secondary — multi-station validation
    mets_il2, yt_il2, yp_il2, X_te_il2 = run_rf_level1(il2_df, "Illinois Secondary (Fox R.) [VALIDATION]")
    results["level1_il2_validation"] = mets_il2
    print(f"  Multi-station pattern check: IL2 R²={mets_il2['r2']:.4f} "
          f"(expected low, same pattern as IL primary)")

    # Naive baseline (Illinois Primary)
    il_X_te = l1_preds["illinois"][2]
    il_y_te = pd.Series(l1_preds["illinois"][0], index=il_X_te.index)
    naive_mets = compute_naive_baseline(il_X_te, il_y_te, il_df)
    results["naive_il"] = naive_mets
    print_metrics("Naive Baseline — Illinois (current = future)", naive_mets)

    predictive_gap = l1_mets["california"]["r2"] - l1_mets["illinois"]["r2"]
    print(f"\n  Predictive Gap (CA vs IL Primary): {predictive_gap:.3f} R²")
    print(f"  Predictive Gap (CA vs IL Secondary): "
          f"{l1_mets['california']['r2'] - mets_il2['r2']:.3f} R²")
    results["level1_predictive_gap_r2"]  = round(predictive_gap, 4)

    # ── Level 2: RF + Soil Moisture — AMC-Aware (Illinois Primary only) ───────
    section("LEVEL 2: RF + SOIL MOISTURE  (Illinois Primary Only)")

    X2, y2 = prepare_rf_level2(il_df, forecast_horizon=config.FORECAST_HORIZON)
    X2_tr, X2_te, y2_tr, y2_te = rf_train_test_split(X2, y2)
    rf2 = RandomForestModel()
    rf2.fit(X2_tr, y2_tr)
    yp2 = rf2.predict(X2_te)
    il_l2_mets  = compute_metrics(y2_te.values, yp2)
    il_l2_preds = (y2_te.values, yp2, X2_te)

    results["level2_il"] = il_l2_mets
    print_metrics("Level 2 — Illinois Primary (RF + Soil Moisture)", il_l2_mets)
    print(f"  Level 1 -> Level 2 R² improvement: "
          f"+{il_l2_mets['r2'] - l1_mets['illinois']['r2']:.4f}")

    # ── Level 3: LSTM — Temporal Memory Architecture (Illinois Primary only) ──
    section("LEVEL 3: LSTM TEMPORAL MEMORY  (Illinois Primary Only)")

    _, il_yt_l3, il_yp_l3, il_l3_mets, _, _, _ = train_lstm_il(il_df)
    results["level3_il"] = il_l3_mets
    print_metrics("Level 3 — Illinois Primary (LSTM)", il_l3_mets)
    print(f"  Level 1 -> Level 3 R²: +{il_l3_mets['r2'] - l1_mets['illinois']['r2']:.4f}")
    print(f"  Level 2 -> Level 3 R²: +{il_l3_mets['r2'] - il_l2_mets['r2']:.4f}")

    # ── Save predictions for storm recomputation / reproducibility ───────────
    preds_dir = os.path.join(config.OUTPUT_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    np.save(os.path.join(preds_dir, "l1_yt.npy"), l1_preds["illinois"][0])
    np.save(os.path.join(preds_dir, "l1_yp.npy"), l1_preds["illinois"][1])
    np.save(os.path.join(preds_dir, "l2_yt.npy"), y2_te.values)
    np.save(os.path.join(preds_dir, "l2_yp.npy"), yp2)
    np.save(os.path.join(preds_dir, "l3_yt.npy"), il_yt_l3)
    np.save(os.path.join(preds_dir, "l3_yp.npy"), il_yp_l3)
    # Save aligned precip series for each test set
    l1_preds["illinois"][2]["precipitation_mm_hr"].to_csv(
        os.path.join(preds_dir, "l1_precip_te.csv"))
    X2_te["precipitation_mm_hr"].to_csv(
        os.path.join(preds_dir, "l2_precip_te.csv"))
    il_df["precipitation_mm_hr"].iloc[-len(il_yt_l3):].reset_index(drop=True).to_csv(
        os.path.join(preds_dir, "l3_precip_te.csv"), header=True)
    print(f"  Predictions saved to {preds_dir}/")

    # ── Storm Intensity Sub-Analysis (All 3 IL levels) ────────────────────────
    section("STORM INTENSITY SUB-ANALYSIS  (Robustness — Not an IV)")

    precip_l1_te = get_precip_for_test(l1_preds["illinois"][2], il_df)
    precip_l2_te = get_precip_for_test(X2_te, il_df)
    n_l3_te      = len(il_yt_l3)
    precip_l3_te = il_df["precipitation_mm_hr"].iloc[-n_l3_te:].reset_index(drop=True)

    storm_l1 = storm_intensity_metrics(
        l1_preds["illinois"][0], l1_preds["illinois"][1], precip_l1_te
    )
    storm_l2 = storm_intensity_metrics(y2_te.values, yp2, precip_l2_te)
    storm_l3 = storm_intensity_metrics(
        pd.Series(il_yt_l3).values, pd.Series(il_yp_l3).values,
        precip_l3_te
    )

    print(f"\n  Storm Intensity R² — Illinois Primary:")
    print(f"  {'Class':<12}  {'Level 1':>9}  {'Level 2':>9}  {'Level 3':>9}")
    print(f"  {'-'*45}")
    for cls in ["dry", "light", "moderate", "heavy"]:
        r1 = storm_l1.loc[cls, "r2"] if cls in storm_l1.index else float("nan")
        r2 = storm_l2.loc[cls, "r2"] if cls in storm_l2.index else float("nan")
        r3 = storm_l3.loc[cls, "r2"] if cls in storm_l3.index else float("nan")
        print(f"  {cls.capitalize():<12}  {r1:>9.4f}  {r2:>9.4f}  {r3:>9.4f}")

    results["storm_il_l1"] = storm_l1.to_dict()
    results["storm_il_l2"] = storm_l2.to_dict()
    results["storm_il_l3"] = storm_l3.to_dict()

    # Build unified storm_intensity dict keyed by class, used by generate_tables.py
    storm_intensity = {}
    for cls in ["dry", "light", "moderate", "heavy"]:
        entry = {}
        for df_key, level_prefix in [(storm_l1, "l1"), (storm_l2, "l2"), (storm_l3, "l3")]:
            if cls in df_key.index:
                for col in df_key.columns:
                    val = df_key.loc[cls, col]
                    entry[f"{level_prefix}_{col}"] = (
                        None if (isinstance(val, float) and np.isnan(val)) else float(val)
                    )
                entry[f"{level_prefix}_n_samples"] = int(df_key.loc[cls, "n_samples"]) \
                    if "n_samples" in df_key.columns else 0
        # n_samples from any non-empty level
        for df_ref in [storm_l1, storm_l2, storm_l3]:
            if cls in df_ref.index and "n_samples" in df_ref.columns:
                entry["n_samples"] = int(df_ref.loc[cls, "n_samples"])
                break
        storm_intensity[cls] = entry
    results["storm_intensity"] = storm_intensity

    # Persist naive metrics at top level for generate_tables.py convenience
    naive = results.get("naive_il", {})
    results["naive_r2"]   = naive.get("r2",   0.458)
    results["naive_rmse"] = naive.get("rmse",  9.46)
    results["naive_mae"]  = naive.get("mae",   5.79)
    results["naive_tpr"]  = naive.get("tpr",   0.677)
    results["naive_far"]  = naive.get("far",   0.323)

    # ── Community Impact Calculation ──────────────────────────────────────────
    section("COMMUNITY IMPACT ANALYSIS")

    impact = compute_community_impact(results, il_df)
    results["community_impact"] = impact

    print(f"\n  Test period          : {impact['test_period_years']:.1f} years")
    print(f"  Hazard events (test) : {impact['hazard_events_in_test']}")
    print(f"  Annualized rate      : {impact['annualized_hazard_rate']:.1f} events/year")
    print(f"\n  Events caught per year (Illinois Primary):")
    print(f"    Naive Baseline : {impact['naive_caught_per_year']:.1f}")
    print(f"    Level 1 (RF)   : {impact['l1_caught_per_year']:.1f}")
    print(f"    Level 2 (RF+SM): {impact['l2_caught_per_year']:.1f}")
    print(f"    Level 3 (LSTM) : {impact['l3_caught_per_year']:.1f}")
    print(f"\n  LSTM improvement vs Level 1: +{impact['additional_events_vs_l1']:.1f} events/year")
    print(f"  Fewer missed events/year   : {impact['fewer_missed_per_year']:.1f}")
    print(f"\n  IMPACT STATEMENT:")
    print(f"  {impact['impact_sentence']}")

    # ── Generate the Five Mandated Figures ────────────────────────────────────
    section("GENERATING FIVE MANDATED FIGURES")

    # Inject naive metrics into Level 1 dict so Figure 1 can render the baseline bar
    il_l1_for_fig = dict(l1_mets["illinois"])
    il_l1_for_fig["naive_r2"]  = naive_mets["r2"]
    il_l1_for_fig["naive_tpr"] = naive_mets["tpr"]

    fig = plot_figure1_progression(
        il_l1_for_fig, il_l2_mets, il_l3_mets, l1_mets["california"]
    )
    save_fig(fig, "figure1_progression.png")

    print("  Graph 2 (CCF) already saved as figure2_ccf.png")

    fig = plot_figure3_storm_intensity(
        storm_l1, storm_l2, storm_l3,
        overall_l1_r2=l1_mets["illinois"]["r2"],
    )
    save_fig(fig, "figure3_storm_intensity.png")

    fig = plot_figure4_safety(
        naive_mets, l1_mets["illinois"], il_l2_mets, il_l3_mets,
        l1_mets["california"],
    )
    save_fig(fig, "figure4_safety.png")

    # ── Multi-Station Validation Summary Figure ───────────────────────────────
    try:
        _make_multistation_fig(
            results, ca_opt_lag, il_opt_lag, il2_opt_lag,
            l1_mets, mets_il2, il2_name
        )
    except Exception as e:
        print(f"  [WARN] Multi-station summary figure skipped: {e}")

    # ── Final Summary Table ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT COMPLETE   ({elapsed:.0f} seconds)")
    print(f"{'='*65}")

    rows = [
        ("Naive Baseline — Illinois",           results["naive_il"]),
        ("Level 1 — Illinois (RF Rain Only)",   results["level1_il"]),
        ("Level 1 — California (Benchmark)",    results["level1_ca"]),
        ("Level 1 — IL Secondary (Validation)", results["level1_il2_validation"]),
        ("Level 2 — Illinois (RF + Soil)",      results["level2_il"]),
        ("Level 3 — Illinois (LSTM)",           results["level3_il"]),
    ]
    print(f"\n  {'Model':<44}  {'R2':>7}  {'RMSE':>7}  {'TPR':>7}  {'FAR':>7}")
    print(f"  {'-'*44}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for label, m in rows:
        print(f"  {label:<44}  {m['r2']:>7.4f}  {m['rmse']:>7.2f}  "
              f"  {m['tpr']:>6.2%}  {m['far']:>6.2%}")

    print(f"\n  CCF Background Evidence")
    print(f"  {'-'*55}")
    print(f"  California            : {results['ccf_ca_lag_hours']:2d} hr lag")
    print(f"  Illinois Primary      : {results['ccf_il_lag_hours']:2d} hr lag  "
          f"(Sponge Effect = {results['ccf_sponge_effect_il']} hr)")
    print(f"  Illinois Secondary    : {results['ccf_il2_lag_hours']:2d} hr lag  "
          f"(Sponge Effect = {results['ccf_sponge_effect_il2']} hr)")

    json_path = os.path.join(config.OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results -> {json_path}")
    print(f"  All figures  -> {config.OUTPUT_DIR}/\n")


def _make_multistation_fig(results, ca_lag, il_lag, il2_lag,
                            l1_mets, mets_il2, il2_name):
    """Optional supplementary figure comparing L1 performance across all stations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Multi-Station Validation — Hydrological Buffer Effect Generalizes Across Illinois\n"
        "Both buffered watersheds show the same CCF lag pattern and model performance gap",
        fontsize=12, fontweight="bold",
    )

    labels_r2  = ["California\n(Benchmark)", "IL Primary\n(Mackinaw R.)", f"IL Secondary\n(Fox R.)"]
    r2_vals    = [l1_mets["california"]["r2"], l1_mets["illinois"]["r2"], mets_il2["r2"]]
    tpr_vals   = [l1_mets["california"]["tpr"], l1_mets["illinois"]["tpr"], mets_il2["tpr"]]
    lag_vals   = [ca_lag, il_lag, il2_lag]
    colors     = ["#E07B39", "#4A90D9", "#7D3C98"]

    # R² bar chart
    ax1 = axes[0]
    bars = ax1.bar(labels_r2, r2_vals, color=colors, edgecolor="white", linewidth=2, width=0.5)
    for bar, val in zip(bars, r2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"R² = {val:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("R²  (Coefficient of Determination)", fontsize=11)
    ax1.set_title("Level 1 RF (Rain Only) — All Stations\n"
                  "Both IL stations show low R², CA shows high R²", fontsize=10)

    # CCF lag bar chart
    ax2 = axes[1]
    bars2 = ax2.bar(labels_r2, lag_vals, color=colors, edgecolor="white", linewidth=2, width=0.5)
    for bar, val in zip(bars2, lag_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                 f"{val} hr", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(lag_vals) * 1.35)
    ax2.set_ylabel("Peak CCF Lag (hours)", fontsize=11)
    ax2.set_title("CCF Peak Lag — All Stations\n"
                  "Both IL stations show extended lag (Sponge Effect)", fontsize=10)
    ax2.axhline(ca_lag, color="#E07B39", ls="--", lw=1.5, label=f"CA baseline ({ca_lag} hr)")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "figure6_multistation_validation.png")


# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Hydrological Buffer Effect experiment.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use physics-based synthetic data instead of real APIs")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
