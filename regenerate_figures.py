"""
regenerate_figures.py — Rebuild all graphs and tables from saved results/predictions.

Run this instead of run_experiment.py whenever you only need to update visuals.
CCF is recomputed from the cached raw CSVs (fast — no ML training needed).

Requires:
  outputs/results.json
  outputs/predictions/l1_yt.npy  l1_yp.npy  l1_precip_te.csv
  outputs/predictions/l2_yt.npy  l2_yp.npy  l2_precip_te.csv
  outputs/predictions/l3_yt.npy  l3_yp.npy  l3_precip_te.csv
  data/raw/california_2012_2023.csv
  data/raw/illinois_2012_2023.csv
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.visualizations import (
    plot_figure1_progression,
    plot_figure2_ccf,
    plot_figure3_storm_intensity,
    plot_figure4_safety,
    save_fig,
)
from src.analysis import compute_ccf, storm_intensity_metrics
from generate_tables import make_table1, make_table2, make_table3, make_table4

PREDS   = os.path.join(config.OUTPUT_DIR, "predictions")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def load_arr(name):
    return np.load(os.path.join(PREDS, name))


def load_precip(name):
    return pd.read_csv(os.path.join(PREDS, name), index_col=0).squeeze()


def load_raw_csv(name):
    path = os.path.join(RAW_DIR, name)
    df = pd.read_csv(path)
    # Try to parse a datetime column if present
    for col in ["datetime", "date", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.set_index(col)
            break
    return df


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def main():
    # ── Load results ──────────────────────────────────────────────────────────
    results_path = os.path.join(config.OUTPUT_DIR, "results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run run_experiment.py first.")
        sys.exit(1)
    with open(results_path) as f:
        res = json.load(f)

    naive_mets = res["naive_il"]
    l1_mets    = res["level1_il"]
    l2_mets    = res["level2_il"]
    l3_mets    = res["level3_il"]
    ca_mets    = res.get("level1_ca", {})

    # Inject naive metrics into l1 dict for Graph 1
    l1_for_fig = dict(l1_mets)
    l1_for_fig["naive_r2"]  = naive_mets["r2"]
    l1_for_fig["naive_tpr"] = naive_mets["tpr"]

    # ── Load saved predictions ────────────────────────────────────────────────
    section("Loading saved predictions")
    l1_yt = load_arr("l1_yt.npy"); l1_yp = load_arr("l1_yp.npy")
    l2_yt = load_arr("l2_yt.npy"); l2_yp = load_arr("l2_yp.npy")
    l3_yt = load_arr("l3_yt.npy"); l3_yp = load_arr("l3_yp.npy")
    p1 = load_precip("l1_precip_te.csv")
    p2 = load_precip("l2_precip_te.csv")
    p3 = load_precip("l3_precip_te.csv")
    print("  Loaded prediction arrays.")

    # ── Storm intensity (picks up current config thresholds) ──────────────────
    section("Storm intensity metrics")
    storm_l1 = storm_intensity_metrics(l1_yt, l1_yp, p1)
    storm_l2 = storm_intensity_metrics(l2_yt, l2_yp, p2)
    storm_l3 = storm_intensity_metrics(l3_yt, l3_yp, p3)
    for cls in ["dry", "light", "moderate", "heavy"]:
        r1 = storm_l1.loc[cls, "r2"] if cls in storm_l1.index else float("nan")
        r2 = storm_l2.loc[cls, "r2"] if cls in storm_l2.index else float("nan")
        r3 = storm_l3.loc[cls, "r2"] if cls in storm_l3.index else float("nan")
        print(f"  {cls.capitalize():<12}  L1={r1:>7.4f}  L2={r2:>7.4f}  L3={r3:>7.4f}")

    # ── Graph 1: Accuracy Progression ─────────────────────────────────────────
    section("Graph 1 — Accuracy Progression")
    fig = plot_figure1_progression(l1_for_fig, l2_mets, l3_mets, ca_mets)
    save_fig(fig, "figure1_progression.png")

    # ── Graph 2: CCF (recomputed from raw CSVs, 2-panel only) ─────────────────
    section("Graph 2 — CCF Lag Analysis")
    try:
        ca_df = load_raw_csv("california_2012_2023.csv")
        il_df = load_raw_csv("illinois_2012_2023.csv")
        ca_lags, ca_corrs = compute_ccf(
            ca_df["precipitation_mm_hr"], ca_df["turbidity"])
        il_lags, il_corrs = compute_ccf(
            il_df["precipitation_mm_hr"], il_df["turbidity"])
        ca_lag = int(ca_lags[np.argmax(ca_corrs)])
        il_lag = int(il_lags[np.argmax(il_corrs)])
        print(f"  CA lag = {ca_lag} hr  |  IL lag = {il_lag} hr")
        fig = plot_figure2_ccf(ca_lags, ca_corrs, ca_lag,
                               il_lags, il_corrs, il_lag)
        save_fig(fig, "figure2_ccf.png")
    except Exception as e:
        print(f"  [WARN] CCF recomputation failed: {e} — existing file kept.")

    # ── Graph 3: Storm Intensity Stress Test ──────────────────────────────────
    section("Graph 3 — Storm Intensity Stress Test")
    fig = plot_figure3_storm_intensity(
        storm_l1, storm_l2, storm_l3,
        overall_l1_r2=l1_mets["r2"],
    )
    save_fig(fig, "figure3_storm_intensity.png")

    # ── Graph 4: Hazard Detection Performance ─────────────────────────────────
    section("Graph 4 — Hazard Detection Performance")
    fig = plot_figure4_safety(naive_mets, l1_mets, l2_mets, l3_mets, ca_mets)
    save_fig(fig, "figure4_safety.png")

    # ── Tables 1–4 ────────────────────────────────────────────────────────────
    section("Tables 1–4")
    make_table1(res)
    make_table2(res)
    make_table3(res)
    make_table4(res)

    print(f"\n  All outputs written to {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
