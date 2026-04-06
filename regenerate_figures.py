"""
regenerate_figures.py — Rebuild all graphs and tables from saved results/predictions.

Run this instead of run_experiment.py whenever you only need to update visuals
(label changes, color fixes, layout tweaks) without retraining the LSTM.

Requires:
  outputs/results.json
  outputs/predictions/l1_yt.npy  l1_yp.npy  l1_precip_te.csv
  outputs/predictions/l2_yt.npy  l2_yp.npy  l2_precip_te.csv
  outputs/predictions/l3_yt.npy  l3_yp.npy  l3_precip_te.csv
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.visualizations import (
    plot_figure1_progression,
    plot_figure3_timeseries,
    plot_figure4_storm_intensity,
    plot_figure5_safety,
    save_fig,
)
from src.analysis import storm_intensity_metrics
from generate_tables import make_table1, make_table2, make_table3, make_table4

PREDS = os.path.join(config.OUTPUT_DIR, "predictions")


def load_arr(name):
    return np.load(os.path.join(PREDS, name))


def load_precip(name):
    return pd.read_csv(os.path.join(PREDS, name), index_col=0).squeeze()


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

    # Inject naive R² / TPR into l1 dict for Graph 1
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
    print("  Loaded all prediction arrays.")

    # ── Recompute storm intensity (picks up new thresholds from config) ───────
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

    # ── Graph 2: CCF — skip (no CCF arrays saved; existing file is correct) ───
    section("Graph 2 — CCF")
    print("  Skipped (CCF arrays not cached — existing figure2_ccf.png retained).")

    # ── Graph 3: Time Series ──────────────────────────────────────────────────
    section("Graph 3 — Time Series")
    fig = plot_figure3_timeseries(
        l1_yt, l1_yp, l2_yt, l2_yp, l3_yt, l3_yp,
        l1_mets, l2_mets, l3_mets,
    )
    save_fig(fig, "figure3_timeseries.png")

    # ── Graph 4: Storm Intensity ──────────────────────────────────────────────
    section("Graph 4 — Storm Intensity")
    fig = plot_figure4_storm_intensity(storm_l1, storm_l2, storm_l3)
    save_fig(fig, "figure4_storm_intensity.png")

    # ── Graph 5: Safety Metrics ───────────────────────────────────────────────
    section("Graph 5 — Safety Metrics")
    fig = plot_figure5_safety(naive_mets, l1_mets, l2_mets, l3_mets, ca_mets)
    save_fig(fig, "figure5_safety.png")

    # ── Tables ────────────────────────────────────────────────────────────────
    section("Tables 1–4")
    make_table1(res)
    make_table2(res)
    make_table3(res)
    make_table4(res)

    print(f"\n  All outputs written to {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
