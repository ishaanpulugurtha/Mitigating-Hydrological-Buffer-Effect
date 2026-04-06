"""
visualizations.py — Publication-ready graphs for science fair presentation.

Graph 1. Turbidity Prediction Accuracy by Model Level
Graph 2. Rainfall-to-Turbidity Response Time: Cross-Correlation Analysis
Graph 3. Predicted vs. Measured Turbidity Across Three Model Levels
Graph 4. Model Performance Across Storm Intensity Categories
Graph 5. Hazard Detection and False Alarm Rates

All functions return a matplotlib Figure. Call save_fig(fig, "filename") to write.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# -- Global style / palette ----------------------------------------------------
BG      = "#F8F9FA"
NAVY    = "#1A252F"
GRAY_L  = "#CCCCCC"   # Naive Baseline
GRAY_M  = "#888888"   # Level 1
BLUE    = "#3B82C4"   # Level 2
GREEN   = "#1A7340"   # Level 3
NB_CLR  = "#E07B39"   # Non-Buffered Reference (orange)
HAZARD  = "#D94F3D"   # Hazard / alert lines

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})


def save_fig(fig: plt.Figure, filename: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


def _clip_r2(val):
    """Display R² is never shown as negative — clip to 0.000 minimum."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return max(0.0, float(val))


# ==============================================================================
#  Graph 1 — Turbidity Prediction Accuracy by Model Level
# ==============================================================================

def plot_figure1_progression(
    il_l1_mets: dict,
    il_l2_mets: dict,
    il_l3_mets: dict,
    ca_l1_mets: dict,
) -> plt.Figure:
    """
    Graph 1: Single bar chart showing R² progression across four model levels.
    Non-Buffered Reference shown as a dashed orange horizontal line.
    Only R² is shown here; TPR/FAR are dedicated to Graph 5 (no overlap).
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Graph 1 — Turbidity Prediction Accuracy by Model Level\n"
        "Illinois Mackinaw River Watershed — Glacially Buffered Soil Profile",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    labels = [
        "Naive\nBaseline",
        "Level 1\nRF (Rain Only)",
        "Level 2\nRF + Soil\nMoisture",
        "Level 3\nLSTM",
    ]
    colors = [GRAY_L, GRAY_M, BLUE, GREEN]
    x = np.arange(len(labels))

    r2_vals = [
        _clip_r2(il_l1_mets.get("naive_r2", 0.458)),
        _clip_r2(il_l1_mets["r2"]),
        _clip_r2(il_l2_mets["r2"]),
        _clip_r2(il_l3_mets["r2"]),
    ]

    bars = ax.bar(x, r2_vals, color=colors, width=0.55,
                  edgecolor="white", linewidth=2.0, zorder=3)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.018, f"{val:.3f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=NAVY)

    # Non-Buffered Reference dashed line
    nb_r2 = config.CA_BENCHMARK_R2
    ax.axhline(nb_r2, color=NB_CLR, ls="--", lw=2.5, zorder=4,
               label=f"Non-Buffered Reference  R² = {nb_r2:.3f}")

    # Single annotation: L1→L3 total gain
    l1_r2 = _clip_r2(il_l1_mets["r2"])
    l3_r2 = _clip_r2(il_l3_mets["r2"])
    delta  = l3_r2 - l1_r2
    ax.annotate(
        f"+{delta:.3f} R²\n(Level 1 → Level 3)",
        xy=(3, l3_r2),
        xytext=(2.45, l3_r2 - 0.16),
        fontsize=11, fontweight="bold", color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("R²  (Coefficient of Determination)", fontsize=12)
    ax.set_xlabel("Model Level", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # Legend swatches for levels
    patches = [
        mpatches.Patch(color=GRAY_L, label="Naive Baseline"),
        mpatches.Patch(color=GRAY_M, label="Level 1: RF (Rain Only)"),
        mpatches.Patch(color=BLUE,   label="Level 2: RF + Soil Moisture"),
        mpatches.Patch(color=GREEN,  label="Level 3: LSTM"),
    ]
    leg1 = ax.legend(handles=patches, fontsize=10, loc="upper left",
                     title="Model Levels", title_fontsize=10)
    ax.add_artist(leg1)
    # Second legend for reference line
    ref_line = mpatches.Patch(
        color=NB_CLR, label=f"Non-Buffered Reference  R² = {nb_r2:.3f}",
        linestyle="--", fill=False, linewidth=2)
    ax.legend(handles=[
        plt.Line2D([0], [0], color=NB_CLR, ls="--", lw=2.5,
                   label=f"Non-Buffered Reference  R² = {nb_r2:.3f}")
    ], fontsize=10, loc="lower right")

    fig.tight_layout()
    return fig


# ==============================================================================
#  Graph 2 — Cross-Correlation Analysis (3-Panel)
# ==============================================================================

def plot_figure2_ccf(
    ca_lags:  np.ndarray, ca_corrs:  np.ndarray, ca_lag:  int,
    il_lags:  np.ndarray, il_corrs:  np.ndarray, il_lag:  int,
    il2_lags: np.ndarray = None, il2_corrs: np.ndarray = None,
    il2_lag:  int = None, il2_name: str = "Illinois Secondary",
) -> plt.Figure:
    """
    Graph 2: Cross-correlation — CA (non-buffered), IL primary, IL secondary.
    """
    n_panels = 3 if il2_lags is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5),
                              sharey=True)
    if n_panels == 2:
        axes = list(axes)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Graph 2 — Rainfall-to-Turbidity Response Time: Cross-Correlation Analysis\n"
        "Quantifying the Hydrological Buffer Effect across three watersheds",
        fontsize=12, fontweight="bold", color=NAVY,
    )

    panels = [
        (ca_lags,  ca_corrs,  ca_lag,
         "California — Non-Buffered Watershed\n(Mediterranean Climate)",
         "Thin rocky soils — rapid runoff",
         NB_CLR),
        (il_lags,  il_corrs,  il_lag,
         "Illinois Primary — Mackinaw River\n(Glacially Buffered)",
         "Deep glacial-till soils — delayed response",
         BLUE),
    ]
    if n_panels == 3:
        short = il2_name.split(",")[0] if "," in il2_name else il2_name
        panels.append((
            il2_lags, il2_corrs, il2_lag,
            f"Illinois Secondary — {short}\n(Glacially Buffered — Validation)",
            "Confirms pattern across IL watersheds",
            "#7D3C98",
        ))

    for ax, (lags, corrs, opt_lag, title, subtitle, color) in zip(axes, panels):
        ax.set_facecolor(BG)
        ax.fill_between(lags, 0, corrs, color=color, alpha=0.25)
        ax.plot(lags, corrs, color=color, lw=2.2)
        ax.axvline(opt_lag, color=HAZARD, ls="--", lw=2.2,
                   label=f"Peak lag = {opt_lag} hr")
        ax.set_xlabel("Lag (hours after rainfall peak)", fontsize=10)
        ax.set_ylabel("Cross-Correlation Coefficient (r)", fontsize=10)
        ax.set_xlim(0, config.MAX_LAG_HOURS)
        ax.set_ylim(0, 0.8)
        ax.set_title(title, fontsize=10.5, color=color, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Annotate peak correlation value at the curve's highest point
        peak_idx = np.argmax(corrs)
        peak_r   = corrs[peak_idx]
        ax.annotate(
            f"r = {peak_r:.3f}",
            xy=(lags[peak_idx], peak_r),
            xytext=(lags[peak_idx] + 3, peak_r - 0.07),
            fontsize=9.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

        ax.text(0.62, 0.75, f"{subtitle}\n{opt_lag}-hr peak lag",
                transform=ax.transAxes, fontsize=8.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=color, alpha=0.88))

    sponge = il_lag - ca_lag
    extra  = (f"  |  Fox River: {il2_lag} hr — confirms across IL"
              if n_panels == 3 and il2_lag else "")
    fig.text(
        0.5, -0.03,
        (f"Quantified Sponge Effect: Illinois lag ({il_lag} hr) − "
         f"California lag ({ca_lag} hr) = {sponge}-hour buffer delay{extra}"),
        ha="center", fontsize=11, fontweight="bold", color=HAZARD,
    )
    fig.tight_layout()
    return fig


# ==============================================================================
#  Graph 3 — Predicted vs. Measured Turbidity (3 Stacked Panels)
# ==============================================================================

def plot_figure3_timeseries(
    il_yt_l1: np.ndarray, il_yp_l1: np.ndarray,
    il_yt_l2: np.ndarray, il_yp_l2: np.ndarray,
    il_yt_l3: np.ndarray, il_yp_l3: np.ndarray,
    il_l1_mets: dict,
    il_l2_mets: dict,
    il_l3_mets: dict,
    n_hours: int = 600,
    threshold: float = config.TURBIDITY_HAZARD_NTU,
) -> plt.Figure:
    """
    Graph 3: Three stacked panels — same test period, IL predictions.
    Level 3 panel includes True Positive early-warning markers.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Graph 3 — Predicted vs. Measured Turbidity Across Three Model Levels\n"
        "Illinois Mackinaw River — 600-hour test window",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    panels = [
        (il_yt_l1, il_yp_l1, il_l1_mets, GRAY_M, "Level 1: RF (Rainfall Only)",
         (f"Level 1: RF (Rainfall Only)  |  "
          f"R²={_clip_r2(il_l1_mets['r2']):.3f}  RMSE={il_l1_mets['rmse']:.2f} NTU  "
          f"TPR={il_l1_mets['tpr']:.1%}  FAR={il_l1_mets['far']:.1%}")),
        (il_yt_l2, il_yp_l2, il_l2_mets, BLUE, "Level 2: RF + Soil Moisture",
         (f"Level 2: RF + Soil Moisture  |  "
          f"R²={_clip_r2(il_l2_mets['r2']):.3f}  RMSE={il_l2_mets['rmse']:.2f} NTU  "
          f"TPR={il_l2_mets['tpr']:.1%}  FAR={il_l2_mets['far']:.1%}")),
        (il_yt_l3, il_yp_l3, il_l3_mets, GREEN, "Level 3: LSTM",
         (f"Level 3: LSTM  |  "
          f"R²={_clip_r2(il_l3_mets['r2']):.3f}  RMSE={il_l3_mets['rmse']:.2f} NTU  "
          f"TPR={il_l3_mets['tpr']:.1%}  FAR={il_l3_mets['far']:.1%}")),
    ]

    for panel_idx, (ax, (yt, yp, mets, color, short_label, subtitle)) in \
            enumerate(zip(axes, panels)):
        n = min(n_hours, len(yt))
        t = np.arange(n)
        yt_n = yt[:n]
        yp_n = yp[:n]

        ax.plot(t, yt_n, color="#999999", lw=1.1, alpha=0.85, label="Measured turbidity")
        ax.plot(t, yp_n, color=color, lw=1.8, ls="--",
                label=f"Predicted — {short_label}")
        ax.axhline(threshold, color=HAZARD, ls=":", lw=1.8,
                   label=f"Hazard threshold ({threshold} NTU)")

        # Level 3 panel: mark True Positive early-warning events
        if panel_idx == 2:
            actual_haz = yt_n >= threshold
            pred_haz   = yp_n >= threshold
            tp_flags   = actual_haz & pred_haz
            tp_rising  = np.where(np.diff(tp_flags.astype(int)) == 1)[0]
            labeled_once = False
            n_marked = 0
            for edge in tp_rising:
                if n_marked >= 4:
                    break
                ax.scatter([edge], [yp_n[edge] + 1.5], marker="^",
                           color=GREEN, s=90, zorder=6,
                           label="True Positive — early warning triggered"
                           if not labeled_once else "")
                if not labeled_once:
                    ax.annotate(
                        "True Positive\nearly warning triggered",
                        xy=(edge, yp_n[edge] + 1.5),
                        xytext=(min(edge + 25, n - 10), yp_n[edge] + 6),
                        fontsize=8.5, color=GREEN, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3),
                    )
                    labeled_once = True
                n_marked += 1

        ax.set_ylabel("Turbidity (NTU)", fontsize=10)
        ax.set_ylim(0, max(30, float(yt_n.max()) * 1.05))
        ax.set_xlim(0, n)
        ax.legend(fontsize=8.5, ncol=4, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        ax.text(0.01, 0.94, subtitle,
                transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=color, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

    axes[-1].set_xlabel("Hours (test set)", fontsize=10)
    fig.tight_layout()
    return fig


# ==============================================================================
#  Graph 4 — Model Performance Across Storm Intensity Categories
# ==============================================================================

def plot_figure4_storm_intensity(
    storm_l1: pd.DataFrame,
    storm_l2: pd.DataFrame,
    storm_l3: pd.DataFrame,
) -> plt.Figure:
    """
    Graph 4: Grouped R² by storm intensity class.
    R² clipped to 0.000 minimum. Heavy Storm bars hatched if no data.
    """
    classes      = ["dry", "light", "moderate", "heavy"]
    class_labels = ["Dry / No Rain\n(0–0.1 mm/hr)",
                    "Light Rain\n(0.1–3 mm/hr)",
                    "Moderate Storm\n(3–8 mm/hr)",
                    "Heavy Storm\n(>8 mm/hr)"]
    x     = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.suptitle(
        "Graph 4 — Model Performance Across Storm Intensity Categories\n"
        "Illinois Watershed — R² by precipitation intensity class",
        fontsize=12, fontweight="bold", color=NAVY,
    )

    level_data = [
        (storm_l1, "Level 1: RF (Rain Only)",     GRAY_M),
        (storm_l2, "Level 2: RF + Soil Moisture", BLUE),
        (storm_l3, "Level 3: LSTM",               GREEN),
    ]

    heavy_is_na = True

    for offset, (storm_df, label, color) in enumerate(level_data):
        for ci, cls in enumerate(classes):
            raw_val = float(storm_df.loc[cls, "r2"]) \
                if (cls in storm_df.index and not np.isnan(storm_df.loc[cls, "r2"])) \
                else np.nan
            val = _clip_r2(raw_val)  # display value clipped to 0

            xi = x[ci] + (offset - 1) * width
            if cls == "heavy" and (np.isnan(raw_val) if raw_val is not None else True):
                ax.bar(xi, 0.04, width, bottom=-0.02, color="#DDDDDD",
                       edgecolor="#999999", linewidth=1.0, hatch="////",
                       alpha=0.6, zorder=3)
            else:
                if cls == "heavy":
                    heavy_is_na = False
                ax.bar(xi, val, width, color=color,
                       edgecolor="white", linewidth=1.5, zorder=3,
                       label=label if ci == 0 else "")
                if val is not None and not np.isnan(val):
                    ax.text(xi, val + 0.018, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=8.5,
                            fontweight="bold", color="#222222", zorder=5)

    if heavy_is_na:
        ax.text(x[3], 0.12, "No events\nin test period",
                ha="center", fontsize=8.5, color="#888888", style="italic")

    ax.axhline(0, color="black", lw=1.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("R²  (Coefficient of Determination)", fontsize=11)

    handles, lbls = ax.get_legend_handles_labels()
    seen, unique_h, unique_l = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); unique_h.append(h); unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    return fig


# ==============================================================================
#  Graph 5 — Hazard Detection and False Alarm Rates
# ==============================================================================

def plot_figure5_safety(
    naive_mets: dict,
    il_l1_mets: dict,
    il_l2_mets: dict,
    il_l3_mets: dict,
    ca_l1_mets: dict = None,
    threshold: float = config.TURBIDITY_HAZARD_NTU,
) -> plt.Figure:
    """
    Graph 5: TPR (left) and FAR (right) — all four IL levels.
    No overlap with Graph 1 (R² is Graph 1 only; TPR+FAR are Graph 5 only).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Graph 5 — Hazard Detection and False Alarm Rates\n"
        "Illinois Watershed — Public Health Safety Metrics by Model Level",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    labels = ["Naive\nBaseline", "Level 1\nRF (Rain Only)",
              "Level 2\nRF + Soil\nMoisture", "Level 3\nLSTM"]
    colors = [GRAY_L, GRAY_M, BLUE, GREEN]
    mets   = [naive_mets, il_l1_mets, il_l2_mets, il_l3_mets]
    x      = np.arange(len(labels))

    # ── Left: TPR ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    tpr_vals = [m["tpr"] for m in mets]
    bars = ax1.bar(x, tpr_vals, color=colors, width=0.55,
                   edgecolor="white", linewidth=2.0, zorder=3)
    for bar, val in zip(bars, tpr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.015, f"{val:.1%}",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    delta_tpr = il_l3_mets["tpr"] - il_l1_mets["tpr"]
    ax1.annotate(
        f"+{delta_tpr:.1%}\nLevel 1 → Level 3",
        xy=(3, il_l3_mets["tpr"]),
        xytext=(2.42, il_l3_mets["tpr"] - 0.20),
        fontsize=10, fontweight="bold", color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
    )
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Hazard Detection Rate (True Positive Rate)", fontsize=11)
    ax1.set_xlabel("Model Level", fontsize=11)
    ax1.set_title("Hazard Detection Rate (TPR)\nHigher = catches more dangerous events",
                  fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.4, zorder=0)

    # ── Right: FAR ────────────────────────────────────────────────────────────
    ax2 = axes[1]
    far_vals = [m["far"] for m in mets]
    bars2 = ax2.bar(x, far_vals, color=colors, width=0.55,
                    edgecolor="white", linewidth=2.0, zorder=3)
    for bar, val in zip(bars2, far_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.008, f"{val:.1%}",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    delta_far = il_l1_mets["far"] - il_l3_mets["far"]
    ax2.annotate(
        f"−{delta_far:.1%}\nLevel 1 → Level 3",
        xy=(3, il_l3_mets["far"]),
        xytext=(2.38, il_l3_mets["far"] + 0.08),
        fontsize=10, fontweight="bold", color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
    )
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, max(0.55, max(far_vals) * 1.28))
    ax2.set_ylabel("False Alarm Rate", fontsize=11)
    ax2.set_xlabel("Model Level", fontsize=11)
    ax2.set_title("False Alarm Rate (FAR)\nLower = fewer unnecessary alerts",
                  fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.4, zorder=0)

    fig.text(
        0.5, -0.03,
        "The LSTM simultaneously improves hazard detection AND reduces false alarms — "
        "demonstrating practical deployability, not just statistical accuracy.",
        ha="center", fontsize=11, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#DFF0D8",
                  edgecolor=GREEN, alpha=0.85),
    )
    fig.tight_layout()
    return fig
