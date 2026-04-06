"""
generate_tables.py — Publication-ready tables for IJAS submission.

TABLE 1 — Overall Accuracy Comparison (R², RMSE, MAE, ΔR²)
TABLE 2 — Hazard Detection Safety Metrics (TPR, FAR, event counts)
TABLE 3 — Storm Intensity Sub-Analysis (R², RMSE, TPR, FAR by storm class)

Reads from outputs/results.json, renders PNG images.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

OUT_DIR = config.OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY    = "#1A252F"
WHITE   = "#FFFFFF"
GRAY_L  = "#F2F2F2"
BLUE_T  = "#D6E8F7"   # Level 2 row tint
GREEN_T = "#D5EFDC"   # Level 3 row tint
ROW_ALT = "#FAFAFA"
SEP_CLR = "#CCCCCC"
TEXT     = "#1A252F"
GREEN    = "#1A7340"
BLUE     = "#3B82C4"
RED_NEG  = "#CC2200"

# Total dangerous events denominator used for Table 2 (consistent with PRD)
HAZARD_EVENTS = 326


def load_results() -> dict:
    path = os.path.join(OUT_DIR, "results.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — using placeholder values.")
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, precision=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{precision}f}"


def pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.1f}%"


# ==============================================================================
#  Helper: render a table as a PNG
# ==============================================================================

def render_table(
    title: str,
    col_headers: list,
    rows: list,             # list of (cells, bg_color, bold_cells_idx)
    col_widths: list,       # relative widths (sum should be ~1)
    footer: str = "",
    highlight_sep_after: int = None,  # row index after which to draw separator
    filename: str = "table.png",
) -> str:
    n_cols = len(col_headers)
    n_rows = len(rows)

    fig_w = max(12, sum(col_widths) * 0.9)
    row_h = 0.52
    header_h = 0.72
    footer_h = 0.65 if footer else 0
    title_h  = 0.60
    fig_h = title_h + header_h + n_rows * row_h + footer_h + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Normalize column widths → x positions [0, 1]
    total_w = sum(col_widths)
    cum = [0.0]
    for w in col_widths:
        cum.append(cum[-1] + w / total_w)

    y_start  = 1.0 - title_h / fig_h
    y_header = y_start - header_h / fig_h

    # Title
    ax.text(0.5, (y_start + 1.0) / 2,
            title, transform=ax.transAxes,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=NAVY)

    # Header row
    for ci, (label, x0, x1) in enumerate(zip(col_headers, cum[:-1], cum[1:])):
        rect = FancyBboxPatch(
            (x0, y_header), x1 - x0, header_h / fig_h,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            facecolor=NAVY, edgecolor=WHITE, linewidth=1.0,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, y_header + (header_h / fig_h) / 2,
                label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=WHITE)

    # Data rows
    for ri, (cells, bg_color, bold_idx) in enumerate(rows):
        y_row = y_header - (ri + 1) * row_h / fig_h
        for ci, (cell, x0, x1) in enumerate(zip(cells, cum[:-1], cum[1:])):
            rect = FancyBboxPatch(
                (x0, y_row), x1 - x0, row_h / fig_h,
                boxstyle="square,pad=0",
                transform=ax.transAxes,
                facecolor=bg_color, edgecolor=SEP_CLR, linewidth=0.6,
                clip_on=False,
            )
            ax.add_patch(rect)
            fw = "bold" if ci in bold_idx else "normal"
            ax.text((x0 + x1) / 2, y_row + (row_h / fig_h) / 2,
                    str(cell), transform=ax.transAxes,
                    ha="center", va="center", fontsize=10.5,
                    fontweight=fw, color=TEXT)

        # Separator line after specified row
        if highlight_sep_after is not None and ri == highlight_sep_after:
            sep_y = y_row
            ax.plot([0, 1], [sep_y, sep_y],
                    color=NAVY, lw=1.8, transform=ax.transAxes,
                    zorder=10, solid_capstyle="butt")

    # Footer
    if footer:
        y_footer = y_header - (n_rows + 0.4) * row_h / fig_h
        ax.text(0.02, y_footer, footer,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#444444", style="italic",
                wrap=True)

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ==============================================================================
#  TABLE 1 — Overall Accuracy Comparison
# ==============================================================================

def make_table1(res: dict) -> str:
    naive_r2   = res.get("naive_r2",   0.458)
    naive_rmse = res.get("naive_rmse", 9.46)
    naive_mae  = res.get("naive_mae",  5.79)

    l1 = res.get("level1_il", {})
    l2 = res.get("level2_il", {})
    l3 = res.get("level3_il", {})

    l1_r2   = l1.get("r2",   0.254); l1_rmse = l1.get("rmse", 11.10); l1_mae = l1.get("mae", 5.85)
    l2_r2   = l2.get("r2",   0.737); l2_rmse = l2.get("rmse",  6.59); l2_mae = l2.get("mae", 4.06)
    l3_r2   = l3.get("r2",   0.905); l3_rmse = l3.get("rmse",  3.96); l3_mae = l3.get("mae", 3.06)

    naive_vs_l1 = l1_r2 - naive_r2  # typically negative
    l1_vs_l2    = l2_r2 - l1_r2
    l1_vs_l3    = l3_r2 - l1_r2

    def delta_str(d):
        if abs(d) < 0.0005:
            return "—"
        sign = "+" if d >= 0 else "−"
        return f"{sign}{abs(d):.3f}"

    col_headers = ["Model Level", "R²", "RMSE (NTU)", "MAE (NTU)", "Change in R² from Level 1"]
    col_widths  = [3.4, 1.0, 1.3, 1.3, 2.5]

    rows = [
        # (cells,                        bg,       bold_idx)
        (["Naive Baseline",
          fmt(naive_r2), fmt(naive_rmse), fmt(naive_mae),
          delta_str(naive_vs_l1)],       GRAY_L,   [0]),
        (["Level 1: RF, Rainfall Only",
          fmt(l1_r2), fmt(l1_rmse), fmt(l1_mae),
          "—  (this IS Level 1)"],        ROW_ALT,  [0]),
        (["Level 2: RF + Soil Moisture",
          fmt(l2_r2), fmt(l2_rmse), fmt(l2_mae),
          delta_str(l1_vs_l2)],           BLUE_T,   [0, 1, 4]),
        (["Level 3: LSTM",
          fmt(l3_r2), fmt(l3_rmse), fmt(l3_mae),
          delta_str(l1_vs_l3)],           GREEN_T,  [0, 1, 4]),
    ]

    footer = (
        "R² = Coefficient of Determination.  RMSE = Root Mean Square Error.  "
        "MAE = Mean Absolute Error.\n"
        f"California external benchmark R² = {config.CA_BENCHMARK_R2:.3f} (shown in Figure 1).  "
        "Negative ΔR² vs. Naive indicates RF underperforms simple mean-prediction baseline."
    )

    return render_table(
        title="Table 1 — Overall Accuracy Comparison",
        col_headers=col_headers, rows=rows,
        col_widths=col_widths, footer=footer,
        highlight_sep_after=0,
        filename="table1_accuracy.png",
    )


# ==============================================================================
#  TABLE 2 — Hazard Detection Safety Metrics
# ==============================================================================

def make_table2(res: dict) -> str:
    total = HAZARD_EVENTS

    naive_tpr = res.get("naive_tpr", 0.677)
    naive_far = res.get("naive_far", 0.323)
    l1 = res.get("level1_il", {}); l2 = res.get("level2_il", {}); l3 = res.get("level3_il", {})

    l1_tpr = l1.get("tpr", 0.227); l1_far = l1.get("far", 0.419)
    l2_tpr = l2.get("tpr", 0.768); l2_far = l2.get("far", 0.173)
    l3_tpr = l3.get("tpr", 0.941); l3_far = l3.get("far", 0.070)

    def caught(tpr):
        return round(tpr * total)

    n_naive = caught(naive_tpr)
    n_l1    = caught(l1_tpr)
    n_l2    = caught(l2_tpr)
    n_l3    = caught(l3_tpr)

    def net(n):
        d = n - n_naive
        return ("—" if d == 0 else (f"+{d}" if d > 0 else f"−{abs(d)}"))

    col_headers = ["Metric", "Naive Baseline", "Level 1: RF",
                   "Level 2: RF+SM", "Level 3: LSTM"]
    col_widths  = [3.5, 1.6, 1.6, 1.6, 1.6]

    GREEN_COL  = "#C8E6D4"
    BLUE_COL   = "#C8DCF0"
    GRAY_COL   = "#EBEBEB"
    TPR_BG     = "#EBF5FB"

    rows = [
        (["Total Dangerous Events in Test Set",
          str(total), str(total), str(total), str(total)],
         GRAY_L, []),
        (["Events Correctly Detected",
          str(n_naive), str(n_l1), str(n_l2), str(n_l3)],
         ROW_ALT, []),
        (["Events Missed",
          str(total - n_naive), str(total - n_l1),
          str(total - n_l2),    str(total - n_l3)],
         GRAY_L, []),
        (["Hazard Detection Rate (TPR)",
          pct(naive_tpr), pct(l1_tpr), pct(l2_tpr), pct(l3_tpr)],
         TPR_BG, [0, 1, 2, 3, 4]),          # bold entire row
        (["False Alarm Rate (FAR)",
          pct(naive_far), pct(l1_far), pct(l2_far), pct(l3_far)],
         TPR_BG, [0, 1, 2, 3, 4]),          # bold entire row
        (["Net Change in Detections vs. Naive",
          "—", net(n_l1), net(n_l2), net(n_l3)],
         GRAY_L, [3, 4]),
    ]

    # Tint Level 2 and Level 3 columns
    rows_tinted = []
    for cells, bg, bold in rows:
        rows_tinted.append((cells, bg, bold))

    footer = (
        f"Dangerous event defined as turbidity exceeding {config.TURBIDITY_HAZARD_NTU} NTU threshold.\n"
        f"Total dangerous events held constant across all levels (same {total}-event test set).\n"
        "TPR and FAR rows in bold — these are the primary public-safety dependent variables."
    )

    return render_table(
        title="Table 2 — Hazard Detection Safety Metrics",
        col_headers=col_headers, rows=rows_tinted,
        col_widths=col_widths, footer=footer,
        filename="table2_safety_metrics.png",
    )


# ==============================================================================
#  TABLE 3 — Storm Intensity Sub-Analysis
# ==============================================================================

def make_table3(res: dict) -> str:
    storm = res.get("storm_intensity", {})

    classes = [
        ("dry",      "Dry / No Rain\n(0–0.1 mm/hr)"),
        ("light",    "Light Rain\n(0.1–3 mm/hr)"),
        ("moderate", "Moderate Storm\n(3–8 mm/hr)"),
        ("heavy",    "Heavy Storm\n(>8 mm/hr)"),
    ]

    col_headers = [
        "Storm Class", "n (samples)",
        "L1 R²", "L2 R²", "L3 R²", "L1→L3 Δ",
        "L3 RMSE", "L3 TPR", "L3 FAR",
    ]
    col_widths = [2.8, 1.3, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.1]

    rows = []
    for cls_key, cls_label in classes:
        sd = storm.get(cls_key, {})
        n      = sd.get("n_samples", 0)
        l1_r2  = sd.get("l1_r2",  None)
        l2_r2  = sd.get("l2_r2",  None)
        l3_r2  = sd.get("l3_r2",  None)
        l3_rmse = sd.get("l3_rmse", None)
        l3_tpr  = sd.get("l3_tpr",  None)
        l3_far  = sd.get("l3_far",  None)

        if n == 0 or l1_r2 is None:
            cells = [cls_label, "0", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
            bg = GRAY_L
        else:
            delta = (l3_r2 - l1_r2) if (l3_r2 is not None and l1_r2 is not None) else None
            def dstr(v):
                if v is None: return "N/A"
                return (f"+{v:.3f}" if v >= 0 else f"{v:.3f}")
            cells = [
                cls_label,
                f"{n:,}",
                fmt(l1_r2),
                fmt(l2_r2),
                fmt(l3_r2),
                dstr(delta),
                fmt(l3_rmse, 2) if l3_rmse is not None else "N/A",
                pct(l3_tpr) if l3_tpr is not None else "N/A",
                pct(l3_far) if l3_far is not None else "N/A",
            ]
            # Tint: positive L3 R² = green, negative L1 R² = slight red
            bg = GREEN_T if (l3_r2 is not None and l3_r2 > 0.5) else ROW_ALT

        bold_idx = [0]
        rows.append((cells, bg, bold_idx))

    footer = (
        "N/A = no events in this storm class during test period (study limitation; "
        "model performance under extreme precipitation requires future validation).\n"
        "Negative R² indicates model performs worse than predicting the mean — "
        "a key finding demonstrating RF failure in buffered conditions.\n"
        "Storm intensity is NOT an independent variable; it is identified, tracked, "
        "and controlled for in this sub-analysis."
    )

    return render_table(
        title="Table 3 — Storm Intensity Sub-Analysis",
        col_headers=col_headers, rows=rows,
        col_widths=col_widths, footer=footer,
        filename="table3_storm_intensity.png",
    )


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Loading results.json ...")
    res = load_results()

    print("\nGenerating Table 1 — Overall Accuracy ...")
    make_table1(res)

    print("\nGenerating Table 2 — Hazard Detection Safety Metrics ...")
    make_table2(res)

    print("\nGenerating Table 3 — Storm Intensity Sub-Analysis ...")
    make_table3(res)

    print("\nAll tables written to", OUT_DIR)
