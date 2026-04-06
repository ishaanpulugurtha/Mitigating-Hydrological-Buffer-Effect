# Mitigating the Hydrological Buffer Effect
### Enhancing Early Warning of Storm-Driven Water Quality Risks in Buffered Watersheds Using LSTM Architectures

> **IJAS State Exposition — Science Fair Project**  
> Investigates whether adding soil moisture awareness and transitioning to an LSTM architecture can overcome the hydrological buffering limitations that prevent accurate storm-driven turbidity prediction in Illinois, using California as an external performance benchmark.

---

## Experimental Design

**One Independent Variable: Model Type — tested at three levels**

| Level | Model | One Change from Previous |
|-------|-------|--------------------------|
| Level 1 | Random Forest — Rainfall Only | Baseline. Establishes the predictive gap. |
| Level 2 | Random Forest + Soil Moisture | Adds Antecedent Moisture Conditions (AMC). |
| Level 3 | LSTM (Long Short-Term Memory) | Replaces RF architecture. Same features as L2. |

**Control:** Naive baseline (future turbidity = current turbidity)  
**Comparison Group:** California (Mediterranean climate — no Hydrological Buffer Effect)  
**Background Evidence:** CCF Lag Analysis — quantifies the Sponge Effect *before* the experiment begins

> California is **not** an independent variable. It is an external benchmark. The gap between CA and IL performance at Level 1 IS the quantified problem this experiment solves.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full experiment

```bash
# Use real USGS + NASA data (recommended)
python run_experiment.py

# Force synthetic data (instant, no internet — great for testing)
python run_experiment.py --synthetic
```

All five figures are saved to `outputs/` and all numeric results to `outputs/results.json`.

---

## Output Files (5 Figures + Results)

| File | Description |
|------|-------------|
| `outputs/figure1_progression.png` | **Hero graphic** — Illinois R² progression: RF → RF+SM → LSTM |
| `outputs/figure2_ccf.png` | CCF side-by-side (CA vs IL) — the visual Sponge Effect |
| `outputs/figure3_timeseries.png` | 3-panel actual vs. predicted for IL at all three levels |
| `outputs/figure4_storm_intensity.png` | R² by storm class for all 3 IL levels (robustness) |
| `outputs/figure5_safety.png` | Hazard detection rate: Naive / L1 / L2 / L3 |
| `outputs/results.json` | All numeric metrics, lag values, and storm sub-analysis |

---

## Key Scientific Concepts

### The Hydrological Buffer Effect ("Sponge Effect")
Deep glacial-till soils in Illinois absorb rainfall before any reaches streams. The soil must approach saturation (high Antecedent Moisture Conditions) before runoff begins. This creates a non-linear, variable time delay between a storm event and the resulting turbidity spike — a delay a standard model cannot account for without being told the soil's current state.

### Why Standard AI Fails in Illinois
A Random Forest has **no memory**. It treats each prediction independently and cannot know that the soil is already saturated from a storm 15 hours ago. This is why R² collapses in Illinois.

### The Two Targeted Fixes

| Fix | Level | Mechanism |
|-----|-------|-----------|
| **Feature fix** | Level 2 | Add `soil_moisture` (AMC) so the RF knows the "Sponge Status" before a storm |
| **Architecture fix** | Level 3 | LSTM retains a 36-hour memory window, learning the variable lag naturally |

### Why This is a Controlled Experiment
Between each level, **exactly one thing changes**. This enables causal inference about what drives predictive failure in buffered watersheds.

---

## Constants Across All Levels

- USGS monitoring data source (same station, same time period)
- Illinois watershed region only (CA = benchmark)
- 80/20 train/test split
- Hazard threshold: Turbidity > 25 NTU
- **6-hour forecasting window** (all models predict turbidity 6 hours ahead)
- Python / TensorFlow (Keras)

---

## Data Sources

| Variable | Source | Fetch method |
|----------|--------|-------------|
| Turbidity (NTU) | USGS NWIS, param 63680 | `dataretrieval` Python package |
| Precipitation (mm/hr) | USGS NWIS, param 00045 → meteostat fallback | automatic |
| Soil Moisture (m³/m³) | NASA SMAP / USDA SCAN network → computed fallback | automatic |

### USGS Station IDs

| Region | Site | USGS ID |
|--------|------|---------|
| California (Benchmark — Flashy) | Cache Creek near Yolo, CA | 11452500 |
| Illinois (Test — Buffered) | Mackinaw River near Congerville, IL | 05568000 |

---

## Configuration

Open `config.py` to adjust:

```python
START_DATE = "2019-01-01"       # date range for all data
END_DATE   = "2023-12-31"

TURBIDITY_HAZARD_NTU = 25       # "hazardous event" threshold (NTU)
WINDOW_SIZE          = 36       # LSTM look-back window (hours)
FORECAST_HORIZON     = 6        # hours ahead all models predict (constant)
MAX_LAG_HOURS        = 48       # max lag tested in CCF analysis
```

---

## Project Structure

```
sjwp9th/
├── config.py               ← Central configuration
├── requirements.txt
├── run_experiment.py       ← Master script — runs background evidence + 3 levels
├── data/
│   ├── raw/                ← Downloaded CSV cache (auto-created)
│   └── processed/
├── outputs/                ← Five figures + results.json (auto-created)
└── src/
    ├── data_pipeline.py    ← USGS + NASA/SCAN fetch + synthetic fallback
    ├── preprocessing.py    ← Feature engineering (prepare_rf_level1/2, LSTM sequences)
    ├── models.py           ← RandomForestModel, LSTMModel, compute_metrics()
    ├── analysis.py         ← CCF, storm classification, storm_intensity_metrics()
    └── visualizations.py   ← Five mandated IJAS figures
```
