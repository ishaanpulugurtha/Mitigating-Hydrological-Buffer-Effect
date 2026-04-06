"""
config.py — Central configuration for the Hydrological Buffer Effect experiment.

Mitigating the Hydrological Buffer Effect: Enhancing Early Warning of
Storm-Driven Water Quality Risks in Buffered Watersheds Using LSTM Architectures

Modify SITES to swap in different USGS monitoring stations.
To find sites with turbidity data, visit:
  https://waterdata.usgs.gov/nwis/rt  (select parameter 63680 = Turbidity)
"""

# -- Date Range ----------------------------------------------------------------
START_DATE = "2012-01-01"   # 12 years → ~105,000 hourly data points per region
END_DATE   = "2023-12-31"

# -- USGS Monitoring Sites -----------------------------------------------------
#
#  Each entry requires:
#    usgs_site_id  -> 8-digit USGS station number
#    scan_site_id  -> USDA SCAN site number (for soil moisture)
#    lat / lon     -> station coordinates (decimal degrees)
#
SITES = {
    "california": {
        "name":           "Cache Creek near Yolo, CA",
        "usgs_site_id":   "11452500",
        "state":          "CA",
        "watershed_type": "Flashy (Rocky/Steep)",
        "lat":            38.729,
        "lon":           -121.982,
        # SCAN Zamora, CA — closest to Cache Creek with soil moisture
        "scan_site_id":   "2099",
        "description": (
            "Small Coast Range watershed with thin soils; precipitation produces "
            "rapid surface runoff with 1–4 hr turbidity response."
        ),
    },
    "illinois": {
        "name":           "Mackinaw River near Congerville, IL",
        "usgs_site_id":   "05568000",
        "state":          "IL",
        "watershed_type": "Buffered (Flat/Deep Soil)",
        "lat":            40.622,
        "lon":           -89.198,
        # SCAN Bondville, IL — long-running soil moisture record
        "scan_site_id":   "2012",
        "description": (
            "Agricultural flatland watershed with deep glacial-till soils; "
            "the 'sponge' delays turbidity response by 10–18 hours."
        ),
    },

    # Second Illinois station — multi-station validation
    # Fox River at Algonquin, IL: northern IL glacial terrain; validates
    # that the Hydrological Buffer Effect pattern generalizes across the state.
    "illinois2": {
        "name":           "Fox River at Algonquin, IL",
        "usgs_site_id":   "05550000",
        "state":          "IL",
        "watershed_type": "Buffered (Flat/Deep Soil)",
        "lat":            42.165,
        "lon":           -88.321,
        # SCAN DeKalb, IL — northern IL soil moisture station
        "scan_site_id":   "2062",
        "description": (
            "Northern Illinois glacially-buffered watershed in the Fox River basin; "
            "McHenry/Kane county area. Used for multi-station validation to confirm "
            "the Hydrological Buffer Effect generalizes across Illinois buffered watersheds."
        ),
    },
}

# -- USGS Parameter Codes ------------------------------------------------------
PARAMS = {
    "turbidity":     "63680",   # Turbidity, water, unfiltered (FNU ≈ NTU)
    "precipitation": "00045",   # Precipitation, total, inches
    "discharge":     "00060",   # Streamflow, cubic feet per second
}

# -- Hydrological Thresholds ---------------------------------------------------
TURBIDITY_HAZARD_NTU = 25       # Turbidity threshold that triggers a "hazard alert"

STORM_CLASSES = {
    "dry":      (0.0,  0.1),    # mm/hr  ->  no measurable rainfall
    "light":    (0.1,  3.0),    # mm/hr  ->  drizzle / light rain
    "moderate": (3.0,  8.0),    # mm/hr  ->  moderate rain
    "heavy":    (8.0,  999.0),  # mm/hr  ->  heavy / intense storm (lowered threshold for real data)
}

# -- California External Benchmark (from original RF study, current-time prediction) --------
# Used in figures/tables to show what model performance looks like without the Buffer Effect.
CA_BENCHMARK_R2  = 0.843   # Non-buffered reference R² (prior validated study, no forecast lag)
CA_BENCHMARK_TPR = 0.255   # Non-buffered reference hazard detection rate (our CA run)

# -- LSTM Architecture ---------------------------------------------------------
WINDOW_SIZE      = 36    # Hours of history fed into LSTM (1.5-day look-back)
FORECAST_HORIZON = 6     # Hours ahead all models predict (6-hr early-warning window, constant)
MAX_LAG_HOURS    = 48    # Maximum lag tested in the Cross-Correlation analysis
LSTM_EPOCHS      = 50
LSTM_BATCH       = 512   # Large batches → fewer steps/epoch → much faster on CPU with 100k+ rows
LSTM_PATIENCE    = 10    # Early-stopping patience

# -- Reproducibility -----------------------------------------------------------
RANDOM_SEED = 42

# -- I/O Paths -----------------------------------------------------------------
DATA_DIR    = "data"
RAW_DIR     = "data/raw"
PROC_DIR    = "data/processed"
OUTPUT_DIR  = "outputs"
