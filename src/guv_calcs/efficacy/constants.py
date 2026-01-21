"""Constants for the efficacy module."""

# Log reduction labels
LOG_LABELS = {1: "90%", 2: "99%", 3: "99.9%", 4: "99.99%", 5: "99.999%"}

# Ordering for display/plots
CATEGORY_ORDER = ["Bacteria", "Viruses", "Bacterial spores", "Fungi", "Protists"]
MEDIUM_ORDER = ["Aerosol", "Surface", "Liquid"]

# Column names (from disinfection CSV)
COL_CATEGORY = "Category"
COL_SPECIES = "Species"
COL_STRAIN = "Strain"
COL_WAVELENGTH = "wavelength [nm]"
COL_K1 = "k1 [cm2/mJ]"
COL_K2 = "k2 [cm2/mJ]"
COL_RESISTANT = "% resistant"
COL_MEDIUM = "Medium"
COL_CONDITION = "Condition"
COL_REFERENCE = "Reference"
COL_LINK = "Link"

# Computed column names
COL_EACH = "eACH-UV"
COL_CADR_LPS = "CADR-UV [lps]"
COL_CADR_CFM = "CADR-UV [cfm]"

# Base columns for display (always shown in this order)
BASE_DISPLAY_COLS = [
    COL_CATEGORY,
    COL_SPECIES,
    COL_STRAIN,
    COL_WAVELENGTH,
    COL_K1,
    COL_K2,
    COL_RESISTANT,
    COL_MEDIUM,
    COL_CONDITION,
    COL_REFERENCE,
    COL_LINK,
]

# Axis input aliases for forgiving parsing
AXIS_ALIASES = {
    "each": COL_EACH,
    "each-uv": COL_EACH,
    "eachuv": COL_EACH,
    "k1": COL_K1,
    "k2": COL_K2,
    # CADR handled specially (depends on use_metric)
}

# Log level aliases (map to log number)
LOG_ALIASES = {
    "log1": 1, "90": 1, "0.9": 1,
    "log2": 2, "99": 2, "0.99": 2,
    "log3": 3, "99.9": 3, "0.999": 3,
    "log4": 4, "99.99": 4, "0.9999": 4,
    "log5": 5, "99.999": 5, "0.99999": 5,
}

# Time unit aliases
TIME_UNIT_ALIASES = {
    "seconds": "seconds", "sec": "seconds", "s": "seconds",
    "minutes": "minutes", "min": "minutes", "m": "minutes",
    "hours": "hours", "hr": "hours", "h": "hours",
}

# Compatible column groups (columns that can be co-plotted)
RATE_COLS = {COL_EACH, COL_CADR_LPS, COL_CADR_CFM}
SUSCEPTIBILITY_COLS = {COL_K1, COL_K2}
