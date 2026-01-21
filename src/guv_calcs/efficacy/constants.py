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
