"""Lamp configuration data and keyword resolution utilities."""

import re

LAMP_CONFIGS = {
    "aerolamp": {
        "ies_file": "aerolamp.ies",
        "spectrum_file": "aerolamp.csv",
        "aliases": ["aerolamp devkit", "devkit", "aerolamp devkit v1"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "edge",
        },
        "fixture": {
            "housing_width": 0.1,
            "housing_length": 0.118,
            "housing_height": 0.076,
        },
    },
    "beacon": {
        "ies_file": "beacon.ies",
        "spectrum_file": "beacon.csv",
        "aliases": [],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "edge",
            # "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.12,
            "housing_length": 0.12,
            "housing_height": 0.08,
        },
    },
    "lumenizer_zone": {
        "ies_file": "lumenizer_zone.ies",
        "spectrum_file": "lumenizer_zone.csv",
        "aliases": ["lumenizer"],
        # "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "downlight",
            "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.2,
            "housing_length": 0.2,
            "housing_height": 0.08,
        },
    },
    "nukit_lantern": {
        "ies_file": "nukit_lantern.ies",
        "spectrum_file": "nukit_lantern.csv",
        "aliases": ["lantern"],
        # "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "edge",
        },
        "fixture": {
            "housing_width": 0.086,
            "housing_length": 0.140,
            "housing_height": 0.045,
        },
    },
    "nukit_torch": {
        "ies_file": "nukit_torch.ies",
        "spectrum_file": "nukit_torch.csv",
        "aliases": ["torch"],
        # "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            # "mode": "corner",
            # "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.154,
            "housing_length": 0.042,
            "housing_height": 0.05,
        },
    },
    "sterilray": {
        "ies_file": "sterilray.ies",
        "spectrum_file": "sterilray.csv",
        "aliases": ["sabre","germbuster_sabre","sterilray_germbuster","sterilray_germbuster_sabre"],
        # "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "horizontal",
        },
        "fixture": {
            "housing_width": 0.60,
            "housing_length": 0.10,
            "housing_height": 0.08,
        },
    },
    "ushio_b1": {
        "ies_file": "ushio_b1.ies",
        "spectrum_file": "ushio_b1.csv",
        "aliases": ["ushio_diffused", "b1"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "corner",
            "max_tilt": 45,
        },
        # "fixture": {
            # "housing_width": 0.15,
            # "housing_length": 0.08,
            # "housing_height": 0.06,
        # },
    },
    "ushio_b1.5": {
        "ies_file": "ushio_b1.5.ies",
        "spectrum_file": "ushio_b1.5.csv",
        "aliases": ["ushio_diffused" "b1.5","b15"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "corner",
            "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.18,
            "housing_length": 0.10,
            "housing_height": 0.07,
        },
    },
    "uvpro222_b1": {
        "ies_file": "uvpro222_b1.ies",
        "spectrum_file": "uvpro222_b1.csv",
        "aliases": ["uvpro_b1", "bioabundance_b1"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "corner",
            # "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.15,
            "housing_length": 0.08,
            "housing_height": 0.06,
        },
    },
    "uvpro222_b2": {
        "ies_file": "uvpro222_b2.ies",
        "spectrum_file": "uvpro222_b2.csv",
        "aliases": ["uvpro_b2", "bioabundance_b2"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "edge",
            # "max_tilt": 45,
        },
        "fixture": {
            "housing_width": 0.20,
            "housing_length": 0.12,
            "housing_height": 0.08,
        },
    },
    "visium": {
        "ies_file": "visium.ies",
        "spectrum_file": "visium.csv",
        "aliases": ["visium_diffused"],
        "guv_type": "krcl",
        "peak_wavelength": 222,
        "placement": {
            "mode": "downlight",
        },
        "fixture": {
            "housing_width": 0.12,
            "housing_length": 0.12,
            "housing_height": 0.06,
        },
    },
}


def _normalize(token: str) -> str:
    """Normalize keyword: lowercase, collapse whitespace/underscores/dashes to underscore."""
    return re.sub(r"[\s_\-]+", "_", str(token).strip().lower())


def resolve_keyword(token: str) -> tuple[str, dict]:
    """
    Return (canonical_key, config) for a keyword or alias.

    Supports flexible matching:
    - Case insensitive: "Ushio_B1" -> "ushio_b1"
    - Whitespace/underscores/dashes normalized: "ushio b1" -> "ushio_b1"
    - Alias lookup: "ushio" -> "ushio_b1"

    Raises:
        KeyError: If token doesn't match any key or alias
    """
    normalized = _normalize(token)

    # Direct key match (keys are already normalized in LAMP_CONFIGS)
    if normalized in LAMP_CONFIGS:
        return normalized, LAMP_CONFIGS[normalized]

    # Alias lookup - normalize aliases too for flexible matching
    for key, config in LAMP_CONFIGS.items():
        normalized_aliases = [_normalize(a) for a in config.get("aliases", [])]
        if normalized in normalized_aliases:
            return key, config

    raise KeyError(f"{token!r} not found. Valid keys: {list(LAMP_CONFIGS.keys())}")


def get_valid_keys() -> list[str]:
    """Return list of valid canonical lamp keys."""
    return list(LAMP_CONFIGS.keys())


def get_config(key: str) -> dict:
    """Get config dict for a lamp key or alias."""
    _, config = resolve_keyword(key)
    return config
