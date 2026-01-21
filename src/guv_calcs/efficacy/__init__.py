"""Efficacy calculation module for UV disinfection."""

from .data import Data
from .math import (
    eACH_UV,
    log1,
    log2,
    log3,
    log4,
    log5,
    seconds_to_S,
    CADR_CFM,
    CADR_LPS,
)


def get_disinfection_table(fluence=None):
    """Return disinfection table with optional fluence-based computed columns."""
    return Data(fluence=fluence).df


__all__ = [
    "Data",
    "get_disinfection_table",
    "eACH_UV",
    "log1",
    "log2",
    "log3",
    "log4",
    "log5",
    "seconds_to_S",
    "CADR_CFM",
    "CADR_LPS",
]
