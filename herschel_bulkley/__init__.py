"""
Herschel-Bulkley Rheology Library

A comprehensive library for Herschel-Bulkley fluid modeling including:
- Constitutive and inverse equations
- Parameter fitting from rheometer data
- Elliptical duct flow solver
- Validation against analytical limits
- CLI and plotting capabilities
"""

from .core import HerschelBulkley
from .fitting import ParameterFitter
from .flow import EllipticalDuctFlow
from .validation import Validator
from .plotting import RheologyPlotter

__version__ = "1.0.0"
__author__ = "Rheology Library"

__all__ = [
    "HerschelBulkley",
    "ParameterFitter", 
    "EllipticalDuctFlow",
    "Validator",
    "RheologyPlotter"
]
