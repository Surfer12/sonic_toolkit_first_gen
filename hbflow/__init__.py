from .models import (
    hb_tau_from_gamma,
    hb_gamma_from_tau,
    hb_apparent_viscosity,
    hb_regularized_tau,
    hb_regularized_apparent_viscosity,
)
from .fit import HBParams, fit_herschel_bulkley
from .duct import solve_elliptical_hb
from .advanced_duct import solve_elliptical_hb_pde, EllipticalHBDuctSolver
from .validation import validate_limits, benchmark_against_analytical
from .rheometry import simulate_rheometer_data, add_noise_to_data

__all__ = [
    "hb_tau_from_gamma",
    "hb_gamma_from_tau",
    "hb_apparent_viscosity",
    "hb_regularized_tau",
    "hb_regularized_apparent_viscosity",
    "HBParams",
    "fit_herschel_bulkley",
    "solve_elliptical_hb",
    "solve_elliptical_hb_pde",
    "EllipticalHBDuctSolver",
    "validate_limits",
    "benchmark_against_analytical",
    "simulate_rheometer_data",
    "add_noise_to_data",
]