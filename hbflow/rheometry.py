"""
Rheometry module for Herschel-Bulkley fluid characterization.

This module provides tools for:
- Simulating rheometer measurements
- Adding realistic noise to data
- Generating test datasets
- Analyzing measurement quality
"""

from __future__ import annotations

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from numpy.typing import ArrayLike

from .models import hb_tau_from_gamma


def simulate_rheometer_data(
    tau_y: float,
    K: float,
    n: float,
    gamma_dot_min: float = 0.1,
    gamma_dot_max: float = 1000.0,
    num_points: int = 50,
    measurement_type: str = 'controlled_shear_rate',
    ramp_time: Optional[float] = None
) -> dict:
    """
    Simulate rheometer measurements for HB fluid.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    gamma_dot_min : float
        Minimum shear rate [1/s]
    gamma_dot_max : float
        Maximum shear rate [1/s]
    num_points : int
        Number of measurement points
    measurement_type : str
        'controlled_shear_rate' or 'controlled_stress'
    ramp_time : float, optional
        Time for stress ramp in controlled stress mode [s]

    Returns
    -------
    dict with simulated measurement data
    """
    if measurement_type == 'controlled_shear_rate':
        # Controlled shear rate experiment (most common)
        gamma_dot = np.logspace(np.log10(gamma_dot_min), np.log10(gamma_dot_max), num_points)
        tau = hb_tau_from_gamma(gamma_dot, tau_y, K, n)

        # Add time information
        time = np.linspace(0, 10.0, num_points)  # 10 seconds total

        result = {
            'measurement_type': 'controlled_shear_rate',
            'gamma_dot': gamma_dot,
            'tau': tau,
            'time': time,
            'parameters': {
                'tau_y': tau_y,
                'K': K,
                'n': n
            }
        }

    elif measurement_type == 'controlled_stress':
        # Controlled stress experiment
        tau_min = tau_y + K * (gamma_dot_min ** n)
        tau_max = tau_y + K * (gamma_dot_max ** n)
        tau = np.linspace(tau_min, tau_max, num_points)

        # Calculate corresponding shear rates
        gamma_dot = np.zeros(num_points)
        for i, tau_val in enumerate(tau):
            if tau_val <= tau_y:
                gamma_dot[i] = 0.0
            else:
                gamma_dot[i] = ((tau_val - tau_y) / K) ** (1.0 / n)

        # Time with stress ramp
        if ramp_time is None:
            ramp_time = 60.0  # 1 minute
        time = np.linspace(0, ramp_time, num_points)

        result = {
            'measurement_type': 'controlled_stress',
            'gamma_dot': gamma_dot,
            'tau': tau,
            'time': time,
            'parameters': {
                'tau_y': tau_y,
                'K': K,
                'n': n
            }
        }

    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")

    return result


def add_noise_to_data(
    gamma_dot: ArrayLike,
    tau: ArrayLike,
    noise_type: str = 'proportional',
    noise_level: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic noise to rheometer data.

    Parameters
    ----------
    gamma_dot : array-like
        Shear rates [1/s]
    tau : array-like
        Shear stresses [Pa]
    noise_type : str
        'proportional', 'absolute', or 'combined'
    noise_level : float
        Noise level (fraction for proportional, Pa for absolute)
    random_state : int, optional
        Random number generator seed

    Returns
    -------
    tuple of (gamma_dot_noisy, tau_noisy)
    """
    rng = np.random.default_rng(random_state)

    gamma_dot = np.asarray(gamma_dot)
    tau = np.asarray(tau)

    gamma_dot_noisy = gamma_dot.copy()
    tau_noisy = tau.copy()

    if noise_type == 'proportional':
        # Proportional noise (percentage of signal)
        tau_noise = rng.normal(0, noise_level * tau, size=len(tau))
        gamma_dot_noise = rng.normal(0, noise_level * gamma_dot, size=len(gamma_dot))

        tau_noisy += tau_noise
        gamma_dot_noisy += gamma_dot_noise

    elif noise_type == 'absolute':
        # Absolute noise (fixed amount)
        tau_noise = rng.normal(0, noise_level, size=len(tau))
        gamma_dot_noise = rng.normal(0, noise_level * 0.1, size=len(gamma_dot))

        tau_noisy += tau_noise
        gamma_dot_noisy += gamma_dot_noise

    elif noise_type == 'combined':
        # Both proportional and absolute noise
        tau_proportional = rng.normal(0, 0.03 * tau, size=len(tau))
        tau_absolute = rng.normal(0, noise_level, size=len(tau))

        gamma_dot_proportional = rng.normal(0, 0.01 * gamma_dot, size=len(gamma_dot))
        gamma_dot_absolute = rng.normal(0, noise_level * 0.1, size=len(gamma_dot))

        tau_noisy += tau_proportional + tau_absolute
        gamma_dot_noisy += gamma_dot_proportional + gamma_dot_absolute

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Ensure non-negative values
    tau_noisy = np.maximum(tau_noisy, 0.0)
    gamma_dot_noisy = np.maximum(gamma_dot_noisy, 0.0)

    return gamma_dot_noisy, tau_noisy


def generate_synthetic_dataset(
    fluid_types: List[Dict[str, float]],
    samples_per_fluid: int = 50,
    noise_type: str = 'combined',
    noise_level: float = 0.05,
    random_state: Optional[int] = None
) -> dict:
    """
    Generate a comprehensive synthetic rheometer dataset.

    Parameters
    ----------
    fluid_types : list of dict
        List of fluid parameter dictionaries with keys 'tau_y', 'K', 'n', 'name'
    samples_per_fluid : int
        Number of samples per fluid type
    noise_type : str
        Type of noise to add
    noise_level : float
        Noise level
    random_state : int, optional
        Random seed

    Returns
    -------
    dict with dataset and metadata
    """
    rng = np.random.default_rng(random_state)

    all_gamma_dot = []
    all_tau = []
    labels = []
    metadata = []

    for fluid in fluid_types:
        # Simulate measurements
        data = simulate_rheometer_data(
            fluid['tau_y'], fluid['K'], fluid['n'],
            num_points=samples_per_fluid,
            measurement_type='controlled_shear_rate'
        )

        # Add noise
        gamma_dot_noisy, tau_noisy = add_noise_to_data(
            data['gamma_dot'], data['tau'],
            noise_type=noise_type,
            noise_level=noise_level,
            random_state=random_state
        )

        all_gamma_dot.extend(gamma_dot_noisy)
        all_tau.extend(tau_noisy)
        labels.extend([fluid['name']] * samples_per_fluid)

        metadata.extend([{
            'true_parameters': {
                'tau_y': fluid['tau_y'],
                'K': fluid['K'],
                'n': fluid['n']
            },
            'fluid_name': fluid['name'],
            'measurement_type': 'controlled_shear_rate'
        }] * samples_per_fluid)

    return {
        'gamma_dot': np.array(all_gamma_dot),
        'tau': np.array(all_tau),
        'labels': labels,
        'metadata': metadata,
        'noise_type': noise_type,
        'noise_level': noise_level
    }


def analyze_measurement_quality(
    gamma_dot: ArrayLike,
    tau: ArrayLike,
    time: Optional[ArrayLike] = None,
    window_size: int = 10
) -> dict:
    """
    Analyze the quality of rheometer measurements.

    Parameters
    ----------
    gamma_dot : array-like
        Shear rates [1/s]
    tau : array-like
        Shear stresses [Pa]
    time : array-like, optional
        Time stamps [s]
    window_size : int
        Window size for local analysis

    Returns
    -------
    dict with quality metrics
    """
    gamma_dot = np.asarray(gamma_dot)
    tau = np.asarray(tau)

    results = {
        'basic_stats': {
            'n_points': len(gamma_dot),
            'gamma_dot_range': [float(np.min(gamma_dot)), float(np.max(gamma_dot))],
            'tau_range': [float(np.min(tau)), float(np.max(tau))],
            'gamma_dot_span': float(np.max(gamma_dot) / np.min(gamma_dot)),
            'tau_span': float(np.max(tau) / np.min(tau))
        },
        'data_quality': {},
        'rheological_assessment': {}
    }

    # Check for data quality issues
    quality_issues = []

    # Check for negative values
    if np.any(gamma_dot < 0):
        quality_issues.append('negative_shear_rates')
    if np.any(tau < 0):
        quality_issues.append('negative_stresses')

    # Check for zeros
    if np.any(gamma_dot == 0):
        quality_issues.append('zero_shear_rates')
    if np.any(tau == 0):
        quality_issues.append('zero_stresses')

    # Check for monotonicity
    if not np.all(np.diff(gamma_dot) >= 0):
        quality_issues.append('non_monotonic_shear_rate')
    if not np.all(np.diff(tau) >= 0):
        quality_issues.append('non_monotonic_stress')

    # Check for outliers using IQR method
    tau_q1, tau_q3 = np.percentile(tau, [25, 75])
    tau_iqr = tau_q3 - tau_q1
    tau_outliers = np.sum((tau < tau_q1 - 1.5 * tau_iqr) | (tau > tau_q3 + 1.5 * tau_iqr))

    gamma_dot_q1, gamma_dot_q3 = np.percentile(gamma_dot, [25, 75])
    gamma_dot_iqr = gamma_dot_q3 - gamma_dot_q1
    gamma_dot_outliers = np.sum((gamma_dot < gamma_dot_q1 - 1.5 * gamma_dot_iqr) |
                               (gamma_dot > gamma_dot_q3 + 1.5 * gamma_dot_iqr))

    results['data_quality'] = {
        'quality_issues': quality_issues,
        'tau_outliers': int(tau_outliers),
        'gamma_dot_outliers': int(gamma_dot_outliers),
        'data_completeness': 1.0 if len(quality_issues) == 0 else 0.5 if len(quality_issues) <= 2 else 0.0
    }

    # Rheological assessment
    assessment = {}

    # Estimate flow behavior
    if len(gamma_dot) > 10:
        # Calculate apparent viscosity
        eta_app = tau / gamma_dot

        # Check for yield stress (plateau at low shear rates)
        low_shear_idx = gamma_dot < np.percentile(gamma_dot, 30)
        if np.any(low_shear_idx):
            eta_low = np.mean(eta_app[low_shear_idx])
            eta_high = np.mean(eta_app[~low_shear_idx])
            viscosity_ratio = eta_low / eta_high if eta_high > 0 else 1.0

            assessment['yield_stress_indicator'] = viscosity_ratio > 10
            assessment['viscosity_ratio'] = float(viscosity_ratio)

        # Estimate flow index
        log_gamma = np.log(gamma_dot)
        log_tau = np.log(tau)

        # Linear regression on log-log plot
        A = np.vstack([np.ones_like(log_gamma), log_gamma]).T
        coeffs = np.linalg.lstsq(A, log_tau, rcond=None)[0]
        estimated_n = coeffs[1]

        assessment['estimated_flow_index'] = float(estimated_n)
        assessment['shear_thinning'] = estimated_n < 1.0
        assessment['newtonian_like'] = abs(estimated_n - 1.0) < 0.1

    results['rheological_assessment'] = assessment

    # Time-based analysis if time data available
    if time is not None:
        time = np.asarray(time)
        results['temporal_analysis'] = {
            'total_time': float(np.max(time) - np.min(time)),
            'measurement_frequency': len(time) / (np.max(time) - np.min(time)),
            'time_uniformity': float(1.0 / (1.0 + np.std(np.diff(time))))
        }

    return results


def simulate_thixotropic_effects(
    tau_y: float,
    K: float,
    n: float,
    gamma_dot: ArrayLike,
    time: ArrayLike,
    thixotropic_rate: float = 0.1,
    recovery_rate: float = 0.05
) -> np.ndarray:
    """
    Simulate thixotropic effects in HB fluids.

    Parameters
    ----------
    tau_y : float
        Base yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    gamma_dot : array-like
        Shear rate history [1/s]
    time : array-like
        Time points [s]
    thixotropic_rate : float
        Rate of structure breakdown under shear [1/s]
    recovery_rate : float
        Rate of structure recovery at rest [1/s]

    Returns
    -------
    tau_history : np.ndarray
        Time-dependent shear stress history [Pa]
    """
    gamma_dot = np.asarray(gamma_dot)
    time = np.asarray(time)

    if len(gamma_dot) != len(time):
        raise ValueError("gamma_dot and time must have same length")

    tau_history = np.zeros_like(gamma_dot)
    structure_parameter = 1.0  # 1.0 = fully structured, 0.0 = fully broken down

    for i in range(len(gamma_dot)):
        # Structure evolution
        if i > 0:
            dt = time[i] - time[i-1]

            if gamma_dot[i] > 0:
                # Breakdown under shear
                structure_parameter -= thixotropic_rate * gamma_dot[i] * dt
            else:
                # Recovery at rest
                structure_parameter += recovery_rate * dt

            structure_parameter = np.clip(structure_parameter, 0.1, 1.0)

        # Current stress with structure effect
        tau_current = hb_tau_from_gamma(gamma_dot[i], tau_y, K, n)
        tau_history[i] = tau_current * structure_parameter

    return tau_history


def simulate_viscoelastic_effects(
    tau_y: float,
    K: float,
    n: float,
    gamma_dot: ArrayLike,
    time: ArrayLike,
    G_prime: float = 1000.0,
    G_double_prime: float = 100.0,
    relaxation_time: float = 1.0
) -> np.ndarray:
    """
    Simulate viscoelastic effects in HB fluids using Jeffreys model.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    gamma_dot : array-like
        Shear rate history [1/s]
    time : array-like
        Time points [s]
    G_prime : float
        Elastic modulus [Pa]
    G_double_prime : float
        Viscous modulus [Pa]
    relaxation_time : float
        Relaxation time [s]

    Returns
    -------
    tau_history : np.ndarray
        Time-dependent shear stress with viscoelastic effects [Pa]
    """
    gamma_dot = np.asarray(gamma_dot)
    time = np.asarray(time)

    if len(gamma_dot) != len(time):
        raise ValueError("gamma_dot and time must have same length")

    tau_history = np.zeros_like(gamma_dot)
    gamma_history = np.zeros_like(gamma_dot)  # Strain history

    # Initial conditions
    tau_history[0] = hb_tau_from_gamma(gamma_dot[0], tau_y, K, n)
    gamma_history[0] = gamma_dot[0] * time[0] if time[0] > 0 else 0

    for i in range(1, len(gamma_dot)):
        dt = time[i] - time[i-1]

        # Strain increment
        dgamma = gamma_dot[i] * dt
        gamma_history[i] = gamma_history[i-1] + dgamma

        # Viscoelastic stress calculation (simplified Jeffreys model)
        # τ = τ_viscous + τ_elastic * exp(-t/λ)
        tau_viscous = hb_tau_from_gamma(gamma_dot[i], tau_y, K, n)
        tau_elastic = G_prime * gamma_history[i] * np.exp(-time[i] / relaxation_time)

        tau_history[i] = tau_viscous + tau_elastic

    return tau_history
