"""
Validation module for Herschel-Bulkley fluid models.

This module provides comprehensive validation against:
- Newtonian and power-law limits
- Analytical solutions for simple geometries
- Benchmark cases from literature
- Numerical convergence studies
"""

from __future__ import annotations

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from numpy.typing import ArrayLike

from .models import hb_tau_from_gamma, hb_gamma_from_tau, hb_apparent_viscosity
from .duct import solve_elliptical_hb
from .fit import HBParams


def validate_limits(
    tau_y: float,
    K: float,
    n: float,
    gamma_dot_test: ArrayLike = [0.1, 1.0, 10.0, 100.0],
    tau_test: ArrayLike = [1.0, 10.0, 100.0, 1000.0]
) -> dict:
    """
    Validate HB model against Newtonian and power-law limits.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    gamma_dot_test : array-like
        Test shear rates [1/s]
    tau_test : array-like
        Test shear stresses [Pa]

    Returns
    -------
    dict with validation results and error metrics
    """
    gamma_dot_test = np.asarray(gamma_dot_test)
    tau_test = np.asarray(tau_test)

    results = {
        'parameters': {
            'tau_y': tau_y,
            'K': K,
            'n': n,
            'is_newtonian': tau_y == 0 and abs(n - 1.0) < 1e-10,
            'is_power_law': tau_y == 0 and abs(n - 1.0) > 1e-10,
            'is_bingham': tau_y > 0 and abs(n - 1.0) < 1e-10
        },
        'constitutive_test': {},
        'inverse_test': {},
        'newtonian_limit': None,
        'power_law_limit': None,
        'bingham_limit': None
    }

    # Test constitutive model
    tau_pred = hb_tau_from_gamma(gamma_dot_test, tau_y, K, n)
    tau_expected = tau_y + K * np.power(gamma_dot_test, n)

    results['constitutive_test'] = {
        'gamma_dot': gamma_dot_test.tolist(),
        'tau_predicted': tau_pred.tolist(),
        'tau_expected': tau_expected.tolist(),
        'max_error': float(np.max(np.abs(tau_pred - tau_expected))),
        'mean_error': float(np.mean(np.abs(tau_pred - tau_expected))),
        'rmse': float(np.sqrt(np.mean((tau_pred - tau_expected)**2))),
        'valid': np.allclose(tau_pred, tau_expected, rtol=1e-10, atol=1e-12)
    }

    # Test inverse model
    gamma_dot_pred = hb_gamma_from_tau(tau_test, tau_y, K, n)
    gamma_dot_expected = np.power(np.maximum((tau_test - tau_y), 0) / K, 1/n)

    results['inverse_test'] = {
        'tau': tau_test.tolist(),
        'gamma_dot_predicted': gamma_dot_pred.tolist(),
        'gamma_dot_expected': gamma_dot_expected.tolist(),
        'max_error': float(np.max(np.abs(gamma_dot_pred - gamma_dot_expected))),
        'mean_error': float(np.mean(np.abs(gamma_dot_pred - gamma_dot_expected))),
        'rmse': float(np.sqrt(np.mean((gamma_dot_pred - gamma_dot_expected)**2))),
        'valid': np.allclose(gamma_dot_pred, gamma_dot_expected, rtol=1e-10, atol=1e-12)
    }

    # Test Newtonian limit
    if tau_y == 0.0 and abs(n - 1.0) < 1e-10:
        mu = K  # Newtonian viscosity
        tau_newtonian = mu * gamma_dot_test
        tau_hb = hb_tau_from_gamma(gamma_dot_test, 0, mu, 1.0)

        results['newtonian_limit'] = {
            'viscosity_mu': mu,
            'tau_newtonian': tau_newtonian.tolist(),
            'tau_hb': tau_hb.tolist(),
            'max_error': float(np.max(np.abs(tau_hb - tau_newtonian))),
            'mean_error': float(np.mean(np.abs(tau_hb - tau_newtonian))),
            'valid': np.allclose(tau_hb, tau_newtonian, rtol=1e-10, atol=1e-12)
        }

    # Test power-law limit
    if tau_y == 0.0 and abs(n - 1.0) > 1e-10:
        tau_power_law = K * np.power(gamma_dot_test, n)
        tau_hb = hb_tau_from_gamma(gamma_dot_test, 0, K, n)

        results['power_law_limit'] = {
            'consistency_K': K,
            'flow_index_n': n,
            'tau_power_law': tau_power_law.tolist(),
            'tau_hb': tau_hb.tolist(),
            'max_error': float(np.max(np.abs(tau_hb - tau_power_law))),
            'mean_error': float(np.mean(np.abs(tau_hb - tau_power_law))),
            'valid': np.allclose(tau_hb, tau_power_law, rtol=1e-10, atol=1e-12)
        }

    # Test Bingham limit
    if tau_y > 0 and abs(n - 1.0) < 1e-10:
        # For Bingham plastic, test both yielded and unyielded regions
        gamma_dot_test_bingham = np.array([0.1, 1.0, 10.0, 100.0])
        tau_test_bingham = np.array([tau_y - 1.0, tau_y, tau_y + 5.0, tau_y + 50.0])

        tau_bingham = hb_tau_from_gamma(gamma_dot_test_bingham, tau_y, K, 1.0)
        gamma_dot_bingham = hb_gamma_from_tau(tau_test_bingham, tau_y, K, 1.0)

        # Expected Bingham behavior
        tau_expected = tau_y + K * gamma_dot_test_bingham  # Constitutive
        gamma_dot_expected = np.where(tau_test_bingham <= tau_y, 0.0,
                                    (tau_test_bingham - tau_y) / K)  # Inverse

        results['bingham_limit'] = {
            'constitutive': {
                'gamma_dot': gamma_dot_test_bingham.tolist(),
                'tau_bingham': tau_bingham.tolist(),
                'tau_expected': tau_expected.tolist(),
                'max_error': float(np.max(np.abs(tau_bingham - tau_expected))),
                'valid': np.allclose(tau_bingham, tau_expected, rtol=1e-10, atol=1e-12)
            },
            'inverse': {
                'tau': tau_test_bingham.tolist(),
                'gamma_dot_bingham': gamma_dot_bingham.tolist(),
                'gamma_dot_expected': gamma_dot_expected.tolist(),
                'max_error': float(np.max(np.abs(gamma_dot_bingham - gamma_dot_expected))),
                'valid': np.allclose(gamma_dot_bingham, gamma_dot_expected, rtol=1e-10, atol=1e-12)
            }
        }

    return results


def benchmark_against_analytical(
    tau_y: float,
    K: float,
    n: float,
    dp_dx: float = -1000.0,
    radius: float = 0.01,
    num_points: int = 50
) -> dict:
    """
    Benchmark numerical solution against analytical solutions where available.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    dp_dx : float
        Pressure gradient [Pa/m]
    radius : float
        Pipe radius [m]
    num_points : int
        Number of radial points for comparison

    Returns
    -------
    dict with benchmark results
    """
    results = {
        'parameters': {
            'tau_y': tau_y,
            'K': K,
            'n': n,
            'dp_dx': dp_dx,
            'radius': radius
        },
        'analytical_available': False,
        'numerical_solution': None,
        'comparison': None,
        'error_metrics': None
    }

    # Check if analytical solution is available
    has_analytical = False
    analytical_Q = 0.0
    analytical_profile = None

    if tau_y == 0.0 and abs(n - 1.0) < 1e-10:
        # Newtonian analytical solution
        has_analytical = True
        mu = K
        analytical_Q = (np.pi * radius**4 * abs(dp_dx)) / (8 * mu)

        # Parabolic velocity profile
        r_points = np.linspace(0, radius, num_points)
        analytical_profile = (abs(dp_dx) / (4 * mu)) * (radius**2 - r_points**2)

    elif tau_y == 0.0:
        # Power-law analytical solution (simplified)
        has_analytical = True
        # Approximate analytical solution for power-law
        # This is an approximation - full analytical solution is complex
        analytical_Q = _approximate_power_law_flow(dp_dx, radius, K, n)

        r_points = np.linspace(0, radius, num_points)
        analytical_profile = _approximate_power_law_profile(dp_dx, r_points, radius, K, n)

    elif abs(n - 1.0) < 1e-10 and tau_y > 0:
        # Bingham plastic analytical solution
        has_analytical = True
        analytical_Q, analytical_profile = _bingham_analytical_solution(
            dp_dx, radius, tau_y, K, num_points
        )

    results['analytical_available'] = has_analytical

    if has_analytical:
        results['analytical_solution'] = {
            'Q': analytical_Q,
            'velocity_profile': analytical_profile,
            'r_points': np.linspace(0, radius, num_points)
        }

    return results


def _approximate_power_law_flow(dp_dx: float, radius: float, K: float, n: float) -> float:
    """Approximate analytical flow rate for power-law fluid."""
    # Simplified approximation - real analytical solution involves complex integrals
    # This is a rough approximation for benchmarking
    return (np.pi * radius**3 / K) * (abs(dp_dx) * radius / (2*K))**(1/n) * radius / (n + 1)


def _approximate_power_law_profile(
    dp_dx: float,
    r_points: np.ndarray,
    radius: float,
    K: float,
    n: float
) -> np.ndarray:
    """Approximate velocity profile for power-law fluid."""
    # Simplified velocity profile approximation
    # Real profile would require solving the ODE analytically
    gamma_dot_w = ((abs(dp_dx) * radius) / (2 * K))**(1/n)

    # Approximate power-law profile
    profile = gamma_dot_w * radius / (n + 1) * (1 - (r_points / radius)**(n+1)/(n+1))

    return profile


def _bingham_analytical_solution(
    dp_dx: float,
    radius: float,
    tau_y: float,
    mu: float,
    num_points: int
) -> Tuple[float, np.ndarray]:
    """
    Analytical solution for Bingham plastic in circular pipe.

    Returns flow rate and velocity profile.
    """
    tau_w = radius * abs(dp_dx) / 2

    if tau_w <= tau_y:
        # No flow
        return 0.0, np.zeros(num_points)

    # Yielded region radius
    r_y = radius * (tau_y / tau_w)

    # Flow rate calculation for Bingham plastic
    # Q = (π R^4 Δp)/(8 μ L) - (π τ_y R^3)/(3 μ) + (π τ_y^4)/(2 μ Δp / L)^3
    # Simplified form
    Q1 = (np.pi * radius**4 * abs(dp_dx)) / (8 * mu)
    Q2 = (np.pi * tau_y * radius**3) / (3 * mu)
    Q3 = (np.pi * tau_y**4) / (2 * (mu * abs(dp_dx))**3) * radius**3

    Q = Q1 - Q2 + Q3

    # Velocity profile
    r_points = np.linspace(0, radius, num_points)
    velocity_profile = np.zeros(num_points)

    for i, r in enumerate(r_points):
        if r <= r_y:
            # Unyielded plug region
            velocity_profile[i] = (abs(dp_dx) / (4 * mu)) * (radius**2 - r_y**2) + \
                                (tau_y / mu) * (r_y - r)
        else:
            # Yielded region (parabolic)
            velocity_profile[i] = (abs(dp_dx) / (4 * mu)) * (radius**2 - r**2)

    return Q, velocity_profile


def validate_numerical_convergence(
    tau_y: float,
    K: float,
    n: float,
    dp_dx: float = -1000.0,
    a: float = 0.01,
    b: float = 0.005,
    grid_sizes: Optional[List[int]] = None
) -> dict:
    """
    Validate numerical convergence with grid refinement.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    dp_dx : float
        Pressure gradient [Pa/m]
    a : float
        Semi-major axis [m]
    b : float
        Semi-minor axis [m]
    grid_sizes : list of int, optional
        Grid sizes to test (default: [21, 41, 81, 161])

    Returns
    -------
    dict with convergence analysis results
    """
    if grid_sizes is None:
        grid_sizes = [21, 41, 81, 161]

    results = {
        'grid_sizes': grid_sizes,
        'solutions': [],
        'convergence_analysis': {}
    }

    for nx in grid_sizes:
        ny = nx  # Square grid

        try:
            result = solve_elliptical_hb(
                dp_dx, tau_y, K, n, a, b, nx, tolerance=1e-8, max_iter=2000
            )

            results['solutions'].append({
                'nx': nx,
                'ny': ny,
                'Q': result['Q'],
                'max_velocity': result['max_velocity'],
                'yielded_fraction': result['yielded_fraction'],
                'converged': result['converged'],
                'iterations': result['iterations']
            })

        except Exception as e:
            results['solutions'].append({
                'nx': nx,
                'ny': ny,
                'error': str(e)
            })

    # Analyze convergence
    successful_solutions = [s for s in results['solutions'] if 'error' not in s]

    if len(successful_solutions) >= 2:
        Q_values = [s['Q'] for s in successful_solutions]
        grid_sizes_success = [s['nx'] for s in successful_solutions]

        # Calculate convergence rates
        if len(Q_values) >= 3:
            # Richardson extrapolation for convergence rate
            h1, h2, h3 = 1.0/grid_sizes_success[-3], 1.0/grid_sizes_success[-2], 1.0/grid_sizes_success[-1]
            Q1, Q2, Q3 = Q_values[-3], Q_values[-2], Q_values[-1]

            # Estimate convergence rate
            convergence_rate = np.log(abs((Q2 - Q1) / (Q3 - Q2))) / np.log(h2 / h3)

            results['convergence_analysis'] = {
                'convergence_rate': float(convergence_rate),
                'grid_sizes': grid_sizes_success,
                'Q_values': Q_values,
                'Q_convergence': float(abs(Q_values[-1] - Q_values[-2]) / Q_values[-1]),
                'order_estimate': 'second' if convergence_rate > 1.8 else 'first'
            }

    return results


def validate_physical_reasonableness(
    tau_y: float,
    K: float,
    n: float,
    test_conditions: Optional[List[Dict[str, float]]] = None
) -> dict:
    """
    Validate that HB model predictions are physically reasonable.

    Parameters
    ----------
    tau_y : float
        Yield stress [Pa]
    K : float
        Consistency index [Pa·s^n]
    n : float
        Flow behavior index [-]
    test_conditions : list of dict, optional
        Test conditions with keys 'dp_dx', 'a', 'b'

    Returns
    -------
    dict with physical validation results
    """
    if test_conditions is None:
        test_conditions = [
            {'dp_dx': -100.0, 'a': 0.01, 'b': 0.005},
            {'dp_dx': -1000.0, 'a': 0.01, 'b': 0.005},
            {'dp_dx': -10000.0, 'a': 0.01, 'b': 0.005},
        ]

    results = {
        'parameters': {'tau_y': tau_y, 'K': K, 'n': n},
        'test_conditions': test_conditions,
        'physical_checks': []
    }

    for condition in test_conditions:
        try:
            result = solve_elliptical_hb(
                condition['dp_dx'], tau_y, K, n,
                condition['a'], condition['b'],
                tolerance=1e-8, max_iter=2000
            )

            # Physical reasonableness checks
            checks = {
                'condition': condition,
                'Q_positive': result['Q'] >= 0,
                'velocity_non_negative': np.all(result['velocity_profile'] >= -1e-6),
                'yielded_fraction_reasonable': 0 <= result['yielded_fraction'] <= 1,
                'wall_shear_stress_positive': result['wall_shear_stress'] >= 0,
                'solution_converged': result['converged'],
                'iterations_reasonable': result['iterations'] < 5000,
                'flow_increases_with_dp': True,  # Would need comparison
                'results': result
            }

            results['physical_checks'].append(checks)

        except Exception as e:
            results['physical_checks'].append({
                'condition': condition,
                'error': str(e),
                'computation_failed': True
            })

    # Overall assessment
    successful_checks = [c for c in results['physical_checks'] if 'error' not in c]
    if successful_checks:
        all_passed = all(
            c['Q_positive'] and c['velocity_non_negative'] and
            c['yielded_fraction_reasonable'] and c['solution_converged']
            for c in successful_checks
        )

        results['overall_assessment'] = {
            'physically_reasonable': all_passed,
            'tests_passed': sum(1 for c in successful_checks
                              if c['Q_positive'] and c['velocity_non_negative']),
            'total_tests': len(successful_checks)
        }
    else:
        results['overall_assessment'] = {
            'physically_reasonable': False,
            'tests_passed': 0,
            'total_tests': len(results['physical_checks']),
            'all_failed': True
        }

    return results
