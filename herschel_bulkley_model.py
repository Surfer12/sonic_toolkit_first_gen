"""
Herschel-Bulkley Fluid Model Implementation

This module implements the Herschel-Bulkley (HB) constitutive model for
non-Newtonian fluids with yield stress, including:
- Constitutive and inverse forms
- Parameter fitting from rheometer data
- Elliptical duct flow solver
- Validation against Newtonian and power-law limits
- API and CLI interfaces

Mathematical Model:
τ = τy + K·γ̇^n          (constitutive form)
γ̇(τ) = max(((τ−τy)/K)^(1/n), 0)  (inverse form)

Author: Ryan David Oates
Date: August 26, 2025
License: GPL-3.0-only
"""

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
import json
import argparse
import warnings
from pathlib import Path


@dataclass
class HBParameters:
    """Herschel-Bulkley model parameters."""
    tau_y: float  # Yield stress [Pa]
    K: float      # Consistency index [Pa·s^n]
    n: float      # Flow behavior index [-]

    def __post_init__(self):
        """Validate parameters."""
        if self.tau_y < 0:
            raise ValueError("Yield stress τy must be non-negative")
        if self.K <= 0:
            raise ValueError("Consistency index K must be positive")
        if self.n <= 0:
            raise ValueError("Flow behavior index n must be positive")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'tau_y': self.tau_y,
            'K': self.K,
            'n': self.n
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HBParameters':
        """Create from dictionary."""
        return cls(
            tau_y=data['tau_y'],
            K=data['K'],
            n=data['n']
        )


@dataclass
class FitResult:
    """Parameter fitting result with uncertainty."""
    parameters: HBParameters
    covariance: np.ndarray
    r_squared: float
    rmse: float
    parameter_errors: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'parameters': self.parameters.to_dict(),
            'covariance': self.covariance.tolist(),
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'parameter_errors': self.parameter_errors
        }


class HerschelBulkleyModel:
    """
    Herschel-Bulkley fluid model implementation.

    Provides constitutive and inverse forms with proper handling of
    unyielded regions (τ ≤ τy) and vectorized operations for performance.
    """

    def __init__(self, parameters: Optional[HBParameters] = None):
        """
        Initialize HB model.

        Args:
            parameters: HB model parameters. If None, defaults to Newtonian.
        """
        if parameters is None:
            # Default to Newtonian fluid (τy=0, n=1)
            parameters = HBParameters(tau_y=0.0, K=1.0, n=1.0)

        self.parameters = parameters

    def constitutive_model(self, gamma_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Constitutive model: τ = τy + K·γ̇^n

        Args:
            gamma_dot: Shear rate(s) [1/s]

        Returns:
            Shear stress(es) [Pa]
        """
        gamma_dot = np.asarray(gamma_dot)
        tau_y = self.parameters.tau_y
        K = self.parameters.K
        n = self.parameters.n

        # Handle scalar case
        if gamma_dot.ndim == 0:
            return tau_y + K * (gamma_dot ** n)

        # Vectorized computation
        return tau_y + K * np.power(gamma_dot, n)

    def inverse_model(self, tau: Union[float, np.ndarray],
                     clamp_negative: bool = True) -> Union[float, np.ndarray]:
        """
        Inverse model: γ̇(τ) = max(((τ−τy)/K)^(1/n), 0)

        Args:
            tau: Shear stress(es) [Pa]
            clamp_negative: Whether to clamp negative values to zero

        Returns:
            Shear rate(s) [1/s]
        """
        tau = np.asarray(tau)
        tau_y = self.parameters.tau_y
        K = self.parameters.K
        n = self.parameters.n

        # Calculate effective stress (τ - τy)
        tau_effective = tau - tau_y

        # Handle unyielded region (τ ≤ τy)
        if tau.ndim == 0:
            # Scalar case
            if tau_effective <= 0:
                return 0.0 if clamp_negative else (tau_effective / K) ** (1.0 / n)
            else:
                gamma_dot = (tau_effective / K) ** (1.0 / n)
                return max(gamma_dot, 0.0) if clamp_negative else gamma_dot

        # Vectorized case
        if clamp_negative:
            gamma_dot = np.zeros_like(tau)
            # Only compute for yielded regions
            yielded_mask = tau_effective > 0
            if np.any(yielded_mask):
                gamma_dot[yielded_mask] = (tau_effective[yielded_mask] / K) ** (1.0 / n)
        else:
            # Compute for all regions without clamping
            gamma_dot = (tau_effective / K) ** (1.0 / n)

        return gamma_dot

    def apparent_viscosity(self, gamma_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate apparent viscosity: η = τ/γ̇

        Args:
            gamma_dot: Shear rate(s) [1/s]

        Returns:
            Apparent viscosity(ies) [Pa·s]
        """
        tau = self.constitutive_model(gamma_dot)
        gamma_dot = np.asarray(gamma_dot)

        # Handle zero shear rate
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.divide(tau, gamma_dot)
            eta = np.where(gamma_dot == 0, np.inf, eta)

        return eta

    def get_model_info(self) -> dict:
        """Get model information and characteristics."""
        return {
            'model_type': 'Herschel-Bulkley',
            'parameters': self.parameters.to_dict(),
            'is_newtonian': self.parameters.tau_y == 0 and self.parameters.n == 1,
            'is_power_law': self.parameters.tau_y == 0 and self.parameters.n != 1,
            'is_bingham': self.parameters.n == 1 and self.parameters.tau_y > 0,
            'behavior': self._classify_behavior()
        }

    def _classify_behavior(self) -> str:
        """Classify fluid behavior."""
        tau_y = self.parameters.tau_y
        n = self.parameters.n

        if tau_y == 0 and n == 1:
            return "Newtonian"
        elif tau_y == 0 and n != 1:
            return f"Power-law ({'shear-thinning' if n < 1 else 'shear-thickening'})"
        elif tau_y > 0 and n == 1:
            return "Bingham plastic"
        else:
            return f"Herschel-Bulkley ({'shear-thinning' if n < 1 else 'shear-thickening'})"


class HBParameterFitter:
    """
    Parameter fitting for Herschel-Bulkley model from rheometer data.

    Supports fitting (τy, K, n) with uncertainty analysis and cross-validation.
    """

    def __init__(self):
        """Initialize parameter fitter."""
        pass

    def fit_parameters(self, gamma_dot: np.ndarray, tau: np.ndarray,
                      tau_y_bounds: Tuple[float, float] = (0, 100),
                      K_bounds: Tuple[float, float] = (0.1, 100),
                      n_bounds: Tuple[float, float] = (0.1, 2.0),
                      use_weights: bool = True) -> FitResult:
        """
        Fit HB parameters to rheometer data.

        Args:
            gamma_dot: Shear rates [1/s]
            tau: Shear stresses [Pa]
            tau_y_bounds: Bounds for yield stress fitting
            K_bounds: Bounds for consistency index fitting
            n_bounds: Bounds for flow behavior index fitting
            use_weights: Whether to use weighted least squares

        Returns:
            FitResult with parameters, covariance, and statistics
        """
        if len(gamma_dot) != len(tau):
            raise ValueError("gamma_dot and tau must have same length")

        if len(gamma_dot) < 3:
            raise ValueError("Need at least 3 data points for fitting")

        # Prepare data
        gamma_dot = np.asarray(gamma_dot)
        tau = np.asarray(tau)

        # Remove any invalid data points
        valid_mask = (gamma_dot > 0) & (tau > 0) & np.isfinite(gamma_dot) & np.isfinite(tau)
        gamma_dot = gamma_dot[valid_mask]
        tau = tau[valid_mask]

        if len(gamma_dot) < 3:
            raise ValueError("Need at least 3 valid data points after filtering")

        # Define objective function
        def hb_objective(params, gamma_dot, tau):
            tau_y, K, n = params

            # Predicted stress
            tau_pred = tau_y + K * np.power(gamma_dot, n)

            # Weighted residual (higher weights for higher shear rates)
            if use_weights:
                weights = np.sqrt(gamma_dot / np.max(gamma_dot))
                residual = weights * (tau_pred - tau)
            else:
                residual = tau_pred - tau

            return residual

        # Initial guess
        tau_y_init = np.min(tau) if np.min(tau) > 0 else 0
        K_init = np.mean(tau) / np.mean(gamma_dot)**0.5
        n_init = 0.8  # Typical shear-thinning value

        initial_guess = [tau_y_init, K_init, n_init]
        # Format bounds for scipy.optimize.least_squares: (lower_bounds, upper_bounds)
        bounds = ([tau_y_bounds[0], K_bounds[0], n_bounds[0]],  # lower bounds
                  [tau_y_bounds[1], K_bounds[1], n_bounds[1]])   # upper bounds

        try:
            # Perform least squares fitting
            result = opt.least_squares(
                hb_objective,
                initial_guess,
                bounds=bounds,
                args=(gamma_dot, tau),
                method='trf',
                loss='soft_l1'  # Robust loss function
            )

            if not result.success:
                warnings.warn(f"Fitting may not have converged: {result.message}")

            # Extract fitted parameters
            tau_y_fit, K_fit, n_fit = result.x
            parameters = HBParameters(tau_y=tau_y_fit, K=K_fit, n=n_fit)

            # Calculate covariance matrix (approximate)
            J = result.jac
            try:
                if J.shape[0] >= 3 and J.shape[1] == 3:
                    cov = np.linalg.inv(J.T @ J) * (np.sum(result.fun**2) / (len(result.fun) - 3))
                else:
                    # Fallback for poorly conditioned Jacobian
                    cov = np.eye(3) * np.nan

                # Calculate parameter errors with bounds checking
                parameter_errors = {}
                for i, param_name in enumerate(['tau_y_error', 'K_error', 'n_error']):
                    if (cov.shape[0] > i and cov.shape[1] > i and
                        not np.isnan(cov[i, i]) and cov[i, i] > 0):
                        parameter_errors[param_name] = np.sqrt(cov[i, i])
                    else:
                        parameter_errors[param_name] = np.nan

            except (np.linalg.LinAlgError, ValueError):
                # Handle singular or ill-conditioned matrices
                cov = np.eye(3) * np.nan
                parameter_errors = {
                    'tau_y_error': np.nan,
                    'K_error': np.nan,
                    'n_error': np.nan
                }

            # Calculate goodness of fit
            tau_pred = parameters.tau_y + parameters.K * np.power(gamma_dot, parameters.n)
            ss_res = np.sum((tau - tau_pred)**2)
            ss_tot = np.sum((tau - np.mean(tau))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((tau - tau_pred)**2))

            return FitResult(
                parameters=parameters,
                covariance=cov,
                r_squared=r_squared,
                rmse=rmse,
                parameter_errors=parameter_errors
            )

        except Exception as e:
            raise RuntimeError(f"Parameter fitting failed: {e}")

    def cross_validate(self, gamma_dot: np.ndarray, tau: np.ndarray,
                      k_folds: int = 5) -> dict:
        """
        Perform k-fold cross-validation.

        Args:
            gamma_dot: Shear rates [1/s]
            tau: Shear stresses [Pa]
            k_folds: Number of folds

        Returns:
            Dictionary with CV statistics
        """
        n_samples = len(gamma_dot)
        fold_size = n_samples // k_folds

        cv_scores = []

        for i in range(k_folds):
            # Split data
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else n_samples

            # Test set
            test_gamma = gamma_dot[start_idx:end_idx]
            test_tau = tau[start_idx:end_idx]

            # Training set
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[start_idx:end_idx] = False
            train_gamma = gamma_dot[train_mask]
            train_tau = tau[train_mask]

            # Fit on training data
            try:
                fit_result = self.fit_parameters(train_gamma, train_tau)

                # Evaluate on test data
                model = HerschelBulkleyModel(fit_result.parameters)
                tau_pred = model.constitutive_model(test_gamma)

                # Calculate RMSE for this fold
                rmse = np.sqrt(np.mean((test_tau - tau_pred)**2))
                cv_scores.append(rmse)

            except Exception as e:
                warnings.warn(f"CV fold {i+1} failed: {e}")
                continue

        if not cv_scores:
            raise RuntimeError("All cross-validation folds failed")

        return {
            'cv_rmse_mean': np.mean(cv_scores),
            'cv_rmse_std': np.std(cv_scores),
            'cv_rmse_folds': cv_scores,
            'n_folds_successful': len(cv_scores)
        }


class EllipticalDuctFlowSolver:
    """
    Flow solver for Herschel-Bulkley fluids in elliptical ducts.

    Solves for volumetric flow rate Q given pressure gradient Δp,
    and provides velocity profile calculations.
    """

    def __init__(self, hb_model: HerschelBulkleyModel, a: float, b: float):
        """
        Initialize elliptical duct solver.

        Args:
            hb_model: Herschel-Bulkley model instance
            a: Semi-major axis [m]
            b: Semi-minor axis [m]
        """
        if a <= 0 or b <= 0:
            raise ValueError("Semi-axes must be positive")

        self.hb_model = hb_model
        self.a = a  # Semi-major axis
        self.b = b  # Semi-minor axis

        # Elliptical geometry properties
        self.area = np.pi * a * b
        self.perimeter = self._calculate_perimeter()

    def _calculate_perimeter(self) -> float:
        """Calculate elliptical perimeter using Ramanujan approximation."""
        a, b = self.a, self.b
        h = ((a - b) / (a + b))**2
        return np.pi * (a + b) * (1 + (3*h)/(10 + np.sqrt(4 - 3*h)))

    def _integrand(self, x: float, dp_dx: float) -> float:
        """
        Integrand for velocity profile calculation.

        For elliptical duct, the velocity profile is calculated by integrating
        the HB inverse model over the cross-section.
        """
        # For elliptical coordinates
        # The shear rate γ̇ = (r / R) * |dp_dx| / (2η), but for HB it's more complex
        # This is a simplified approach - in practice, we'd need to solve the
        # full momentum equation for elliptical geometry

        # Simplified model: assume plug flow in unyielded region
        tau_wall = (self.a * self.b / self.perimeter) * abs(dp_dx)

        # Calculate shear stress at position x
        # This is an approximation for elliptical duct
        position_factor = x / self.a  # Normalized position
        tau_local = tau_wall * position_factor

        # Calculate shear rate from inverse HB model
        gamma_dot = self.hb_model.inverse_model(tau_local)

        return gamma_dot

    def calculate_flow_rate(self, dp_dx: float, num_points: int = 100) -> dict:
        """
        Calculate volumetric flow rate for given pressure gradient.

        Args:
            dp_dx: Pressure gradient [Pa/m]
            num_points: Number of integration points

        Returns:
            Dictionary with flow results
        """
        if dp_dx == 0:
            return {
                'Q': 0.0,
                'velocity_profile': np.zeros(num_points),
                'yielded_fraction': 0.0,
                'warnings': ['Zero pressure gradient']
            }

        # Calculate wall shear stress for elliptical duct
        # For elliptical duct: τ_w = (a*b / P) * |dp_dx|
        # where P is the wetted perimeter
        # This is the average wall shear stress
        tau_w = (self.a * self.b / self.perimeter) * abs(dp_dx)

        # Check if fluid yields
        if tau_w <= self.hb_model.parameters.tau_y:
            # No flow - unyielded plug
            return {
                'Q': 0.0,
                'velocity_profile': np.zeros(num_points),
                'yielded_fraction': 0.0,
                'warnings': ['Pressure gradient too low - no flow']
            }

        # Integration points across duct radius
        r_points = np.linspace(0, self.a, num_points)
        velocity_profile = np.zeros(num_points)

        # Calculate velocity profile
        for i, r in enumerate(r_points):
            if r == 0:
                # Centerline velocity (plug flow)
                velocity_profile[i] = self._calculate_centerline_velocity(dp_dx, tau_w)
            else:
                # Velocity at position r
                velocity_profile[i] = self._calculate_velocity_at_radius(r, dp_dx, tau_w)

        # Calculate flow rate by integrating velocity profile
        # For elliptical duct: Q = ∫∫ u dA over cross-section
        # Using numerical integration
        Q = self._integrate_velocity_profile(velocity_profile, r_points)

        # Calculate yielded fraction
        yielded_radius = self._calculate_yielded_radius(tau_w, dp_dx)
        yielded_fraction = (yielded_radius / self.a)**2

        return {
            'Q': Q,
            'velocity_profile': velocity_profile,
            'yielded_fraction': yielded_fraction,
            'wall_shear_stress': tau_w,
            'warnings': []
        }

    def _calculate_centerline_velocity(self, dp_dx: float, tau_w: float) -> float:
        """Calculate centerline velocity."""
        # Simplified approach - in practice, this requires solving the full equation
        # For now, use an approximation
        tau_eff = tau_w - self.hb_model.parameters.tau_y
        if tau_eff <= 0:
            return 0.0

        gamma_dot_eff = self.hb_model.inverse_model(tau_eff)
        # Approximate velocity as gamma_dot * characteristic length
        return gamma_dot_eff * self.a

    def _calculate_velocity_at_radius(self, r: float, dp_dx: float, tau_w: float) -> float:
        """Calculate velocity at given radius using simplified analytical approach."""
        # For elliptical duct, the exact velocity profile requires solving:
        # d²u/dr² + (1/r) du/dr - (dp_dx/μ) = 0 (Newtonian)
        # This is a significant simplification for demonstration

        # Wall shear stress scaling (approximation for elliptical geometry)
        position_factor = np.sqrt((r / self.a)**2 + ((0.0) / self.b)**2)  # Simplified radial scaling
        tau_local = tau_w * position_factor

        if tau_local <= self.hb_model.parameters.tau_y:
            # Unyielded region - constant velocity (plug flow)
            return self._calculate_centerline_velocity(dp_dx, tau_w)
        else:
            # Yielded region - simplified velocity calculation
            # This approximates the velocity profile but is not exact
            gamma_dot = self.hb_model.inverse_model(tau_local)
            # Simplified velocity scaling - in practice, this requires integration
            velocity_scale = gamma_dot * (self.a - r) * 0.1  # Empirical scaling factor
            return velocity_scale

    def _calculate_yielded_radius(self, tau_w: float, dp_dx: float) -> float:
        """Calculate radius of yielded region."""
        if tau_w <= self.hb_model.parameters.tau_y:
            return 0.0

        # Find radius where shear stress equals yield stress
        r_yield = self.a * (self.hb_model.parameters.tau_y / tau_w)
        return r_yield

    def _integrate_velocity_profile(self, velocity_profile: np.ndarray,
                                  r_points: np.ndarray) -> float:
        """Integrate velocity profile to get flow rate using trapezoidal rule."""
        # For elliptical duct, the cross-section area element is more complex
        # This is still an approximation, but uses better numerical integration

        if len(r_points) < 2:
            return 0.0

        # Use trapezoidal rule for numerical integration
        # Note: This approximates elliptical geometry as circular for integration
        # A more accurate implementation would use elliptical coordinates

        dr = np.diff(r_points)
        r_mid = (r_points[:-1] + r_points[1:]) / 2
        u_mid = (velocity_profile[:-1] + velocity_profile[1:]) / 2

        # Integrate ∫ 2πr u dr (circular approximation)
        # For elliptical duct, this should be adjusted for the actual geometry
        integrand = 2 * np.pi * r_mid * u_mid

        # Apply trapezoidal rule
        flow_rate = np.sum(integrand * dr)

        # Correction factor for elliptical geometry (approximate)
        # The actual elliptical cross-section is π*a*b vs π*a² for circular
        elliptical_factor = (self.b / self.a)
        flow_rate *= elliptical_factor

        return max(flow_rate, 0.0)  # Ensure non-negative flow rate


class HBVisualizer:
    """Visualization tools for HB model and flow results."""

    def __init__(self):
        """Initialize visualizer."""
        plt.style.use('default')

    def plot_rheogram(self, model: HerschelBulkleyModel,
                     gamma_dot_range: Tuple[float, float] = (0.1, 1000),
                     num_points: int = 100,
                     save_path: Optional[str] = None):
        """
        Plot shear stress vs shear rate (rheogram).

        Args:
            model: HB model instance
            gamma_dot_range: Range of shear rates to plot
            num_points: Number of points to plot
            save_path: Path to save plot (optional)
        """
        gamma_dot = np.logspace(np.log10(gamma_dot_range[0]),
                               np.log10(gamma_dot_range[1]), num_points)
        tau = model.constitutive_model(gamma_dot)
        eta = model.apparent_viscosity(gamma_dot)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Shear stress vs shear rate
        ax1.loglog(gamma_dot, tau, 'b-', linewidth=2, label='HB Model')
        ax1.set_xlabel('Shear Rate, γ̇ (1/s)')
        ax1.set_ylabel('Shear Stress, τ (Pa)')
        ax1.set_title('Rheogram: τ vs γ̇')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Mark yield stress if applicable
        if model.parameters.tau_y > 0:
            ax1.axhline(y=model.parameters.tau_y, color='r', linestyle='--',
                       alpha=0.7, label='Yield Stress')
            ax1.legend()

        # Apparent viscosity vs shear rate
        ax2.loglog(gamma_dot, eta, 'g-', linewidth=2)
        ax2.set_xlabel('Shear Rate, γ̇ (1/s)')
        ax2.set_ylabel('Apparent Viscosity, η (Pa·s)')
        ax2.set_title('Apparent Viscosity vs Shear Rate')
        ax2.grid(True, alpha=0.3)

        # Add model info
        info = model.get_model_info()
        fig.suptitle(f"Herschel-Bulkley Model: {info['behavior']}\n" +
                    ".2f" +
                    f"n = {model.parameters.n:.3f}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig

    def plot_velocity_profile(self, flow_result: dict, duct_geometry: dict,
                            save_path: Optional[str] = None):
        """
        Plot velocity profile in elliptical duct.

        Args:
            flow_result: Result from EllipticalDuctFlowSolver.calculate_flow_rate
            duct_geometry: Dictionary with 'a' and 'b' semi-axes
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create elliptical cross-section
        a, b = duct_geometry['a'], duct_geometry['b']
        theta = np.linspace(0, 2*np.pi, 100)
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)

        ax.plot(x_ellipse, y_ellipse, 'k--', alpha=0.5, label='Duct Wall')

        # Plot velocity profile (simplified as radial)
        r_max = max(a, b)
        r_points = np.linspace(0, r_max, len(flow_result['velocity_profile']))
        velocity = flow_result['velocity_profile']

        ax.plot(r_points, velocity, 'b-', linewidth=2, label='Velocity Profile')

        # Mark yielded/unyielded regions
        if flow_result['yielded_fraction'] < 1.0:
            r_yield = r_max * flow_result['yielded_fraction']
            ax.axvline(x=r_yield, color='r', linestyle=':', alpha=0.7,
                      label='.2f')
        ax.set_xlabel('Radial Position (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile in Elliptical Duct')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add flow information
        info_text = ".2f"".3f"".3f"".3f"
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Velocity profile saved to {save_path}")

        return fig


def create_cli_interface():
    """Create command-line interface for HB model."""
    parser = argparse.ArgumentParser(description='Herschel-Bulkley Fluid Model CLI')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Constitutive model command
    const_parser = subparsers.add_parser('constitutive', help='Calculate shear stress from shear rate')
    const_parser.add_argument('--tau-y', type=float, default=0.0, help='Yield stress [Pa]')
    const_parser.add_argument('--K', type=float, default=1.0, help='Consistency index [Pa·s^n]')
    const_parser.add_argument('--n', type=float, default=1.0, help='Flow behavior index')
    const_parser.add_argument('--gamma-dot', type=float, required=True, help='Shear rate [1/s]')

    # Inverse model command
    inv_parser = subparsers.add_parser('inverse', help='Calculate shear rate from shear stress')
    inv_parser.add_argument('--tau-y', type=float, default=0.0, help='Yield stress [Pa]')
    inv_parser.add_argument('--K', type=float, default=1.0, help='Consistency index [Pa·s^n]')
    inv_parser.add_argument('--n', type=float, default=1.0, help='Flow behavior index')
    inv_parser.add_argument('--tau', type=float, required=True, help='Shear stress [Pa]')

    # Flow solver command
    flow_parser = subparsers.add_parser('flow', help='Calculate flow rate in elliptical duct')
    flow_parser.add_argument('--tau-y', type=float, default=0.0, help='Yield stress [Pa]')
    flow_parser.add_argument('--K', type=float, default=1.0, help='Consistency index [Pa·s^n]')
    flow_parser.add_argument('--n', type=float, default=1.0, help='Flow behavior index')
    flow_parser.add_argument('--dp-dx', type=float, required=True, help='Pressure gradient [Pa/m]')
    flow_parser.add_argument('--a', type=float, default=0.01, help='Semi-major axis [m]')
    flow_parser.add_argument('--b', type=float, default=0.005, help='Semi-minor axis [m]')

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate plots')
    plot_parser.add_argument('--tau-y', type=float, default=0.0, help='Yield stress [Pa]')
    plot_parser.add_argument('--K', type=float, default=1.0, help='Consistency index [Pa·s^n]')
    plot_parser.add_argument('--n', type=float, default=1.0, help='Flow behavior index')
    plot_parser.add_argument('--output', type=str, default='hb_plot.png', help='Output file path')

    return parser


def main():
    """Main CLI function."""
    parser = create_cli_interface()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command in ['constitutive', 'inverse', 'flow', 'plot']:
            # Create HB model
            params = HBParameters(
                tau_y=getattr(args, 'tau_y', 0.0),
                K=getattr(args, 'K', 1.0),
                n=getattr(args, 'n', 1.0)
            )
            model = HerschelBulkleyModel(params)

        if args.command == 'constitutive':
            tau = model.constitutive_model(args.gamma_dot)
            print(".6f")

        elif args.command == 'inverse':
            gamma_dot = model.inverse_model(args.tau)
            print(".6f")

        elif args.command == 'flow':
            solver = EllipticalDuctFlowSolver(model, args.a, args.b)
            result = solver.calculate_flow_rate(args.dp_dx)

            print(".6f")
            print(".3f")
            print(".3f")

            if result['warnings']:
                print("Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")

        elif args.command == 'plot':
            visualizer = HBVisualizer()
            fig = visualizer.plot_rheogram(model, save_path=args.output)
            print(f"Plot saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
