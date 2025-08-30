"""
Core Herschel-Bulkley constitutive equations and inverse functions.

The Herschel-Bulkley model describes the relationship between shear stress (τ) 
and shear rate (γ̇) for non-Newtonian fluids with yield stress:

τ = τy + K * γ̇^n  for τ > τy
γ̇ = 0             for τ ≤ τy

Units:
- τ: shear stress [Pa]
- γ̇: shear rate [1/s]
- τy: yield stress [Pa]
- K: consistency index [Pa·s^n]
- n: flow behavior index [dimensionless]

Domain constraints:
- τy ≥ 0
- K > 0
- n > 0 (typically 0.1 ≤ n ≤ 2.0)
"""

import numpy as np
from typing import Union, Tuple
import warnings


class HerschelBulkley:
    """
    Herschel-Bulkley constitutive model with robust inverse calculations.
    """
    
    def __init__(self, tau_y: float, K: float, n: float, 
                 gamma_min: float = 1e-12, gamma_max: float = 1e6):
        """
        Initialize Herschel-Bulkley model parameters.
        
        Parameters:
        -----------
        tau_y : float
            Yield stress [Pa], must be ≥ 0
        K : float
            Consistency index [Pa·s^n], must be > 0
        n : float
            Flow behavior index [dimensionless], must be > 0
        gamma_min : float
            Minimum shear rate for clamping [1/s]
        gamma_max : float
            Maximum shear rate for clamping [1/s]
        """
        self._validate_parameters(tau_y, K, n)
        
        self.tau_y = tau_y
        self.K = K
        self.n = n
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    
    @staticmethod
    def _validate_parameters(tau_y: float, K: float, n: float) -> None:
        """Validate model parameters."""
        if tau_y < 0:
            raise ValueError(f"Yield stress must be ≥ 0, got {tau_y}")
        if K <= 0:
            raise ValueError(f"Consistency index must be > 0, got {K}")
        if n <= 0:
            raise ValueError(f"Flow behavior index must be > 0, got {n}")
        if n > 2.0:
            warnings.warn(f"Flow behavior index n={n} > 2.0 is unusual")
    
    def stress_from_rate(self, gamma_dot: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate shear stress from shear rate using HB model.
        
        τ = τy + K * γ̇^n
        
        Parameters:
        -----------
        gamma_dot : float or array-like
            Shear rate [1/s]
            
        Returns:
        --------
        tau : float or array
            Shear stress [Pa]
        """
        gamma_dot = np.asarray(gamma_dot)
        return self.tau_y + self.K * np.power(np.abs(gamma_dot), self.n) * np.sign(gamma_dot)
    
    def rate_from_stress(self, tau: Union[float, np.ndarray], 
                        clamp: bool = True) -> Union[float, np.ndarray]:
        """
        Calculate shear rate from shear stress (inverse HB model).
        
        γ̇ = ((τ - τy) / K)^(1/n)  for |τ| > τy
        γ̇ = 0                     for |τ| ≤ τy
        
        Parameters:
        -----------
        tau : float or array-like
            Shear stress [Pa]
        clamp : bool
            Whether to clamp results to [gamma_min, gamma_max]
            
        Returns:
        --------
        gamma_dot : float or array
            Shear rate [1/s]
        """
        tau = np.asarray(tau)
        tau_abs = np.abs(tau)
        tau_sign = np.sign(tau)
        
        # Initialize output array
        gamma_dot = np.zeros_like(tau, dtype=float)
        
        # Only calculate for stresses above yield
        yielding = tau_abs > self.tau_y
        
        if np.any(yielding):
            tau_excess = tau_abs[yielding] - self.tau_y
            gamma_dot[yielding] = tau_sign[yielding] * np.power(tau_excess / self.K, 1.0 / self.n)
        
        # Apply clamping if requested
        if clamp:
            gamma_dot = np.clip(np.abs(gamma_dot), self.gamma_min, self.gamma_max) * np.sign(gamma_dot)
        
        return gamma_dot if tau.ndim > 0 else float(gamma_dot)
    
    def apparent_viscosity(self, gamma_dot: Union[float, np.ndarray], 
                          regularization: str = 'papanastasiou', 
                          m_reg: float = 1000.0) -> Union[float, np.ndarray]:
        """
        Calculate apparent viscosity with optional regularization.
        
        η_app = τ / γ̇
        
        For regularization near γ̇ = 0, Papanastasiou model:
        τ = (τy * (1 - exp(-m * γ̇)) + K * γ̇^n) / γ̇
        
        Parameters:
        -----------
        gamma_dot : float or array-like
            Shear rate [1/s]
        regularization : str
            'none', 'papanastasiou', or 'simple'
        m_reg : float
            Regularization parameter for Papanastasiou model
            
        Returns:
        --------
        eta_app : float or array
            Apparent viscosity [Pa·s]
        """
        gamma_dot = np.asarray(gamma_dot)
        gamma_abs = np.abs(gamma_dot)
        
        if regularization == 'papanastasiou':
            # Papanastasiou regularization
            tau_reg = (self.tau_y * (1.0 - np.exp(-m_reg * gamma_abs)) + 
                      self.K * np.power(gamma_abs, self.n))
            eta_app = tau_reg / np.maximum(gamma_abs, self.gamma_min)
            
        elif regularization == 'simple':
            # Simple regularization: add small viscosity
            tau = self.stress_from_rate(gamma_dot)
            eta_app = np.abs(tau) / np.maximum(gamma_abs, self.gamma_min)
            
        else:  # no regularization
            tau = self.stress_from_rate(gamma_dot)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                eta_app = np.abs(tau) / gamma_abs
                eta_app = np.where(gamma_abs == 0, np.inf, eta_app)
        
        return eta_app if gamma_dot.ndim > 0 else float(eta_app)
    
    def flow_curve_data(self, gamma_range: Tuple[float, float] = (0.01, 1000), 
                       n_points: int = 100, log_space: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate flow curve data for plotting.
        
        Parameters:
        -----------
        gamma_range : tuple
            (min, max) shear rate range [1/s]
        n_points : int
            Number of points to generate
        log_space : bool
            Whether to use logarithmic spacing
            
        Returns:
        --------
        gamma_dot : array
            Shear rate values [1/s]
        tau : array
            Corresponding shear stress values [Pa]
        """
        if log_space:
            gamma_dot = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_points)
        else:
            gamma_dot = np.linspace(gamma_range[0], gamma_range[1], n_points)
        
        tau = self.stress_from_rate(gamma_dot)
        return gamma_dot, tau
    
    def __repr__(self) -> str:
        return f"HerschelBulkley(τy={self.tau_y:.3f} Pa, K={self.K:.3f} Pa·s^n, n={self.n:.3f})"


# Convenience functions for backward compatibility
def hb_stress_from_rate(gamma_dot: Union[float, np.ndarray], 
                       tau_y: float, K: float, n: float) -> Union[float, np.ndarray]:
    """Convenience function for HB stress calculation."""
    hb = HerschelBulkley(tau_y, K, n)
    return hb.stress_from_rate(gamma_dot)


def hb_rate_from_stress(tau: Union[float, np.ndarray], 
                       tau_y: float, K: float, n: float, 
                       clamp: bool = True) -> Union[float, np.ndarray]:
    """Convenience function for HB inverse calculation."""
    hb = HerschelBulkley(tau_y, K, n)
    return hb.rate_from_stress(tau, clamp)
