from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def hb_tau_from_gamma(gamma_dot: np.ndarray | float, tau_y: float, consistency_K: float, flow_index_n: float) -> np.ndarray:
    """
    Compute shear stress for Herschel–Bulkley model.

    τ(γ̇) = τ_y + K · γ̇^n

    Parameters
    - gamma_dot: shear rate γ̇ [1/s]
    - tau_y: yield stress τ_y [Pa]
    - consistency_K: consistency index K [Pa·s^n]
    - flow_index_n: flow index n [-]

    Returns
    - τ [Pa] with same shape as gamma_dot
    """
    gamma = np.asarray(gamma_dot, dtype=float)
    return tau_y + consistency_K * np.power(np.maximum(gamma, 0.0), flow_index_n)


def hb_gamma_from_tau(tau: np.ndarray | float, tau_y: float, consistency_K: float, flow_index_n: float) -> np.ndarray:
    """
    Inverse of Herschel–Bulkley curve: γ̇(τ).

    γ̇(τ) = max( ((τ - τ_y)/K)^(1/n), 0 )

    Parameters
    - tau: shear stress τ [Pa]
    - tau_y: yield stress τ_y [Pa]
    - consistency_K: consistency index K [Pa·s^n]
    - flow_index_n: flow index n [-]

    Returns
    - γ̇ [1/s] with same shape as tau
    """
    t = np.asarray(tau, dtype=float)
    excess = (t - tau_y) / consistency_K
    # Clamp negative values to zero and avoid negative roots
    return np.power(np.maximum(excess, 0.0), 1.0 / flow_index_n)


def hb_apparent_viscosity(gamma_dot: np.ndarray | float, tau_y: float, consistency_K: float, flow_index_n: float, epsilon: float = 1e-12) -> np.ndarray:
    """
    Apparent viscosity μ_app(γ̇) = τ(γ̇)/γ̇ for HB law.

    μ_app(γ̇) = τ_y/γ̇ + K·γ̇^{n-1}

    Parameters
    - gamma_dot: shear rate γ̇ [1/s]
    - tau_y: yield stress τ_y [Pa]
    - consistency_K: consistency index K [Pa·s^n]
    - flow_index_n: flow index n [-]
    - epsilon: lower bound for γ̇ to avoid division by zero [1/s]

    Returns
    - μ_app [Pa·s]
    """
    g = np.asarray(gamma_dot, dtype=float)
    g_safe = np.maximum(g, epsilon)
    return tau_y / g_safe + consistency_K * np.power(g_safe, flow_index_n - 1.0)


# Papanastasiou regularization for yield term for numerical stability in PDE solvers
# τ(γ̇) = τ_y (1 - exp(-m γ̇)) + K γ̇^n

def hb_regularized_tau(
    gamma_dot: np.ndarray | float,
    tau_y: float,
    consistency_K: float,
    flow_index_n: float,
    m_reg: float = 1000.0,
) -> np.ndarray:
    g = np.asarray(gamma_dot, dtype=float)
    return tau_y * (1.0 - np.exp(-m_reg * g)) + consistency_K * np.power(np.maximum(g, 0.0), flow_index_n)


def hb_regularized_apparent_viscosity(
    gamma_dot: np.ndarray | float,
    tau_y: float,
    consistency_K: float,
    flow_index_n: float,
    m_reg: float = 1000.0,
    gamma_min: float = 1e-8,
) -> np.ndarray:
    g = np.asarray(gamma_dot, dtype=float)
    g_safe = np.maximum(g, gamma_min)
    tau = hb_regularized_tau(g_safe, tau_y, consistency_K, flow_index_n, m_reg)
    return tau / g_safe
