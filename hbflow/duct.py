from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .models import hb_regularized_apparent_viscosity


@dataclass
class EllipticalHBResult:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray  # velocity field [m/s]
    Q: float  # volumetric flow rate [m^3/s]
    mean_velocity: float  # Q / (pi a b)
    iterations: int
    converged: bool
    meta: Dict[str, Any]


def _compute_gradients_centered(w: np.ndarray, hx: float, hy: float, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Central differences inside mask; zeros outside mask act as boundary
    wx = np.zeros_like(w)
    wy = np.zeros_like(w)
    # Use central differences where possible
    wx[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2.0 * hx)
    wy[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * hy)
    # For near-boundary points, one-sided differences using boundary value 0 outside mask
    # Left/right borders
    wx[:, 0] = (w[:, 1] - 0.0) / (hx)
    wx[:, -1] = (0.0 - w[:, -2]) / (hx)
    # Bottom/top borders
    wy[0, :] = (w[1, :] - 0.0) / (hy)
    wy[-1, :] = (0.0 - w[-2, :]) / (hy)
    # Mask out values outside domain
    wx = wx * mask
    wy = wy * mask
    return wx, wy


def solve_elliptical_hb(
    a: float,
    b: float,
    dpdz: float,
    tau_y: float,
    consistency_K: float,
    flow_index_n: float,
    nx: int = 129,
    ny: int = 129,
    m_reg: float = 1000.0,
    gamma_min: float = 1e-6,
    picard_tol: float = 1e-5,
    picard_max_iter: int = 50,
    sor_omega: float = 1.6,
    sor_tol: float = 1e-6,
    sor_max_iter: int = 10000,
) -> EllipticalHBResult:
    """
    Solve steady laminar pressure-driven flow of a Herschel–Bulkley fluid
    in an elliptical pipe of semi-axes a (x-direction) and b (y-direction).

    Governing equation for unidirectional axial velocity w(x,y):
        div( μ_app(|grad w|) grad w ) = -dp/dz  [Pa/m]
    with w = 0 on boundary.

    We employ Papanastasiou regularization for the yield term and a Picard
    iteration on μ_app, while solving the variable-coefficient Poisson equation
    at each step via SOR.

    Parameters
    - a, b: semi-axes [m]
    - dpdz: pressure gradient ∂p/∂z [Pa/m] (negative for positive flow)
    - tau_y, consistency_K, flow_index_n: HB parameters
    - nx, ny: grid resolution (odd numbers recommended)
    - m_reg: Papanastasiou regularization parameter [1/s]
    - gamma_min: min shear rate used in μ_app to avoid singularity [1/s]
    - picard_tol: relative change tolerance on w
    - picard_max_iter: max Picard iterations
    - sor_omega: SOR relaxation factor (1<ω<2)
    - sor_tol: SOR residual tolerance
    - sor_max_iter: max SOR iterations per Picard step

    Returns EllipticalHBResult with velocity field, flow rate, etc.
    """
    assert a > 0 and b > 0
    # Positive driving term G = -dpdz
    G = -float(dpdz)
    if G <= 0:
        raise ValueError("dpdz must be negative (pressure drop along +z).")

    # Grid and mask for ellipse: domain [-a, a] x [-b, b]
    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = ((X / a) ** 2 + (Y / b) ** 2) <= 1.0
    mask = mask.astype(float)

    # Initialize velocity with Newtonian-like quadratic profile scaled by K (if n~1)
    w = np.zeros((ny, nx), dtype=float)
    quad = 1.0 - (X / a) ** 2 - (Y / b) ** 2
    quad[quad < 0.0] = 0.0
    # If n ~ 1, K approximates μ
    if abs(flow_index_n - 1.0) < 1e-6 and consistency_K > 0:
        mu = consistency_K
        w_newtonian = (G / (2.0 * mu)) * (a * a * b * b) / (a * a + b * b) * quad
        w = w_newtonian
    w *= mask

    # Picard iterations
    for picard_iter in range(1, picard_max_iter + 1):
        wx, wy = _compute_gradients_centered(w, hx, hy, mask)
        gamma_dot = np.hypot(wx, wy)
        mu_app = hb_regularized_apparent_viscosity(
            gamma_dot, tau_y, consistency_K, flow_index_n, m_reg=m_reg, gamma_min=gamma_min
        )

        # Build SOR iteration for variable coefficient Poisson: div(mu grad w) = G
        # Precompute face viscosities (harmonic means)
        mu_center = mu_app
        # Shifted arrays for neighbors; pad with own value near boundary
        mu_left = np.roll(mu_center, shift=1, axis=1)
        mu_right = np.roll(mu_center, shift=-1, axis=1)
        mu_down = np.roll(mu_center, shift=1, axis=0)
        mu_up = np.roll(mu_center, shift=-1, axis=0)

        # Harmonic mean for interior; where neighbor is outside mask, use center value
        eps = 1e-30
        hm = lambda a_, b_: 2 * a_ * b_ / (a_ + b_ + eps)
        mu_xm = hm(mu_center, mu_left)
        mu_xp = hm(mu_center, mu_right)
        mu_ym = hm(mu_center, mu_down)
        mu_yp = hm(mu_center, mu_up)

        # Zero out coefficients where neighbor is outside domain
        mask_bool = mask.astype(bool)
        # Neighbor masks
        mask_left = np.roll(mask_bool, shift=1, axis=1)
        mask_right = np.roll(mask_bool, shift=-1, axis=1)
        mask_down = np.roll(mask_bool, shift=1, axis=0)
        mask_up = np.roll(mask_bool, shift=-1, axis=0)

        mu_xm = np.where(mask_left, mu_xm, mu_center)
        mu_xp = np.where(mask_right, mu_xp, mu_center)
        mu_ym = np.where(mask_down, mu_ym, mu_center)
        mu_yp = np.where(mask_up, mu_yp, mu_center)

        # Precompute coefficients
        Axm = mu_xm / (hx * hx)
        Axp = mu_xp / (hx * hx)
        Aym = mu_ym / (hy * hy)
        Ayp = mu_yp / (hy * hy)
        Aii = Axm + Axp + Aym + Ayp + 1e-30

        # Right-hand side
        bvec = G * mask

        # SOR iterations on w
        # We sweep only interior masked points
        resid = np.inf
        sor_iter = 0
        while resid > sor_tol and sor_iter < sor_max_iter:
            max_delta = 0.0
            sor_iter += 1
            # Red-black ordering could speed up; here simple Gauss-Seidel SOR
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if mask_bool[j, i]:
                        # Neighbor values; outside mask treated as 0
                        w_l = w[j, i - 1] if mask_bool[j, i - 1] else 0.0
                        w_r = w[j, i + 1] if mask_bool[j, i + 1] else 0.0
                        w_d = w[j - 1, i] if mask_bool[j - 1, i] else 0.0
                        w_u = w[j + 1, i] if mask_bool[j + 1, i] else 0.0

                        rhs = bvec[j, i] + Axm[j, i] * w_l + Axp[j, i] * w_r + Aym[j, i] * w_d + Ayp[j, i] * w_u
                        w_new = (1.0 - sor_omega) * w[j, i] + sor_omega * (rhs / Aii[j, i])
                        delta = abs(w_new - w[j, i])
                        if delta > max_delta:
                            max_delta = delta
                        w[j, i] = w_new
            resid = max_delta

        # Convergence check for Picard loop
        if picard_iter == 1:
            prev_w = w.copy()
            continue
        rel_change = np.linalg.norm((w - prev_w) * mask) / (np.linalg.norm(prev_w * mask) + 1e-30)
        if rel_change < picard_tol:
            # Compute Q and return
            area = np.pi * a * b
            Q = float(np.sum(w * mask) * hx * hy)
            mean_u = Q / area
            return EllipticalHBResult(x=x, y=y, w=w * mask, Q=Q, mean_velocity=mean_u, iterations=picard_iter, converged=True, meta={
                "G": G,
                "dpdz": dpdz,
                "sor_iterations_last": sor_iter,
            })
        prev_w = w.copy()

    # If not converged, still compute Q
    area = np.pi * a * b
    Q = float(np.sum(w * mask) * hx * hy)
    mean_u = Q / area
    return EllipticalHBResult(x=x, y=y, w=w * mask, Q=Q, mean_velocity=mean_u, iterations=picard_max_iter, converged=False, meta={
        "G": G,
        "dpdz": dpdz,
        "note": "Picard did not converge",
    })


def analytic_newtonian_ellipse_profile(a: float, b: float, dpdz: float, mu: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytic velocity profile for Newtonian fluid in elliptical duct:
    w(x,y) = ( -dp/dz / (2 μ) ) * (a^2 b^2)/(a^2 + b^2) * (1 - x^2/a^2 - y^2/b^2)
    """
    G = -float(dpdz)
    quad = 1.0 - (x / a) ** 2 - (y / b) ** 2
    quad = np.maximum(quad, 0.0)
    return (G / (2.0 * mu)) * (a * a * b * b) / (a * a + b * b) * quad
