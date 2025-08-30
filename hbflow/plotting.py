from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .models import hb_tau_from_gamma, hb_gamma_from_tau


def plot_stress_rate_curve(
    tau_y: float,
    consistency_K: float,
    flow_index_n: float,
    gamma_min: float = 1e-3,
    gamma_max: float = 1e3,
    num: int = 200,
    data_gamma: Optional[np.ndarray] = None,
    data_tau: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    gamma = np.logspace(np.log10(gamma_min), np.log10(gamma_max), num=num)
    tau = hb_tau_from_gamma(gamma, tau_y, consistency_K, flow_index_n)
    ax.loglog(gamma, tau, label="HB model")
    if data_gamma is not None and data_tau is not None:
        ax.scatter(data_gamma, data_tau, s=12, color="C1", alpha=0.7, label="data")
    ax.set_xlabel("Shear rate γ̇ [1/s]")
    ax.set_ylabel("Shear stress τ [Pa]")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    return ax


def plot_velocity_profile(x: np.ndarray, y: np.ndarray, w: np.ndarray, a: float, b: float, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = ((X / a) ** 2 + (Y / b) ** 2) <= 1.0
    w_plot = np.where(mask, w, np.nan)
    im = ax.pcolormesh(X, Y, w_plot, shading="auto", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Velocity w [m/s]")
    return ax
