from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike

try:
    # SciPy may not be available in all environments; we guard imports
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False
    least_squares = None  # type: ignore


@dataclass
class HBParams:
    tau_y: float  # [Pa]
    consistency_K: float  # [Pa·s^n]
    flow_index_n: float  # [-]


def _hb_residuals(params_vec: np.ndarray, gamma_dot: np.ndarray, tau_obs: np.ndarray) -> np.ndarray:
    tau_y, K, n = params_vec
    pred = tau_y + K * np.power(np.maximum(gamma_dot, 0.0), n)
    return pred - tau_obs


def fit_herschel_bulkley(
    gamma_dot: ArrayLike,
    tau: ArrayLike,
    initial: Optional[Tuple[float, float, float]] = None,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.0, 1e-12, 0.1), (1e6, 1e6, 2.0)),
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 10000,
    bootstrap_samples: int = 0,
    random_state: Optional[int] = None,
) -> dict:
    """
    Fit Herschel–Bulkley parameters (τ_y, K, n) to rheometer data.

    Parameters
    - gamma_dot: array of shear rates γ̇ [1/s]
    - tau: array of shear stresses τ [Pa]
    - initial: optional initial guess (τ_y, K, n). If None, estimated from data.
    - bounds: lower and upper bounds ((τ_y_min, K_min, n_min), (τ_y_max, K_max, n_max))
    - robust_loss: loss function for robust fitting (e.g., 'linear', 'soft_l1', 'huber', 'cauchy')
    - f_scale: scaling parameter for robust loss
    - max_nfev: maximum function evaluations
    - bootstrap_samples: number of bootstrap resamples for uncertainty; 0 disables
    - random_state: RNG seed for bootstrap

    Returns
    - dict with keys: 'params' (HBParams), 'cov' (3x3 covariance or None),
      'stderr', 'cv', 'r2', 'loss', and optionally 'bootstrap' stats
    """
    g = np.asarray(gamma_dot, dtype=float).ravel()
    t = np.asarray(tau, dtype=float).ravel()
    assert g.shape == t.shape, "gamma_dot and tau must have same shape"

    if initial is None:
        # Heuristic initial guess
        tau_y0 = max(0.0, float(np.percentile(t - t.min(), 5)))
        # Avoid zeros
        positive = g > 0
        if np.any(positive):
            # Estimate n from slope in log-log after subtracting tau_y0 where possible
            g_pos = g[positive]
            t_pos = t[positive]
            # Rough K,n via linear regression on log-log
            y = np.log(np.maximum(t_pos - tau_y0, 1e-12))
            x = np.log(g_pos)
            A = np.vstack([np.ones_like(x), x]).T
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            n0 = float(max(0.2, min(1.5, coef[1])))
            K0 = float(np.exp(coef[0]))
        else:
            n0 = 1.0
            K0 = max(1e-3, float(np.median(t)))
        initial = (tau_y0, K0, n0)

    if not SCIPY_AVAILABLE:
        # Fallback: simple grid search around initial (coarse). Not as accurate as SciPy.
        tau_y0, K0, n0 = initial
        best = (np.inf, (tau_y0, K0, n0))
        for tau_y in np.linspace(max(bounds[0][0], 0.5 * tau_y0), min(bounds[1][0], 2.0 * tau_y0 + 1e-9), 9):
            for K in np.logspace(np.log10(max(bounds[0][1], K0 / 10)), np.log10(min(bounds[1][1], K0 * 10 + 1e-9)), 9):
                for n in np.linspace(max(bounds[0][2], 0.5 * n0), min(bounds[1][2], 1.5 * n0), 9):
                    res = _hb_residuals(np.array([tau_y, K, n]), g, t)
                    cost = np.sum(res * res)
                    if cost < best[0]:
                        best = (cost, (tau_y, K, n))
        params_vec = np.array(best[1])
        jac = None
        cost = best[0]
    else:
        result = least_squares(
            fun=_hb_residuals,
            x0=np.array(initial, dtype=float),
            bounds=(np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)),
            args=(g, t),
            loss=robust_loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        params_vec = result.x
        cost = 2 * result.cost  # SciPy reports 1/2 sum(res^2)
        # Approximate covariance from Jacobian at solution (if available and well-conditioned)
        jac = result.jac if hasattr(result, "jac") else None

    tau_y, K, n = params_vec.tolist()

    # R^2
    pred = tau_y + K * np.power(np.maximum(g, 0.0), n)
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    cov = None
    stderr = np.array([np.nan, np.nan, np.nan])
    if SCIPY_AVAILABLE and jac is not None and jac.size:
        # Covariance: σ^2 (J^T J)^{-1}, with σ^2 = SSR/(N - p)
        J = np.asarray(jac)
        dof = max(1, g.size - 3)
        sigma2 = ss_res / dof
        try:
            JTJ = J.T @ J
            cov = sigma2 * np.linalg.pinv(JTJ)
            stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except Exception:
            cov = None

    # Coefficient of variation (CV = stderr/estimate)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = stderr / np.array([tau_y, K, n])

    out = {
        "params": HBParams(tau_y=tau_y, consistency_K=K, flow_index_n=n),
        "cov": cov,
        "stderr": stderr,
        "cv": cv,
        "r2": r2,
        "loss": cost,
    }

    if bootstrap_samples and SCIPY_AVAILABLE:
        rng = np.random.default_rng(random_state)
        boot = []
        residuals = t - pred
        for _ in range(bootstrap_samples):
            # Resample residuals (paired bootstrap also possible)
            t_boot = pred + rng.choice(residuals, size=residuals.size, replace=True)
            r = least_squares(
                fun=_hb_residuals,
                x0=params_vec,
                bounds=(np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)),
                args=(g, t_boot),
                loss=robust_loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
            )
            boot.append(r.x)
        boot_arr = np.asarray(boot)
        out["bootstrap"] = {
            "samples": boot_arr,
            "mean": np.mean(boot_arr, axis=0),
            "std": np.std(boot_arr, axis=0, ddof=1),
            "cov": np.std(boot_arr, axis=0, ddof=1) / np.mean(boot_arr, axis=0),
        }

    return out
