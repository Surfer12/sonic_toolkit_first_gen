import numpy as np
from hbflow.fit import fit_herschel_bulkley, SCIPY_AVAILABLE


def test_fit_recovery_noiseless():
    rng = np.random.default_rng(0)
    tau_y_true, K_true, n_true = 5.0, 2.0, 0.8
    gamma = np.logspace(-3, 3, 200)
    tau = tau_y_true + K_true * gamma ** n_true

    res = fit_herschel_bulkley(gamma, tau, initial=(1.0, 1.0, 1.0), bootstrap_samples=0)
    p = res["params"]
    if SCIPY_AVAILABLE:
        tol = 1e-2
    else:
        # Fallback grid search is coarse
        tol = 2e-1
    assert abs(p.tau_y - tau_y_true) / tau_y_true < tol
    assert abs(p.consistency_K - K_true) / K_true < tol
    assert abs(p.flow_index_n - n_true) / n_true < tol
