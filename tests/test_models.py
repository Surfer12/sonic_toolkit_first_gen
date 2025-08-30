import numpy as np
from hbflow.models import hb_tau_from_gamma, hb_gamma_from_tau, hb_apparent_viscosity, hb_regularized_apparent_viscosity


def test_forward_inverse_consistency():
    tau_y, K, n = 5.0, 2.0, 0.8
    gamma = np.logspace(-3, 3, 50)
    tau = hb_tau_from_gamma(gamma, tau_y, K, n)
    gamma_back = hb_gamma_from_tau(tau, tau_y, K, n)
    # For tau >= tau_y, inverse should recover gamma (within tolerance)
    assert np.allclose(gamma, gamma_back, rtol=1e-6, atol=1e-12)


def test_apparent_viscosity_limits():
    tau_y, K, n = 3.0, 1.5, 1.0
    gamma = np.array([1e-12, 1e-3, 1.0])
    mu_app = hb_apparent_viscosity(gamma, tau_y, K, n)
    assert mu_app.shape == gamma.shape


def test_regularized_viscosity_positive():
    tau_y, K, n = 10.0, 0.5, 0.5
    gamma = np.logspace(-6, 2, 40)
    mu_reg = hb_regularized_apparent_viscosity(gamma, tau_y, K, n, m_reg=500.0)
    assert np.all(mu_reg > 0)
