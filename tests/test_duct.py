import numpy as np
from hbflow.duct import solve_elliptical_hb, analytic_newtonian_ellipse_profile


def test_newtonian_limit_matches_analytic():
    a, b = 0.01, 0.02
    mu = 1.0
    dpdz = -100.0
    # Set HB to Newtonian: tau_y=0, n=1, K=mu
    res = solve_elliptical_hb(
        a,
        b,
        dpdz,
        tau_y=0.0,
        consistency_K=mu,
        flow_index_n=1.0,
        nx=49,
        ny=49,
        picard_max_iter=3,
        sor_tol=1e-5,
        sor_max_iter=2000,
    )
    X, Y = np.meshgrid(res.x, res.y, indexing="xy")
    w_analytic = analytic_newtonian_ellipse_profile(a, b, dpdz, mu, X, Y)
    # Compare normalized profiles to reduce sensitivity to discretization
    mask = ((X / a) ** 2 + (Y / b) ** 2) <= 1.0
    w_num = res.w[mask]
    w_an = w_analytic[mask]
    w_num /= np.max(w_num)
    w_an /= np.max(w_an)
    assert np.allclose(w_num, w_an, atol=0.1)


def test_flow_rate_positive():
    a, b = 0.01, 0.01
    dpdz = -100.0
    res = solve_elliptical_hb(a, b, dpdz, tau_y=1.0, consistency_K=1.0, flow_index_n=0.8, nx=33, ny=33)
    assert res.Q > 0
