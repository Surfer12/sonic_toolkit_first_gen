from __future__ import annotations

import argparse
import sys
import json
import numpy as np

from .models import hb_gamma_from_tau
from .fit import fit_herschel_bulkley
from .duct import solve_elliptical_hb
from .advanced_duct import solve_elliptical_hb_pde
from .validation import validate_limits, benchmark_against_analytical
from .rheometry import simulate_rheometer_data, add_noise_to_data
from .plotting import plot_stress_rate_curve, plot_velocity_profile


def cmd_predict(args: argparse.Namespace) -> int:
    tau = np.array(args.tau, dtype=float)
    gd = hb_gamma_from_tau(tau, args.tau_y, args.K, args.n)
    print(json.dumps({"gamma_dot": gd.tolist()}, indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import pandas as pd
    df = pd.read_csv(args.csv)
    g = df[args.gamma_col].to_numpy(dtype=float)
    t = df[args.tau_col].to_numpy(dtype=float)
    result = fit_herschel_bulkley(g, t, bootstrap_samples=args.bootstrap)
    params = result["params"]
    out = {
        "tau_y": params.tau_y,
        "K": params.consistency_K,
        "n": params.flow_index_n,
        "stderr": result.get("stderr").tolist() if result.get("stderr") is not None else None,
        "cv": result.get("cv").tolist() if result.get("cv") is not None else None,
        "r2": result.get("r2"),
    }
    print(json.dumps(out, indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        ax = plot_stress_rate_curve(params.tau_y, params.consistency_K, params.flow_index_n, data_gamma=g, data_tau=t)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    res = solve_elliptical_hb(
        a=args.a,
        b=args.b,
        dpdz=args.dpdz,
        tau_y=args.tau_y,
        consistency_K=args.K,
        flow_index_n=args.n,
        nx=args.nx,
        ny=args.ny,
        m_reg=args.m_reg,
        gamma_min=args.gamma_min,
    )
    print(json.dumps({
        "Q_m3_per_s": res.Q,
        "mean_velocity_m_per_s": res.mean_velocity,
        "iterations": res.iterations,
        "converged": res.converged,
    }, indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        ax = plot_velocity_profile(res.x, res.y, res.w, args.a, args.b)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
    return 0


def cmd_pde_flow(args: argparse.Namespace) -> int:
    from .advanced_duct import solve_elliptical_hb_pde

    res = solve_elliptical_hb_pde(
        a=args.a,
        b=args.b,
        dpdz=args.dpdz,
        tau_y=args.tau_y,
        consistency_K=args.K,
        flow_index_n=args.n,
        nx=args.nx,
        ny=args.ny,
        m_reg=args.m_reg,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )
    print(json.dumps({
        "Q_m3_per_s": res.Q,
        "mean_velocity_m_per_s": res.mean_velocity,
        "iterations": res.iterations,
        "converged": res.converged,
        "residual": res.residual,
    }, indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        ax = plot_velocity_profile(res.x, res.y, res.w, args.a, args.b)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .validation import validate_limits

    results = validate_limits(
        tau_y=args.tau_y,
        K=args.K,
        n=args.n,
        gamma_dot_test=args.gamma_dot_test,
        tau_test=args.tau_test
    )
    print(json.dumps(results, indent=2))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .rheometry import simulate_rheometer_data, add_noise_to_data
    import pandas as pd

    # Simulate data
    data = simulate_rheometer_data(
        tau_y=args.tau_y,
        K=args.K,
        n=args.n,
        num_points=args.points,
        measurement_type=args.type
    )

    # Add noise
    gamma_dot_noisy, tau_noisy = add_noise_to_data(
        data['gamma_dot'],
        data['tau'],
        noise_type=args.noise_type,
        noise_level=args.noise_level
    )

    # Prepare output
    output_data = {
        'gamma_dot': gamma_dot_noisy.tolist(),
        'tau': tau_noisy.tolist(),
        'time': data['time'].tolist(),
        'measurement_type': data['measurement_type'],
        'true_parameters': data['parameters']
    }

    print(json.dumps(output_data, indent=2))

    # Save to CSV if requested
    if args.output:
        df = pd.DataFrame({
            'gamma_dot': gamma_dot_noisy,
            'tau': tau_noisy,
            'time': data['time']
        })
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hbflow", description="Herschel–Bulkley utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Compute γ̇ from τ for HB")
    p_pred.add_argument("--tau", type=float, nargs="+", required=True, help="Shear stress values [Pa]")
    p_pred.add_argument("--tau-y", dest="tau_y", type=float, required=True, help="Yield stress τ_y [Pa]")
    p_pred.add_argument("--K", type=float, required=True, help="Consistency K [Pa·s^n]")
    p_pred.add_argument("--n", type=float, required=True, help="Flow index n [-]")
    p_pred.set_defaults(func=cmd_predict)

    p_fit = sub.add_parser("fit", help="Fit HB parameters from CSV data")
    p_fit.add_argument("--csv", type=str, required=True, help="CSV with columns for shear rate and stress")
    p_fit.add_argument("--gamma-col", type=str, default="gamma_dot", help="Column name for shear rate")
    p_fit.add_argument("--tau-col", type=str, default="tau", help="Column name for shear stress")
    p_fit.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples for uncertainty (requires SciPy)")
    p_fit.add_argument("--plot", type=str, default=None, help="Path to save stress-rate plot")
    p_fit.set_defaults(func=cmd_fit)

    p_flow = sub.add_parser("flow", help="Solve elliptical duct HB flow")
    p_flow.add_argument("--a", type=float, required=True, help="Ellipse semi-axis a [m]")
    p_flow.add_argument("--b", type=float, required=True, help="Ellipse semi-axis b [m]")
    p_flow.add_argument("--dpdz", type=float, required=True, help="Pressure gradient ∂p/∂z [Pa/m] (negative)")
    p_flow.add_argument("--tau-y", dest="tau_y", type=float, required=True, help="Yield stress τ_y [Pa]")
    p_flow.add_argument("--K", type=float, required=True, help="Consistency K [Pa·s^n]")
    p_flow.add_argument("--n", type=float, required=True, help="Flow index n [-]")
    p_flow.add_argument("--nx", type=int, default=129, help="Grid points in x")
    p_flow.add_argument("--ny", type=int, default=129, help="Grid points in y")
    p_flow.add_argument("--m-reg", dest="m_reg", type=float, default=1000.0, help="Papanastasiou regularization m [1/s]")
    p_flow.add_argument("--gamma-min", dest="gamma_min", type=float, default=1e-6, help="Min shear rate for μ_app [1/s]")
    p_flow.add_argument("--plot", type=str, default=None, help="Path to save velocity plot")
    p_flow.set_defaults(func=cmd_flow)

    # Advanced PDE flow solver
    p_pde = sub.add_parser("pde-flow", help="Solve elliptical HB flow using advanced PDE method")
    p_pde.add_argument("--a", type=float, required=True, help="Ellipse semi-axis a [m]")
    p_pde.add_argument("--b", type=float, required=True, help="Ellipse semi-axis b [m]")
    p_pde.add_argument("--dpdz", type=float, required=True, help="Pressure gradient ∂p/∂z [Pa/m] (negative)")
    p_pde.add_argument("--tau-y", dest="tau_y", type=float, required=True, help="Yield stress τ_y [Pa]")
    p_pde.add_argument("--K", type=float, required=True, help="Consistency K [Pa·s^n]")
    p_pde.add_argument("--n", type=float, required=True, help="Flow index n [-]")
    p_pde.add_argument("--nx", type=int, default=65, help="Grid points in x")
    p_pde.add_argument("--ny", type=int, default=65, help="Grid points in y")
    p_pde.add_argument("--m-reg", dest="m_reg", type=float, default=1000.0, help="Papanastasiou regularization m [1/s]")
    p_pde.add_argument("--max-iter", dest="max_iter", type=int, default=1000, help="Maximum iterations")
    p_pde.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    p_pde.add_argument("--plot", type=str, default=None, help="Path to save velocity plot")
    p_pde.set_defaults(func=cmd_pde_flow)

    # Validation commands
    p_validate = sub.add_parser("validate", help="Validate HB model against limits")
    p_validate.add_argument("--tau-y", dest="tau_y", type=float, required=True, help="Yield stress τ_y [Pa]")
    p_validate.add_argument("--K", type=float, required=True, help="Consistency K [Pa·s^n]")
    p_validate.add_argument("--n", type=float, required=True, help="Flow index n [-]")
    p_validate.add_argument("--gamma-dot-test", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0], help="Test shear rates [1/s]")
    p_validate.add_argument("--tau-test", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0], help="Test shear stresses [Pa]")
    p_validate.set_defaults(func=cmd_validate)

    # Rheometer data simulation
    p_simulate = sub.add_parser("simulate", help="Simulate rheometer data")
    p_simulate.add_argument("--tau-y", dest="tau_y", type=float, required=True, help="Yield stress τ_y [Pa]")
    p_simulate.add_argument("--K", type=float, required=True, help="Consistency K [Pa·s^n]")
    p_simulate.add_argument("--n", type=float, required=True, help="Flow index n [-]")
    p_simulate.add_argument("--points", type=int, default=50, help="Number of measurement points")
    p_simulate.add_argument("--type", choices=["controlled_shear_rate", "controlled_stress"], default="controlled_shear_rate", help="Measurement type")
    p_simulate.add_argument("--noise-type", choices=["proportional", "absolute", "combined"], default="combined", help="Type of noise to add")
    p_simulate.add_argument("--noise-level", type=float, default=0.05, help="Noise level")
    p_simulate.add_argument("--output", type=str, default=None, help="Output CSV file")
    p_simulate.set_defaults(func=cmd_simulate)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
