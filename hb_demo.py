#!/usr/bin/env python3
"""
Herschel-Bulkley Model Demonstration

This script demonstrates the Herschel-Bulkley fluid model implementation
with examples of different fluid types and flow calculations.

Author: Ryan David Oates
Date: August 26, 2025
License: GPL-3.0-only
"""

import numpy as np
import matplotlib.pyplot as plt
from herschel_bulkley_model import (
    HBParameters, HerschelBulkleyModel,
    EllipticalDuctFlowSolver, HBVisualizer
)


def demo_fluid_types():
    """Demonstrate different fluid types."""
    print("🔬 Herschel-Bulkley Fluid Model Demonstration")
    print("=" * 60)

    # Define different fluid types
    fluids = {
        'Newtonian': HBParameters(tau_y=0.0, K=1.0, n=1.0),
        'Bingham Plastic': HBParameters(tau_y=2.0, K=1.5, n=1.0),
        'Shear-thinning HB': HBParameters(tau_y=1.0, K=2.0, n=0.6),
        'Power-law': HBParameters(tau_y=0.0, K=1.8, n=0.7)
    }

    # Shear rate range
    gamma_dot = np.logspace(-2, 2, 100)

    print("Fluid Type Analysis:")
    print("-" * 40)

    for name, params in fluids.items():
        model = HerschelBulkleyModel(params)
        info = model.get_model_info()

        # Calculate properties at reference shear rate
        gamma_ref = 1.0
        tau = model.constitutive_model(gamma_ref)
        eta = model.apparent_viscosity(gamma_ref)

        print("15")
        print(f"    Parameters: τy={params.tau_y:.1f} Pa, K={params.K:.1f} Pa·s^n, n={params.n:.1f}")
        print(f"    At γ̇={gamma_ref:.1f} 1/s: τ={tau:.2f} Pa, η={eta:.2f} Pa·s")
        print()

    return fluids


def demo_flow_calculations():
    """Demonstrate flow calculations in elliptical ducts."""
    print("🌊 Elliptical Duct Flow Calculations")
    print("=" * 50)

    # Define test fluids
    newtonian = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.0, n=1.0))
    bingham = HerschelBulkleyModel(HBParameters(tau_y=2.0, K=1.5, n=1.0))

    # Duct geometry
    a, b = 0.01, 0.005  # Semi-major and semi-minor axes [m]
    dp_dx_values = [-5000, -2000, -500, -100]  # Pressure gradients [Pa/m]

    print(f"Duct geometry: a={a*1000:.0f} mm, b={b*1000:.0f} mm (elliptical)")
    print()

    for fluid_name, model in [('Newtonian', newtonian), ('Bingham', bingham)]:
        print(f"{fluid_name} Fluid Results:")
        print("-" * 30)

        solver = EllipticalDuctFlowSolver(model, a, b)

        for dp_dx in dp_dx_values:
            result = solver.calculate_flow_rate(dp_dx)

            if result['Q'] > 0:
                print("6.0f"
                      "5.0f")
            else:
                print("6.0f")
        print()


def demo_rheological_plots():
    """Generate rheological plots."""
    print("📊 Generating Rheological Plots...")
    print("-" * 40)

    # Create models
    models = {
        'Newtonian': HBParameters(tau_y=0.0, K=1.0, n=1.0),
        'Bingham': HBParameters(tau_y=2.0, K=1.5, n=1.0),
        'HB (shear-thinning)': HBParameters(tau_y=1.0, K=2.0, n=0.6)
    }

    visualizer = HBVisualizer()

    # Create plots for each model
    for name, params in models.items():
        model = HerschelBulkleyModel(params)

        # Generate rheogram
        fig = visualizer.plot_rheogram(model, save_path=f'hb_rheogram_{name.lower().replace(" ", "_")}.png')
        plt.close(fig)

        print(f"✅ Generated rheogram for {name}")

    print("All plots saved to current directory.")
    print()


def demo_api_usage():
    """Demonstrate API usage."""
    print("🔧 API Usage Examples")
    print("=" * 30)

    # Create a Herschel-Bulkley model
    params = HBParameters(tau_y=1.5, K=2.0, n=0.8)
    model = HerschelBulkleyModel(params)

    print("Model parameters:")
    print(f"  Yield stress (τy): {params.tau_y} Pa")
    print(f"  Consistency index (K): {params.K} Pa·s^n")
    print(f"  Flow behavior index (n): {params.n}")
    print()

    # Example calculations
    gamma_dot_test = 2.0
    tau_test = 5.0

    # Forward calculation
    tau_calculated = model.constitutive_model(gamma_dot_test)
    print("2.1f")

    # Inverse calculation
    gamma_dot_calculated = model.inverse_model(tau_test)
    print("2.1f")

    # Vectorized operations
    gamma_dot_array = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    tau_array = model.constitutive_model(gamma_dot_array)

    print("\nVectorized calculations:")
    print("6s")
    for gd, tau in zip(gamma_dot_array, tau_array):
        print("6.1f")
    print()

    # Model information
    info = model.get_model_info()
    print("Model classification:")
    print(f"  Behavior: {info['behavior']}")
    print(f"  Type: {info['model_type']}")
    print(f"  Newtonian: {info['is_newtonian']}")
    print(f"  Power-law: {info['is_power_law']}")
    print()


def demo_duct_flow():
    """Demonstrate duct flow calculations."""
    print("🔄 Duct Flow Analysis")
    print("=" * 30)

    # Create a shear-thinning fluid
    params = HBParameters(tau_y=2.0, K=1.5, n=0.8)
    model = HerschelBulkleyModel(params)

    print("Fluid: Herschel-Bulkley (shear-thinning)")
    print(f"Parameters: τy={params.tau_y} Pa, K={params.K} Pa·s^n, n={params.n}")
    print()

    # Duct geometry
    a, b = 0.01, 0.005  # 10mm x 5mm elliptical duct
    solver = EllipticalDuctFlowSolver(model, a, b)

    # Test different pressure gradients
    dp_dx_range = np.logspace(3, 5, 10)  # 1000 to 100000 Pa/m
    flow_rates = []
    wall_stresses = []
    valid_dp_dx = []

    print("Pressure Gradient Analysis:")
    print("8s")
    print("-" * 50)

    for dp_dx in dp_dx_range:
        result = solver.calculate_flow_rate(-dp_dx)  # Negative for flow direction

        if result['Q'] > 0:
            print("8.0f")
            flow_rates.append(result['Q'])
            wall_stresses.append(result['wall_shear_stress'])
            valid_dp_dx.append(dp_dx)
        else:
            print("8.0f")

    # Use only valid pressure gradients for plotting
    if len(flow_rates) == 0:
        print("\nNo flow occurred for any pressure gradient - fluid always yielded")
        return
    print()

    # Plot flow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(valid_dp_dx, flow_rates, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Pressure Gradient |dp/dx| (Pa/m)')
    ax.set_ylabel('Flow Rate Q (m³/s)')
    ax.set_title('Flow Rate vs Pressure Gradient\nElliptical Duct, HB Fluid')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95,
           '.1f',
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           verticalalignment='top')

    plt.tight_layout()
    plt.savefig('hb_flow_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Flow curve saved as 'hb_flow_curve.png'")


def main():
    """Run all demonstrations."""
    try:
        demo_fluid_types()
        demo_api_usage()
        demo_flow_calculations()
        demo_duct_flow()
        demo_rheological_plots()

        print("🎉 All demonstrations completed successfully!")
        print("\\nGenerated files:")
        print("- hb_rheogram_*.png (rheological plots)")
        print("- hb_flow_curve.png (flow analysis)")
        print("\\nTo run individual components:")
        print("  python3 herschel_bulkley_model.py constitutive --gamma-dot 2.0")
        print("  python3 herschel_bulkley_model.py flow --dp-dx -5000")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
