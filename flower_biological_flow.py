"""
🌸 Flower-Inspired Biological Flow Analysis
==========================================

Advanced multi-phase flow analysis inspired by natural biological systems:
- Plant vascular transport (xylem/phloem)
- Nutrient flow in complex networks
- Biological pattern formation
- Flower-like fluid dynamics

Author: Ryan David Oates
Date: Current
License: GPL-3.0-only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import time
from typing import Dict, List, Tuple, Optional
import warnings

# Import our multi-phase flow framework
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from multi_phase_flow_analysis import (
        MultiPhaseFlowAnalyzer, MultiPhaseFlowConfig,
        PhaseProperties, InterfaceMethod
    )
except ImportError:
    print("Warning: Multi-phase flow framework not found.")
    print("Please run multi_phase_flow_analysis.py first.")


class BiologicalFlowSystem:
    """
    Biological flow system inspired by plant vascular networks and flowers.

    Models complex transport phenomena found in:
    - Plant stems (xylem/phloem)
    - Leaf venation patterns
    - Flower petal fluid dynamics
    - Root system nutrient transport
    """

    def __init__(self):
        self.flow_analyzer = None
        self.vascular_network = {}
        self.nutrient_concentrations = {}
        print("🌸 Biological Flow System initialized")
        print("   Modeling plant vascular transport and flower-like patterns")

    def create_plant_stem_model(self) -> MultiPhaseFlowAnalyzer:
        """
        Create a model of fluid transport in a plant stem.

        Features:
        - Xylem (water transport) - low viscosity, high volume
        - Phloem (nutrient transport) - higher viscosity, complex rheology
        - Vascular bundle interfaces
        - Transpiration-driven flow
        """
        print("\n🌱 MODELING PLANT STEM VASCULAR SYSTEM")

        # Plant stem dimensions (2cm diameter, 10cm height)
        config = MultiPhaseFlowConfig(
            domain_size=(0.02, 0.10),  # 2cm × 10cm
            grid_points=(40, 200),
            interface_method=InterfaceMethod.VOF,
            surface_tension=0.055,  # Biological surface tension
            time_step=5e-5,
            max_time=0.5
        )

        analyzer = MultiPhaseFlowAnalyzer(config)

        # Xylem sap (water transport) - low viscosity Newtonian fluid
        xylem_sap = PhaseProperties(
            name="Xylem Sap",
            density=1000,  # kg/m³
            viscosity=0.001,  # Pa·s (water-like)
            color="lightblue"
        )

        # Phloem sap (nutrient transport) - complex rheological fluid
        phloem_sap = PhaseProperties(
            name="Phloem Sap",
            density=1100,  # kg/m³ (denser due to sugars)
            viscosity=0.05,  # Pa·s (higher viscosity due to sugars/proteins)
            hb_params={
                'tau_y': 2.0,    # Yield stress due to protein networks
                'K': 10.0,       # Consistency index
                'n': 0.6         # Shear-thinning behavior
            },
            color="darkgreen"
        )

        analyzer.add_phase(xylem_sap)
        analyzer.add_phase(phloem_sap)

        print("✅ Plant vascular system model created")
        print("   Xylem: Water transport (Newtonian)")
        print("   Phloem: Nutrient transport (Herschel-Bulkley)")
        print(f"   Domain: {config.domain_size[0]*100:.0f}cm × {config.domain_size[1]*100:.0f}cm")

        return analyzer

    def create_leaf_venation_model(self) -> MultiPhaseFlowAnalyzer:
        """
        Create a model of fluid flow in leaf venation patterns.

        Features:
        - Complex branching network
        - Transpiration-driven flow
        - Stomatal resistance effects
        - Network connectivity
        """
        print("\n🍃 MODELING LEAF VENATION SYSTEM")

        # Leaf section (5cm × 5cm)
        config = MultiPhaseFlowConfig(
            domain_size=(0.05, 0.05),  # 5cm × 5cm
            grid_points=(100, 100),
            interface_method=InterfaceMethod.VOF,
            surface_tension=0.04,
            time_step=1e-5,
            max_time=0.1
        )

        analyzer = MultiPhaseFlowAnalyzer(config)

        # Vascular sap
        vascular_fluid = PhaseProperties(
            name="Vascular Fluid",
            density=1020,
            viscosity=0.002,
            hb_params={'tau_y': 0.5, 'K': 2.0, 'n': 0.7},
            color="green"
        )

        # Intercellular fluid
        intercellular_fluid = PhaseProperties(
            name="Intercellular Fluid",
            density=1010,
            viscosity=0.0015,
            color="lightgreen"
        )

        analyzer.add_phase(vascular_fluid)
        analyzer.add_phase(intercellular_fluid)

        print("✅ Leaf venation system model created")
        print("   Vascular bundles: Complex rheology")
        print("   Intercellular spaces: Lower viscosity")
        print(f"   Domain: {config.domain_size[0]*100:.0f}cm × {config.domain_size[1]*100:.0f}cm")

        return analyzer

    def create_flower_petal_model(self) -> MultiPhaseFlowAnalyzer:
        """
        Create a model of fluid dynamics in flower petals.

        Features:
        - Petal surface tension effects
        - Color pigment transport
        - Structural fluid dynamics
        - Environmental interaction
        """
        print("\n🌺 MODELING FLOWER PETAL FLUID DYNAMICS")

        # Petal section (1cm × 0.5cm)
        config = MultiPhaseFlowConfig(
            domain_size=(0.01, 0.005),  # 1cm × 0.5cm
            grid_points=(50, 25),
            interface_method=InterfaceMethod.VOF,
            surface_tension=0.035,  # Petal surface tension
            time_step=2e-6,
            max_time=0.05
        )

        analyzer = MultiPhaseFlowAnalyzer(config)

        # Cellular fluid (main petal fluid)
        cellular_fluid = PhaseProperties(
            name="Cellular Fluid",
            density=1040,
            viscosity=0.008,
            hb_params={'tau_y': 1.0, 'K': 5.0, 'n': 0.65},
            viscoelastic_params={'relaxation_time': 0.1, 'elastic_modulus': 2000},
            color="pink"
        )

        # Pigment solution
        pigment_fluid = PhaseProperties(
            name="Pigment Solution",
            density=1080,
            viscosity=0.015,
            hb_params={'tau_y': 3.0, 'K': 15.0, 'n': 0.55},
            color="red"
        )

        analyzer.add_phase(cellular_fluid)
        analyzer.add_phase(pigment_fluid)

        print("✅ Flower petal model created")
        print("   Cellular fluid: Viscoelastic properties")
        print("   Pigment solution: Higher viscosity, yield stress")
        print(f"   Domain: {config.domain_size[0]*100:.1f}cm × {config.domain_size[1]*100:.1f}cm")

        return analyzer

    def simulate_biological_transport(self, analyzer: MultiPhaseFlowAnalyzer,
                                    system_type: str, inlet_velocity: float = 0.001):
        """
        Simulate biological transport for different system types.

        Args:
            analyzer: MultiPhaseFlowAnalyzer instance
            system_type: Type of biological system
            inlet_velocity: Inlet velocity (m/s)
        """
        print(f"\n🚀 SIMULATING {system_type.upper()} BIOLOGICAL TRANSPORT")

        # Initialize geometry based on system type
        if "PLANT STEM" in system_type.upper():
            # Multiple vascular bundles
            centers = [(0.008, 0.02), (0.012, 0.04), (0.006, 0.06)]
            radius = 0.0015

            for i, center in enumerate(centers):
                if i % 2 == 0:
                    analyzer.initialize_droplet((center[0], center[1]), radius, "Xylem Sap", "Phloem Sap")
                else:
                    analyzer.initialize_droplet((center[0], center[1]), radius, "Phloem Sap", "Xylem Sap")

        elif "LEAF" in system_type.upper():
            # Branching venation pattern
            analyzer.initialize_droplet((0.025, 0.025), 0.003, "Vascular Fluid", "Intercellular Fluid")

        elif "FLOWER" in system_type.upper():
            # Petal surface patterns
            analyzer.initialize_droplet((0.005, 0.0025), 0.001, "Pigment Solution", "Cellular Fluid")

        print(f"   Inlet velocity: {inlet_velocity*1000:.1f} mm/s")
        print("   Simulating biological transport patterns...")

        # Solve the flow problem
        start_time = time.time()
        solution = analyzer.solve_flow(inlet_velocity)
        solve_time = time.time() - start_time

        # Analyze results
        analysis = analyzer.analyze_interface_dynamics(solution)

        print("✅ Biological transport simulation completed")
        print(f"   Solution time: {solve_time:.3f} seconds")
        print("   Analysis results:")
        print(f"     Flow rate: {analysis.get('flow_rate', 0)*1e6:.2f} μL/s")
        print(f"     Pressure drop: {analysis.get('pressure_drop', 0):.1f} Pa")
        print(f"     Average stress: {analysis.get('avg_stress', 0):.2f} Pa")

        if 'interface_mean_y' in analysis:
            print(f"     Interface position: {analysis['interface_mean_y']*1000:.1f} mm")

        return solution, analysis

    def visualize_biological_patterns(self, analyzer: MultiPhaseFlowAnalyzer,
                                    solution: Dict, analysis: Dict, system_type: str):
        """
        Create specialized visualization for biological flow patterns.
        """
        print("\n🎨 Generating biological flow visualization...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"images/flower_biological_flow_{system_type.replace(' ', '_').lower()}_{timestamp}.png"

        # Enhanced visualization for biological systems
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🌸 {system_type} - Biological Flow Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Volume fraction with biological interpretation
        im1 = axes[0, 0].imshow(solution['volume_fraction'], extent=[0, analyzer.Lx*100, 0, analyzer.Ly*100],
                               origin='lower', cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_title('Biological Interface (Volume Fraction)', fontweight='bold')
        axes[0, 0].set_xlabel('x [mm]')
        axes[0, 0].set_ylabel('y [mm]')
        plt.colorbar(im1, ax=axes[0, 0], label='Volume Fraction')

        # Add biological annotations
        if "PLANT" in system_type.upper():
            axes[0, 0].text(0.1, 0.9, 'Vascular Bundles', transform=axes[0, 0].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        elif "FLOWER" in system_type.upper():
            axes[0, 0].text(0.1, 0.9, 'Petal Structure', transform=axes[0, 0].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.7))

        # Plot 2: Velocity field with flow interpretation
        im2 = axes[0, 1].imshow(solution['velocity_x']*1000, extent=[0, analyzer.Lx*100, 0, analyzer.Ly*100],
                               origin='lower', cmap='Blues', aspect='auto')
        axes[0, 1].set_title('Transport Velocity (mm/s)', fontweight='bold')
        axes[0, 1].set_xlabel('x [mm]')
        axes[0, 1].set_ylabel('y [mm]')
        plt.colorbar(im2, ax=axes[0, 1], label='Velocity [mm/s]')

        # Plot 3: Rheological stress with biological context
        im3 = axes[0, 2].imshow(solution['stress'], extent=[0, analyzer.Lx*100, 0, analyzer.Ly*100],
                               origin='lower', cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Biological Stress Field', fontweight='bold')
        axes[0, 2].set_xlabel('x [mm]')
        axes[0, 2].set_ylabel('y [mm]')
        plt.colorbar(im3, ax=axes[0, 2], label='Stress [Pa]')

        # Plot 4: Strain rate (deformation)
        im4 = axes[1, 0].imshow(solution['strain_rate'], extent=[0, analyzer.Lx*100, 0, analyzer.Ly*100],
                               origin='lower', cmap='plasma', aspect='auto')
        axes[1, 0].set_title('Deformation Rate (1/s)', fontweight='bold')
        axes[1, 0].set_xlabel('x [mm]')
        axes[1, 0].set_ylabel('y [mm]')
        plt.colorbar(im4, ax=axes[1, 0], label='Strain Rate [1/s]')

        # Plot 5: Flow efficiency analysis
        mid_x = analyzer.nx // 2
        y_coords = np.linspace(0, analyzer.Ly*100, analyzer.ny)
        velocity_profile = solution['velocity_x'][:, mid_x] * 1000

        axes[1, 1].plot(velocity_profile, y_coords, 'g-', linewidth=2, label='Transport Profile')
        axes[1, 1].set_title('Biological Transport Profile', fontweight='bold')
        axes[1, 1].set_xlabel('Velocity [mm/s]')
        axes[1, 1].set_ylabel('Position [mm]')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # Plot 6: System performance metrics
        metrics_labels = ['Flow Rate', 'Pressure Drop', 'Max Stress', 'Interface Stability']
        metrics_values = [
            analysis.get('flow_rate', 0) * 1e6,  # Convert to μL/s
            analysis.get('pressure_drop', 0),
            np.max(solution['stress']),
            analysis.get('interface_std_y', 0) * 1000  # Convert to mm
        ]

        bars = axes[1, 2].bar(metrics_labels, metrics_values, color=['lightblue', 'lightgreen', 'orange', 'pink'])
        axes[1, 2].set_title('Biological System Metrics', fontweight='bold')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.02,
                          '.2f', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Biological flow visualization saved to: {save_path}")

        plt.show()

        return save_path


def main():
    """Main demonstration of flower-inspired biological flows."""
    print("🌸 FLOWER-INSPIRED BIOLOGICAL FLOW ANALYSIS")
    print("=" * 60)
    print("Exploring the beauty of natural flow patterns through science!")

    # Initialize biological flow system
    bio_system = BiologicalFlowSystem()

    # Create different biological models
    models = []

    try:
        # Plant stem vascular system
        plant_stem = bio_system.create_plant_stem_model()
        models.append(("Plant Stem Vascular System", plant_stem))

        # Leaf venation system
        leaf_venation = bio_system.create_leaf_venation_model()
        models.append(("Leaf Venation System", leaf_venation))

        # Flower petal system
        flower_petal = bio_system.create_flower_petal_model()
        models.append(("Flower Petal Fluid Dynamics", flower_petal))

    except Exception as e:
        print(f"❌ Error creating biological models: {e}")
        return

    print("\n🫧 BIOLOGICAL SYSTEMS CREATED:")
    for name, _ in models:
        print(f"  • {name}")

    # Simulate each biological system
    results = []

    for system_name, analyzer in models:
        try:
            # Simulate with biologically relevant velocities
            if "PLANT STEM" in system_name.upper():
                velocity = 0.0005  # 0.5 mm/s (typical sap flow)
            elif "LEAF" in system_name.upper():
                velocity = 0.002   # 2 mm/s (transpiration driven)
            else:  # Flower
                velocity = 0.001   # 1 mm/s (petal fluid dynamics)

            solution, analysis = bio_system.simulate_biological_transport(
                analyzer, system_name, velocity
            )

            # Create visualization
            viz_path = bio_system.visualize_biological_patterns(
                analyzer, solution, analysis, system_name
            )

            results.append((system_name, analyzer, solution, analysis, viz_path))

            print()

        except Exception as e:
            print(f"❌ Error in {system_name}: {e}")
            continue

    # Summary
    print("=" * 60)
    print("🌸 BIOLOGICAL FLOW ANALYSIS SUMMARY")
    print("=" * 60)

    for system_name, _, _, analysis, viz_path in results:
        print(f"\n🌱 {system_name}:")
        print(f"   Flow Rate: {analysis.get('flow_rate', 0)*1e6:.2f} μL/s")
        print(f"   Pressure Drop: {analysis.get('pressure_drop', 0):.1f} Pa")
        print(f"   Max Velocity: {analysis.get('max_velocity', 0)*1000:.2f} mm/s")
        print(f"   Visualization: {viz_path}")

    print("\n🌸 NATURE'S FLOW PATTERNS REVEALED!")
    print("   From plant vascular systems to flower petals,")
    print("   complex multi-phase flows create the beauty we see.")

    return results


if __name__ == "__main__":
    # Run the complete flower-inspired biological flow analysis
    results = main()

    print("\n🎯 What aspect of biological flow patterns would you like to explore further?")
    print("   • Plant vascular system optimization")
    print("   • Leaf venation pattern analysis")
    print("   • Flower petal fluid dynamics")
    print("   • Natural pattern formation")
    print("   • Biological system modeling")
