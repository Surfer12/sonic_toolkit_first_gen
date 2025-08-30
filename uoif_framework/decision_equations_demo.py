"""
UOIF Decision Equations with Trust Region Optimization Demo

This script demonstrates the integration of UOIF decision equations with the
scientific computing toolkit's trust region methods for confidence-constrained optimization.

Demonstrates:
1. Ψ(x) decision equation optimization with trust region constraints
2. Confidence-constrained parameter optimization
3. Integration with toolkit performance metrics
4. Adaptive optimization with historical parameter adjustment
5. Sensitivity analysis of decision parameters

Author: Scientific Computing Toolkit Team
"""

import numpy as np
from datetime import datetime
import json

from .decision_equations import (
    DecisionEquations,
    DecisionParameters,
    OptimizationResult
)


def demo_basic_psi_optimization():
    """
    Demonstrate basic Ψ(x) optimization using trust region methods.
    """
    print("🔬 Basic Ψ(x) Optimization Demo")
    print("=" * 50)

    # Initialize decision equations system
    decision_system = DecisionEquations()

    # Example evidence data for HB model validation
    evidence_data = {
        'evidence_strength': 0.85,      # Strong experimental evidence
        'canonical_strength': 0.78,     # Good theoretical foundation
        'authority_risk': 0.15,         # Low authority risk
        'verifiability_risk': 0.08      # Low verifiability risk
    }

    print("📊 Evidence Data:")
    for key, value in evidence_data.items():
        print(".3f")

    # Perform optimization
    print("\n🔄 Optimizing Ψ(x) Parameters with Trust Region Method...")
    optimization_result = decision_system.optimize_decision_parameters(evidence_data)

    print("\n📈 Optimization Results:")
    print(f"  Optimization Success: {optimization_result.optimization_success}")
    print(f"  Iterations: {optimization_result.iterations}")
    print(".6f")
    print(".4f")
    print(".4f")

    print("\n⚙️ Optimal Parameters:")
    params = optimization_result.optimal_parameters
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\n✅ Constraint Satisfaction:")
    for constraint, satisfied in optimization_result.constraint_satisfaction.items():
        status = "✅" if satisfied else "❌"
        print(f"  {constraint}: {status}")

    return optimization_result


def demo_consciousness_framework_optimization():
    """
    Demonstrate Ψ(x) optimization for consciousness framework evaluation.
    """
    print("\n🧠 Consciousness Framework Optimization Demo")
    print("=" * 50)

    decision_system = DecisionEquations()

    # Consciousness framework evidence data
    consciousness_evidence = {
        'evidence_strength': 0.75,      # Moderate empirical evidence
        'canonical_strength': 0.82,     # Strong theoretical foundation
        'authority_risk': 0.25,         # Higher authority risk (emerging field)
        'verifiability_risk': 0.18      # Moderate verifiability risk
    }

    print("🧠 Consciousness Evidence Data:")
    for key, value in consciousness_evidence.items():
        print(".3f")

    # Add additional constraints for consciousness evaluation
    consciousness_constraints = {
        'min_beta': 1.3,                # Higher uplift for consciousness
        'max_lambda_total': 1.0,        # Balanced risk assessment
        'evidence_dominance': True      # Prefer evidence over canonical
    }

    print("\n🔄 Optimizing Consciousness Ψ(x) with Custom Constraints...")
    optimization_result = decision_system.optimize_decision_parameters(
        consciousness_evidence,
        constraints=consciousness_constraints
    )

    print("\n📊 Consciousness Optimization Results:")
    print(".6f")
    print(".4f")
    print(".4f")

    print("\n🎯 Consciousness-Specific Parameters:")
    params = optimization_result.optimal_parameters
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    return optimization_result


def demo_toolkit_integration_optimization():
    """
    Demonstrate optimization with full toolkit integration including performance metrics.
    """
    print("\n🔧 Toolkit Integration Optimization Demo")
    print("=" * 50)

    decision_system = DecisionEquations()

    # Fluid dynamics evidence with toolkit performance
    fluid_dynamics_evidence = {
        'evidence_strength': 0.88,      # Strong experimental evidence
        'canonical_strength': 0.85,     # Good theoretical foundation
        'authority_risk': 0.12,         # Low authority risk
        'verifiability_risk': 0.06      # Very low verifiability risk
    }

    # Toolkit performance metrics
    toolkit_performance = {
        'correlation': 0.9987,          # Toolkit precision achieved
        'bootstrap_samples': 1000,      # Standard bootstrap count
        'computation_time': 0.234,      # Fast execution
        'memory_usage': 45.6,           # Efficient memory usage
        'convergence_iterations': 15    # Quick convergence
    }

    print("🌊 Fluid Dynamics Evidence with Toolkit Performance:")
    print("Evidence Data:")
    for key, value in fluid_dynamics_evidence.items():
        print(".3f")
    print("\nToolkit Performance:")
    for key, value in toolkit_performance.items():
        if isinstance(value, float):
            print(".4f")
        else:
            print(f"  {key}: {value}")

    # Perform optimization with toolkit integration
    print("\n🔄 Optimizing with Full Toolkit Integration...")
    optimization_result = decision_system.optimize_decision_parameters(
        fluid_dynamics_evidence,
        toolkit_performance=toolkit_performance
    )

    print("\n🚀 Integrated Optimization Results:")
    print(".6f")
    print(".4f")
    print(".4f")
    print(f"  Iterations: {optimization_result.iterations}")
    print(".6f")

    return optimization_result


def demo_adaptive_optimization():
    """
    Demonstrate adaptive optimization with historical parameter adjustment.
    """
    print("\n🔄 Adaptive Optimization Demo")
    print("=" * 50)

    decision_system = DecisionEquations()

    # Current evidence for LSTM convergence validation
    lstm_evidence = {
        'evidence_strength': 0.80,      # Good empirical evidence
        'canonical_strength': 0.88,     # Strong theoretical foundation
        'authority_risk': 0.10,         # Low authority risk
        'verifiability_risk': 0.05      # Very low verifiability risk
    }

    # Mock historical optimization results
    adaptation_history = [
        {
            'optimization_success': True,
            'optimal_parameters': DecisionParameters(
                alpha=0.55, lambda1=0.45, lambda2=0.35, beta=1.25,
                evidence_strength=0.78, canonical_strength=0.85,
                authority_risk=0.12, verifiability_risk=0.08
            )
        },
        {
            'optimization_success': True,
            'optimal_parameters': DecisionParameters(
                alpha=0.60, lambda1=0.40, lambda2=0.30, beta=1.30,
                evidence_strength=0.82, canonical_strength=0.87,
                authority_risk=0.08, verifiability_risk=0.06
            )
        }
    ]

    print("🧠 LSTM Convergence Evidence:")
    for key, value in lstm_evidence.items():
        print(".3f")

    print(f"\n📚 Using {len(adaptation_history)} Historical Optimizations for Adaptation")

    # Perform adaptive optimization
    print("\n🔄 Performing Adaptive Optimization...")
    adaptive_result = decision_system.adaptive_optimization(
        lstm_evidence,
        adaptation_history=adaptation_history
    )

    print("\n🎯 Adaptive Optimization Results:")
    print(".6f")
    print(".4f")
    print(".4f")

    print("\n🔧 Adapted Parameters:")
    params = adaptive_result.optimal_parameters
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    return adaptive_result


def demo_sensitivity_analysis():
    """
    Demonstrate sensitivity analysis of decision parameters.
    """
    print("\n📊 Sensitivity Analysis Demo")
    print("=" * 50)

    decision_system = DecisionEquations()

    # Base evidence for sensitivity analysis
    base_evidence = {
        'evidence_strength': 0.82,
        'canonical_strength': 0.79,
        'authority_risk': 0.14,
        'verifiability_risk': 0.09
    }

    # Parameter ranges for sensitivity analysis
    parameter_ranges = {
        'evidence_strength': (0.6, 0.9),
        'canonical_strength': (0.6, 0.9),
        'authority_risk': (0.05, 0.25),
        'verifiability_risk': (0.03, 0.18)
    }

    print("🔍 Base Evidence Configuration:")
    for key, value in base_evidence.items():
        print(".3f")

    print("\n📈 Parameter Sensitivity Ranges:")
    for param, (min_val, max_val) in parameter_ranges.items():
        print(".1f")

    # Perform sensitivity analysis
    print("\n🔬 Performing Sensitivity Analysis...")
    sensitivity_results = decision_system.sensitivity_analysis(
        base_evidence,
        parameter_ranges=parameter_ranges
    )

    print("\n📋 Sensitivity Analysis Results:")
    print(".4f")

    print(f"\n🎯 Most Sensitive Parameter: {sensitivity_results['most_sensitive']}")

    print("\n📊 Parameter Sensitivity Details:")
    for param_name, sensitivity_data in sensitivity_results['sensitivity_results'].items():
        if sensitivity_data:
            max_change = max(abs(entry['relative_change']) for entry in sensitivity_data)
            print(".1f")

    return sensitivity_results


def demo_comprehensive_workflow():
    """
    Demonstrate comprehensive workflow from evidence to optimization results.
    """
    print("\n🔄 Comprehensive Decision Equations Workflow Demo")
    print("=" * 60)

    decision_system = DecisionEquations()

    # Multi-domain evidence examples
    domains_data = {
        'fluid_dynamics': {
            'evidence_strength': 0.87,
            'canonical_strength': 0.84,
            'authority_risk': 0.11,
            'verifiability_risk': 0.07
        },
        'consciousness_framework': {
            'evidence_strength': 0.76,
            'canonical_strength': 0.83,
            'authority_risk': 0.23,
            'verifiability_risk': 0.16
        },
        'biological_transport': {
            'evidence_strength': 0.81,
            'canonical_strength': 0.79,
            'authority_risk': 0.13,
            'verifiability_risk': 0.08
        }
    }

    workflow_results = {}

    for domain, evidence in domains_data.items():
        print(f"\n🌟 Processing {domain.replace('_', ' ').title()} Domain")

        # Perform optimization
        result = decision_system.optimize_decision_parameters(evidence)

        # Store results
        workflow_results[domain] = {
            'psi_value': result.psi_value,
            'confidence_bounds': result.confidence_bounds,
            'optimal_alpha': result.optimal_parameters.alpha,
            'success': result.optimization_success
        }

        print(".4f")
        print(".3f")
        print(".4f")
        print(f"  Optimization Success: {result.optimization_success}")

    # Cross-domain comparison
    print("\n📊 Cross-Domain Comparison:")
    print("Domain                  | Ψ(x)   | Confidence | α    | Status")
    print("-" * 60)
    for domain, results in workflow_results.items():
        domain_name = domain.replace('_', ' ').title()[:20]
        psi = ".3f"
        conf = ".2f"
        alpha = ".2f"
        status = "✅" if results['success'] else "❌"
        print("22")

    return workflow_results


def run_all_decision_equations_demos():
    """
    Run all decision equations demonstrations.
    """
    print("🎯 UOIF Decision Equations with Trust Region Optimization Demonstrations")
    print("=" * 80)
    print("This demo showcases the integration of UOIF decision equations Ψ(x)")
    print("with the scientific computing toolkit's trust region optimization methods.")
    print("")

    # Demo 1: Basic Ψ(x) optimization
    basic_result = demo_basic_psi_optimization()

    # Demo 2: Consciousness framework optimization
    consciousness_result = demo_consciousness_framework_optimization()

    # Demo 3: Toolkit integration optimization
    toolkit_result = demo_toolkit_integration_optimization()

    # Demo 4: Adaptive optimization
    adaptive_result = demo_adaptive_optimization()

    # Demo 5: Sensitivity analysis
    sensitivity_result = demo_sensitivity_analysis()

    # Demo 6: Comprehensive workflow
    workflow_results = demo_comprehensive_workflow()

    # Summary
    print("\n🎉 Decision Equations Demo Summary")
    print("=" * 80)
    print("✅ Basic Ψ(x) optimization with trust region methods")
    print("✅ Consciousness framework parameter optimization")
    print("✅ Full toolkit integration with performance metrics")
    print("✅ Adaptive optimization with historical parameter adjustment")
    print("✅ Comprehensive sensitivity analysis")
    print("✅ Multi-domain workflow demonstration")

    # Performance summary
    all_results = [
        basic_result, consciousness_result, toolkit_result,
        adaptive_result, workflow_results
    ]

    successful_optimizations = sum(1 for r in all_results[:-1] if hasattr(r, 'optimization_success') and r.optimization_success)
    total_optimizations = len(all_results) - 1  # Exclude workflow_results

    print("
📊 Performance Summary:"    print(f"  Successful Optimizations: {successful_optimizations}/{total_optimizations}")
    print(".1f")

    # Save comprehensive results
    comprehensive_summary = {
        'timestamp': datetime.now().isoformat(),
        'basic_optimization': {
            'psi_value': basic_result.psi_value,
            'success': basic_result.optimization_success
        },
        'consciousness_optimization': {
            'psi_value': consciousness_result.psi_value,
            'success': consciousness_result.optimization_success
        },
        'toolkit_integration': {
            'psi_value': toolkit_result.psi_value,
            'success': toolkit_result.optimization_success
        },
        'adaptive_optimization': {
            'psi_value': adaptive_result.psi_value,
            'success': adaptive_result.optimization_success
        },
        'sensitivity_analysis': {
            'base_psi': sensitivity_result['base_psi'],
            'most_sensitive': sensitivity_result['most_sensitive']
        },
        'workflow_results': workflow_results,
        'performance_metrics': {
            'successful_optimizations': successful_optimizations,
            'total_optimizations': total_optimizations,
            'success_rate': successful_optimizations / total_optimizations if total_optimizations > 0 else 0
        }
    }

    with open('uoif_decision_equations_demo_results.json', 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)

    print("\n💾 Comprehensive results saved to: uoif_decision_equations_demo_results.json")

    return {
        'basic': basic_result,
        'consciousness': consciousness_result,
        'toolkit': toolkit_result,
        'adaptive': adaptive_result,
        'sensitivity': sensitivity_result,
        'workflow': workflow_results
    }


if __name__ == "__main__":
    # Run all demonstrations
    results = run_all_decision_equations_demos()

    print("
🎯 Decision equations demonstrations completed successfully!"    print("The UOIF framework now includes comprehensive Ψ(x) optimization")
    print("with trust region methods and toolkit integration.")
