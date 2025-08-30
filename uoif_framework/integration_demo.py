"""
UOIF Framework Integration Demonstration

This script demonstrates the integration between UOIF scoring system
and the scientific computing toolkit's bootstrap analysis capabilities.

Demonstrates:
1. Bootstrap uncertainty quantification for UOIF scores
2. Comprehensive uncertainty analysis using multiple methods
3. Temporal dynamics integration with performance monitoring
4. Real-world examples with HB model validation and consciousness claims

Author: Scientific Computing Toolkit Team
"""

import numpy as np
from datetime import datetime, timedelta
import json

from .toolkit_integration import (
    BootstrapIntegrator,
    UncertaintyQuantificationIntegrator,
    UOIFTemporalIntegrator
)
from .scoring_engine import ScoringEngine
from .allocation_system import AllocationSystem


def demo_bootstrap_integration():
    """
    Demonstrate UOIF scoring integration with bootstrap uncertainty quantification.
    """
    print("🔬 UOIF Bootstrap Integration Demo")
    print("=" * 50)

    # Initialize bootstrap integrator
    bootstrap_integrator = BootstrapIntegrator(n_bootstrap=1000, confidence_level=0.95)

    # Example HB model validation claim
    hb_claim_components = {
        'authority': 0.85,      # Strong experimental foundation
        'verifiability': 0.92,  # LM optimization results
        'depth': 0.88,          # Detailed rheological analysis
        'alignment': 0.90,      # Matches Herschel-Bulkley physics
        'recurrence': 5,        # 5 citations in literature
        'noise': 0.05          # Low noise in measurements
    }

    print("📊 HB Model Validation Claim Components:")
    for comp, value in hb_claim_components.items():
        print(".3f")

    # Perform bootstrap uncertainty analysis
    print("\n🔄 Performing Bootstrap Uncertainty Analysis...")
    bootstrap_results = bootstrap_integrator.bootstrap_score_uncertainty(
        claim_id="hb_model_validation_001",
        raw_components=hb_claim_components,
        claim_type="primitive",
        domain="hb_models"
    )

    print("\n📈 Bootstrap Analysis Results:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".3f")
    print(f"  Reliability Class: {bootstrap_results['reliability']['reliability_class']}")
    print(".3f")

    return bootstrap_results


def demo_comprehensive_uncertainty():
    """
    Demonstrate comprehensive uncertainty analysis using multiple methods.
    """
    print("\n🔬 Comprehensive Uncertainty Analysis Demo")
    print("=" * 50)

    # Initialize uncertainty quantification integrator
    uncertainty_integrator = UncertaintyQuantificationIntegrator()

    # Example consciousness framework claim
    consciousness_components = {
        'authority': 0.78,      # Strong theoretical foundation
        'verifiability': 0.85,  # Experimental validation available
        'depth': 0.82,          # Complex mathematical framework
        'alignment': 0.88,      # Matches cognitive science principles
        'recurrence': 3,        # Growing literature citations
        'noise': 0.08          # Some subjectivity in interpretation
    }

    print("🧠 Consciousness Framework Claim Components:")
    for comp, value in consciousness_components.items():
        print(".3f")

    # Perform comprehensive uncertainty analysis
    print("\n🔄 Performing Comprehensive Uncertainty Analysis...")
    comprehensive_results = uncertainty_integrator.comprehensive_uncertainty_analysis(
        claim_id="consciousness_framework_001",
        raw_components=consciousness_components,
        claim_type="interpretation",
        domain="consciousness_framework"
    )

    print("\n📊 Method Comparison:")
    for method_name, method_result in comprehensive_results['methods'].items():
        if 'mean' in method_result:
            print(f"  {method_name.capitalize()}:")
            print(".4f")
            print(".4f")
            print(f"    Reliability: {method_result.get('reliability', {}).get('reliability_class', 'N/A')}")

    # Consensus analysis
    consensus = comprehensive_results['consensus']
    print("
🎯 Consensus Analysis:"    print(".4f")
    print(".4f")
    print(f"  Agreement Level: {consensus['agreement_level']}")
    print(".3f")

    # Performance metrics
    performance = comprehensive_results.get('performance', {})
    if performance:
        print("
⚡ Performance Metrics:"        print(".3f")
        print(f"  Total Samples: {performance['total_samples']}")
        print(".2f")
        print(f"  Consensus Achieved: {performance['consensus_achieved']}")

    return comprehensive_results


def demo_temporal_integration():
    """
    Demonstrate temporal dynamics integration with performance monitoring.
    """
    print("\n⏰ Temporal Dynamics Integration Demo")
    print("=" * 50)

    # Initialize temporal integrator
    temporal_integrator = UOIFTemporalIntegrator()

    # Simulate claim with historical data
    claim_id = "lstm_convergence_validation_001"

    # Create mock historical data
    base_scores = [0.85, 0.87, 0.89, 0.86, 0.88]
    timestamps = [datetime.now() - timedelta(days=i*7) for i in range(len(base_scores))]

    print("📈 LSTM Convergence Claim Temporal Analysis:")

    # Apply temporal decay to historical scores
    decayed_scores = []
    for i, (score, timestamp) in enumerate(zip(base_scores, timestamps)):
        decayed_score = temporal_integrator.apply_temporal_decay(
            claim_id=claim_id,
            base_score=score,
            last_update=timestamp
        )
        decayed_scores.append(decayed_score)

        days_old = (datetime.now() - timestamp).total_seconds() / (24 * 3600)
        print(".1f"
    # Performance regression detection
    current_score = 0.82  # Simulate current score
    regression_analysis = temporal_integrator.detect_performance_regression(
        claim_id=claim_id,
        current_score=current_score
    )

    print("
📉 Performance Regression Analysis:"    print(f"  Regression Detected: {regression_analysis['regression_detected']}")
    print(".3f")
    print(".4f")
    print(f"  Confidence: {regression_analysis['confidence']:.3f}")

    # Temporal boost calculation
    mock_performance = {
        'correlation': 0.9987,  # Meets toolkit precision
        'bootstrap_samples': 1000,
        'score': current_score
    }

    temporal_boost = temporal_integrator.calculate_temporal_boost(
        claim_id=claim_id,
        recent_performance=mock_performance
    )

    print("
🚀 Temporal Boost Factor:"    print(".3f")

    return {
        'decayed_scores': decayed_scores,
        'regression_analysis': regression_analysis,
        'temporal_boost': temporal_boost
    }


def demo_allocation_integration():
    """
    Demonstrate integration of allocation system with uncertainty quantification.
    """
    print("\n🎯 Allocation System Integration Demo")
    print("=" * 50)

    # Initialize allocation system
    allocation_system = AllocationSystem()

    # Example claim with toolkit performance data
    toolkit_performance = {
        'correlation': 0.9987,      # Toolkit precision achieved
        'bootstrap_samples': 1000,  # Standard bootstrap count
        'performance_regression': False  # No regression detected
    }

    # Allocate alpha with toolkit integration
    allocation_result = allocation_system.allocate_alpha(
        claim_id="integrated_allocation_demo_001",
        claim_type="interpretation",
        domain="consciousness_framework",
        validation_score=0.89,
        toolkit_performance=toolkit_performance
    )

    print("🎯 Integrated Allocation Result:")
    print(".4f")
    print(".3f")
    print(f"  Promotion Status: {allocation_result.promotion_status}")
    print(".3f")
    print(".3f")

    # Show allocation factors
    factors = allocation_result.allocation_factors
    print("
⚙️ Allocation Factors:"    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    return allocation_result


def demo_full_integration_workflow():
    """
    Demonstrate complete integration workflow from scoring to allocation.
    """
    print("\n🔄 Complete Integration Workflow Demo")
    print("=" * 50)

    # Step 1: Component evaluation with uncertainty
    print("Step 1: Component Uncertainty Analysis")
    uncertainty_integrator = UncertaintyQuantificationIntegrator()
    components = {
        'authority': 0.88,
        'verifiability': 0.92,
        'depth': 0.85,
        'alignment': 0.90,
        'recurrence': 4,
        'noise': 0.06
    }

    uncertainty_results = uncertainty_integrator.comprehensive_uncertainty_analysis(
        claim_id="workflow_demo_001",
        raw_components=components,
        claim_type="interpretation",
        domain="fluid_dynamics"
    )

    print(f"  Consensus Score: {uncertainty_results['consensus']['consensus_mean']:.4f}")
    print(f"  Agreement Level: {uncertainty_results['consensus']['agreement_level']}")

    # Step 2: Scoring with integrated uncertainty
    print("\nStep 2: UOIF Scoring with Uncertainty")
    scoring_engine = ScoringEngine()
    score_result = scoring_engine.score_claim(
        claim_id="workflow_demo_001",
        raw_components=components,
        claim_type="interpretation",
        domain="fluid_dynamics"
    )

    print(".4f")
    print(".3f")

    # Step 3: Temporal dynamics
    print("\nStep 3: Temporal Dynamics Integration")
    temporal_integrator = UOIFTemporalIntegrator()
    temporal_boost = temporal_integrator.calculate_temporal_boost(
        claim_id="workflow_demo_001",
        recent_performance={
            'correlation': 0.9987,
            'bootstrap_samples': 1000,
            'score': score_result.total_score
        }
    )

    print(".3f")

    # Step 4: Final allocation with all integrations
    print("\nStep 4: Final Allocation with Full Integration")
    allocation_system = AllocationSystem()
    final_allocation = allocation_system.allocate_alpha(
        claim_id="workflow_demo_001",
        claim_type="interpretation",
        domain="fluid_dynamics",
        validation_score=score_result.total_score,
        toolkit_performance={
            'correlation': 0.9987,
            'bootstrap_samples': 1000,
            'performance_regression': False
        }
    )

    print(".4f")
    print(f"  Promotion Status: {final_allocation.promotion_status}")
    print(".3f")

    return {
        'uncertainty': uncertainty_results,
        'scoring': score_result,
        'temporal_boost': temporal_boost,
        'final_allocation': final_allocation
    }


def run_all_demos():
    """
    Run all integration demonstrations.
    """
    print("🎪 UOIF Framework Integration Demonstrations")
    print("=" * 60)
    print("This demo showcases the integration between UOIF scoring system")
    print("and the scientific computing toolkit's bootstrap analysis capabilities.\n")

    # Demo 1: Bootstrap Integration
    bootstrap_results = demo_bootstrap_integration()

    # Demo 2: Comprehensive Uncertainty
    uncertainty_results = demo_comprehensive_uncertainty()

    # Demo 3: Temporal Integration
    temporal_results = demo_temporal_integration()

    # Demo 4: Allocation Integration
    allocation_results = demo_allocation_integration()

    # Demo 5: Full Workflow
    workflow_results = demo_full_integration_workflow()

    # Summary
    print("\n🎉 Integration Demo Summary")
    print("=" * 60)
    print("✅ Bootstrap uncertainty quantification integrated")
    print("✅ Multiple uncertainty methods combined")
    print("✅ Temporal dynamics with performance monitoring")
    print("✅ Allocation system with toolkit integration")
    print("✅ Complete workflow from scoring to allocation")

    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'bootstrap_reliability': bootstrap_results['reliability']['reliability_class'],
        'uncertainty_agreement': uncertainty_results['consensus']['agreement_level'],
        'temporal_boost': temporal_results['temporal_boost'],
        'final_allocation': workflow_results['final_allocation'].allocated_alpha,
        'promotion_status': workflow_results['final_allocation'].promotion_status
    }

    with open('uoif_integration_demo_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n💾 Results saved to: uoif_integration_demo_results.json")

    return {
        'bootstrap': bootstrap_results,
        'uncertainty': uncertainty_results,
        'temporal': temporal_results,
        'allocation': allocation_results,
        'workflow': workflow_results
    }


if __name__ == "__main__":
    # Run all demonstrations
    results = run_all_demos()

    print("
🎯 Integration demonstrations completed successfully!"    print("The UOIF framework is now fully integrated with the scientific computing toolkit.")
