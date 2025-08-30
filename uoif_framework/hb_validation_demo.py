"""
UOIF HB Model Validation with Promotion Triggers Demo

This script demonstrates the application of UOIF promotion triggers to validate
Herschel-Bulkley (HB) rheological models and promote them to "Empirically Grounded" status.

Demonstrates:
1. HB model validation using UOIF scoring components
2. Promotion triggers based on R² > 0.98 threshold
3. Material-specific validation criteria
4. Confidence assessment for rheological predictions
5. Batch validation of multiple HB models

Author: Scientific Computing Toolkit Team
"""

import numpy as np
from datetime import datetime
import json

from .hb_validation import (
    HBModelValidator,
    HBModelParameters
)


def demo_single_hb_validation():
    """
    Demonstrate validation of a single HB model with promotion triggers.
    """
    print("🔬 Single HB Model Validation Demo")
    print("=" * 50)

    # Initialize HB validator
    hb_validator = HBModelValidator()

    # Example polymer melt HB model
    model_id = "polymer_melt_hb_001"

    # Simulated experimental data (polymer melt)
    shear_rates = np.logspace(-2, 3, 50)  # 0.01 to 1000 1/s
    true_params = HBModelParameters(
        tau_y=50.0, K=1000.0, n=0.6,
        tau_y_uncertainty=5.0, K_uncertainty=50.0, n_uncertainty=0.02
    )

    # Generate synthetic experimental data with noise
    np.random.seed(42)
    true_stresses = true_params.tau_y + true_params.K * shear_rates**true_params.n
    noise_level = 0.02  # 2% noise
    measured_stresses = true_stresses * (1 + np.random.normal(0, noise_level, len(shear_rates)))

    experimental_data = {
        'shear_rates': shear_rates.tolist(),
        'stresses': measured_stresses.tolist()
    }

    print("🧪 Polymer Melt HB Model Validation:")
    print(f"  True Parameters: τ_y={true_params.tau_y} Pa, K={true_params.K} Pa·s^n, n={true_params.n}")
    print(f"  Shear Rate Range: {shear_rates.min():.3f} - {shear_rates.max():.1f} 1/s")
    print(f"  Data Points: {len(shear_rates)}")

    # Perform HB model validation
    print("\n🔄 Performing HB Model Validation with UOIF Scoring...")
    validation_result = hb_validator.validate_hb_model(
        model_id=model_id,
        experimental_data=experimental_data,
        fitted_parameters=true_params,
        material_type="polymer_melt"
    )

    print("\n📊 Validation Results:")
    print(".4f")
    print(".3f")
    print(".3f")
    print(".4f")
    print(f"  Promotion Status: {validation_result.promotion_status}")

    print("\n⚙️ UOIF Components:")
    components = validation_result.uoif_components
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".1f")
    print(".3f")

    print("\n✅ Validation Criteria:")
    for criterion, satisfied in validation_result.validation_criteria.items():
        status = "✅" if satisfied else "❌"
        print(f"  {criterion}: {status}")

    # Check if promoted to "Empirically Grounded"
    if validation_result.r_squared > 0.98:
        print("
🎉 SUCCESS: Model promoted to 'Empirically Grounded' status!"        print("   R² > 0.98 threshold met for high-confidence rheological predictions")
    else:
        print("
⚠️  Model needs improvement to reach 'Empirically Grounded' status"        print("   R² must exceed 0.98 for promotion"

    return validation_result


def demo_material_specific_validation():
    """
    Demonstrate material-specific validation criteria.
    """
    print("\n🧫 Material-Specific Validation Demo")
    print("=" * 50)

    hb_validator = HBModelValidator()

    # Test different material types
    materials_data = {
        'polymer_melt': {
            'params': HBModelParameters(tau_y=100.0, K=2000.0, n=0.5,
                                      tau_y_uncertainty=10.0, K_uncertainty=100.0, n_uncertainty=0.03),
            'shear_range': (0.01, 1000),  # Wide range typical for melts
            'expected_performance': 'high'
        },
        'biological_fluid': {
            'params': HBModelParameters(tau_y=5.0, K=10.0, n=0.8,
                                      tau_y_uncertainty=0.5, K_uncertainty=1.0, n_uncertainty=0.05),
            'shear_range': (0.1, 100),    # Moderate range for biological fluids
            'expected_performance': 'medium'
        },
        'suspension': {
            'params': HBModelParameters(tau_y=25.0, K=500.0, n=0.7,
                                      tau_y_uncertainty=2.5, K_uncertainty=25.0, n_uncertainty=0.04),
            'shear_range': (0.01, 100),   # Broad range for suspensions
            'expected_performance': 'medium-high'
        }
    }

    results = {}

    for material_type, material_data in materials_data.items():
        print(f"\n🌟 Testing {material_type.replace('_', ' ').title()}:")

        # Generate synthetic data
        shear_rates = np.logspace(np.log10(material_data['shear_range'][0]),
                                np.log10(material_data['shear_range'][1]), 30)

        # Generate stresses with material-appropriate noise
        true_stresses = (material_data['params'].tau_y +
                        material_data['params'].K * shear_rates**material_data['params'].n)

        if material_type == 'biological_fluid':
            noise_level = 0.05  # Higher noise for biological data
        elif material_type == 'polymer_melt':
            noise_level = 0.03  # Moderate noise for controlled experiments
        else:
            noise_level = 0.04  # Standard noise for suspensions

        measured_stresses = true_stresses * (1 + np.random.normal(0, noise_level, len(shear_rates)))

        experimental_data = {
            'shear_rates': shear_rates.tolist(),
            'stresses': measured_stresses.tolist()
        }

        # Validate model
        result = hb_validator.validate_hb_model(
            model_id=f"{material_type}_validation",
            experimental_data=experimental_data,
            fitted_parameters=material_data['params'],
            material_type=material_type
        )

        results[material_type] = result

        print(".4f")
        print(".3f")
        print(".4f")
        print(f"  Status: {result.promotion_status}")

    # Cross-material comparison
    print("\n📊 Material Performance Comparison:")
    print("Material Type      | R²    | RMSE  | Confidence | Status")
    print("-" * 60)
    for material, result in results.items():
        material_name = material.replace('_', ' ').title()[:16]
        r2 = ".3f"
        rmse = ".2f"
        conf = ".2f"
        status = result.promotion_status[:12]
        print("15")

    return results


def demo_batch_hb_validation():
    """
    Demonstrate batch validation of multiple HB models.
    """
    print("\n🔄 Batch HB Model Validation Demo")
    print("=" * 50)

    hb_validator = HBModelValidator()

    # Generate batch of HB models with varying quality
    batch_models = []

    for i in range(10):
        # Vary model quality
        quality_factor = np.random.beta(2, 2)  # Bias toward good models

        # Generate parameters
        base_tau_y = 50 * quality_factor
        base_K = 1000 * quality_factor
        base_n = 0.5 + 0.3 * quality_factor

        params = HBModelParameters(
            tau_y=base_tau_y,
            K=base_K,
            n=base_n,
            tau_y_uncertainty=base_tau_y * 0.1,
            K_uncertainty=base_K * 0.05,
            n_uncertainty=0.02
        )

        # Generate experimental data with quality-dependent noise
        shear_rates = np.logspace(-2, 2, 25)
        true_stresses = params.tau_y + params.K * shear_rates**params.n

        # Noise inversely related to quality
        noise_level = 0.1 * (1 - quality_factor) + 0.01
        measured_stresses = true_stresses * (1 + np.random.normal(0, noise_level, len(shear_rates)))

        model_data = {
            'model_id': f'batch_model_{i:03d}',
            'experimental_data': {
                'shear_rates': shear_rates.tolist(),
                'stresses': measured_stresses.tolist()
            },
            'fitted_parameters': params,
            'material_type': np.random.choice(['polymer_melt', 'biological_fluid', 'suspension'])
        }

        batch_models.append(model_data)

    print(f"🔬 Processing {len(batch_models)} HB models in batch...")

    # Perform batch validation
    batch_results = hb_validator.batch_validate_hb_models(batch_models)

    # Analyze results
    promotion_counts = {}
    r_squared_values = []

    for result in batch_results:
        status = result.promotion_status
        promotion_counts[status] = promotion_counts.get(status, 0) + 1
        r_squared_values.append(result.r_squared)

    print("\n📈 Batch Validation Summary:")
    print(f"  Total Models: {len(batch_results)}")
    print(".3f")
    print(".3f")
    print(".4f")

    print("\n🏆 Promotion Status Distribution:")
    for status, count in promotion_counts.items():
        percentage = (count / len(batch_results)) * 100
        print(".1f")

    # Identify top performers
    top_models = sorted(batch_results, key=lambda x: x.r_squared, reverse=True)[:3]
    print("\n🥇 Top 3 Performing Models:")
    for i, model in enumerate(top_models, 1):
        print(f"  {i}. {model.model_id}: R²={model.r_squared:.4f}, Status={model.promotion_status}")

    return batch_results


def demo_validation_improvements():
    """
    Demonstrate validation improvement recommendations.
    """
    print("\n💡 Validation Improvement Recommendations Demo")
    print("=" * 50)

    hb_validator = HBModelValidator()

    # Create a model that needs improvement (moderate performance)
    model_id = "improvement_example"

    # Parameters with moderate uncertainty
    params = HBModelParameters(
        tau_y=75.0, K=1500.0, n=0.55,
        tau_y_uncertainty=15.0,    # 20% uncertainty
        K_uncertainty=300.0,       # 20% uncertainty
        n_uncertainty=0.08         # High uncertainty
    )

    # Generate data with moderate noise
    shear_rates = np.logspace(-2, 2, 30)
    true_stresses = params.tau_y + params.K * shear_rates**params.n
    measured_stresses = true_stresses * (1 + np.random.normal(0, 0.06, len(shear_rates)))

    experimental_data = {
        'shear_rates': shear_rates.tolist(),
        'stresses': measured_stresses.tolist()
    }

    # Validate and get recommendations
    validation_result = hb_validator.validate_hb_model(
        model_id=model_id,
        experimental_data=experimental_data,
        fitted_parameters=params,
        material_type="polymer_melt"
    )

    print("🔧 Model Needing Improvement:")
    print(".4f")
    print(".3f")
    print(".3f")
    print(f"  Status: {validation_result.promotion_status}")

    # Get improvement recommendations
    recommendations = hb_validator.recommend_improvements(validation_result)

    print("\n💡 Improvement Recommendations:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"  {i}. {recommendation}")

    # Simulate improved model
    print("\n🔄 Simulating Improved Model...")
    improved_params = HBModelParameters(
        tau_y=75.0, K=1500.0, n=0.55,
        tau_y_uncertainty=7.5,     # Reduced uncertainty
        K_uncertainty=75.0,        # Reduced uncertainty
        n_uncertainty=0.03         # Reduced uncertainty
    )

    # Better data quality
    improved_stresses = true_stresses * (1 + np.random.normal(0, 0.02, len(shear_rates)))
    improved_data = {
        'shear_rates': shear_rates.tolist(),
        'stresses': improved_stresses.tolist()
    }

    improved_result = hb_validator.validate_hb_model(
        model_id="improvement_example_improved",
        experimental_data=improved_data,
        fitted_parameters=improved_params,
        material_type="polymer_melt"
    )

    print("\n✨ Improved Model Results:")
    print(".4f")
    print(".3f")
    print(".3f")
    print(f"  Status: {improved_result.promotion_status}")

    if improved_result.r_squared > 0.98:
        print("  🎉 Successfully promoted to 'Empirically Grounded'!")

    return validation_result, improved_result


def demo_comprehensive_validation_workflow():
    """
    Demonstrate comprehensive HB validation workflow with statistics.
    """
    print("\n🔬 Comprehensive HB Validation Workflow Demo")
    print("=" * 60)

    hb_validator = HBModelValidator()

    # Run all individual demos to build comprehensive dataset
    print("Step 1: Running individual validations...")
    single_result = demo_single_hb_validation()
    material_results = demo_material_specific_validation()
    batch_results = demo_batch_hb_validation()

    # Combine all results
    all_results = [single_result] + list(material_results.values()) + batch_results

    print("
📊 Comprehensive Validation Statistics:"    stats = hb_validator.get_validation_statistics()

    print(f"  Total Models Validated: {stats['total_models']}")
    print(".4f")
    print(".3f")
    print(".4f")
    print(".1%")

    print("\n🏆 Promotion Status Distribution:")
    for status, count in stats['promotion_distribution'].items():
        percentage = (count / stats['total_models']) * 100
        print(".1f")

    if stats['material_performance']:
        print("\n🧫 Material-Specific Performance:")
        for material, perf in stats['material_performance'].items():
            material_name = material.replace('_', ' ').title()
            print(f"  {material_name}:")
            print(".3f")
            print(".3f")
            print(".1%")

    # Performance insights
    empirically_grounded = [r for r in all_results if r.promotion_status == "Empirically Grounded"]
    high_confidence = [r for r in all_results if r.confidence_score > 0.9]

    print("
💡 Performance Insights:"    print(f"  Empirically Grounded Models: {len(empirically_grounded)}/{len(all_results)}")
    print(f"  High Confidence Models (≥0.9): {len(high_confidence)}/{len(all_results)}")
    print(".1%")
    print(".1%")

    return {
        'statistics': stats,
        'all_results': all_results,
        'empirically_grounded': empirically_grounded,
        'high_confidence': high_confidence
    }


def run_all_hb_validation_demos():
    """
    Run all HB validation demonstrations.
    """
    print("🏆 UOIF HB Model Validation with Promotion Triggers Demonstrations")
    print("=" * 80)
    print("This demo showcases the application of UOIF promotion triggers for")
    print("Herschel-Bulkley model validation, promoting models to 'Empirically Grounded'")
    print("status when R² > 0.98 threshold is met.")
    print("")

    # Demo 1: Single model validation
    single_result = demo_single_hb_validation()

    # Demo 2: Material-specific validation
    material_results = demo_material_specific_validation()

    # Demo 3: Batch validation
    batch_results = demo_batch_hb_validation()

    # Demo 4: Improvement recommendations
    original_result, improved_result = demo_validation_improvements()

    # Demo 5: Comprehensive workflow
    workflow_results = demo_comprehensive_validation_workflow()

    # Final summary
    print("\n🎉 HB Validation Demo Summary")
    print("=" * 80)
    print("✅ Single model validation with promotion triggers")
    print("✅ Material-specific validation criteria")
    print("✅ Batch processing of multiple HB models")
    print("✅ Improvement recommendations for model enhancement")
    print("✅ Comprehensive validation statistics and insights")

    # Calculate overall success metrics
    all_results = ([single_result] + list(material_results.values()) +
                   batch_results + [original_result, improved_result] +
                   workflow_results['all_results'])

    empirically_grounded_count = sum(1 for r in all_results if r.promotion_status == "Empirically Grounded")
    high_confidence_count = sum(1 for r in all_results if r.confidence_score > 0.9)
    excellent_fit_count = sum(1 for r in all_results if r.r_squared > 0.98)

    print("
📊 Overall Performance Metrics:"    print(f"  Models Processed: {len(all_results)}")
    print(f"  Empirically Grounded: {empirically_grounded_count} ({empirically_grounded_count/len(all_results)*100:.1f}%)")
    print(f"  High Confidence (≥0.9): {high_confidence_count} ({high_confidence_count/len(all_results)*100:.1f}%)")
    print(f"  Excellent Fit (R²>0.98): {excellent_fit_count} ({excellent_fit_count/len(all_results)*100:.1f}%)")

    # Save comprehensive results
    comprehensive_summary = {
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': {
            'total_models': len(all_results),
            'empirically_grounded': empirically_grounded_count,
            'high_confidence': high_confidence_count,
            'excellent_fit': excellent_fit_count,
            'success_rates': {
                'empirically_grounded': empirically_grounded_count / len(all_results),
                'high_confidence': high_confidence_count / len(all_results),
                'excellent_fit': excellent_fit_count / len(all_results)
            }
        },
        'validation_statistics': workflow_results['statistics'],
        'demo_results': {
            'single_validation': {
                'r_squared': single_result.r_squared,
                'promotion_status': single_result.promotion_status,
                'confidence_score': single_result.confidence_score
            },
            'material_performance': {
                material: {
                    'r_squared': result.r_squared,
                    'promotion_status': result.promotion_status,
                    'confidence_score': result.confidence_score
                }
                for material, result in material_results.items()
            },
            'batch_summary': {
                'total_batch_models': len(batch_results),
                'avg_r_squared': sum(r.r_squared for r in batch_results) / len(batch_results),
                'promotion_distribution': workflow_results['statistics']['promotion_distribution']
            }
        }
    }

    with open('uoif_hb_validation_demo_results.json', 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)

    print("\n💾 Comprehensive results saved to: uoif_hb_validation_demo_results.json")

    return {
        'single': single_result,
        'materials': material_results,
        'batch': batch_results,
        'improvements': (original_result, improved_result),
        'workflow': workflow_results
    }


if __name__ == "__main__":
    # Run all HB validation demonstrations
    results = run_all_hb_validation_demos()

    print("
🎯 HB validation demonstrations completed successfully!"    print("The UOIF framework now includes comprehensive promotion triggers")
    print("for HB model validation with 'Empirically Grounded' status.")
