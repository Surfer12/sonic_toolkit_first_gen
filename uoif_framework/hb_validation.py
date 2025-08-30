"""
UOIF HB Model Validation with Promotion Triggers

This module implements specialized validation for Herschel-Bulkley (HB) rheological models
using UOIF promotion triggers to determine "Empirically Grounded" status.

Key Features:
- HB model parameter validation using UOIF scoring
- R² threshold-based promotion triggers (R² > 0.98)
- Integration with toolkit's LM optimization results
- Confidence assessment for rheological predictions
- Material-specific validation criteria

Author: Scientific Computing Toolkit Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from .scoring_engine import ScoringEngine, ScoreComponents
from .allocation_system import AllocationSystem
from .source_hierarchy import SourceHierarchy


@dataclass
class HBValidationResult:
    """Result of HB model validation."""
    model_id: str
    validation_timestamp: datetime
    r_squared: float
    rmse: float
    mae: float
    parameter_uncertainty: Dict[str, float]
    promotion_status: str
    confidence_score: float
    uoif_components: ScoreComponents
    material_type: str
    shear_rate_range: Tuple[float, float]
    validation_criteria: Dict[str, bool]


@dataclass
class HBModelParameters:
    """HB model parameters with uncertainty."""
    tau_y: float          # Yield stress [Pa]
    K: float             # Consistency index [Pa·s^n]
    n: float             # Flow index [-]

    # Uncertainties
    tau_y_uncertainty: float
    K_uncertainty: float
    n_uncertainty: float


class HBModelValidator:
    """
    Validates HB models using UOIF promotion triggers.

    This class applies UOIF scoring and promotion criteria specifically
    to HB rheological models to determine empirical grounding status.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HB model validator.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.scoring_engine = ScoringEngine()
        self.allocation_system = AllocationSystem()
        self.source_hierarchy = SourceHierarchy()
        self.validation_history: Dict[str, List[HBValidationResult]] = {}

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for HB validation."""
        return {
            'promotion_thresholds': {
                'r_squared': 0.98,        # R² > 0.98 for promotion
                'rmse_max': 2.0,          # RMSE < 2.0 Pa
                'mae_max': 1.5,           # MAE < 1.5 Pa
                'confidence_min': 0.85,   # Minimum confidence score
                'parameter_uncertainty_max': 0.15  # Max relative uncertainty
            },
            'material_criteria': {
                'polymer_melt': {
                    'min_shear_rate': 0.01,    # 1/s
                    'max_shear_rate': 1000,    # 1/s
                    'expected_r_squared': 0.97
                },
                'biological_fluid': {
                    'min_shear_rate': 0.1,     # 1/s
                    'max_shear_rate': 100,     # 1/s
                    'expected_r_squared': 0.95
                },
                'suspension': {
                    'min_shear_rate': 0.01,    # 1/s
                    'max_shear_rate': 100,     # 1/s
                    'expected_r_squared': 0.96
                },
                'yield_stress_fluid': {
                    'min_shear_rate': 0.001,   # 1/s
                    'max_shear_rate': 10,      # 1/s
                    'expected_r_squared': 0.94
                }
            },
            'uoif_components': {
                'authority_base': 0.85,      # Strong experimental foundation
                'verifiability_base': 0.90,  # LM optimization results
                'depth_base': 0.80,          # Detailed rheological analysis
                'alignment_base': 0.88,      # Matches HB physics
                'recurrence_base': 4,        # Citations in literature
                'noise_base': 0.08          # Measurement uncertainty
            }
        }

    def validate_hb_model(self, model_id: str, experimental_data: Dict[str, Any],
                         fitted_parameters: HBModelParameters, material_type: str) -> HBValidationResult:
        """
        Validate HB model using UOIF promotion triggers.

        Args:
            model_id: Unique model identifier
            experimental_data: Experimental stress-strain data
            fitted_parameters: Fitted HB model parameters
            material_type: Type of material (polymer_melt, biological_fluid, etc.)

        Returns:
            HBValidationResult: Complete validation result
        """
        # Extract experimental data
        shear_rates = np.array(experimental_data['shear_rates'])
        measured_stresses = np.array(experimental_data['stresses'])

        # Calculate model predictions
        predicted_stresses = self._predict_hb_stresses(shear_rates, fitted_parameters)

        # Calculate validation metrics
        r_squared, rmse, mae = self._calculate_validation_metrics(
            measured_stresses, predicted_stresses
        )

        # Assess parameter uncertainties
        parameter_uncertainty = self._assess_parameter_uncertainty(fitted_parameters)

        # Generate UOIF components based on validation results
        uoif_components = self._generate_uoif_components(
            r_squared, rmse, material_type, fitted_parameters
        )

        # Calculate UOIF score
        confidence_score = self._calculate_uoif_confidence(uoif_components)

        # Determine promotion status
        promotion_status = self._determine_promotion_status(
            r_squared, rmse, mae, confidence_score, parameter_uncertainty, material_type
        )

        # Check validation criteria
        validation_criteria = self._check_validation_criteria(
            r_squared, rmse, mae, shear_rates, material_type
        )

        # Create validation result
        result = HBValidationResult(
            model_id=model_id,
            validation_timestamp=datetime.now(),
            r_squared=r_squared,
            rmse=rmse,
            mae=mae,
            parameter_uncertainty=parameter_uncertainty,
            promotion_status=promotion_status,
            confidence_score=confidence_score,
            uoif_components=uoif_components,
            material_type=material_type,
            shear_rate_range=(float(np.min(shear_rates)), float(np.max(shear_rates))),
            validation_criteria=validation_criteria
        )

        # Store in history
        if model_id not in self.validation_history:
            self.validation_history[model_id] = []
        self.validation_history[model_id].append(result)

        return result

    def _predict_hb_stresses(self, shear_rates: np.ndarray,
                           parameters: HBModelParameters) -> np.ndarray:
        """
        Predict stresses using HB model.

        Args:
            shear_rates: Array of shear rates
            parameters: HB model parameters

        Returns:
            Array of predicted stresses
        """
        tau_y = parameters.tau_y
        K = parameters.K
        n = parameters.n

        # HB constitutive equation: τ = τ_y + K·γ̇^n
        stresses = tau_y + K * np.power(np.maximum(shear_rates, 1e-10), n)

        return stresses

    def _calculate_validation_metrics(self, measured: np.ndarray,
                                    predicted: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate validation metrics: R², RMSE, MAE.

        Args:
            measured: Measured stress values
            predicted: Predicted stress values

        Returns:
            Tuple of (R², RMSE, MAE)
        """
        # Remove NaN and infinite values
        valid_mask = np.isfinite(measured) & np.isfinite(predicted)
        measured = measured[valid_mask]
        predicted = predicted[valid_mask]

        if len(measured) == 0:
            return 0.0, float('inf'), float('inf')

        # R² calculation
        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE calculation
        rmse = np.sqrt(np.mean((measured - predicted) ** 2))

        # MAE calculation
        mae = np.mean(np.abs(measured - predicted))

        return r_squared, rmse, mae

    def _assess_parameter_uncertainty(self, parameters: HBModelParameters) -> Dict[str, float]:
        """
        Assess relative parameter uncertainties.

        Args:
            parameters: HB model parameters with uncertainties

        Returns:
            Dict of relative uncertainties
        """
        return {
            'tau_y_relative': abs(parameters.tau_y_uncertainty / parameters.tau_y) if parameters.tau_y != 0 else float('inf'),
            'K_relative': abs(parameters.K_uncertainty / parameters.K) if parameters.K != 0 else float('inf'),
            'n_relative': abs(parameters.n_uncertainty / parameters.n) if parameters.n != 0 else float('inf'),
            'max_relative': max(
                abs(parameters.tau_y_uncertainty / parameters.tau_y) if parameters.tau_y != 0 else 0,
                abs(parameters.K_uncertainty / parameters.K) if parameters.K != 0 else 0,
                abs(parameters.n_uncertainty / parameters.n) if parameters.n != 0 else 0
            )
        }

    def _generate_uoif_components(self, r_squared: float, rmse: float,
                                material_type: str, parameters: HBModelParameters) -> ScoreComponents:
        """
        Generate UOIF components based on HB model validation results.

        Args:
            r_squared: Model R² value
            rmse: Root mean square error
            material_type: Type of material
            parameters: Fitted parameters

        Returns:
            ScoreComponents for UOIF scoring
        """
        base_components = self.config['uoif_components']

        # Adjust components based on validation results
        authority = base_components['authority_base']
        verifiability = base_components['verifiability_base']
        depth = base_components['depth_base']
        alignment = base_components['alignment_base']
        recurrence = base_components['recurrence_base']
        noise = base_components['noise_base']

        # R²-based adjustments
        if r_squared > 0.98:
            authority *= 1.1    # Stronger authority with excellent fit
            verifiability *= 1.1 # Higher verifiability
        elif r_squared > 0.95:
            authority *= 1.05
            verifiability *= 1.05
        else:
            authority *= 0.9    # Weaker authority with poor fit
            verifiability *= 0.9

        # RMSE-based adjustments
        if rmse < 1.0:
            depth *= 1.1        # Better depth with low error
            alignment *= 1.05
        elif rmse > 5.0:
            depth *= 0.85
            alignment *= 0.9
            noise *= 1.2       # Higher noise with high error

        # Material-specific adjustments
        material_config = self.config['material_criteria'].get(material_type, {})
        expected_r_squared = material_config.get('expected_r_squared', 0.95)

        if r_squared >= expected_r_squared:
            recurrence += 1     # Higher citation potential
        else:
            recurrence -= 1

        # Parameter uncertainty adjustments
        param_uncertainty = self._assess_parameter_uncertainty(parameters)
        max_uncertainty = param_uncertainty['max_relative']

        if max_uncertainty < 0.1:
            noise *= 0.8       # Lower noise with precise parameters
        elif max_uncertainty > 0.2:
            noise *= 1.3       # Higher noise with uncertain parameters

        return ScoreComponents(
            authority=min(1.0, authority),
            verifiability=min(1.0, verifiability),
            depth=min(1.0, depth),
            alignment=min(1.0, alignment),
            recurrence=max(0, recurrence),
            noise=min(1.0, noise)
        )

    def _calculate_uoif_confidence(self, components: ScoreComponents) -> float:
        """
        Calculate confidence score using UOIF scoring engine.

        Args:
            components: UOIF score components

        Returns:
            Confidence score (0-1)
        """
        # Use scoring engine for HB model domain
        result = self.scoring_engine.score_claim(
            claim_id="hb_validation_temp",
            raw_components={
                'authority': components.authority,
                'verifiability': components.verifiability,
                'depth': components.depth,
                'alignment': components.alignment,
                'recurrence': components.recurrence,
                'noise': components.noise
            },
            claim_type="primitive",
            domain="hb_models"
        )

        return result.confidence_level

    def _determine_promotion_status(self, r_squared: float, rmse: float, mae: float,
                                  confidence_score: float, parameter_uncertainty: Dict[str, float],
                                  material_type: str) -> str:
        """
        Determine promotion status based on validation results.

        Args:
            r_squared: Model R² value
            rmse, mae: Error metrics
            confidence_score: UOIF confidence score
            parameter_uncertainty: Parameter uncertainty assessment
            material_type: Type of material

        Returns:
            Promotion status string
        """
        thresholds = self.config['promotion_thresholds']

        # Check all promotion criteria
        criteria_met = {
            'r_squared': r_squared >= thresholds['r_squared'],
            'rmse': rmse <= thresholds['rmse_max'],
            'mae': mae <= thresholds['mae_max'],
            'confidence': confidence_score >= thresholds['confidence_min'],
            'parameter_uncertainty': parameter_uncertainty['max_relative'] <= thresholds['parameter_uncertainty_max']
        }

        if all(criteria_met.values()):
            return "Empirically Grounded"
        elif sum(criteria_met.values()) >= 3:
            return "Well Validated"
        elif sum(criteria_met.values()) >= 2:
            return "Acceptable"
        else:
            return "Needs Improvement"

    def _check_validation_criteria(self, r_squared: float, rmse: float,
                                 shear_rates: np.ndarray, material_type: str) -> Dict[str, bool]:
        """
        Check additional validation criteria for HB models.

        Args:
            r_squared: Model R² value
            rmse: Root mean square error
            shear_rates: Experimental shear rates
            material_type: Type of material

        Returns:
            Dict of validation criteria satisfaction
        """
        material_config = self.config['material_criteria'].get(material_type, {})
        min_shear_rate = material_config.get('min_shear_rate', 0.01)
        max_shear_rate = material_config.get('max_shear_rate', 1000)

        criteria = {}

        # Shear rate range coverage
        actual_min = np.min(shear_rates)
        actual_max = np.max(shear_rates)
        criteria['shear_rate_coverage'] = (
            actual_min <= min_shear_rate * 1.5 and  # Allow some tolerance
            actual_max >= max_shear_rate * 0.7
        )

        # R² meets material expectations
        expected_r_squared = material_config.get('expected_r_squared', 0.95)
        criteria['material_expectations'] = r_squared >= expected_r_squared

        # Sufficient data points
        criteria['sufficient_data'] = len(shear_rates) >= 10

        # No extreme outliers (RMSE not excessively high)
        criteria['reasonable_error'] = rmse < 10.0  # Pa

        return criteria

    def batch_validate_hb_models(self, models_data: List[Dict[str, Any]]) -> List[HBValidationResult]:
        """
        Validate multiple HB models in batch.

        Args:
            models_data: List of model validation data

        Returns:
            List of validation results
        """
        results = []

        for model_data in models_data:
            model_id = model_data['model_id']
            experimental_data = model_data['experimental_data']
            fitted_parameters = model_data['fitted_parameters']
            material_type = model_data['material_type']

            result = self.validate_hb_model(
                model_id, experimental_data, fitted_parameters, material_type
            )
            results.append(result)

        return results

    def get_validation_history(self, model_id: Optional[str] = None) -> List[HBValidationResult]:
        """
        Get validation history for HB models.

        Args:
            model_id: Optional specific model ID

        Returns:
            List of validation results
        """
        if model_id:
            return self.validation_history.get(model_id, [])
        return [result for results in self.validation_history.values() for result in results]

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive validation statistics.

        Returns:
            Dict containing validation statistics
        """
        all_results = self.get_validation_history()

        if not all_results:
            return {'total_models': 0, 'message': 'No validation data available'}

        # Basic statistics
        total_models = len(all_results)
        avg_r_squared = np.mean([r.r_squared for r in all_results])
        avg_rmse = np.mean([r.rmse for r in all_results])
        avg_confidence = np.mean([r.confidence_score for r in all_results])

        # Promotion status distribution
        promotion_counts = {}
        for result in all_results:
            status = result.promotion_status
            promotion_counts[status] = promotion_counts.get(status, 0) + 1

        # Material type distribution
        material_counts = {}
        for result in all_results:
            material = result.material_type
            material_counts[material] = material_counts.get(material, 0) + 1

        # Performance by material type
        material_performance = {}
        for material in material_counts.keys():
            material_results = [r for r in all_results if r.material_type == material]
            if material_results:
                material_performance[material] = {
                    'count': len(material_results),
                    'avg_r_squared': np.mean([r.r_squared for r in material_results]),
                    'avg_confidence': np.mean([r.confidence_score for r in material_results]),
                    'promotion_rate': sum(1 for r in material_results if r.promotion_status == "Empirically Grounded") / len(material_results)
                }

        return {
            'total_models': total_models,
            'avg_r_squared': avg_r_squared,
            'avg_rmse': avg_rmse,
            'avg_confidence': avg_confidence,
            'promotion_distribution': promotion_counts,
            'material_distribution': material_counts,
            'material_performance': material_performance,
            'empirically_grounded_rate': promotion_counts.get("Empirically Grounded", 0) / total_models if total_models > 0 else 0
        }

    def recommend_improvements(self, validation_result: HBValidationResult) -> List[str]:
        """
        Recommend improvements based on validation results.

        Args:
            validation_result: HB validation result

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        thresholds = self.config['promotion_thresholds']

        if validation_result.r_squared < thresholds['r_squared']:
            recommendations.append(".3f"
        if validation_result.rmse > thresholds['rmse_max']:
            recommendations.append(".1f"
        if validation_result.confidence_score < thresholds['confidence_min']:
            recommendations.append(".3f"
        if not all(validation_result.validation_criteria.values()):
            failed_criteria = [k for k, v in validation_result.validation_criteria.items() if not v]
            if failed_criteria:
                recommendations.append(f"Address validation criteria: {', '.join(failed_criteria)}")

        # Material-specific recommendations
        material_config = self.config['material_criteria'].get(validation_result.material_type, {})
        expected_r_squared = material_config.get('expected_r_squared', 0.95)

        if validation_result.r_squared < expected_r_squared:
            recommendations.append(f"Improve fit for {validation_result.material_type} (target R²: {expected_r_squared})")

        if not recommendations:
            recommendations.append("Model validation is strong - consider collecting additional data for robustness")

        return recommendations
