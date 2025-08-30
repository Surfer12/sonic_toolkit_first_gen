"""
Inverse Precision Framework - Mojo Implementation

This module implements the 0.9987 precision convergence criterion for high-performance
inverse analysis of complex multi-component systems. Optimized for computational
rheology and viscoelastic parameter extraction.
"""

from math import sqrt, abs, log, exp, pi
from memory import stack_allocation, heap_allocation
from collections import Dict, List
from algorithm import sort, min, max
from time import time
from random import random_float64
from utils import StringRef

# Import numerical libraries
from numojo import Matrix, Vector, SVD, QR, Eig, LinAlg
from numojo.linalg import inv, det, norm, cond, pinv, svd
from numojo.stats import mean, std, var, chi2_ppf

# Type aliases for clarity
alias FloatType = Float64
alias MatrixType = Matrix[FloatType]
alias VectorType = Vector[FloatType]
alias PrecisionType = Float64

# Core precision criteria structure
@register_passable("trivial")
struct PrecisionCriteria:
    """Precision convergence criteria for inverse analysis."""
    var relative_tolerance: PrecisionType
    var absolute_tolerance: PrecisionType
    var max_iterations: Int
    var min_step_size: PrecisionType
    var condition_number_threshold: PrecisionType
    var confidence_level: PrecisionType

    fn __init__(
        relative_tolerance: PrecisionType = 0.0013,  # 0.9987 precision
        absolute_tolerance: PrecisionType = 1e-8,
        max_iterations: Int = 100,
        min_step_size: PrecisionType = 1e-12,
        condition_number_threshold: PrecisionType = 1e12,
        confidence_level: PrecisionType = 0.95
    ) -> Self:
        return Self {
            relative_tolerance: relative_tolerance,
            absolute_tolerance: absolute_tolerance,
            max_iterations: max_iterations,
            min_step_size: min_step_size,
            condition_number_threshold: condition_number_threshold,
            confidence_level: confidence_level
        }

@register_passable("trivial")
struct ConvergenceMetrics:
    """Detailed convergence tracking metrics."""
    var iteration_count: Int
    var relative_errors: List[PrecisionType]
    var absolute_errors: List[PrecisionType]
    var condition_numbers: List[PrecisionType]
    var final_precision: PrecisionType
    var convergence_reason: StringRef

    fn __init__() -> Self:
        return Self {
            iteration_count: 0,
            relative_errors: List[PrecisionType](),
            absolute_errors: List[PrecisionType](),
            condition_numbers: List[PrecisionType](),
            final_precision: 0.0,
            convergence_reason: StringRef("Not converged")
        }

    fn reset(inout self):
        """Reset convergence metrics."""
        self.iteration_count = 0
        self.relative_errors.clear()
        self.absolute_errors.clear()
        self.condition_numbers.clear()
        self.final_precision = 0.0
        self.convergence_reason = StringRef("Not converged")

# Main inverse precision framework class
struct InversePrecisionFramework:
    """High-precision inverse analysis framework with 0.9987 convergence criterion."""

    var criteria: PrecisionCriteria
    var logger: Logger

    fn __init__(criteria: PrecisionCriteria = PrecisionCriteria()) -> Self:
        return Self {
            criteria: criteria,
            logger: Logger("InversePrecisionFramework")
        }

    fn check_convergence(self, k_current: VectorType, k_previous: VectorType) -> (Bool, Dict[StringRef, PrecisionType]):
        """
        Check convergence using 0.9987 precision criterion.

        Args:
            k_current: Current parameter estimates
            k_previous: Previous parameter estimates

        Returns:
            Tuple of (converged: Bool, metrics: Dict)
        """
        var metrics = Dict[StringRef, PrecisionType]()

        # Compute absolute difference
        var absolute_diff = norm(k_current - k_previous)

        # Compute relative difference (0.9987 criterion)
        var norm_current = norm(k_current)
        var relative_diff: PrecisionType = 0.0

        if norm_current > 0.0:
            relative_diff = absolute_diff / norm_current
        else:
            relative_diff = absolute_diff

        # Check convergence criteria
        var absolute_converged = absolute_diff <= self.criteria.absolute_tolerance
        var relative_converged = relative_diff <= self.criteria.relative_tolerance
        var converged = absolute_converged and relative_converged

        # Store metrics
        metrics["absolute_error"] = absolute_diff
        metrics["relative_error"] = relative_diff
        metrics["absolute_converged"] = PrecisionType(absolute_converged)
        metrics["relative_converged"] = PrecisionType(relative_converged)
        metrics["precision_achieved"] = 1.0 - relative_diff

        return converged, metrics

    fn inverse_extract_precise(
        self,
        measured_data: VectorType,
        component_matrix: MatrixType,
        initial_guess: VectorType,
        bounds: Optional[Tuple[VectorType, VectorType]] = None
    ) -> Dict[StringRef, AnyType]:
        """
        Perform high-precision inverse extraction with 0.9987 convergence.

        Args:
            measured_data: Measured system response [n_measurements]
            component_matrix: Component contribution matrix [n_measurements, n_components]
            initial_guess: Initial parameter estimates [n_components]
            bounds: Optional parameter bounds (lower, upper)

        Returns:
            Comprehensive extraction results with precision metrics
        """
        self.logger.info("Starting inverse extraction with 0.9987 precision criterion")
        self.logger.info("Problem size: " + str(measured_data.size) + " measurements, " +
                        str(component_matrix.cols) + " components")

        # Initialize convergence tracking
        var convergence_metrics = ConvergenceMetrics()
        var k_current = initial_guess
        var k_previous = VectorType.zeros(component_matrix.cols)

        # Pre-allocate arrays for efficiency
        var component_inv = MatrixType.zeros(component_matrix.cols, component_matrix.rows)

        # Main optimization loop with precision control
        for iteration in range(self.criteria.max_iterations):

            # Check matrix conditioning and compute inverse
            var condition_number = self._compute_condition_number(component_matrix)
            convergence_metrics.condition_numbers.append(condition_number)

            if condition_number > self.criteria.condition_number_threshold:
                self.logger.warning("Ill-conditioned matrix at iteration " + str(iteration) +
                                  ": κ = " + str(condition_number))
                # Use pseudo-inverse for stability
                component_inv = self._pseudo_inverse(component_matrix)
            else:
                component_inv = self._matrix_inverse(component_matrix)

            # Update parameter estimates
            k_previous = k_current
            k_current = component_inv @ measured_data

            # Apply bounds if specified
            if bounds:
                var lower_bounds, upper_bounds = bounds.value()
                k_current = self._apply_bounds(k_current, lower_bounds, upper_bounds)

            # Check convergence
            var converged, metrics = self.check_convergence(k_current, k_previous)

            # Store convergence history
            convergence_metrics.iteration_count = iteration + 1
            convergence_metrics.relative_errors.append(metrics["relative_error"])
            convergence_metrics.absolute_errors.append(metrics["absolute_error"])

            # Check for minimum step size
            var step_size = norm(k_current - k_previous)
            if step_size < self.criteria.min_step_size:
                convergence_metrics.convergence_reason = StringRef("Minimum step size reached")
                self.logger.info("Converged at iteration " + str(iteration + 1) + ": minimum step size")
                break

            if converged:
                convergence_metrics.convergence_reason = StringRef("0.9987 precision criterion met")
                convergence_metrics.final_precision = metrics["precision_achieved"]
                self.logger.info("Converged at iteration " + str(iteration + 1) +
                               ": precision = " + str(metrics["precision_achieved"]))
                break

        else:
            convergence_metrics.convergence_reason = StringRef("Maximum iterations reached")
            self.logger.warning("Maximum iterations reached without convergence")

        # Compute final results
        var predicted_data = component_matrix @ k_current
        var residuals = measured_data - predicted_data
        var success = convergence_metrics.final_precision > 0.99

        # Compute statistical metrics
        var statistical_metrics = self._compute_statistical_metrics(
            measured_data, predicted_data, component_matrix.cols
        )

        # Package results
        var results = Dict[StringRef, AnyType]()
        results["extracted_parameters"] = k_current
        results["convergence_metrics"] = convergence_metrics
        results["predicted_data"] = predicted_data
        results["residuals"] = residuals
        results["success"] = success
        results.update(statistical_metrics)

        return results

    fn _compute_condition_number(self, matrix: MatrixType) -> PrecisionType:
        """Compute matrix condition number using SVD."""
        var U, s, Vt = svd(matrix)

        if s.size > 0 and s[0] > 0.0 and s[s.size - 1] > 0.0:
            return s[0] / s[s.size - 1]
        else:
            return 1e15  # Very ill-conditioned

    fn _matrix_inverse(self, matrix: MatrixType) -> MatrixType:
        """Compute matrix inverse with error handling."""
        try:
            return inv(matrix)
        except:
            self.logger.error("Matrix inversion failed, using pseudo-inverse")
            return self._pseudo_inverse(matrix)

    fn _pseudo_inverse(self, matrix: MatrixType) -> MatrixType:
        """Compute Moore-Penrose pseudo-inverse."""
        var U, s, Vt = svd(matrix)

        # Apply regularization to singular values
        var s_inv = VectorType.zeros(s.size)
        var rcond = 1e-15

        for i in range(s.size):
            if s[i] > rcond * s[0]:
                s_inv[i] = 1.0 / s[i]

        # Compute pseudo-inverse
        var S_inv = MatrixType.diag(s_inv)
        return Vt.T @ S_inv @ U.T

    fn _apply_bounds(self, params: VectorType, lower_bounds: VectorType, upper_bounds: VectorType) -> VectorType:
        """Apply parameter bounds."""
        var bounded_params = params

        for i in range(params.size):
            if params[i] < lower_bounds[i]:
                bounded_params[i] = lower_bounds[i]
            elif params[i] > upper_bounds[i]:
                bounded_params[i] = upper_bounds[i]

        return bounded_params

    fn _compute_statistical_metrics(
        self,
        measured: VectorType,
        predicted: VectorType,
        n_parameters: Int
    ) -> Dict[StringRef, AnyType]:
        """
        Compute statistical metrics for parameter estimation quality.

        Args:
            measured: Measured data
            predicted: Predicted data
            n_parameters: Number of fitted parameters

        Returns:
            Statistical metrics dictionary
        """
        var residuals = measured - predicted
        var n_data = measured.size
        var n_dof = n_data - n_parameters

        if n_dof <= 0:
            var metrics = Dict[StringRef, AnyType]()
            metrics["warning"] = StringRef("Insufficient degrees of freedom")
            return metrics

        # Basic statistics
        var mse = mean(residuals * residuals)
        var rmse = sqrt(mse)
        var mae = mean(abs(residuals))

        # R-squared
        var measured_mean = mean(measured)
        var ss_res = sum(residuals * residuals)
        var ss_tot = sum((measured - measured_mean) * (measured - measured_mean))
        var r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        # Chi-squared test
        var chi_squared = sum(residuals * residuals)
        var chi_squared_reduced = chi_squared / PrecisionType(n_dof)

        # Confidence intervals (simplified)
        var confidence_interval = sqrt(chi2_ppf(self.criteria.confidence_level, n_dof)) * rmse

        # Package metrics
        var metrics = Dict[StringRef, AnyType]()
        metrics["mse"] = mse
        metrics["rmse"] = rmse
        metrics["mae"] = mae
        metrics["r_squared"] = r_squared
        metrics["chi_squared"] = chi_squared
        metrics["chi_squared_reduced"] = chi_squared_reduced
        metrics["confidence_interval"] = confidence_interval
        metrics["degrees_of_freedom"] = n_dof

        return metrics

# High-performance logger for Mojo
struct Logger:
    """Simple high-performance logger."""
    var name: StringRef

    fn __init__(name: StringRef) -> Self:
        return Self { name: name }

    fn info(self, message: StringRef):
        """Log info message."""
        print("[INFO] " + self.name + ": " + message)

    fn warning(self, message: StringRef):
        """Log warning message."""
        print("[WARNING] " + self.name + ": " + message)

    fn error(self, message: StringRef):
        """Log error message."""
        print("[ERROR] " + self.name + ": " + message)

# Utility functions for enhanced performance
fn vector_norm(v: VectorType) -> PrecisionType:
    """Compute vector Euclidean norm."""
    var sum_squares: PrecisionType = 0.0
    for i in range(v.size):
        sum_squares += v[i] * v[i]
    return sqrt(sum_squares)

fn matrix_norm(matrix: MatrixType) -> PrecisionType:
    """Compute matrix Frobenius norm."""
    var sum_squares: PrecisionType = 0.0
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            sum_squares += matrix[i, j] * matrix[i, j]
    return sqrt(sum_squares)

fn compute_svd(matrix: MatrixType) -> Tuple[MatrixType, VectorType, MatrixType]:
    """Compute SVD decomposition."""
    return svd(matrix)

# Enhanced viscoelastic parameter extraction
struct VEHBPrecisionExtractor:
    """
    High-precision VEHB parameter extraction with 0.9987 convergence.

    Specialized for viscoelastic Herschel-Bulkley model parameter estimation.
    """

    var precision_framework: InversePrecisionFramework
    var vehb_parameters: Dict[StringRef, PrecisionType]

    fn __init__(base_params: Dict[StringRef, PrecisionType] = Dict[StringRef, PrecisionType]()) -> Self:
        var criteria = PrecisionCriteria()
        var framework = InversePrecisionFramework(criteria)

        return Self {
            precision_framework: framework,
            vehb_parameters: base_params
        }

    fn extract_parameters(
        self,
        experimental_stress: VectorType,
        shear_rates: VectorType,
        times: VectorType,
        known_params: Dict[StringRef, PrecisionType]
    ) -> Dict[StringRef, AnyType]:
        """
        Extract VEHB parameters using high-precision inverse analysis.

        Args:
            experimental_stress: Measured stress data [Pa]
            shear_rates: Applied shear rates [1/s]
            times: Time points [s]
            known_params: Known parameters

        Returns:
            Extraction results with precision metrics
        """
        # Build component matrix for VEHB model
        var component_matrix = self._build_vehb_component_matrix(shear_rates, times, known_params)

        # Create initial guess for unknown parameters
        var initial_guess = self._create_initial_guess(known_params)

        # Set up parameter bounds
        var bounds = self._get_parameter_bounds(known_params)

        # Perform high-precision extraction
        var extraction_results = self.precision_framework.inverse_extract_precise(
            experimental_stress, component_matrix, initial_guess, bounds
        )

        # Map results back to parameter names
        if extraction_results["success"]:
            var extracted_params = self._map_results_to_parameters(
                extraction_results["extracted_parameters"], known_params
            )
            extraction_results["extracted_parameters_named"] = extracted_params

        return extraction_results

    fn _build_vehb_component_matrix(
        self,
        shear_rates: VectorType,
        times: VectorType,
        known_params: Dict[StringRef, PrecisionType]
    ) -> MatrixType:
        """
        Build component matrix for VEHB model inverse analysis.

        Each column represents the contribution of one unknown parameter
        to the total stress prediction.
        """
        var n_measurements = shear_rates.size
        var unknown_params = self._get_unknown_parameters(known_params)
        var n_unknown = len(unknown_params)

        var component_matrix = MatrixType.zeros(n_measurements, n_unknown)

        # Create base parameter set
        var base_params = self._create_base_parameters(known_params)

        for i in range(n_unknown):
            var param_name = unknown_params[i]
            for j in range(n_measurements):
                # Compute sensitivity of stress to this parameter
                component_matrix[j, i] = self._compute_parameter_sensitivity(
                    base_params, param_name, shear_rates[j], times[j]
                )

        return component_matrix

    fn _compute_parameter_sensitivity(
        self,
        params: Dict[StringRef, PrecisionType],
        parameter_name: StringRef,
        shear_rate: PrecisionType,
        time: PrecisionType
    ) -> PrecisionType:
        """
        Compute sensitivity of stress to a parameter change.

        Uses finite difference sensitivity analysis.
        """
        var epsilon = 1e-8

        # Create perturbed parameter set
        var perturbed_params = params.copy()
        perturbed_params[parameter_name] = params[parameter_name] + epsilon

        # Compute stresses with original and perturbed parameters
        var stress_original = self._compute_vehb_stress(params, shear_rate, time)
        var stress_perturbed = self._compute_vehb_stress(perturbed_params, shear_rate, time)

        # Return sensitivity (finite difference)
        return (stress_perturbed - stress_original) / epsilon

    fn _compute_vehb_stress(
        self,
        params: Dict[StringRef, PrecisionType],
        shear_rate: PrecisionType,
        time: PrecisionType
    ) -> PrecisionType:
        """
        Compute VEHB stress for given parameters.

        Simplified implementation - in practice, this would include
        full viscoelastic constitutive equations.
        """
        var tau_y = params.get("tau_y", 0.0)
        var K = params.get("K", 1000.0)
        var n = params.get("n", 0.8)
        var G0 = params.get("G0", 1e6)
        var Ge = params.get("Ge", 1e5)
        var tau_relax = params.get("tau_relax", 1.0)
        var eta_inf = params.get("eta_inf", 100.0)

        # HB stress contribution
        var tau_hb: PrecisionType = 0.0
        if shear_rate > 0.0:
            tau_hb = K * pow(shear_rate, n)

        # Viscoelastic contribution (simplified relaxation)
        var tau_ve = (G0 - Ge) * exp(-time / tau_relax) * shear_rate

        # Total stress
        var tau_total = tau_y + tau_hb + tau_ve

        return max(0.0, tau_total)  # Ensure non-negative stress

    fn _get_unknown_parameters(self, known_params: Dict[StringRef, PrecisionType]) -> List[StringRef]:
        """Get list of unknown parameters."""
        var all_params = List[StringRef]()
        all_params.append("tau_y")
        all_params.append("K")
        all_params.append("n")
        all_params.append("G0")
        all_params.append("Ge")
        all_params.append("tau_relax")
        all_params.append("eta_inf")

        var unknown_params = List[StringRef]()

        for i in range(len(all_params)):
            var param = all_params[i]
            if not known_params.contains(param):
                unknown_params.append(param)

        return unknown_params

    fn _create_base_parameters(self, known_params: Dict[StringRef, PrecisionType]) -> Dict[StringRef, PrecisionType]:
        """Create base parameter set with known and default values."""
        var base_params = Dict[StringRef, PrecisionType]()

        # Default values
        base_params["tau_y"] = 100.0      # Pa
        base_params["K"] = 1000.0         # Pa·s^n
        base_params["n"] = 0.8            # Dimensionless
        base_params["G0"] = 1e6           # Pa
        base_params["Ge"] = 1e5           # Pa
        base_params["tau_relax"] = 1.0    # s
        base_params["eta_inf"] = 100.0    # Pa·s

        # Override with known parameters
        for key in known_params.keys():
            base_params[key] = known_params[key]

        return base_params

    fn _create_initial_guess(self, known_params: Dict[StringRef, PrecisionType]) -> VectorType:
        """Create initial guess for unknown parameters."""
        var unknown_params = self._get_unknown_parameters(known_params)
        var initial_guess = VectorType.zeros(len(unknown_params))

        for i in range(len(unknown_params)):
            var param = unknown_params[i]
            if param == "tau_y":
                initial_guess[i] = 10.0
            elif param == "K":
                initial_guess[i] = 1000.0
            elif param == "n":
                initial_guess[i] = 0.8
            elif param == "G0":
                initial_guess[i] = 1e6
            elif param == "Ge":
                initial_guess[i] = 1e5
            elif param == "tau_relax":
                initial_guess[i] = 1.0
            elif param == "eta_inf":
                initial_guess[i] = 100.0

        return initial_guess

    fn _get_parameter_bounds(self, known_params: Dict[StringRef, PrecisionType]) -> Tuple[VectorType, VectorType]:
        """Get parameter bounds for optimization."""
        var unknown_params = self._get_unknown_parameters(known_params)
        var lower_bounds = VectorType.zeros(len(unknown_params))
        var upper_bounds = VectorType.zeros(len(unknown_params))

        for i in range(len(unknown_params)):
            var param = unknown_params[i]
            if param == "tau_y":
                lower_bounds[i] = 0.0
                upper_bounds[i] = 1000.0
            elif param == "K":
                lower_bounds[i] = 1.0
                upper_bounds[i] = 1e6
            elif param == "n":
                lower_bounds[i] = 0.1
                upper_bounds[i] = 1.5
            elif param in ["G0", "Ge"]:
                lower_bounds[i] = 1e3
                upper_bounds[i] = 1e9
            elif param == "tau_relax":
                lower_bounds[i] = 0.001
                upper_bounds[i] = 100.0
            elif param == "eta_inf":
                lower_bounds[i] = 1.0
                upper_bounds[i] = 1e6

        return (lower_bounds, upper_bounds)

    fn _map_results_to_parameters(
        self,
        extracted_values: VectorType,
        known_params: Dict[StringRef, PrecisionType]
    ) -> Dict[StringRef, PrecisionType]:
        """Map extracted values back to parameter names."""
        var unknown_params = self._get_unknown_parameters(known_params)
        var extracted_params = known_params.copy()

        for i in range(len(unknown_params)):
            var param_name = unknown_params[i]
            extracted_params[param_name] = extracted_values[i]

        return extracted_params

# Example usage and testing functions
fn test_precision_framework():
    """Test the inverse precision framework."""
    print("Testing Inverse Precision Framework with 0.9987 convergence criterion...")

    # Create synthetic test data
    var n_measurements = 50
    var n_components = 5

    # Generate component matrix
    var component_matrix = MatrixType.randn(n_measurements, n_components)
    var scales = VectorType.ones(n_components)
    for i in range(n_components):
        scales[i] = 1.0 / vector_norm(component_matrix.col(i))

    for i in range(n_components):
        for j in range(n_measurements):
            component_matrix[j, i] *= scales[i]

    # Generate true parameters and measured data
    var true_parameters = VectorType([1.0, 2.0, -1.5, 3.0, 0.5])
    var measured_data = component_matrix @ true_parameters

    # Add noise
    var noise_level = 0.01
    for i in range(n_measurements):
        measured_data[i] *= 1.0 + random_float64() * noise_level * 2.0 - noise_level

    # Initial guess
    var initial_guess = VectorType.ones(n_components)

    # Create framework and perform extraction
    var framework = InversePrecisionFramework()
    var results = framework.inverse_extract_precise(measured_data, component_matrix, initial_guess)

    # Print results
    if results["success"]:
        print("✓ Extraction successful!")
        print("Final precision: " + str(results["convergence_metrics"].final_precision))
        print("Iterations: " + str(results["convergence_metrics"].iteration_count))

        var extracted_params = results["extracted_parameters"]
        print("Extracted parameters:")
        for i in range(extracted_params.size):
            print("  param_" + str(i) + ": " + str(extracted_params[i]) +
                  " (true: " + str(true_parameters[i]) + ")")
    else:
        print("✗ Extraction failed")

fn test_vehb_extraction():
    """Test VEHB parameter extraction."""
    print("\nTesting VEHB Parameter Extraction...")

    # Create synthetic VEHB data
    var n_points = 20
    var shear_rates = VectorType.zeros(n_points)
    var times = VectorType.zeros(n_points)
    var experimental_stress = VectorType.zeros(n_points)

    # Generate test data
    var true_params = Dict[StringRef, PrecisionType]()
    true_params["tau_y"] = 100.0
    true_params["K"] = 1000.0
    true_params["n"] = 0.8
    true_params["G0"] = 1e6
    true_params["Ge"] = 1e5
    true_params["tau_relax"] = 1.0
    true_params["eta_inf"] = 100.0

    for i in range(n_points):
        shear_rates[i] = pow(10.0, -2.0 + 4.0 * PrecisionType(i) / PrecisionType(n_points))
        times[i] = 10.0 * PrecisionType(i) / PrecisionType(n_points)

        # Compute true stress (simplified)
        var tau_y = true_params["tau_y"]
        var K = true_params["K"]
        var n = true_params["n"]
        var G0 = true_params["G0"]
        var Ge = true_params["Ge"]
        var tau_relax = true_params["tau_relax"]

        var tau_hb = K * pow(shear_rates[i], n)
        var tau_ve = (G0 - Ge) * exp(-times[i] / tau_relax) * shear_rates[i]
        experimental_stress[i] = tau_y + tau_hb + tau_ve

    # Add noise
    for i in range(n_points):
        experimental_stress[i] *= 1.0 + 0.05 * (random_float64() * 2.0 - 1.0)

    # Test extraction with some known parameters
    var known_params = Dict[StringRef, PrecisionType]()
    known_params["n"] = 0.8
    known_params["G0"] = 1e6
    known_params["Ge"] = 1e5
    known_params["tau_relax"] = 1.0
    known_params["eta_inf"] = 100.0

    var extractor = VEHBPrecisionExtractor()
    var results = extractor.extract_parameters(experimental_stress, shear_rates, times, known_params)

    if results["success"]:
        print("✓ VEHB extraction successful!")
        print("Final precision: " + str(results["convergence_metrics"].final_precision))

        var extracted_params = results["extracted_parameters_named"]
        print("Extracted parameters:")
        for key in extracted_params.keys():
            var true_value = true_params.get(key, 0.0)
            print("  " + key + ": " + str(extracted_params[key]) +
                  " (true: " + str(true_value) + ")")
    else:
        print("✗ VEHB extraction failed")

# Main execution
fn main():
    """Main execution function."""
    print("=== Inverse Precision Framework - Mojo Implementation ===")
    print("0.9987 Convergence Criterion for Complex Rheology Analysis")
    print("=========================================================")

    # Run tests
    test_precision_framework()
    test_vehb_extraction()

    print("\n=== Framework Ready for High-Precision Inverse Analysis ===")

# Execute main function
main()
