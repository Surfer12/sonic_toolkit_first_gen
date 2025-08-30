"""
UOIF Decision Equations with Toolkit Integration

This module implements the UOIF decision equations integrated with the scientific
computing toolkit's trust region methods for confidence-constrained optimization.

Key Equation: Ψ(x) = [αS + (1-α)N] · exp(-[λ₁R_auth + λ₂R_ver]) · P(H|E,β)

Integration Features:
- Trust region constrained optimization
- Confidence bounds as optimization constraints
- Real-time parameter adaptation
- Performance monitoring integration

Author: Scientific Computing Toolkit Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import numpy as np
from scipy import optimize
from scipy import stats

from .allocation_system import AllocationSystem, AllocationResult


@dataclass
class DecisionParameters:
    """Parameters for UOIF decision equation."""
    alpha: float          # Evidence allocation (0-1)
    lambda1: float        # Authority risk penalty
    lambda2: float        # Verifiability risk penalty
    beta: float          # Uplift factor (≥1)
    evidence_strength: float  # S parameter
    canonical_strength: float  # N parameter
    authority_risk: float     # R_auth parameter
    verifiability_risk: float  # R_ver parameter


@dataclass
class OptimizationResult:
    """Result of confidence-constrained optimization."""
    optimal_parameters: DecisionParameters
    psi_value: float
    confidence_bounds: Tuple[float, float]
    optimization_success: bool
    iterations: int
    convergence_metric: float
    constraint_satisfaction: Dict[str, bool]


class DecisionEquations:
    """
    Implements UOIF decision equations with toolkit trust region integration.

    This class provides confidence-constrained optimization using the Ψ(x) decision
    equation integrated with the scientific computing toolkit's optimization methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decision equations system.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.allocation_system = AllocationSystem()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for decision equations."""
        return {
            'optimization': {
                'method': 'trust-constr',
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'confidence_constraint': 0.95,
                'parameter_bounds': {
                    'alpha': (0.0, 1.0),
                    'lambda1': (0.0, 2.0),
                    'lambda2': (0.0, 2.0),
                    'beta': (1.0, 2.0)
                }
            },
            'constraints': {
                'min_confidence': 0.75,
                'max_risk_penalty': 1.5,
                'evidence_balance_tolerance': 0.1
            },
            'toolkit_integration': {
                'use_bootstrap_ci': True,
                'performance_weight': 0.3,
                'correlation_threshold': 0.9987
            }
        }

    def psi_function(self, params: DecisionParameters) -> float:
        """
        Calculate Ψ(x) value using the UOIF decision equation.

        Ψ(x) = [αS + (1-α)N] · exp(-[λ₁R_auth + λ₂R_ver]) · β

        Args:
            params: Decision parameters

        Returns:
            Ψ(x) value (0-1)
        """
        # Evidence combination
        evidence_term = (params.alpha * params.evidence_strength +
                        (1 - params.alpha) * params.canonical_strength)

        # Risk penalty
        risk_penalty = (params.lambda1 * params.authority_risk +
                       params.lambda2 * params.verifiability_risk)

        # Complete Ψ(x) calculation
        psi_value = evidence_term * np.exp(-risk_penalty) * params.beta

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, psi_value))

    def psi_gradient(self, params: DecisionParameters) -> np.ndarray:
        """
        Calculate gradient of Ψ(x) with respect to parameters.

        Args:
            params: Decision parameters

        Returns:
            Gradient vector
        """
        alpha, lambda1, lambda2, beta = params.alpha, params.lambda1, params.lambda2, params.beta
        S, N = params.evidence_strength, params.canonical_strength
        R_auth, R_ver = params.authority_risk, params.verifiability_risk

        # Current Ψ value
        psi = self.psi_function(params)

        # Gradient components
        dpsi_dalpha = (S - N) * np.exp(-lambda1 * R_auth - lambda2 * R_ver) * beta
        dpsi_dlambda1 = -R_auth * psi / beta  # Simplified
        dpsi_dlambda2 = -R_ver * psi / beta   # Simplified
        dpsi_dbeta = psi / beta

        return np.array([dpsi_dalpha, dpsi_dlambda1, dpsi_dlambda2, dpsi_dbeta])

    def optimize_decision_parameters(self, evidence_data: Dict[str, Any],
                                   constraints: Optional[Dict[str, Any]] = None,
                                   toolkit_performance: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize decision parameters using confidence-constrained trust region method.

        Args:
            evidence_data: Evidence and risk data
            constraints: Optional additional constraints
            toolkit_performance: Optional toolkit performance metrics

        Returns:
            OptimizationResult: Complete optimization result
        """
        # Extract evidence data
        S = evidence_data.get('evidence_strength', 0.8)
        N = evidence_data.get('canonical_strength', 0.7)
        R_auth = evidence_data.get('authority_risk', 0.2)
        R_ver = evidence_data.get('verifiability_risk', 0.1)

        # Initial parameters
        initial_params = np.array([0.5, 0.5, 0.5, 1.2])  # alpha, lambda1, lambda2, beta

        # Parameter bounds
        bounds = self._get_parameter_bounds()

        # Objective function (maximize Ψ(x))
        def objective(x):
            params = DecisionParameters(
                alpha=x[0], lambda1=x[1], lambda2=x[2], beta=x[3],
                evidence_strength=S, canonical_strength=N,
                authority_risk=R_auth, verifiability_risk=R_ver
            )
            return -self.psi_function(params)  # Negative for minimization

        # Constraints
        constraint_list = self._build_constraints(S, N, R_auth, R_ver, constraints)

        # Optimization using trust region method
        result = optimize.minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=constraint_list,
            options={
                'maxiter': self.config['optimization']['max_iterations'],
                'xtol': self.config['optimization']['tolerance'],
                'gtol': self.config['optimization']['tolerance']
            }
        )

        # Extract optimal parameters
        optimal_params = DecisionParameters(
            alpha=result.x[0], lambda1=result.x[1], lambda2=result.x[2], beta=result.x[3],
            evidence_strength=S, canonical_strength=N,
            authority_risk=R_auth, verifiability_risk=R_ver
        )

        # Calculate final Ψ value
        final_psi = self.psi_function(optimal_params)

        # Calculate confidence bounds
        confidence_bounds = self._calculate_confidence_bounds(optimal_params, result, toolkit_performance)

        # Check constraint satisfaction
        constraint_satisfaction = self._check_constraint_satisfaction(
            optimal_params, final_psi, constraints
        )

        optimization_result = OptimizationResult(
            optimal_parameters=optimal_params,
            psi_value=final_psi,
            confidence_bounds=confidence_bounds,
            optimization_success=result.success,
            iterations=result.nit,
            convergence_metric=result.fun,
            constraint_satisfaction=constraint_satisfaction
        )

        return optimization_result

    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds_config = self.config['optimization']['parameter_bounds']
        return [
            bounds_config['alpha'],
            bounds_config['lambda1'],
            bounds_config['lambda2'],
            bounds_config['beta']
        ]

    def _build_constraints(self, S: float, N: float, R_auth: float, R_ver: float,
                          additional_constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build optimization constraints.

        Args:
            S, N: Evidence strengths
            R_auth, R_ver: Risk values
            additional_constraints: Additional constraint specifications

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []

        # Minimum confidence constraint
        def confidence_constraint(x):
            params = DecisionParameters(
                alpha=x[0], lambda1=x[1], lambda2=x[2], beta=x[3],
                evidence_strength=S, canonical_strength=N,
                authority_risk=R_auth, verifiability_risk=R_ver
            )
            psi = self.psi_function(params)
            return psi - self.config['constraints']['min_confidence']

        constraints.append({
            'type': 'ineq',
            'fun': confidence_constraint,
            'jac': self._confidence_constraint_jacobian(S, N, R_auth, R_ver)
        })

        # Risk penalty constraint
        def risk_penalty_constraint(x):
            return self.config['constraints']['max_risk_penalty'] - (x[1] + x[2])

        constraints.append({
            'type': 'ineq',
            'fun': risk_penalty_constraint
        })

        # Evidence balance constraint (optional)
        if self.config['constraints'].get('evidence_balance_tolerance'):
            def evidence_balance_constraint(x):
                alpha = x[0]
                # Prefer balanced evidence allocation
                return self.config['constraints']['evidence_balance_tolerance'] - abs(alpha - 0.5)

            constraints.append({
                'type': 'ineq',
                'fun': evidence_balance_constraint
            })

        return constraints

    def _confidence_constraint_jacobian(self, S: float, N: float, R_auth: float, R_ver: float):
        """
        Calculate Jacobian for confidence constraint.

        Args:
            S, N: Evidence strengths
            R_auth, R_ver: Risk values

        Returns:
            Jacobian function for constraint
        """
        def jacobian(x):
            alpha, lambda1, lambda2, beta = x[0], x[1], x[2], x[3]

            # Current Ψ value and its derivatives
            evidence_term = alpha * S + (1 - alpha) * N
            risk_penalty = lambda1 * R_auth + lambda2 * R_ver
            exp_term = np.exp(-risk_penalty)
            psi = evidence_term * exp_term * beta

            # Derivatives w.r.t. parameters
            dpsi_dalpha = (S - N) * exp_term * beta
            dpsi_dlambda1 = -R_auth * psi
            dpsi_dlambda2 = -R_ver * psi
            dpsi_dbeta = evidence_term * exp_term

            return np.array([dpsi_dalpha, dpsi_dlambda1, dpsi_dlambda2, dpsi_dbeta])

        return jacobian

    def _calculate_confidence_bounds(self, params: DecisionParameters,
                                   optimization_result: Any,
                                   toolkit_performance: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Calculate confidence bounds for the optimized Ψ value.

        Args:
            params: Optimal decision parameters
            optimization_result: Scipy optimization result
            toolkit_performance: Toolkit performance metrics

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Base confidence from optimization
        base_psi = self.psi_function(params)

        # Uncertainty from parameter covariance (if available)
        if hasattr(optimization_result, 'hess_inv'):
            try:
                # Calculate parameter uncertainty
                param_uncertainty = np.sqrt(np.diag(optimization_result.hess_inv.todense()))
                psi_uncertainty = self._propagate_uncertainty(params, param_uncertainty)
            except:
                psi_uncertainty = 0.05  # Default uncertainty
        else:
            psi_uncertainty = 0.05

        # Adjust for toolkit performance
        if toolkit_performance:
            correlation = toolkit_performance.get('correlation', 0.9)
            if correlation >= self.config['toolkit_integration']['correlation_threshold']:
                psi_uncertainty *= 0.8  # Reduce uncertainty for high correlation

        # Calculate bounds
        lower_bound = max(0.0, base_psi - 1.96 * psi_uncertainty)
        upper_bound = min(1.0, base_psi + 1.96 * psi_uncertainty)

        return (lower_bound, upper_bound)

    def _propagate_uncertainty(self, params: DecisionParameters,
                             param_uncertainty: np.ndarray) -> float:
        """
        Propagate parameter uncertainty to Ψ uncertainty.

        Args:
            params: Decision parameters
            param_uncertainty: Parameter uncertainties

        Returns:
            Ψ uncertainty estimate
        """
        # Calculate gradient at optimal point
        gradient = self.psi_gradient(params)

        # Propagate uncertainty using first-order approximation
        psi_uncertainty = np.sqrt(np.sum((gradient * param_uncertainty) ** 2))

        return psi_uncertainty

    def _check_constraint_satisfaction(self, params: DecisionParameters, psi_value: float,
                                     constraints: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Check if all constraints are satisfied.

        Args:
            params: Decision parameters
            psi_value: Final Ψ value
            constraints: Additional constraints

        Returns:
            Dict of constraint satisfaction status
        """
        satisfaction = {}

        # Confidence constraint
        satisfaction['min_confidence'] = psi_value >= self.config['constraints']['min_confidence']

        # Risk penalty constraint
        total_risk_penalty = params.lambda1 + params.lambda2
        satisfaction['max_risk_penalty'] = total_risk_penalty <= self.config['constraints']['max_risk_penalty']

        # Evidence balance constraint
        if self.config['constraints'].get('evidence_balance_tolerance'):
            balance_deviation = abs(params.alpha - 0.5)
            satisfaction['evidence_balance'] = balance_deviation <= self.config['constraints']['evidence_balance_tolerance']
        else:
            satisfaction['evidence_balance'] = True

        # Parameter bounds
        satisfaction['parameter_bounds'] = (
            0.0 <= params.alpha <= 1.0 and
            0.0 <= params.lambda1 <= 2.0 and
            0.0 <= params.lambda2 <= 2.0 and
            1.0 <= params.beta <= 2.0
        )

        return satisfaction

    def adaptive_optimization(self, evidence_data: Dict[str, Any],
                            adaptation_history: List[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Perform adaptive optimization with historical parameter adjustment.

        Args:
            evidence_data: Current evidence data
            adaptation_history: Previous optimization results

        Returns:
            Adaptive optimization result
        """
        if not adaptation_history:
            # First-time optimization
            return self.optimize_decision_parameters(evidence_data)

        # Extract successful historical parameters
        successful_params = [
            result['optimal_parameters']
            for result in adaptation_history
            if result.get('optimization_success', False)
        ]

        if not successful_params:
            return self.optimize_decision_parameters(evidence_data)

        # Calculate parameter adaptation
        adapted_initial = self._adapt_initial_parameters(successful_params, evidence_data)

        # Perform optimization with adapted initial guess
        result = self.optimize_decision_parameters(evidence_data)

        # Update result with adaptation information
        result.optimal_parameters = self._blend_parameters(
            result.optimal_parameters, adapted_initial, blend_factor=0.3
        )

        return result

    def _adapt_initial_parameters(self, historical_params: List[DecisionParameters],
                                current_evidence: Dict[str, Any]) -> DecisionParameters:
        """
        Adapt initial parameters based on historical successful optimizations.

        Args:
            historical_params: List of successful historical parameters
            current_evidence: Current evidence data

        Returns:
            Adapted initial parameters
        """
        if not historical_params:
            return DecisionParameters(
                alpha=0.5, lambda1=0.5, lambda2=0.5, beta=1.2,
                evidence_strength=current_evidence.get('evidence_strength', 0.8),
                canonical_strength=current_evidence.get('canonical_strength', 0.7),
                authority_risk=current_evidence.get('authority_risk', 0.2),
                verifiability_risk=current_evidence.get('verifiability_risk', 0.1)
            )

        # Average historical parameters
        avg_alpha = np.mean([p.alpha for p in historical_params])
        avg_lambda1 = np.mean([p.lambda1 for p in historical_params])
        avg_lambda2 = np.mean([p.lambda2 for p in historical_params])
        avg_beta = np.mean([p.beta for p in historical_params])

        return DecisionParameters(
            alpha=avg_alpha,
            lambda1=avg_lambda1,
            lambda2=avg_lambda2,
            beta=avg_beta,
            evidence_strength=current_evidence.get('evidence_strength', 0.8),
            canonical_strength=current_evidence.get('canonical_strength', 0.7),
            authority_risk=current_evidence.get('authority_risk', 0.2),
            verifiability_risk=current_evidence.get('verifiability_risk', 0.1)
        )

    def _blend_parameters(self, optimal: DecisionParameters, adapted: DecisionParameters,
                        blend_factor: float = 0.3) -> DecisionParameters:
        """
        Blend optimal and adapted parameters.

        Args:
            optimal: Optimization result parameters
            adapted: Adapted initial parameters
            blend_factor: Blending factor (0-1)

        Returns:
            Blended parameters
        """
        return DecisionParameters(
            alpha=(1 - blend_factor) * optimal.alpha + blend_factor * adapted.alpha,
            lambda1=(1 - blend_factor) * optimal.lambda1 + blend_factor * adapted.lambda1,
            lambda2=(1 - blend_factor) * optimal.lambda2 + blend_factor * adapted.lambda2,
            beta=(1 - blend_factor) * optimal.beta + blend_factor * adapted.beta,
            evidence_strength=optimal.evidence_strength,
            canonical_strength=optimal.canonical_strength,
            authority_risk=optimal.authority_risk,
            verifiability_risk=optimal.verifiability_risk
        )

    def sensitivity_analysis(self, evidence_data: Dict[str, Any],
                           parameter_ranges: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on decision parameters.

        Args:
            evidence_data: Base evidence data
            parameter_ranges: Ranges for sensitivity analysis

        Returns:
            Sensitivity analysis results
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'alpha': (0.1, 0.9),
                'lambda1': (0.1, 1.0),
                'lambda2': (0.1, 1.0),
                'beta': (1.0, 1.5)
            }

        sensitivity_results = {}

        # Base case
        base_result = self.optimize_decision_parameters(evidence_data)
        base_psi = base_result.psi_value

        # Test parameter variations
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_sensitivity = []

            test_values = np.linspace(min_val, max_val, 10)
            for test_val in test_values:
                # Modify evidence data with test parameter
                test_evidence = evidence_data.copy()

                # Set test parameter (this is a simplified approach)
                if param_name in ['evidence_strength', 'canonical_strength', 'authority_risk', 'verifiability_risk']:
                    test_evidence[param_name] = test_val

                # Optimize with modified evidence
                test_result = self.optimize_decision_parameters(test_evidence)
                psi_change = test_result.psi_value - base_psi

                param_sensitivity.append({
                    'parameter_value': test_val,
                    'psi_value': test_result.psi_value,
                    'psi_change': psi_change,
                    'relative_change': psi_change / base_psi if base_psi > 0 else 0
                })

            sensitivity_results[param_name] = param_sensitivity

        return {
            'base_psi': base_psi,
            'sensitivity_results': sensitivity_results,
            'most_sensitive': max(sensitivity_results.keys(),
                                key=lambda k: max(abs(r['relative_change'])
                                                 for r in sensitivity_results[k]))
        }
