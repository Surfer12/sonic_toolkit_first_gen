"""
UOIF Allocation System

This module implements the dynamic allocation system for the UOIF Ruleset,
providing α(t) allocation based on claim type, validation history, and temporal factors.

Key Features:
- Dynamic allocation based on validation history
- Temporal decay factors for aging claims
- Confidence calibration with promotion/demotion triggers
- Integration with toolkit's performance monitoring

Author: Scientific Computing Toolkit Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np


@dataclass
class AllocationResult:
    """Result of allocation calculation."""
    claim_id: str
    allocated_alpha: float
    allocation_timestamp: datetime
    allocation_factors: Dict[str, float]
    confidence_range: Tuple[float, float]
    temporal_decay: float
    validation_history: List[float]
    promotion_status: str


class AllocationSystem:
    """
    Implements dynamic allocation system for UOIF confidence measures.

    This class manages α(t) allocation with temporal factors, validation history,
    and integration with toolkit performance monitoring.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize allocation system.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.allocation_history: Dict[str, List[AllocationResult]] = {}
        self.validation_cache: Dict[str, List[float]] = {}

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for allocation system."""
        return {
            'base_allocations': {
                'lipschitz_primitives': 0.12,
                'rspo_dmd': (0.15, 0.20),
                'euler_lagrange': (0.10, 0.15),
                'interpretations': (0.35, 0.45),
                'hb_model_validation': (0.25, 0.35),
                'lstm_convergence': (0.20, 0.30),
                'consciousness_claims': (0.30, 0.40)
            },
            'temporal_decay': {
                'half_life_days': 30.0,
                'decay_rate': 0.02,  # Daily decay factor
                'min_allocation': 0.05,
                'max_decay_factor': 0.8
            },
            'validation_history': {
                'max_history_length': 10,
                'history_weight_decay': 0.9,
                'min_validation_samples': 3,
                'validation_threshold': 0.75
            },
            'promotion_triggers': {
                'lipschitz': {'threshold': 0.97, 'promotion': 0.12, 'beta_boost': 1.15},
                'rspo_dmd': {'threshold': 0.98, 'promotion': 0.18, 'beta_boost': 1.20},
                'confidence_minimum': 0.75,
                'demotion_threshold': 0.60,
                'performance_regression_penalty': 0.9
            },
            'toolkit_integration': {
                'correlation_boost_threshold': 0.9987,
                'correlation_boost_factor': 1.1,
                'bootstrap_samples_boost': 1.05,
                'performance_decay_penalty': 0.95
            }
        }

    def allocate_alpha(self, claim_id: str, claim_type: str, domain: str,
                      validation_score: float, temporal_factor: Optional[float] = None,
                      toolkit_performance: Optional[Dict[str, Any]] = None) -> AllocationResult:
        """
        Calculate dynamic α allocation for a claim.

        Args:
            claim_id: Unique claim identifier
            claim_type: Type of claim
            domain: Research domain
            validation_score: Current validation score (0-1)
            temporal_factor: Optional temporal decay factor
            toolkit_performance: Optional toolkit performance metrics

        Returns:
            AllocationResult: Complete allocation result
        """
        # Get base allocation
        base_alpha = self._get_base_allocation(claim_type, domain)

        # Apply validation history adjustment
        validation_adjustment = self._calculate_validation_adjustment(claim_id, validation_score)

        # Apply temporal decay
        temporal_decay = temporal_factor if temporal_factor is not None else self._calculate_temporal_decay(claim_id)

        # Apply toolkit integration factors
        toolkit_factor = self._calculate_toolkit_factor(toolkit_performance)

        # Calculate final allocation
        allocated_alpha = self._calculate_final_allocation(
            base_alpha, validation_adjustment, temporal_decay, toolkit_factor
        )

        # Calculate confidence range
        confidence_range = self._calculate_confidence_range(allocated_alpha, validation_score)

        # Determine promotion status
        promotion_status = self._determine_promotion_status(claim_id, allocated_alpha, validation_score)

        # Create allocation factors for transparency
        allocation_factors = {
            'base_alpha': base_alpha,
            'validation_adjustment': validation_adjustment,
            'temporal_decay': temporal_decay,
            'toolkit_factor': toolkit_factor,
            'final_alpha': allocated_alpha
        }

        # Create result
        result = AllocationResult(
            claim_id=claim_id,
            allocated_alpha=allocated_alpha,
            allocation_timestamp=datetime.now(),
            allocation_factors=allocation_factors,
            confidence_range=confidence_range,
            temporal_decay=temporal_decay,
            validation_history=self._get_validation_history(claim_id),
            promotion_status=promotion_status
        )

        # Store result
        if claim_id not in self.allocation_history:
            self.allocation_history[claim_id] = []
        self.allocation_history[claim_id].append(result)

        # Update validation cache
        self._update_validation_cache(claim_id, validation_score)

        return result

    def _get_base_allocation(self, claim_type: str, domain: str) -> float:
        """
        Get base allocation for claim type and domain.

        Args:
            claim_type: Type of claim
            domain: Research domain

        Returns:
            float: Base allocation value
        """
        # Map claim type to allocation key
        type_mapping = {
            'primitive': 'lipschitz_primitives',
            'lipschitz_primitive': 'lipschitz_primitives',
            'rspo_dmd': 'rspo_dmd',
            'euler_lagrange': 'euler_lagrange',
            'interpretation': 'interpretations',
            'hb_validation': 'hb_model_validation',
            'lstm_validation': 'lstm_convergence',
            'consciousness': 'consciousness_claims'
        }

        allocation_key = type_mapping.get(claim_type, 'interpretations')
        base_allocation = self.config['base_allocations'][allocation_key]

        # Handle range allocations
        if isinstance(base_allocation, tuple):
            return np.mean(base_allocation)  # Use mean for initial allocation
        return base_allocation

    def _calculate_validation_adjustment(self, claim_id: str, validation_score: float) -> float:
        """
        Calculate validation history adjustment factor.

        Args:
            claim_id: Claim identifier
            validation_score: Current validation score

        Returns:
            float: Adjustment factor
        """
        history = self._get_validation_history(claim_id)

        if len(history) < self.config['validation_history']['min_validation_samples']:
            return 1.0  # No adjustment with insufficient history

        # Calculate weighted average of validation history
        weights = [self.config['validation_history']['history_weight_decay'] ** i
                  for i in range(len(history))]
        weights = np.array(weights) / sum(weights)

        historical_avg = np.average(history, weights=weights)

        # Calculate adjustment based on consistency
        if validation_score >= self.config['validation_history']['validation_threshold']:
            if historical_avg >= self.config['validation_history']['validation_threshold']:
                return 1.1  # Consistent high performance
            else:
                return 1.05  # Improving performance
        else:
            if historical_avg >= self.config['validation_history']['validation_threshold']:
                return 0.95  # Declining performance
            else:
                return 0.9  # Consistently low performance

    def _calculate_temporal_decay(self, claim_id: str) -> float:
        """
        Calculate temporal decay factor for claim aging.

        Args:
            claim_id: Claim identifier

        Returns:
            float: Decay factor (0-1)
        """
        if claim_id not in self.allocation_history:
            return 1.0  # No decay for new claims

        last_allocation = self.allocation_history[claim_id][-1]
        time_since_allocation = datetime.now() - last_allocation.allocation_timestamp
        days_since_allocation = time_since_allocation.total_seconds() / (24 * 3600)

        # Exponential decay with half-life
        half_life = self.config['temporal_decay']['half_life_days']
        decay_rate = np.log(2) / half_life
        decay_factor = np.exp(-decay_rate * days_since_allocation)

        # Clamp to reasonable range
        min_decay = self.config['temporal_decay']['min_allocation']
        max_decay = self.config['temporal_decay']['max_decay_factor']

        return max(min_decay, min(max_decay, decay_factor))

    def _calculate_toolkit_factor(self, toolkit_performance: Optional[Dict[str, Any]]) -> float:
        """
        Calculate toolkit integration factor based on performance metrics.

        Args:
            toolkit_performance: Toolkit performance data

        Returns:
            float: Integration factor
        """
        if not toolkit_performance:
            return 1.0

        factor = 1.0

        # Correlation boost
        correlation = toolkit_performance.get('correlation', 0.9)
        if correlation >= self.config['toolkit_integration']['correlation_boost_threshold']:
            factor *= self.config['toolkit_integration']['correlation_boost_factor']

        # Bootstrap samples boost
        bootstrap_samples = toolkit_performance.get('bootstrap_samples', 1000)
        if bootstrap_samples >= 1000:
            factor *= self.config['toolkit_integration']['bootstrap_samples_boost']

        # Performance regression penalty
        if toolkit_performance.get('performance_regression', False):
            factor *= self.config['toolkit_integration']['performance_decay_penalty']

        return factor

    def _calculate_final_allocation(self, base_alpha: float, validation_adjustment: float,
                                  temporal_decay: float, toolkit_factor: float) -> float:
        """
        Calculate final allocation with all factors.

        Args:
            base_alpha: Base allocation
            validation_adjustment: Validation adjustment factor
            temporal_decay: Temporal decay factor
            toolkit_factor: Toolkit integration factor

        Returns:
            float: Final allocation
        """
        final_alpha = base_alpha * validation_adjustment * temporal_decay * toolkit_factor

        # Clamp to reasonable range
        return max(0.05, min(0.95, final_alpha))

    def _calculate_confidence_range(self, allocated_alpha: float,
                                  validation_score: float) -> Tuple[float, float]:
        """
        Calculate confidence range for allocation.

        Args:
            allocated_alpha: Allocated alpha value
            validation_score: Validation score

        Returns:
            Tuple[float, float]: Confidence range (lower, upper)
        """
        # Base confidence range based on validation score
        base_range = 0.1 * (1 - validation_score)  # Larger range for lower validation

        # Adjust based on allocated alpha
        alpha_factor = 1 + (1 - allocated_alpha) * 0.5  # Higher alpha = tighter range

        range_width = base_range * alpha_factor

        lower = max(0.0, allocated_alpha - range_width)
        upper = min(1.0, allocated_alpha + range_width)

        return (lower, upper)

    def _determine_promotion_status(self, claim_id: str, allocated_alpha: float,
                                  validation_score: float) -> str:
        """
        Determine promotion status based on allocation and validation.

        Args:
            claim_id: Claim identifier
            allocated_alpha: Allocated alpha
            validation_score: Validation score

        Returns:
            str: Promotion status
        """
        # Check promotion thresholds
        if validation_score >= self.config['promotion_triggers']['lipschitz']['threshold']:
            return "Empirically Grounded"
        elif validation_score >= self.config['promotion_triggers']['rspo_dmd']['threshold']:
            return "Validated"
        elif validation_score >= self.config['promotion_triggers']['confidence_minimum']:
            return "Acceptable"
        elif validation_score < self.config['promotion_triggers']['demotion_threshold']:
            return "Demoted"
        else:
            return "Standard"

    def _get_validation_history(self, claim_id: str) -> List[float]:
        """Get validation history for claim."""
        return self.validation_cache.get(claim_id, [])

    def _update_validation_cache(self, claim_id: str, validation_score: float):
        """Update validation cache with new score."""
        if claim_id not in self.validation_cache:
            self.validation_cache[claim_id] = []

        history = self.validation_cache[claim_id]
        history.append(validation_score)

        # Maintain maximum history length
        max_length = self.config['validation_history']['max_history_length']
        if len(history) > max_length:
            history.pop(0)

    def get_allocation_history(self, claim_id: Optional[str] = None) -> List[AllocationResult]:
        """
        Get allocation history for claims.

        Args:
            claim_id: Optional specific claim ID

        Returns:
            List[AllocationResult]: Allocation results
        """
        if claim_id:
            return self.allocation_history.get(claim_id, [])
        return [result for results in self.allocation_history.values() for result in results]

    def get_allocation_stats(self) -> Dict[str, Any]:
        """
        Get allocation statistics.

        Returns:
            Dict[str, Any]: Allocation statistics
        """
        total_allocations = sum(len(results) for results in self.allocation_history.values())
        avg_alpha = np.mean([
            result.allocated_alpha
            for results in self.allocation_history.values()
            for result in results
        ]) if total_allocations > 0 else 0

        # Promotion status distribution
        promotion_counts = {}
        for results in self.allocation_history.values():
            for result in results:
                status = result.promotion_status
                promotion_counts[status] = promotion_counts.get(status, 0) + 1

        return {
            'total_allocations': total_allocations,
            'avg_alpha': avg_alpha,
            'promotion_distribution': promotion_counts,
            'temporal_decay_stats': self._calculate_temporal_stats()
        }

    def _calculate_temporal_stats(self) -> Dict[str, float]:
        """Calculate temporal decay statistics."""
        decays = [
            result.temporal_decay
            for results in self.allocation_history.values()
            for result in results
        ]

        if not decays:
            return {'avg_decay': 1.0, 'min_decay': 1.0, 'max_decay': 1.0}

        return {
            'avg_decay': np.mean(decays),
            'min_decay': np.min(decays),
            'max_decay': np.max(decays)
        }

    def optimize_allocations(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize allocation parameters based on performance data.

        Args:
            performance_data: Historical performance data

        Returns:
            Dict[str, Any]: Optimized allocation parameters
        """
        # Analyze performance patterns
        high_performers = [d for d in performance_data if d.get('validation_score', 0) >= 0.9]
        low_performers = [d for d in performance_data if d.get('validation_score', 0) < 0.7]

        # Calculate optimal base allocations
        optimized_allocations = self.config['base_allocations'].copy()

        if high_performers:
            avg_high_alpha = np.mean([d.get('allocated_alpha', 0.3) for d in high_performers])
            # Adjust base allocations toward successful values
            for key, value in optimized_allocations.items():
                if isinstance(value, tuple):
                    optimized_allocations[key] = tuple(v * 0.9 + avg_high_alpha * 0.1 for v in value)
                else:
                    optimized_allocations[key] = value * 0.9 + avg_high_alpha * 0.1

        return {
            'optimized_allocations': optimized_allocations,
            'performance_analysis': {
                'high_performers': len(high_performers),
                'low_performers': len(low_performers),
                'avg_high_alpha': avg_high_alpha if high_performers else 0.3
            }
        }

    def batch_allocate(self, claims_data: List[Dict[str, Any]]) -> List[AllocationResult]:
        """
        Allocate alpha values for multiple claims in batch.

        Args:
            claims_data: List of claim data dictionaries

        Returns:
            List[AllocationResult]: Allocation results for all claims
        """
        results = []

        for claim_data in claims_data:
            claim_id = claim_data['claim_id']
            claim_type = claim_data['claim_type']
            domain = claim_data['domain']
            validation_score = claim_data['validation_score']
            temporal_factor = claim_data.get('temporal_factor')
            toolkit_performance = claim_data.get('toolkit_performance')

            result = self.allocate_alpha(
                claim_id, claim_type, domain, validation_score,
                temporal_factor, toolkit_performance
            )
            results.append(result)

        return results
