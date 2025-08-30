"""
UOIF Scoring Engine

This module implements the quantitative scoring system for the UOIF Ruleset,
providing claim evaluation through weighted components and noise penalty mechanisms.

Scoring Formula: s(c) = Σwᵢ·componentᵢ - w_noise·Noise

Author: Scientific Computing Toolkit Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class ScoreComponents:
    """Components of the UOIF scoring formula."""
    authority: float      # Authoritative source weight
    verifiability: float  # Verification strength
    depth: float         # Technical depth
    alignment: float     # Intent alignment
    recurrence: float    # Citation frequency
    noise: float         # Noise penalty

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'authority': self.authority,
            'verifiability': self.verifiability,
            'depth': self.depth,
            'alignment': self.alignment,
            'recurrence': self.recurrence,
            'noise': self.noise
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoreComponents':
        """Create from dictionary."""
        return cls(
            authority=data.get('authority', 0.0),
            verifiability=data.get('verifiability', 0.0),
            depth=data.get('depth', 0.0),
            alignment=data.get('alignment', 0.0),
            recurrence=data.get('recurrence', 0.0),
            noise=data.get('noise', 0.0)
        )


@dataclass
class ScoringResult:
    """Result of claim scoring."""
    claim_id: str
    total_score: float
    components: ScoreComponents
    confidence_level: float
    scoring_timestamp: datetime
    scorer_metadata: Dict[str, Any]
    component_contributions: Dict[str, float]


class ScoringEngine:
    """
    Implements the UOIF quantitative scoring system.

    This class evaluates claims using the weighted component formula:
    s(c) = Σwᵢ·componentᵢ - w_noise·Noise

    Where weights are optimized for different claim types and domains.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scoring engine.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.scoring_history: Dict[str, ScoringResult] = {}

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for scoring engine."""
        return {
            'base_weights': {
                'authority': 0.35,
                'verifiability': 0.30,
                'depth': 0.10,
                'alignment': 0.15,
                'recurrence': 0.07,
                'noise': 0.23
            },
            'claim_type_adjustments': {
                'primitive': {
                    'authority': 1.2,      # Higher authority weight for primitives
                    'verifiability': 1.1,  # Higher verification weight
                    'depth': 1.3,          # Higher depth weight
                    'noise': 0.8          # Lower noise penalty
                },
                'interpretation': {
                    'authority': 0.9,      # Slightly lower authority weight
                    'verifiability': 1.0,  # Standard verification weight
                    'depth': 1.2,          # Higher depth weight
                    'alignment': 1.2,      # Higher alignment weight
                    'noise': 1.0          # Standard noise penalty
                },
                'speculative': {
                    'authority': 0.7,      # Lower authority weight
                    'verifiability': 0.8,  # Lower verification weight
                    'depth': 0.8,          # Lower depth weight
                    'alignment': 1.0,      # Standard alignment weight
                    'recurrence': 0.5,     # Lower recurrence weight
                    'noise': 1.5          # Higher noise penalty
                }
            },
            'domain_adjustments': {
                'zeta_dynamics': {
                    'depth': 1.2,
                    'authority': 1.1
                },
                'koopman_operators': {
                    'verifiability': 1.2,
                    'depth': 1.1
                },
                'consciousness_framework': {
                    'alignment': 1.3,
                    'authority': 0.9
                },
                'hb_models': {
                    'verifiability': 1.1,
                    'depth': 1.1
                },
                'lstm_convergence': {
                    'verifiability': 1.2,
                    'recurrence': 1.2
                }
            },
            'scoring_ranges': {
                'authority': (0.0, 1.0),
                'verifiability': (0.0, 1.0),
                'depth': (0.0, 1.0),
                'alignment': (0.0, 1.0),
                'recurrence': (0.0, 5.0),  # Citation count can be higher
                'noise': (0.0, 1.0)
            },
            'confidence_mapping': {
                'excellent': (0.90, 1.00),
                'very_good': (0.80, 0.89),
                'good': (0.70, 0.79),
                'acceptable': (0.60, 0.69),
                'poor': (0.00, 0.59)
            }
        }

    def score_claim(self, claim_id: str, raw_components: Dict[str, float],
                   claim_type: str, domain: str,
                   scorer_metadata: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a claim using the UOIF scoring formula.

        Args:
            claim_id: Unique claim identifier
            raw_components: Raw component values (authority, verifiability, etc.)
            claim_type: Type of claim ('primitive', 'interpretation', 'speculative')
            domain: Research domain
            scorer_metadata: Optional metadata about the scoring process

        Returns:
            ScoringResult: Complete scoring result
        """
        # Validate and normalize components
        normalized_components = self._normalize_components(raw_components)

        # Apply claim type adjustments
        adjusted_weights = self._apply_claim_type_adjustments(claim_type)

        # Apply domain adjustments
        adjusted_weights = self._apply_domain_adjustments(domain, adjusted_weights)

        # Calculate total score
        total_score = self._calculate_total_score(normalized_components, adjusted_weights)

        # Calculate component contributions
        component_contributions = self._calculate_component_contributions(
            normalized_components, adjusted_weights
        )

        # Determine confidence level
        confidence_level = self._calculate_confidence_level(total_score)

        # Create scoring result
        result = ScoringResult(
            claim_id=claim_id,
            total_score=total_score,
            components=ScoreComponents(**normalized_components),
            confidence_level=confidence_level,
            scoring_timestamp=datetime.now(),
            scorer_metadata=scorer_metadata or {},
            component_contributions=component_contributions
        )

        # Store result
        self.scoring_history[claim_id] = result

        return result

    def _normalize_components(self, raw_components: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize component values to their defined ranges.

        Args:
            raw_components: Raw component values

        Returns:
            Dict[str, float]: Normalized component values
        """
        normalized = {}

        for component_name, raw_value in raw_components.items():
            if component_name in self.config['scoring_ranges']:
                min_val, max_val = self.config['scoring_ranges'][component_name]
                # Clamp to range and normalize to [0,1] except for recurrence
                if component_name == 'recurrence':
                    # Recurrence can exceed 1.0, normalize differently
                    normalized[component_name] = min(raw_value / 5.0, 1.0)  # Cap at 5 citations
                else:
                    normalized[component_name] = max(min_val, min(max_val, raw_value))
            else:
                # Default normalization
                normalized[component_name] = max(0.0, min(1.0, raw_value))

        return normalized

    def _apply_claim_type_adjustments(self, claim_type: str) -> Dict[str, float]:
        """
        Apply claim type specific weight adjustments.

        Args:
            claim_type: Type of claim

        Returns:
            Dict[str, float]: Adjusted weights
        """
        base_weights = self.config['base_weights'].copy()
        adjustments = self.config['claim_type_adjustments'].get(claim_type, {})

        for component, adjustment in adjustments.items():
            if component in base_weights:
                base_weights[component] *= adjustment

        # Renormalize weights to sum to 1.0 (except noise which is penalty)
        total_weight = sum(w for c, w in base_weights.items() if c != 'noise')
        if total_weight > 0:
            for component in base_weights:
                if component != 'noise':
                    base_weights[component] /= total_weight

        return base_weights

    def _apply_domain_adjustments(self, domain: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply domain-specific weight adjustments.

        Args:
            domain: Research domain
            weights: Current weights

        Returns:
            Dict[str, float]: Domain-adjusted weights
        """
        adjustments = self.config['domain_adjustments'].get(domain, {})
        adjusted_weights = weights.copy()

        for component, adjustment in adjustments.items():
            if component in adjusted_weights:
                adjusted_weights[component] *= adjustment

        # Renormalize
        total_weight = sum(w for c, w in adjusted_weights.items() if c != 'noise')
        if total_weight > 0:
            for component in adjusted_weights:
                if component != 'noise':
                    adjusted_weights[component] /= total_weight

        return adjusted_weights

    def _calculate_total_score(self, components: Dict[str, float],
                             weights: Dict[str, float]) -> float:
        """
        Calculate total score using the UOIF formula.

        Args:
            components: Normalized component values
            weights: Adjusted weights

        Returns:
            float: Total score
        """
        # Positive contributions
        positive_score = sum(
            components.get(component, 0.0) * weights.get(component, 0.0)
            for component in ['authority', 'verifiability', 'depth', 'alignment', 'recurrence']
        )

        # Noise penalty
        noise_penalty = components.get('noise', 0.0) * weights.get('noise', 0.0)

        # Total score
        total_score = positive_score - noise_penalty

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, total_score))

    def _calculate_component_contributions(self, components: Dict[str, float],
                                        weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate individual component contributions to total score.

        Args:
            components: Normalized component values
            weights: Adjusted weights

        Returns:
            Dict[str, float]: Component contributions
        """
        contributions = {}

        for component in ['authority', 'verifiability', 'depth', 'alignment', 'recurrence']:
            contribution = components.get(component, 0.0) * weights.get(component, 0.0)
            contributions[f"{component}_contribution"] = contribution

        # Noise penalty
        noise_contribution = components.get('noise', 0.0) * weights.get('noise', 0.0)
        contributions['noise_penalty'] = -noise_contribution

        return contributions

    def _calculate_confidence_level(self, total_score: float) -> float:
        """
        Calculate confidence level from total score.

        Args:
            total_score: Total score (0-1)

        Returns:
            float: Confidence level (0-1)
        """
        # Confidence level is derived from score with some calibration
        # Higher scores indicate higher confidence
        confidence_level = total_score * 0.9 + 0.1  # Map [0,1] to [0.1, 1.0]

        return min(1.0, confidence_level)

    def get_scoring_weights(self, claim_type: str, domain: str) -> Dict[str, float]:
        """
        Get the effective scoring weights for a claim type and domain.

        Args:
            claim_type: Type of claim
            domain: Research domain

        Returns:
            Dict[str, float]: Effective weights
        """
        weights = self._apply_claim_type_adjustments(claim_type)
        weights = self._apply_domain_adjustments(domain, weights)
        return weights

    def batch_score_claims(self, claims_data: List[Dict[str, Any]]) -> List[ScoringResult]:
        """
        Score multiple claims in batch.

        Args:
            claims_data: List of claim data dictionaries
                Each dict should contain: claim_id, components, claim_type, domain

        Returns:
            List[ScoringResult]: Scoring results for all claims
        """
        results = []

        for claim_data in claims_data:
            claim_id = claim_data['claim_id']
            components = claim_data['components']
            claim_type = claim_data['claim_type']
            domain = claim_data['domain']
            metadata = claim_data.get('metadata')

            result = self.score_claim(claim_id, components, claim_type, domain, metadata)
            results.append(result)

        return results

    def get_scoring_history(self, claim_id: Optional[str] = None) -> List[ScoringResult]:
        """
        Get scoring history for claims.

        Args:
            claim_id: Optional specific claim ID

        Returns:
            List[ScoringResult]: Scoring results
        """
        if claim_id:
            return [self.scoring_history[claim_id]] if claim_id in self.scoring_history else []
        return list(self.scoring_history.values())

    def get_scoring_stats(self) -> Dict[str, Any]:
        """
        Get scoring statistics.

        Returns:
            Dict[str, Any]: Scoring statistics
        """
        total_claims = len(self.scoring_history)
        avg_score = np.mean([r.total_score for r in self.scoring_history.values()]) if total_claims > 0 else 0
        avg_confidence = np.mean([r.confidence_level for r in self.scoring_history.values()]) if total_claims > 0 else 0

        # Score distribution
        scores = [r.total_score for r in self.scoring_history.values()]
        score_distribution = {
            'excellent': sum(1 for s in scores if s >= 0.90),
            'very_good': sum(1 for s in scores if 0.80 <= s < 0.90),
            'good': sum(1 for s in scores if 0.70 <= s < 0.80),
            'acceptable': sum(1 for s in scores if 0.60 <= s < 0.70),
            'poor': sum(1 for s in scores if s < 0.60)
        }

        return {
            'total_claims': total_claims,
            'avg_score': avg_score,
            'avg_confidence': avg_confidence,
            'score_distribution': score_distribution,
            'component_averages': self._calculate_component_averages()
        }

    def _calculate_component_averages(self) -> Dict[str, float]:
        """Calculate average component values across all scored claims."""
        if not self.scoring_history:
            return {}

        component_sums = {}
        component_counts = {}

        for result in self.scoring_history.values():
            for component_name, value in result.components.to_dict().items():
                component_sums[component_name] = component_sums.get(component_name, 0) + value
                component_counts[component_name] = component_counts.get(component_name, 0) + 1

        averages = {}
        for component in component_sums:
            averages[component] = component_sums[component] / component_counts[component]

        return averages

    def optimize_weights(self, training_data: List[Dict[str, Any]],
                        target_accuracy: float = 0.85) -> Dict[str, float]:
        """
        Optimize scoring weights using training data.

        Args:
            training_data: Training data with known good/bad claims
            target_accuracy: Target accuracy for optimization

        Returns:
            Dict[str, float]: Optimized weights
        """
        # Simple optimization - could be enhanced with more sophisticated methods
        best_weights = self.config['base_weights'].copy()
        best_accuracy = self._evaluate_weights(best_weights, training_data)

        # Try different weight combinations
        weight_ranges = {
            'authority': (0.2, 0.5),
            'verifiability': (0.2, 0.4),
            'depth': (0.05, 0.2),
            'alignment': (0.1, 0.2),
            'recurrence': (0.05, 0.1),
            'noise': (0.1, 0.3)
        }

        # Grid search (simplified)
        for auth_weight in np.linspace(weight_ranges['authority'][0], weight_ranges['authority'][1], 5):
            for ver_weight in np.linspace(weight_ranges['verifiability'][0], weight_ranges['verifiability'][1], 5):
                test_weights = self.config['base_weights'].copy()
                test_weights['authority'] = auth_weight
                test_weights['verifiability'] = ver_weight

                # Normalize other weights
                remaining_weight = 1.0 - auth_weight - ver_weight - test_weights['noise']
                other_components = ['depth', 'alignment', 'recurrence']
                for comp in other_components:
                    test_weights[comp] = (test_weights[comp] / sum(test_weights[c] for c in other_components)) * remaining_weight

                accuracy = self._evaluate_weights(test_weights, training_data)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = test_weights.copy()

        return best_weights

    def _evaluate_weights(self, weights: Dict[str, float],
                         training_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate weight performance on training data.

        Args:
            weights: Weights to evaluate
            training_data: Training data

        Returns:
            float: Accuracy score
        """
        correct_predictions = 0
        total_predictions = len(training_data)

        for claim_data in training_data:
            claim_id = claim_data['claim_id']
            components = claim_data['components']
            claim_type = claim_data['claim_type']
            domain = claim_data['domain']
            expected_quality = claim_data['expected_quality']  # 'high' or 'low'

            # Temporarily set weights
            original_weights = self.config['base_weights'].copy()
            self.config['base_weights'] = weights.copy()

            try:
                result = self.score_claim(claim_id, components, claim_type, domain)
                predicted_quality = 'high' if result.total_score >= 0.75 else 'low'

                if predicted_quality == expected_quality:
                    correct_predictions += 1
            finally:
                # Restore original weights
                self.config['base_weights'] = original_weights

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
