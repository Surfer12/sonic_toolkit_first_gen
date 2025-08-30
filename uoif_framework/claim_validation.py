"""
UOIF Claim Validation and Gating

This module implements the claim validation system for the UOIF Ruleset,
providing primitive, interpretation, and speculative claim classification with gating mechanisms.

Claim Types:
- Primitive: Exact derivations, lemmas, notations
- Interpretation: Convergence proofs, manifold reconstructions
- Speculative: Chaos analogs (labeled speculative, no promotion)

Author: Scientific Computing Toolkit Team
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


class ClaimType(Enum):
    """Claim classification types."""
    PRIMITIVE = "primitive"          # Exact derivations, lemmas, notations
    INTERPRETATION = "interpretation"  # Convergence proofs, reconstructions
    SPECULATIVE = "speculative"      # Chaos analogs, no promotion


@dataclass
class ValidationResult:
    """Result of claim validation."""
    claim_id: str
    claim_type: ClaimType
    is_valid: bool
    confidence_score: float
    validation_timestamp: datetime
    validator_source: str
    validation_criteria: List[str]
    failure_reasons: Optional[List[str]] = None
    supporting_evidence: Optional[List[str]] = None


class ClaimValidator:
    """
    Validates and gates claims according to UOIF Ruleset.

    This class implements the claim validation system with automatic gating
    based on claim type, supporting evidence, and validation criteria.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize claim validator.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.validated_claims: Dict[str, ValidationResult] = {}
        self._setup_validation_patterns()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for claim validation."""
        return {
            'validation_criteria': {
                ClaimType.PRIMITIVE: [
                    'exact_derivation',
                    'mathematical_proof',
                    'notation_consistency',
                    'canonical_reference'
                ],
                ClaimType.INTERPRETATION: [
                    'convergence_proof',
                    'manifold_reconstruction',
                    'supporting_evidence',
                    'cross_validation'
                ],
                ClaimType.SPECULATIVE: [
                    'conceptual_analogy',
                    'exploratory_hypothesis'
                ]
            },
            'gating_requirements': {
                ClaimType.PRIMITIVE: {
                    'min_confidence': 0.85,
                    'required_criteria': 3,
                    'canonical_reference_required': True
                },
                ClaimType.INTERPRETATION: {
                    'min_confidence': 0.75,
                    'required_criteria': 2,
                    'supporting_evidence_required': True
                },
                ClaimType.SPECULATIVE: {
                    'min_confidence': 0.50,
                    'required_criteria': 1,
                    'promotion_blocked': True
                }
            },
            'domains': [
                'zeta_dynamics', 'koopman_operators', 'rspo_convergence',
                'consciousness_framework', 'hb_models', 'lstm_convergence'
            ]
        }

    def _setup_validation_patterns(self):
        """Setup regex patterns for validation criteria detection."""
        self.validation_patterns = {
            'exact_derivation': re.compile(r'\b(?:derivation|proof|lemma|theorem)\b', re.IGNORECASE),
            'mathematical_proof': re.compile(r'\b(?:equation|formula|mathematical|formal)\b', re.IGNORECASE),
            'notation_consistency': re.compile(r'\b(?:notation|symbol|definition|consistent)\b', re.IGNORECASE),
            'canonical_reference': re.compile(r'\b(?:grok_report_\d+|canonical|verified)\b', re.IGNORECASE),
            'convergence_proof': re.compile(r'\b(?:convergence|converges|limit|asymptotic)\b', re.IGNORECASE),
            'manifold_reconstruction': re.compile(r'\b(?:manifold|reconstruction|topology|geometry)\b', re.IGNORECASE),
            'supporting_evidence': re.compile(r'\b(?:evidence|validation|experimental|data)\b', re.IGNORECASE),
            'cross_validation': re.compile(r'\b(?:cross.?validation|comparison|benchmark)\b', re.IGNORECASE),
            'conceptual_analogy': re.compile(r'\b(?:analogy|similar|analogous|conceptual)\b', re.IGNORECASE),
            'exploratory_hypothesis': re.compile(r'\b(?:hypothesis|exploratory|speculative|tentative)\b', re.IGNORECASE)
        }

    def validate_claim(self, claim_id: str, claim_text: str, claim_type: ClaimType,
                      domain: str, source_id: str) -> ValidationResult:
        """
        Validate a claim against UOIF criteria.

        Args:
            claim_id: Unique claim identifier
            claim_text: Text content of the claim
            claim_type: Type of claim being validated
            domain: Research domain
            source_id: Source identifier

        Returns:
            ValidationResult: Complete validation result
        """
        # Validate domain
        if domain not in self.config['domains']:
            raise ValueError(f"Unknown domain: {domain}")

        # Check validation criteria
        criteria_results = self._check_validation_criteria(claim_text, claim_type)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(criteria_results, claim_type)

        # Determine validation result
        gating_requirements = self.config['gating_requirements'][claim_type]
        is_valid = self._apply_gating_rules(confidence_score, criteria_results, gating_requirements)

        # Create validation result
        result = ValidationResult(
            claim_id=claim_id,
            claim_type=claim_type,
            is_valid=is_valid,
            confidence_score=confidence_score,
            validation_timestamp=datetime.now(),
            validator_source=source_id,
            validation_criteria=list(criteria_results.keys()),
            failure_reasons=self._get_failure_reasons(is_valid, confidence_score, criteria_results, gating_requirements),
            supporting_evidence=self._extract_supporting_evidence(claim_text, criteria_results)
        )

        # Store validation result
        self.validated_claims[claim_id] = result

        return result

    def _check_validation_criteria(self, claim_text: str, claim_type: ClaimType) -> Dict[str, bool]:
        """
        Check which validation criteria are met by the claim text.

        Args:
            claim_text: Text to analyze
            claim_type: Type of claim

        Returns:
            Dict[str, bool]: Criteria satisfaction results
        """
        required_criteria = self.config['validation_criteria'][claim_type]
        criteria_results = {}

        for criterion in required_criteria:
            pattern = self.validation_patterns.get(criterion)
            if pattern:
                criteria_results[criterion] = bool(pattern.search(claim_text))
            else:
                criteria_results[criterion] = False

        return criteria_results

    def _calculate_confidence_score(self, criteria_results: Dict[str, bool],
                                  claim_type: ClaimType) -> float:
        """
        Calculate confidence score based on criteria satisfaction.

        Args:
            criteria_results: Results of criteria checks
            claim_type: Type of claim

        Returns:
            float: Confidence score (0-1)
        """
        satisfied_criteria = sum(criteria_results.values())
        total_criteria = len(criteria_results)

        if total_criteria == 0:
            return 0.0

        # Base score from criteria satisfaction
        base_score = satisfied_criteria / total_criteria

        # Adjust based on claim type requirements
        gating_requirements = self.config['gating_requirements'][claim_type]
        min_confidence = gating_requirements['min_confidence']

        # Boost score if above minimum requirements
        if satisfied_criteria >= gating_requirements['required_criteria']:
            confidence_boost = 0.1 * (satisfied_criteria - gating_requirements['required_criteria'] + 1)
            base_score = min(1.0, base_score + confidence_boost)

        return max(min_confidence, base_score)

    def _apply_gating_rules(self, confidence_score: float, criteria_results: Dict[str, bool],
                           gating_requirements: Dict[str, Any]) -> bool:
        """
        Apply gating rules to determine validation result.

        Args:
            confidence_score: Calculated confidence score
            criteria_results: Criteria satisfaction results
            gating_requirements: Gating requirements for claim type

        Returns:
            bool: True if claim passes gating
        """
        # Check minimum confidence
        if confidence_score < gating_requirements['min_confidence']:
            return False

        # Check minimum criteria satisfaction
        satisfied_criteria = sum(criteria_results.values())
        if satisfied_criteria < gating_requirements['required_criteria']:
            return False

        # Check canonical reference requirement for primitives
        if gating_requirements.get('canonical_reference_required', False):
            if not criteria_results.get('canonical_reference', False):
                return False

        # Check supporting evidence requirement
        if gating_requirements.get('supporting_evidence_required', False):
            if not criteria_results.get('supporting_evidence', False):
                return False

        # Check promotion blocking for speculative claims
        if gating_requirements.get('promotion_blocked', False):
            return False  # Speculative claims never pass gating

        return True

    def _get_failure_reasons(self, is_valid: bool, confidence_score: float,
                           criteria_results: Dict[str, bool],
                           gating_requirements: Dict[str, Any]) -> Optional[List[str]]:
        """
        Generate failure reasons if validation failed.

        Args:
            is_valid: Whether validation passed
            confidence_score: Confidence score
            criteria_results: Criteria results
            gating_requirements: Gating requirements

        Returns:
            Optional[List[str]]: List of failure reasons
        """
        if is_valid:
            return None

        reasons = []

        if confidence_score < gating_requirements['min_confidence']:
            reasons.append(".2f"        satisfied_criteria = sum(criteria_results.values())
        if satisfied_criteria < gating_requirements['required_criteria']:
            reasons.append(f"Only {satisfied_criteria}/{gating_requirements['required_criteria']} validation criteria met")

        if gating_requirements.get('canonical_reference_required') and not criteria_results.get('canonical_reference'):
            reasons.append("Canonical reference required but not found")

        if gating_requirements.get('supporting_evidence_required') and not criteria_results.get('supporting_evidence'):
            reasons.append("Supporting evidence required but not found")

        if gating_requirements.get('promotion_blocked'):
            reasons.append("Speculative claims are blocked from promotion")

        return reasons

    def _extract_supporting_evidence(self, claim_text: str,
                                   criteria_results: Dict[str, bool]) -> Optional[List[str]]:
        """
        Extract supporting evidence references from claim text.

        Args:
            claim_text: Claim text to analyze
            criteria_results: Criteria satisfaction results

        Returns:
            Optional[List[str]]: List of evidence references
        """
        evidence_patterns = [
            r'grok_report_\d+',  # Canonical reports
            r'arXiv:\d+\.\d+',   # ArXiv references
            r'equation\s*\(\d+\)',  # Equation references
            r'figure\s*\d+',     # Figure references
            r'theorem\s*\d+',    # Theorem references
            r'lemma\s*\d+'       # Lemma references
        ]

        evidence = []
        for pattern in evidence_patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            evidence.extend(matches)

        return list(set(evidence)) if evidence else None

    def get_validation_history(self, claim_id: Optional[str] = None) -> List[ValidationResult]:
        """
        Get validation history for claims.

        Args:
            claim_id: Optional specific claim ID

        Returns:
            List[ValidationResult]: Validation results
        """
        if claim_id:
            return [self.validated_claims[claim_id]] if claim_id in self.validated_claims else []
        return list(self.validated_claims.values())

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dict[str, Any]: Validation statistics
        """
        total_claims = len(self.validated_claims)
        valid_claims = sum(1 for result in self.validated_claims.values() if result.is_valid)
        avg_confidence = sum(result.confidence_score for result in self.validated_claims.values()) / total_claims if total_claims > 0 else 0

        claim_type_stats = {}
        for claim_type in ClaimType:
            type_claims = [r for r in self.validated_claims.values() if r.claim_type == claim_type]
            if type_claims:
                claim_type_stats[claim_type.value] = {
                    'total': len(type_claims),
                    'valid': sum(1 for r in type_claims if r.is_valid),
                    'avg_confidence': sum(r.confidence_score for r in type_claims) / len(type_claims)
                }

        return {
            'total_claims': total_claims,
            'valid_claims': valid_claims,
            'validation_rate': valid_claims / total_claims if total_claims > 0 else 0,
            'avg_confidence': avg_confidence,
            'claim_type_stats': claim_type_stats
        }

    def revalidate_claim(self, claim_id: str, updated_claim_text: str) -> ValidationResult:
        """
        Revalidate an existing claim with updated text.

        Args:
            claim_id: Claim identifier
            updated_claim_text: Updated claim text

        Returns:
            ValidationResult: New validation result
        """
        if claim_id not in self.validated_claims:
            raise ValueError(f"Claim {claim_id} not found in validation history")

        original_result = self.validated_claims[claim_id]

        # Revalidate with updated text
        new_result = self.validate_claim(
            claim_id=claim_id,
            claim_text=updated_claim_text,
            claim_type=original_result.claim_type,
            domain='unknown',  # Could be enhanced to track domain
            source_id=original_result.validator_source
        )

        return new_result
