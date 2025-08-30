"""
UOIF (Unified Olympiad Information Framework) Ruleset Implementation

This package implements the UOIF Ruleset for managing mathematical proofs and claims
in complex domains like zeta dynamics, reverse Koopman operators, and consciousness frameworks.

The framework provides:
- Hierarchical source classification (canonical, expert, community)
- Quantitative claim validation and scoring
- Dynamic allocation systems for confidence measures
- Integration with scientific computing toolkit capabilities

Author: Scientific Computing Toolkit Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Scientific Computing Toolkit Team"

from .source_hierarchy import SourceHierarchy, SourceType
from .claim_validation import ClaimValidator, ClaimType, ValidationResult
from .scoring_engine import ScoringEngine, ScoreComponents
from .allocation_system import AllocationSystem, AllocationResult
from .toolkit_integration import (
    BootstrapIntegrator,
    UncertaintyQuantificationIntegrator,
    UOIFTemporalIntegrator
)
from .decision_equations import (
    DecisionEquations,
    DecisionParameters,
    OptimizationResult
)

__all__ = [
    'SourceHierarchy', 'SourceType',
    'ClaimValidator', 'ClaimType', 'ValidationResult',
    'ScoringEngine', 'ScoreComponents',
    'AllocationSystem', 'AllocationResult',
    'BootstrapIntegrator',
    'UncertaintyQuantificationIntegrator',
    'UOIFTemporalIntegrator',
    'DecisionEquations',
    'DecisionParameters',
    'OptimizationResult'
]
