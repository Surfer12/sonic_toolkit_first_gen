"""
UOIF Source Hierarchy Management

This module implements the hierarchical source classification system for the UOIF Ruleset,
providing canonical, expert, and community source management with priority ordering.

Priority Order: Canonical > Expert > Community (Mirror sources are context-only)

Author: Scientific Computing Toolkit Team
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class SourceType(Enum):
    """Source classification hierarchy."""
    CANONICAL = "canonical"      # Verified reports (highest priority)
    EXPERT = "expert"           # Expert interpretive sources
    COMMUNITY = "community"     # Community interpretive sources
    MIRROR = "mirror"          # Historical context (no priority)


@dataclass
class SourceMetadata:
    """Metadata for source classification."""
    source_id: str
    source_type: SourceType
    timestamp: datetime
    reliability_score: float
    domain: str
    validation_status: str
    citation_count: int = 0
    last_updated: Optional[datetime] = None


class SourceHierarchy:
    """
    Manages hierarchical source classification and validation.

    This class implements the UOIF source hierarchy system with priority ordering
    and reliability assessment for mathematical proof management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize source hierarchy manager.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.sources: Dict[str, SourceMetadata] = {}
        self._load_canonical_sources()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for source hierarchy."""
        return {
            'reliability_thresholds': {
                SourceType.CANONICAL: 0.95,
                SourceType.EXPERT: 0.85,
                SourceType.COMMUNITY: 0.70,
                SourceType.MIRROR: 0.50
            },
            'priority_weights': {
                SourceType.CANONICAL: 1.0,
                SourceType.EXPERT: 0.8,
                SourceType.COMMUNITY: 0.6,
                SourceType.MIRROR: 0.3
            },
            'domains': [
                'zeta_dynamics', 'koopman_operators', 'rspo_convergence',
                'consciousness_framework', 'hb_models', 'lstm_convergence'
            ]
        }

    def _load_canonical_sources(self):
        """Load predefined canonical sources from configuration."""
        canonical_sources = {
            'grok_report_7': {
                'type': SourceType.CANONICAL,
                'domain': 'koopman_operators',
                'description': 'Reverse Koopman Lipschitz continuity (DMD-based extraction)',
                'reliability': 0.98
            },
            'grok_report_9': {
                'type': SourceType.CANONICAL,
                'domain': 'rspo_convergence',
                'description': 'RSPO convergence via DMD swarm optimization',
                'reliability': 0.97
            },
            'grok_report_6_copy': {
                'type': SourceType.CANONICAL,
                'domain': 'consciousness_framework',
                'description': 'Euler-Lagrange Confidence Theorem (variational E[Ψ])',
                'reliability': 0.96
            }
        }

        for source_id, source_data in canonical_sources.items():
            metadata = SourceMetadata(
                source_id=source_id,
                source_type=source_data['type'],
                timestamp=datetime.now(),
                reliability_score=source_data['reliability'],
                domain=source_data['domain'],
                validation_status='verified',
                citation_count=0
            )
            self.sources[source_id] = metadata

    def classify_source(self, source_id: str, source_type: SourceType,
                       domain: str, reliability_score: float) -> SourceMetadata:
        """
        Classify and register a new source in the hierarchy.

        Args:
            source_id: Unique identifier for the source
            source_type: Classification type
            domain: Research domain
            reliability_score: Initial reliability assessment (0-1)

        Returns:
            SourceMetadata: Registered source metadata
        """
        if source_type == SourceType.MIRROR:
            raise ValueError("Mirror sources cannot be newly classified - use for context only")

        # Validate domain
        if domain not in self.config['domains']:
            raise ValueError(f"Unknown domain: {domain}")

        # Validate reliability threshold
        min_reliability = self.config['reliability_thresholds'][source_type]
        if reliability_score < min_reliability:
            raise ValueError(f"Reliability score {reliability_score} below threshold {min_reliability} for {source_type.value}")

        metadata = SourceMetadata(
            source_id=source_id,
            source_type=source_type,
            timestamp=datetime.now(),
            reliability_score=reliability_score,
            domain=domain,
            validation_status='pending_validation',
            citation_count=0
        )

        self.sources[source_id] = metadata
        return metadata

    def get_source_priority(self, source_id: str) -> float:
        """
        Get priority weight for a source based on its classification.

        Args:
            source_id: Source identifier

        Returns:
            float: Priority weight (0-1)
        """
        if source_id not in self.sources:
            return 0.0  # Unknown sources get lowest priority

        source_type = self.sources[source_id].source_type
        return self.config['priority_weights'][source_type]

    def validate_source(self, source_id: str, validation_result: bool,
                       updated_reliability: Optional[float] = None) -> bool:
        """
        Update source validation status and reliability.

        Args:
            source_id: Source identifier
            validation_result: True if validation passed
            updated_reliability: New reliability score (optional)

        Returns:
            bool: True if source is valid and active
        """
        if source_id not in self.sources:
            return False

        metadata = self.sources[source_id]

        if validation_result:
            metadata.validation_status = 'verified'
            if updated_reliability is not None:
                metadata.reliability_score = min(1.0, max(0.0, updated_reliability))
        else:
            metadata.validation_status = 'invalidated'

        metadata.last_updated = datetime.now()
        return validation_result

    def get_sources_by_domain(self, domain: str, min_reliability: float = 0.0) -> List[SourceMetadata]:
        """
        Get all sources for a specific domain above minimum reliability.

        Args:
            domain: Research domain
            min_reliability: Minimum reliability threshold

        Returns:
            List[SourceMetadata]: Filtered source list
        """
        return [
            metadata for metadata in self.sources.values()
            if metadata.domain == domain and metadata.reliability_score >= min_reliability
        ]

    def get_priority_sources(self, domain: Optional[str] = None,
                           min_priority: float = 0.0) -> List[SourceMetadata]:
        """
        Get sources sorted by priority weight.

        Args:
            domain: Optional domain filter
            min_priority: Minimum priority threshold

        Returns:
            List[SourceMetadata]: Sources sorted by priority (highest first)
        """
        sources = list(self.sources.values())

        if domain:
            sources = [s for s in sources if s.domain == domain]

        # Sort by priority weight (descending)
        sources.sort(key=lambda s: self.get_source_priority(s.source_id), reverse=True)

        # Filter by minimum priority
        return [s for s in sources if self.get_source_priority(s.source_id) >= min_priority]

    def export_hierarchy(self) -> str:
        """Export current source hierarchy as JSON."""
        hierarchy_data = {
            'export_timestamp': datetime.now().isoformat(),
            'config': self.config,
            'sources': {
                source_id: {
                    'type': metadata.source_type.value,
                    'domain': metadata.domain,
                    'reliability': metadata.reliability_score,
                    'validation_status': metadata.validation_status,
                    'citation_count': metadata.citation_count,
                    'timestamp': metadata.timestamp.isoformat(),
                    'last_updated': metadata.last_updated.isoformat() if metadata.last_updated else None
                }
                for source_id, metadata in self.sources.items()
            }
        }
        return json.dumps(hierarchy_data, indent=2)

    def import_hierarchy(self, hierarchy_json: str):
        """Import source hierarchy from JSON."""
        hierarchy_data = json.loads(hierarchy_json)

        # Update config if provided
        if 'config' in hierarchy_data:
            self.config.update(hierarchy_data['config'])

        # Import sources
        if 'sources' in hierarchy_data:
            for source_id, source_data in hierarchy_data['sources'].items():
                metadata = SourceMetadata(
                    source_id=source_id,
                    source_type=SourceType(source_data['type']),
                    timestamp=datetime.fromisoformat(source_data['timestamp']),
                    reliability_score=source_data['reliability'],
                    domain=source_data['domain'],
                    validation_status=source_data['validation_status'],
                    citation_count=source_data.get('citation_count', 0),
                    last_updated=datetime.fromisoformat(source_data['last_updated']) if source_data.get('last_updated') else None
                )
                self.sources[source_id] = metadata
