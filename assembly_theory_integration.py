#!/usr/bin/env python3
"""
🧬 ASSEMBLY THEORY INTEGRATION
==============================

Advanced Assembly Theory Integration for Scientific Computing Toolkit

This module implements Assembly Theory (Sara Imari Walker) integration with
consciousness quantification frameworks, providing quantitative measures of
complexity emergence and sub-assembly reuse patterns.

Features:
- Assembly index calculations (A(x) - minimal construction steps)
- Copy number quantification (N(x) - abundance metrics)
- Assembly complexity metrics
- Sub-assembly reuse efficiency analysis
- Integration with Ψ(x) consciousness framework

Author: Scientific Computing Toolkit
License: GPL-3.0-only
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize_scalar
import networkx as nx
from collections import defaultdict, Counter
import warnings


@dataclass
class AssemblyMetrics:
    """Assembly Theory metrics for consciousness evaluation"""

    assembly_index: float = 0.0
    """A(x) - minimal construction steps required to construct object x"""

    copy_number: int = 1
    """N(x) - number of similar objects observed in the system"""

    assembly_complexity: float = 0.0
    """Complexity measure based on assembly pathways"""

    reuse_efficiency: float = 0.0
    """Sub-assembly reuse metric (0-1 scale)"""

    minimal_assembly_path: List[str] = field(default_factory=list)
    """Minimal sequence of assembly operations"""

    sub_assembly_graph: Dict[str, List[str]] = field(default_factory=dict)
    """Graph representation of sub-assembly relationships"""

    emergence_score: float = 0.0
    """Measure of complexity emergence beyond constituent parts"""


@dataclass
class AssemblyPathway:
    """Represents a pathway of assembly operations"""

    operations: List[str]
    """Sequence of assembly operations"""

    complexity_score: float = 0.0
    """Complexity measure of this pathway"""

    reuse_count: int = 0
    """Number of times this pathway is reused"""

    energy_cost: float = 0.0
    """Thermodynamic cost of assembly"""

    stability_score: float = 0.0
    """Structural stability of assembled object"""


class AssemblyTheoryFramework:
    """
    Core framework for Assembly Theory implementation and analysis

    This framework provides quantitative tools for measuring complexity emergence
    and sub-assembly reuse patterns in biological, chemical, and cognitive systems.
    """

    def __init__(self, max_assembly_depth: int = 10):
        """
        Initialize Assembly Theory framework.

        Parameters
        ----------
        max_assembly_depth : int
            Maximum depth for assembly pathway exploration
        """
        self.max_assembly_depth = max_assembly_depth
        self.assembly_database: Dict[str, AssemblyMetrics] = {}
        self.pathway_cache: Dict[str, List[AssemblyPathway]] = {}
        self.sub_assembly_graph = nx.DiGraph()

    def calculate_assembly_index(self, object_representation: Union[str, List[str], np.ndarray],
                                observation_context: Optional[Dict[str, Any]] = None) -> AssemblyMetrics:
        """
        Calculate Assembly Theory metrics for a given object.

        This implements the core A(x) calculation - finding the minimal number
        of assembly operations required to construct the object from basic components.

        Parameters
        ----------
        object_representation : str, list, or ndarray
            Representation of the object to analyze
        observation_context : dict, optional
            Contextual information about observations

        Returns
        -------
        AssemblyMetrics
            Complete assembly analysis results
        """
        print(f"🔬 Calculating assembly metrics for: {type(object_representation)}")

        # Convert input to standardized format
        if isinstance(object_representation, str):
            components = self._decompose_string(object_representation)
        elif isinstance(object_representation, list):
            components = object_representation
        elif isinstance(object_representation, np.ndarray):
            components = self._decompose_array(object_representation)
        else:
            raise ValueError(f"Unsupported object representation type: {type(object_representation)}")

        # Find minimal assembly pathway
        minimal_pathway = self._find_minimal_assembly_pathway(components)

        # Calculate assembly index A(x)
        assembly_index = len(minimal_pathway.operations)

        # Calculate copy number N(x)
        copy_number = self._calculate_copy_number(object_representation, observation_context)

        # Build sub-assembly graph
        sub_assembly_graph = self._build_sub_assembly_graph(components, minimal_pathway)

        # Calculate complexity metrics
        assembly_complexity = self._calculate_assembly_complexity(minimal_pathway, sub_assembly_graph)

        # Calculate reuse efficiency
        reuse_efficiency = self._calculate_reuse_efficiency(sub_assembly_graph)

        # Calculate emergence score
        emergence_score = self._calculate_emergence_score(components, minimal_pathway)

        # Create metrics object
        metrics = AssemblyMetrics(
            assembly_index=assembly_index,
            copy_number=copy_number,
            assembly_complexity=assembly_complexity,
            reuse_efficiency=reuse_efficiency,
            minimal_assembly_path=minimal_pathway.operations,
            sub_assembly_graph=sub_assembly_graph,
            emergence_score=emergence_score
        )

        # Cache results
        cache_key = str(hash(str(object_representation)))
        self.assembly_database[cache_key] = metrics

        print(".2f"        print(".3f"        print(".3f"        print(".3f"
        return metrics

    def _decompose_string(self, text: str) -> List[str]:
        """Decompose string into basic components"""
        # Simple character-level decomposition
        return list(text)

    def _decompose_array(self, array: np.ndarray) -> List[str]:
        """Decompose array into structural components"""
        # Convert array elements to string representations
        return [f"elem_{i}_{val}" for i, val in enumerate(array.flatten())]

    def _find_minimal_assembly_pathway(self, components: List[str]) -> AssemblyPathway:
        """
        Find the minimal pathway of assembly operations.

        This implements a graph-based search for the shortest assembly sequence
        that can construct the target object from basic components.
        """
        if len(components) <= 2:
            # Trivial case
            return AssemblyPathway(
                operations=["combine"] * max(1, len(components) - 1),
                complexity_score=len(components) * 0.1
            )

        # Build assembly graph
        G = nx.DiGraph()

        # Add basic components as starting nodes
        for i, comp in enumerate(components):
            G.add_node(f"basic_{i}", component=comp, level=0)

        # Generate assembly operations up to max depth
        current_level = 0
        while current_level < self.max_assembly_depth:
            current_nodes = [n for n, d in G.nodes(data=True) if d.get('level', 0) == current_level]

            # Generate combinations of current level nodes
            for i in range(len(current_nodes)):
                for j in range(i + 1, len(current_nodes)):
                    node1 = current_nodes[i]
                    node2 = current_nodes[j]

                    # Create assembly operation
                    new_node = f"assembly_{current_level}_{i}_{j}"
                    G.add_node(new_node, level=current_level + 1)
                    G.add_edge(node1, new_node, operation="combine")
                    G.add_edge(node2, new_node, operation="combine")

                    # Check if we've reached the target complexity
                    if len(G.nodes()) >= len(components) * 2:
                        break
                if len(G.nodes()) >= len(components) * 2:
                    break
            current_level += 1

        # Find shortest path to target complexity
        target_complexity = len(components)
        shortest_path = None
        min_operations = float('inf')

        # Find all paths and select minimal
        for node in G.nodes():
            if G.nodes[node].get('level', 0) >= target_complexity - 1:
                # Count operations in path to this node
                operations = []
                for edge in G.edges():
                    if edge[1] == node:
                        operations.append(G.edges[edge].get('operation', 'combine'))

                if len(operations) < min_operations:
                    min_operations = len(operations)
                    shortest_path = operations

        return AssemblyPathway(
            operations=shortest_path or ["combine"] * max(1, target_complexity - 1),
            complexity_score=min_operations * 0.2
        )

    def _calculate_copy_number(self, object_rep: Any, context: Optional[Dict[str, Any]]) -> int:
        """Calculate copy number N(x) based on observation frequency"""
        if context and 'observation_frequency' in context:
            return max(1, context['observation_frequency'])

        # Default to 1 if no context provided
        return 1

    def _build_sub_assembly_graph(self, components: List[str],
                                 pathway: AssemblyPathway) -> Dict[str, List[str]]:
        """Build graph of sub-assembly relationships"""
        graph = defaultdict(list)

        # Create hierarchical relationships based on assembly pathway
        current_components = components.copy()

        for operation in pathway.operations:
            if len(current_components) >= 2:
                # Combine first two components
                comp1, comp2 = current_components[:2]
                new_comp = f"{comp1}_{operation}_{comp2}"

                graph[new_comp] = [comp1, comp2]
                current_components = [new_comp] + current_components[2:]

        return dict(graph)

    def _calculate_assembly_complexity(self, pathway: AssemblyPathway,
                                     sub_graph: Dict[str, List[str]]) -> float:
        """Calculate assembly complexity metric"""
        # Complexity based on pathway length and branching factor
        base_complexity = len(pathway.operations)

        # Add complexity from sub-assembly reuse
        reuse_factor = len(sub_graph) / max(1, len(pathway.operations))

        # Add complexity from hierarchical depth
        max_depth = max([len(path.split('_')) for path in sub_graph.keys()], default=1)

        return base_complexity * (1 + reuse_factor) * (max_depth ** 0.5)

    def _calculate_reuse_efficiency(self, sub_graph: Dict[str, List[str]]) -> float:
        """Calculate sub-assembly reuse efficiency (0-1 scale)"""
        if not sub_graph:
            return 0.0

        # Count how often sub-assemblies are reused
        sub_assembly_counts = Counter()
        for subs in sub_graph.values():
            for sub in subs:
                if '_' in sub:  # Complex sub-assembly
                    sub_assembly_counts[sub] += 1

        if not sub_assembly_counts:
            return 0.0

        # Calculate reuse efficiency
        total_subs = sum(sub_assembly_counts.values())
        unique_subs = len(sub_assembly_counts)
        avg_reuse = total_subs / unique_subs

        # Normalize to 0-1 scale (more reuse = higher efficiency)
        return min(1.0, (avg_reuse - 1) / 5.0)  # Cap at reasonable reuse level

    def _calculate_emergence_score(self, components: List[str], pathway: AssemblyPathway) -> float:
        """Calculate complexity emergence score"""
        # Emergence = complexity of whole - sum of parts
        individual_complexity = sum(len(comp) for comp in components)

        # Whole complexity based on assembly operations
        whole_complexity = len(pathway.operations) * pathway.complexity_score

        emergence = max(0, whole_complexity - individual_complexity)

        # Normalize to 0-1 scale
        return min(1.0, emergence / max(1, whole_complexity))

    def integrate_with_consciousness_framework(self, psi_function: Any,
                                             assembly_metrics: AssemblyMetrics) -> Dict[str, Any]:
        """
        Integrate Assembly Theory metrics with Ψ(x) consciousness framework.

        This creates a hybrid framework combining information-theoretic measures
        of complexity emergence with consciousness quantification.

        Parameters
        ----------
        psi_function : callable
            Ψ(x) consciousness evaluation function
        assembly_metrics : AssemblyMetrics
            Assembly Theory analysis results

        Returns
        -------
        dict
            Integrated consciousness-complexity analysis
        """
        print("🔗 Integrating Assembly Theory with Ψ(x) consciousness framework")

        # Create hybrid complexity measure
        hybrid_complexity = (
            assembly_metrics.assembly_complexity * 0.6 +
            assembly_metrics.emergence_score * 0.4
        )

        # Modulate consciousness by assembly complexity
        complexity_modulation = 1.0 + (hybrid_complexity - 0.5) * 0.2

        # Calculate assembly-informed consciousness bounds
        consciousness_lower = max(0, psi_function - 0.1 * (1 - assembly_metrics.reuse_efficiency))
        consciousness_upper = min(1, psi_function + 0.1 * assembly_metrics.emergence_score)

        integration_results = {
            'hybrid_complexity_score': hybrid_complexity,
            'consciousness_modulation_factor': complexity_modulation,
            'consciousness_bounds': (consciousness_lower, consciousness_upper),
            'assembly_consciousness_correlation': assembly_metrics.emergence_score * complexity_modulation,
            'integration_confidence': min(1.0, assembly_metrics.reuse_efficiency * 0.8 + 0.2),
            'emergence_consciousness_ratio': assembly_metrics.emergence_score / max(0.01, psi_function)
        }

        print(".3f"        print(".3f"
        return integration_results

    def analyze_biological_system(self, system_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze biological system using Assembly Theory framework.

        Parameters
        ----------
        system_description : dict
            Description of biological system with components, structures, etc.

        Returns
        -------
        dict
            Complete biological assembly analysis
        """
        print("🧬 Analyzing biological system with Assembly Theory")

        # Extract system components
        components = system_description.get('components', [])
        structures = system_description.get('structures', [])
        functions = system_description.get('functions', [])

        # Analyze each component
        component_metrics = {}
        for comp in components:
            metrics = self.calculate_assembly_index(comp, system_description)
            component_metrics[str(comp)] = {
                'assembly_index': metrics.assembly_index,
                'complexity': metrics.assembly_complexity,
                'emergence': metrics.emergence_score
            }

        # Analyze system-level assembly
        system_representation = f"system_{'_'.join(str(c) for c in components)}"
        system_metrics = self.calculate_assembly_index(system_representation, system_description)

        # Calculate biological emergence
        biological_emergence = self._calculate_biological_emergence(
            component_metrics, system_metrics, structures, functions
        )

        analysis_results = {
            'component_analysis': component_metrics,
            'system_assembly_metrics': {
                'assembly_index': system_metrics.assembly_index,
                'complexity': system_metrics.assembly_complexity,
                'emergence_score': system_metrics.emergence_score,
                'reuse_efficiency': system_metrics.reuse_efficiency
            },
            'biological_emergence_score': biological_emergence,
            'assembly_pathway_analysis': system_metrics.minimal_assembly_path,
            'sub_assembly_network': system_metrics.sub_assembly_graph
        }

        print(".3f"
        return analysis_results

    def _calculate_biological_emergence(self, component_metrics: Dict[str, Dict],
                                       system_metrics: AssemblyMetrics,
                                       structures: List[str], functions: List[str]) -> float:
        """Calculate emergence score for biological systems"""
        # Component-level complexity
        component_complexity = np.mean([m['complexity'] for m in component_metrics.values()])

        # Structure-function emergence
        structure_complexity = len(structures) * 0.1
        function_complexity = len(functions) * 0.15

        # Biological emergence = system - (components + structures + functions)
        emergence = max(0, system_metrics.assembly_complexity -
                       (component_complexity + structure_complexity + function_complexity))

        return min(1.0, emergence / max(1, system_metrics.assembly_complexity))

    def generate_assembly_report(self, metrics: AssemblyMetrics,
                               system_name: str = "Unknown System") -> str:
        """Generate comprehensive assembly analysis report"""
        report = []
        report.append("🧬 ASSEMBLY THEORY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        report.append(f"System: {system_name}")
        report.append("")

        report.append("📊 ASSEMBLY METRICS")
        report.append("-" * 30)
        report.append(".2f"        report.append(f"Copy Number (N(x)): {metrics.copy_number}")
        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append("")

        report.append("🔧 ASSEMBLY PATHWAY")
        report.append("-" * 25)
        if metrics.minimal_assembly_path:
            for i, op in enumerate(metrics.minimal_assembly_path, 1):
                report.append(f"{i:2d}. {op}")
        else:
            report.append("No assembly pathway found")
        report.append("")

        report.append("🌐 SUB-ASSEMBLY NETWORK")
        report.append("-" * 28)
        if metrics.sub_assembly_graph:
            for parent, children in metrics.sub_assembly_graph.items():
                report.append(f"  {parent}")
                for child in children:
                    report.append(f"    └─ {child}")
        else:
            report.append("No sub-assembly relationships found")
        report.append("")

        report.append("🎯 INTERPRETATION")
        report.append("-" * 20)
        if metrics.emergence_score > 0.7:
            report.append("High emergence: System shows strong complexity emergence")
        elif metrics.emergence_score > 0.4:
            report.append("Moderate emergence: Some complexity beyond components")
        else:
            report.append("Low emergence: Complexity mostly explained by components")

        if metrics.reuse_efficiency > 0.6:
            report.append("High reuse: Efficient sub-assembly utilization")
        elif metrics.reuse_efficiency > 0.3:
            report.append("Moderate reuse: Some sub-assembly sharing")
        else:
            report.append("Low reuse: Limited sub-assembly optimization")

        return "\n".join(report)


def demonstrate_assembly_theory():
    """Demonstrate Assembly Theory integration capabilities"""
    print("🧬 ASSEMBLY THEORY INTEGRATION DEMONSTRATION")
    print("=" * 60)

    framework = AssemblyTheoryFramework()

    # Example 1: Simple molecule
    print("\n1️⃣ ANALYZING SIMPLE MOLECULE")
    water_components = ["H", "H", "O"]
    water_metrics = framework.calculate_assembly_index(water_components)

    print(f"Assembly Index A(x): {water_metrics.assembly_index}")
    print(f"Emergence Score: {water_metrics.emergence_score:.3f}")

    # Example 2: Biological system
    print("\n2️⃣ ANALYZING BIOLOGICAL SYSTEM")
    biological_system = {
        'components': ['protein', 'dna', 'rna', 'lipid'],
        'structures': ['membrane', 'nucleus', 'ribosome'],
        'functions': ['transcription', 'translation', 'metabolism'],
        'observation_frequency': 1000
    }

    bio_analysis = framework.analyze_biological_system(biological_system)
    print(f"Biological Emergence: {bio_analysis['biological_emergence_score']:.3f}")

    # Example 3: Integration with consciousness framework
    print("\n3️⃣ INTEGRATION WITH Ψ(x) CONSCIOUSNESS FRAMEWORK")

    # Mock Ψ(x) function
    def mock_psi_function():
        return 0.75  # Example consciousness score

    psi_score = mock_psi_function()
    integration = framework.integrate_with_consciousness_framework(
        psi_score, water_metrics
    )

    print(f"Original Ψ(x): {psi_score}")
    print(f"Hybrid Complexity: {integration['hybrid_complexity_score']:.3f}")
    print(f"Consciousness Bounds: {integration['consciousness_bounds']}")

    # Generate report
    print("\n4️⃣ GENERATING COMPREHENSIVE REPORT")
    report = framework.generate_assembly_report(water_metrics, "Water Molecule")
    print(report)


if __name__ == "__main__":
    demonstrate_assembly_theory()
