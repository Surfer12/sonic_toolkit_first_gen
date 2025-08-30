#!/usr/bin/env python3
"""
🧬 PROTEIN FOLDING & PRIME PATTERN ANALYSIS
==============================================

Advanced analysis connecting prime number theory to protein folding patterns,
amino acid sequences, and biological optimization algorithms.

Integrates with our existing scientific computing toolkit:
- Twin prime analysis framework
- Biological sequence pattern recognition
- Optimization algorithms
- Chaos theory (Lorenz attractors)

Author: Scientific Computing Toolkit
License: GPL-3.0-only
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional, Any, Union
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import warnings
import re


@dataclass
class AminoAcidProperties:
    """Complete amino acid properties with prime number mappings."""
    name: str
    code: str
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity: float
    volume: float
    polarity: float
    charge: float
    # Prime number mappings for pattern analysis
    prime_value: int
    binary_prime: int  # 2, 3, 5, 7, 11, 13, 17, 19
    twin_prime_map: Tuple[int, int]  # Maps to twin prime pairs

    def __post_init__(self):
        self.validate_properties()

    def validate_properties(self):
        """Validate amino acid properties."""
        if self.prime_value < 2:
            raise ValueError(f"Prime value must be >= 2, got {self.prime_value}")
        if not (2 <= self.binary_prime <= 19):
            raise ValueError(f"Binary prime must be 2,3,5,7,11,13,17,19, got {self.binary_prime}")
        if len(self.code) != 1:
            raise ValueError(f"Amino acid code must be single character, got '{self.code}'")


# Standard amino acid properties with prime mappings
AMINO_ACID_DATA = {
    'A': AminoAcidProperties('Alanine', 'A', 89.1, 6.00, 1.8, 88.6, 8.1, 0.0, 2, 2, (3, 5)),
    'R': AminoAcidProperties('Arginine', 'R', 174.2, 10.76, -4.5, 173.4, 10.5, 1.0, 3, 3, (5, 7)),
    'N': AminoAcidProperties('Asparagine', 'N', 132.1, 5.41, -3.5, 114.1, 11.6, 0.0, 5, 5, (5, 7)),
    'D': AminoAcidProperties('Aspartic Acid', 'D', 133.1, 2.77, -3.5, 111.1, 13.0, -1.0, 7, 7, (5, 7)),
    'C': AminoAcidProperties('Cysteine', 'C', 121.2, 5.07, 2.5, 108.5, 1.9, 0.0, 11, 2, (11, 13)),
    'Q': AminoAcidProperties('Glutamine', 'Q', 146.2, 5.65, -3.5, 143.8, 10.5, 0.0, 13, 3, (11, 13)),
    'E': AminoAcidProperties('Glutamic Acid', 'E', 147.1, 3.22, -3.5, 138.4, 12.3, -1.0, 17, 5, (17, 19)),
    'G': AminoAcidProperties('Glycine', 'G', 75.1, 5.97, -0.4, 60.1, 9.0, 0.0, 19, 7, (17, 19)),
    'H': AminoAcidProperties('Histidine', 'H', 155.2, 7.59, -3.2, 153.2, 10.4, 0.1, 23, 11, (29, 31)),
    'I': AminoAcidProperties('Isoleucine', 'I', 131.2, 6.02, 4.5, 166.7, 5.2, 0.0, 29, 13, (29, 31)),
    'L': AminoAcidProperties('Leucine', 'L', 131.2, 5.98, 3.8, 166.7, 4.9, 0.0, 31, 17, (29, 31)),
    'K': AminoAcidProperties('Lysine', 'K', 146.2, 9.74, -3.9, 168.6, 11.3, 1.0, 37, 19, (41, 43)),
    'M': AminoAcidProperties('Methionine', 'M', 149.2, 5.74, 1.9, 162.9, 5.7, 0.0, 41, 2, (41, 43)),
    'F': AminoAcidProperties('Phenylalanine', 'F', 165.2, 5.48, 2.8, 189.9, 5.2, 0.0, 43, 3, (41, 43)),
    'P': AminoAcidProperties('Proline', 'P', 115.1, 6.30, -1.6, 112.7, 8.0, 0.0, 47, 5, (47, 49)),
    'S': AminoAcidProperties('Serine', 'S', 105.1, 5.68, -0.8, 89.0, 9.2, 0.0, 53, 7, (53, 55)),
    'T': AminoAcidProperties('Threonine', 'T', 119.1, 5.60, -0.7, 116.1, 8.6, 0.0, 59, 11, (59, 61)),
    'W': AminoAcidProperties('Tryptophan', 'W', 204.2, 5.89, -0.9, 227.8, 10.4, 0.0, 61, 13, (59, 61)),
    'Y': AminoAcidProperties('Tyrosine', 'Y', 181.2, 5.66, -1.3, 193.6, 6.1, 0.0, 67, 17, (67, 69)),
    'V': AminoAcidProperties('Valine', 'V', 117.1, 5.96, 4.2, 140.0, 5.9, 0.0, 71, 19, (71, 73))
}


class ProteinPrimeAnalyzer:
    """Advanced protein sequence analysis using prime number theory."""

    def __init__(self):
        self.amino_acids = AMINO_ACID_DATA
        self.pattern_cache = {}

    def sequence_to_primes(self, sequence: str) -> List[int]:
        """Convert amino acid sequence to prime number representation."""
        prime_sequence = []
        for aa in sequence.upper():
            if aa in self.amino_acids:
                prime_sequence.append(self.amino_acids[aa].prime_value)
        return prime_sequence

    def analyze_prime_patterns(self, prime_sequence: List[int]) -> Dict[str, Any]:
        """Analyze prime patterns in protein sequence."""
        if len(prime_sequence) < 2:
            return {'error': 'Sequence too short'}

        # Twin prime analysis
        twin_patterns = []
        for i in range(len(prime_sequence) - 1):
            diff = abs(prime_sequence[i+1] - prime_sequence[i])
            if diff == 2:  # Twin prime difference
                twin_patterns.append((i, prime_sequence[i], prime_sequence[i+1]))

        # Prime gap analysis
        gaps = []
        prime_positions = [i for i, val in enumerate(prime_sequence) if self.is_prime_cached(val)]
        for i in range(len(prime_positions) - 1):
            gaps.append(prime_positions[i+1] - prime_positions[i])

        # Statistical analysis
        if gaps:
            gap_stats = {
                'mean': np.mean(gaps),
                'std': np.std(gaps),
                'min': np.min(gaps),
                'max': np.max(gaps),
                'unique_gaps': len(set(gaps))
            }
        else:
            gap_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'unique_gaps': 0}

        return {
            'sequence_length': len(prime_sequence),
            'prime_sequence': prime_sequence,
            'twin_patterns': twin_patterns,
            'prime_positions': prime_positions,
            'prime_gaps': gaps,
            'gap_statistics': gap_stats,
            'pattern_density': len(twin_patterns) / len(prime_sequence) if prime_sequence else 0,
            'prime_coverage': len(prime_positions) / len(prime_sequence) if prime_sequence else 0
        }

    def analyze_structural_motifs(self, sequence: str) -> Dict[str, Any]:
        """Analyze structural motifs using prime patterns."""
        prime_seq = self.sequence_to_primes(sequence)
        analysis = self.analyze_prime_patterns(prime_seq)

        # Identify potential structural motifs
        motifs = {
            'helix_prone': [],      # Patterns suggesting alpha helix
            'sheet_prone': [],      # Patterns suggesting beta sheet
            'coil_prone': [],       # Patterns suggesting random coil
            'hydrophobic_core': [], # Hydrophobic clustering patterns
            'charge_clusters': []   # Charge clustering patterns
        }

        # Simple motif detection based on prime patterns
        for i, (pos, p1, p2) in enumerate(analysis['twin_patterns']):
            aa1 = [aa for aa, props in self.amino_acids.items() if props.prime_value == p1][0]
            aa2 = [aa for aa, props in self.amino_acids.items() if props.prime_value == p2][0]

            # Hydrophobic pattern detection
            if (self.amino_acids[aa1].hydrophobicity > 2.0 and
                self.amino_acids[aa2].hydrophobicity > 2.0):
                motifs['hydrophobic_core'].append((pos, aa1, aa2))

            # Charge pattern detection
            charge1 = self.amino_acids[aa1].charge
            charge2 = self.amino_acids[aa2].charge
            if abs(charge1) > 0.5 and abs(charge2) > 0.5:
                if (charge1 * charge2) > 0:  # Same charge
                    motifs['charge_clusters'].append((pos, aa1, aa2, 'repulsive'))
                else:  # Opposite charge
                    motifs['charge_clusters'].append((pos, aa1, aa2, 'attractive'))

        return {
            'base_analysis': analysis,
            'structural_motifs': motifs,
            'motif_statistics': {
                'total_motifs': sum(len(patterns) for patterns in motifs.values()),
                'hydrophobic_density': len(motifs['hydrophobic_core']) / len(sequence),
                'charge_density': len(motifs['charge_clusters']) / len(sequence)
            }
        }

    @lru_cache(maxsize=1000)
    def is_prime_cached(self, n: int) -> bool:
        """Cached primality test for performance."""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False

        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


class ProteinOptimizationFramework:
    """Connect protein analysis to optimization algorithms."""

    def __init__(self):
        self.analyzer = ProteinPrimeAnalyzer()
        self.fitness_cache = {}

    def protein_folding_fitness(self, sequence: str, target_structure: str = 'helix') -> float:
        """Calculate fitness score for protein folding prediction."""
        if sequence in self.fitness_cache:
            return self.fitness_cache[sequence]

        analysis = self.analyzer.analyze_structural_motifs(sequence)

        # Multi-objective fitness function
        twin_density = analysis['base_analysis']['pattern_density']
        hydrophobic_density = analysis['motif_statistics']['hydrophobic_density']
        charge_balance = analysis['motif_statistics']['charge_density']

        # Target-specific scoring
        if target_structure == 'helix':
            # Alpha helices prefer hydrophobic clustering
            fitness = 0.4 * twin_density + 0.4 * hydrophobic_density + 0.2 * (1 - charge_balance)
        elif target_structure == 'sheet':
            # Beta sheets prefer charge balance
            fitness = 0.3 * twin_density + 0.3 * hydrophobic_density + 0.4 * charge_balance
        else:  # Random coil
            fitness = 0.5 * twin_density + 0.3 * (1 - hydrophobic_density) + 0.2 * charge_balance

        self.fitness_cache[sequence] = fitness
        return fitness

    def optimize_protein_sequence(self, initial_sequence: str, target_structure: str = 'helix',
                                generations: int = 50) -> Dict[str, Any]:
        """Simple genetic algorithm for protein sequence optimization."""
        current_sequence = initial_sequence
        best_fitness = self.protein_folding_fitness(current_sequence, target_structure)
        fitness_history = [best_fitness]

        # Amino acid alphabet for mutations
        aa_alphabet = list(self.analyzer.amino_acids.keys())

        for generation in range(generations):
            # Simple mutation: change one amino acid
            new_sequence = list(current_sequence)
            mutation_pos = np.random.randint(len(new_sequence))
            new_aa = np.random.choice(aa_alphabet)
            new_sequence[mutation_pos] = new_aa
            new_sequence_str = ''.join(new_sequence)

            # Evaluate fitness
            new_fitness = self.protein_folding_fitness(new_sequence_str, target_structure)

            # Accept if better (greedy selection)
            if new_fitness > best_fitness:
                current_sequence = new_sequence_str
                best_fitness = new_fitness

            fitness_history.append(best_fitness)

        return {
            'optimized_sequence': current_sequence,
            'final_fitness': best_fitness,
            'fitness_history': fitness_history,
            'generations': generations,
            'target_structure': target_structure
        }


def create_protein_visualization_dashboard():
    """Create comprehensive visualization of protein prime analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧬 Protein Folding & Prime Pattern Analysis', fontsize=16, fontweight='bold')

    # Initialize analyzer
    analyzer = ProteinPrimeAnalyzer()
    optimizer = ProteinOptimizationFramework()

    # Sample protein sequences
    sequences = {
        'Alpha Helix Prone': 'AAAAAAAAAAAA',  # Poly-alanine (helix former)
        'Beta Sheet Prone': 'VVVVVVVVVVVV',   # Poly-valine (sheet former)
        'Mixed Sequence': 'MKWVTFISLLFLFSSAYSRGVFRR',
        'Fibroin Motif': 'GAGAGS'            # Silk fibroin repeat
    }

    # Plot 1: Prime sequence analysis
    ax = axes[0, 0]
    for i, (name, seq) in enumerate(list(sequences.items())[:3]):
        prime_seq = analyzer.sequence_to_primes(seq)
        ax.plot(range(len(prime_seq)), prime_seq, 'o-', label=name[:15], markersize=4)

    ax.set_title('Amino Acid → Prime Number Mapping')
    ax.set_xlabel('Position in Sequence')
    ax.set_ylabel('Prime Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Twin prime pattern density
    ax = axes[0, 1]
    names = []
    densities = []
    for name, seq in sequences.items():
        analysis = analyzer.analyze_prime_patterns(analyzer.sequence_to_primes(seq))
        names.append(name[:12])
        densities.append(analysis['pattern_density'])

    bars = ax.bar(range(len(names)), densities, color='skyblue', alpha=0.7)
    ax.set_title('Twin Prime Pattern Density')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Pattern Density')

    # Add value labels on bars
    for bar, density in zip(bars, densities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '.3f', ha='center', va='bottom', fontsize=9)

    # Plot 3: Structural motif analysis
    ax = axes[0, 2]
    motif_types = ['Hydrophobic', 'Charge\nClusters', 'Total\nMotifs']
    colors = ['orange', 'red', 'purple']

    for i, (name, seq) in enumerate(list(sequences.items())[:2]):  # First 2 sequences
        analysis = analyzer.analyze_structural_motifs(seq)
        stats = analysis['motif_statistics']

        # Create grouped bars
        x_pos = np.arange(len(motif_types)) + i * 0.35
        values = [stats['hydrophobic_density'], stats['charge_density'], stats['total_motifs']]

        ax.bar(x_pos, values, width=0.35, label=name[:12],
               color=colors[i], alpha=0.7)

    ax.set_title('Structural Motif Analysis')
    ax.set_xticks(np.arange(len(motif_types)) + 0.175)
    ax.set_xticklabels(motif_types)
    ax.set_ylabel('Density/Motif Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Optimization evolution
    ax = axes[1, 0]
    initial_seq = 'MKWVTFISLLFLFSSAYSRGVFRR'
    optimization = optimizer.optimize_protein_sequence(
        initial_seq, target_structure='helix', generations=30
    )

    ax.plot(range(len(optimization['fitness_history'])), optimization['fitness_history'],
            'o-', color='green', linewidth=2, markersize=4)
    ax.set_title('Protein Sequence Optimization')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Score')
    ax.grid(True, alpha=0.3)

    # Add final fitness annotation
    final_fitness = optimization['final_fitness']
    ax.annotate('.3f',
                xy=(len(optimization['fitness_history'])-1, final_fitness),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # Plot 5: Prime gap distribution
    ax = axes[1, 1]
    all_gaps = []
    for seq in sequences.values():
        analysis = analyzer.analyze_prime_patterns(analyzer.sequence_to_primes(seq))
        all_gaps.extend(analysis['prime_gaps'])

    if all_gaps:
        ax.hist(all_gaps, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.set_title('Prime Gap Distribution')
        ax.set_xlabel('Gap Size')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_gap = np.mean(all_gaps)
        ax.axvline(mean_gap, color='red', linestyle='--', label='.1f')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No gaps found\nin sequences',
                ha='center', va='center', transform=ax.transAxes)

    # Plot 6: Amino acid prime value distribution
    ax = axes[1, 2]
    prime_values = [props.prime_value for props in analyzer.amino_acids.values()]
    aa_codes = list(analyzer.amino_acids.keys())

    # Create scatter plot with amino acid codes
    scatter = ax.scatter(range(len(prime_values)), prime_values,
                        c=prime_values, cmap='viridis', s=100, alpha=0.8)

    # Add amino acid labels
    for i, (code, prime_val) in enumerate(zip(aa_codes, prime_values)):
        ax.annotate(code, (i, prime_val), xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=8)

    ax.set_title('Amino Acid Prime Value Distribution')
    ax.set_xlabel('Amino Acid Index')
    ax.set_ylabel('Prime Value')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Prime Value')

    plt.tight_layout()
    return fig, axes


def main():
    """Main demonstration of protein prime analysis."""
    print("🧬 PROTEIN FOLDING & PRIME PATTERN ANALYSIS")
    print("=" * 55)

    # Initialize analyzer
    analyzer = ProteinPrimeAnalyzer()
    optimizer = ProteinOptimizationFramework()

    # Example protein sequences
    sequences = {
        'Alpha Helix Former': 'AAAAAAAAAAAA',
        'Beta Sheet Former': 'VVVVVVVVVVVV',
        'Mixed Protein': 'MKWVTFISLLFLFSSAYSRGVFRR',
        'Fibroin Repeat': 'GAGAGS'
    }

    print("\n🔬 Analyzing Protein Sequences:")
    print("-" * 35)

    for name, sequence in sequences.items():
        print(f"\n{name}: {sequence}")

        # Convert to prime representation
        prime_seq = analyzer.sequence_to_primes(sequence)
        print(f"Prime representation: {prime_seq}")

        # Analyze patterns
        analysis = analyzer.analyze_prime_patterns(prime_seq)
        print(f"Sequence length: {analysis['sequence_length']}")
        print(f"Twin patterns found: {len(analysis['twin_patterns'])}")
        print(".3f")

        if analysis['prime_gaps']:
            print(f"Prime gaps: {analysis['prime_gaps']}")
            print(f"Average gap: {analysis['gap_statistics']['mean']:.2f}")

        # Structural motif analysis
        structural = analyzer.analyze_structural_motifs(sequence)
        print(f"Structural motifs: {structural['motif_statistics']['total_motifs']}")
        print(f"Hydrophobic density: {structural['motif_statistics']['hydrophobic_density']:.3f}")

    print("\n🎯 Protein Sequence Optimization:")
    print("-" * 35)

    # Optimize for alpha helix
    initial_seq = 'MKWVTFISLLFLFSSAYSRGVFRR'
    optimization = optimizer.optimize_protein_sequence(
        initial_seq, target_structure='helix', generations=20
    )

    print(f"Initial sequence: {initial_seq}")
    print(f"Optimized sequence: {optimization['optimized_sequence']}")
    print(".3f")
    print(f"Generations: {optimization['generations']}")

    print("\n📊 Generating Visualization Dashboard...")
    fig, axes = create_protein_visualization_dashboard()

    # Save visualization
    output_file = "protein_prime_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")

    print("\n✨ Analysis Complete!")
    print("Protein folding patterns show fascinating connections to prime number theory,")
    print("with structural motifs emerging from mathematical patterns in amino acid sequences.")

    return {
        'sequences_analyzed': len(sequences),
        'optimization_results': optimization,
        'visualization_file': output_file
    }


if __name__ == "__main__":
    results = main()
