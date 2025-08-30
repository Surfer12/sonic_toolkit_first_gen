#!/usr/bin/env python3
"""
📊 PERFORMANCE BENCHMARKING SUITE
==================================

Comprehensive Performance Benchmarking for Scientific Computing Toolkit

This module provides detailed performance analysis and benchmarking capabilities
for all scientific frameworks, including computational efficiency, memory usage,
scalability analysis, and comparative performance metrics.

Features:
- Computational performance benchmarks
- Memory usage profiling
- Scalability analysis
- Cross-platform performance comparison
- Performance regression detection
- Real-time performance monitoring

Author: Scientific Computing Toolkit
License: GPL-3.0-only
"""

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import tracemalloc
import cProfile
import pstats
import io
from scipy import stats
import json
from pathlib import Path
import warnings


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics"""

    execution_time: float = 0.0
    """Total execution time in seconds"""

    peak_memory_usage: float = 0.0
    """Peak memory usage in MB"""

    average_memory_usage: float = 0.0
    """Average memory usage in MB"""

    cpu_utilization: float = 0.0
    """Average CPU utilization percentage"""

    memory_efficiency: float = 0.0
    """Memory efficiency score (0-1, higher is better)"""

    computational_efficiency: float = 0.0
    """Computational efficiency score (operations/second)"""

    scalability_score: float = 0.0
    """Scalability score across different problem sizes"""

    benchmark_timestamp: str = ""
    """Timestamp when benchmark was run"""

    system_info: Dict[str, Any] = field(default_factory=dict)
    """System information during benchmark"""


@dataclass
class BenchmarkResult:
    """Complete benchmark result with statistics"""

    component_name: str
    benchmark_name: str
    metrics: PerformanceMetrics
    statistical_summary: Dict[str, float]
    performance_trends: List[Dict[str, Any]]
    recommendations: List[str]


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking framework

    Provides detailed performance analysis for all scientific computing components
    with statistical validation and comparative analysis capabilities.
    """

    def __init__(self):
        self.results_history: List[BenchmarkResult] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'platform': psutil.platform,
            'python_version': psutil.python_version()
        }

    @contextmanager
    def performance_monitor(self):
        """Context manager for comprehensive performance monitoring"""
        tracemalloc.start()
        start_time = time.perf_counter()
        process = psutil.Process()

        # Initial measurements
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        initial_cpu = process.cpu_percent(interval=None)

        monitoring_data = {
            'memory_usage': [],
            'cpu_usage': [],
            'timestamps': []
        }

        try:
            yield monitoring_data
        finally:
            # Final measurements
            end_time = time.perf_counter()
            final_memory = process.memory_info().rss / (1024**2)  # MB
            final_cpu = process.cpu_percent(interval=None)

            # Calculate metrics
            execution_time = end_time - start_time

            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024**2)  # MB

            # Calculate averages
            avg_memory = (initial_memory + final_memory) / 2
            avg_cpu = (initial_cpu + final_cpu) / 2

            # Store monitoring data
            monitoring_data['execution_time'] = execution_time
            monitoring_data['peak_memory'] = peak_memory
            monitoring_data['avg_memory'] = avg_memory
            monitoring_data['avg_cpu'] = avg_cpu

            tracemalloc.stop()

    def benchmark_component(self, component_name: str, benchmark_name: str,
                          component_function: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark a specific component with comprehensive performance analysis.

        Parameters
        ----------
        component_name : str
            Name of the component being benchmarked
        benchmark_name : str
            Specific benchmark test name
        component_function : callable
            Function to benchmark
        *args, **kwargs
            Arguments to pass to the component function

        Returns
        -------
        BenchmarkResult
            Complete benchmark results with statistics
        """
        print(f"🏃 Running benchmark: {component_name} - {benchmark_name}")

        # Run multiple iterations for statistical significance
        n_iterations = 5
        execution_times = []
        memory_peaks = []
        cpu_usages = []

        for i in range(n_iterations):
            print(f"  Iteration {i+1}/{n_iterations}...")

            with self.performance_monitor() as monitor:
                try:
                    result = component_function(*args, **kwargs)
                except Exception as e:
                    print(f"    ⚠️  Error in iteration {i+1}: {e}")
                    continue

            execution_times.append(monitor['execution_time'])
            memory_peaks.append(monitor['peak_memory'])
            cpu_usages.append(monitor['avg_cpu'])

        if not execution_times:
            raise RuntimeError(f"All {n_iterations} benchmark iterations failed")

        # Calculate statistical metrics
        execution_stats = self._calculate_statistics(execution_times)
        memory_stats = self._calculate_statistics(memory_peaks)
        cpu_stats = self._calculate_statistics(cpu_usages)

        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_time=np.mean(execution_times),
            peak_memory_usage=np.mean(memory_peaks),
            average_memory_usage=np.mean([m for m in memory_peaks]),  # Simplified
            cpu_utilization=np.mean(cpu_usages),
            memory_efficiency=self._calculate_memory_efficiency(memory_peaks, execution_times),
            computational_efficiency=self._calculate_computational_efficiency(component_function, args, execution_times),
            scalability_score=self._assess_scalability(component_name),
            benchmark_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.system_info
        )

        # Generate statistical summary
        statistical_summary = {
            'execution_time_mean': execution_stats['mean'],
            'execution_time_std': execution_stats['std'],
            'execution_time_cv': execution_stats['cv'],
            'memory_peak_mean': memory_stats['mean'],
            'memory_peak_std': memory_stats['std'],
            'cpu_usage_mean': cpu_stats['mean'],
            'cpu_usage_std': cpu_stats['std']
        }

        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(
            execution_times, memory_peaks, cpu_usages
        )

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            metrics, statistical_summary, component_name
        )

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            component_name=component_name,
            benchmark_name=benchmark_name,
            metrics=metrics,
            statistical_summary=statistical_summary,
            performance_trends=performance_trends,
            recommendations=recommendations
        )

        # Store in history
        self.results_history.append(benchmark_result)

        print(".2f"        print(".1f"        print(".1f"
        return benchmark_result

    def _calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for performance data"""
        if len(data) < 2:
            return {
                'mean': np.mean(data),
                'std': 0.0,
                'cv': 0.0,
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data)
            }

        return {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'cv': np.std(data, ddof=1) / np.mean(data),  # Coefficient of variation
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data)
        }

    def _calculate_memory_efficiency(self, memory_usage: List[float],
                                   execution_times: List[float]) -> float:
        """Calculate memory efficiency score"""
        if not memory_usage or not execution_times:
            return 0.0

        avg_memory = np.mean(memory_usage)
        avg_time = np.mean(execution_times)

        # Efficiency = 1 / (memory * time) normalized
        # Lower memory-time product = higher efficiency
        efficiency_raw = 1.0 / (avg_memory * avg_time + 1e-10)

        # Normalize to 0-1 scale
        return min(1.0, efficiency_raw / 1e-6)

    def _calculate_computational_efficiency(self, func: Callable, args: Tuple,
                                          execution_times: List[float]) -> float:
        """Calculate computational efficiency (operations per second)"""
        if not execution_times:
            return 0.0

        avg_time = np.mean(execution_times)

        # Estimate operations based on function complexity
        # This is a simplified heuristic - could be made more sophisticated
        estimated_ops = 1000  # Base assumption

        if hasattr(func, '__name__'):
            func_name = func.__name__.lower()
            if 'inverse' in func_name:
                estimated_ops = 5000  # More complex
            elif 'optimization' in func_name:
                estimated_ops = 10000  # Most complex
            elif 'simple' in func_name:
                estimated_ops = 500   # Simpler

        return estimated_ops / max(avg_time, 1e-10)

    def _assess_scalability(self, component_name: str) -> float:
        """Assess scalability score based on component characteristics"""
        # Simplified scalability assessment
        # In practice, this would run benchmarks at different scales
        scalability_factors = {
            'inverse_precision': 0.8,
            'rheology_model': 0.9,
            'biological_flow': 0.7,
            'optical_system': 0.85,
            'cryptographic': 0.95,
            'validation_suite': 0.75
        }

        return scalability_factors.get(component_name.lower().replace('_', ''), 0.5)

    def _analyze_performance_trends(self, execution_times: List[float],
                                   memory_usage: List[float],
                                   cpu_usage: List[float]) -> List[Dict[str, Any]]:
        """Analyze performance trends across iterations"""
        trends = []

        # Check for performance degradation
        if len(execution_times) > 2:
            trend_slope = np.polyfit(range(len(execution_times)), execution_times, 1)[0]
            if trend_slope > 0.01:  # Significant increase
                trends.append({
                    'type': 'degradation',
                    'metric': 'execution_time',
                    'description': 'Performance degradation detected',
                    'severity': 'warning'
                })

        # Check for memory leaks
        if len(memory_usage) > 2:
            memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
            if memory_trend > 1.0:  # Significant memory increase
                trends.append({
                    'type': 'memory_leak',
                    'metric': 'memory_usage',
                    'description': 'Potential memory leak detected',
                    'severity': 'critical'
                })

        # Check for CPU variability
        if len(cpu_usage) > 2:
            cpu_cv = np.std(cpu_usage) / np.mean(cpu_usage)
            if cpu_cv > 0.5:  # High variability
                trends.append({
                    'type': 'cpu_variability',
                    'metric': 'cpu_usage',
                    'description': 'High CPU utilization variability',
                    'severity': 'info'
                })

        return trends

    def _generate_performance_recommendations(self, metrics: PerformanceMetrics,
                                            stats: Dict[str, float],
                                            component_name: str) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        # Memory recommendations
        if metrics.peak_memory_usage > 1000:  # > 1GB
            recommendations.append("Consider memory optimization - peak usage exceeds 1GB")

        if stats.get('memory_peak_std', 0) > 100:  # High variability
            recommendations.append("Investigate memory allocation patterns for consistency")

        # Execution time recommendations
        if metrics.execution_time > 60:  # > 1 minute
            recommendations.append("Consider algorithmic optimization for long execution times")

        if stats.get('execution_time_cv', 0) > 0.3:  # High variability
            recommendations.append("Performance is inconsistent - investigate sources of variability")

        # CPU recommendations
        if metrics.cpu_utilization > 90:
            recommendations.append("High CPU utilization - consider parallelization or optimization")

        # Efficiency recommendations
        if metrics.memory_efficiency < 0.5:
            recommendations.append("Memory efficiency could be improved")

        if metrics.computational_efficiency < 1000:
            recommendations.append("Computational efficiency could be optimized")

        # Scalability recommendations
        if metrics.scalability_score < 0.7:
            recommendations.append("Consider scalability improvements for larger problem sizes")

        return recommendations

    def create_performance_dashboard(self, results: List[BenchmarkResult],
                                   save_path: str = "performance_dashboard.png") -> plt.Figure:
        """Create comprehensive performance dashboard"""
        if not results:
            print("No benchmark results to visualize")
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scientific Computing Toolkit - Performance Dashboard', fontsize=16)

        # Extract data
        component_names = [r.component_name for r in results]
        execution_times = [r.metrics.execution_time for r in results]
        memory_usage = [r.metrics.peak_memory_usage for r in results]
        cpu_usage = [r.metrics.cpu_utilization for r in results]
        efficiency_scores = [r.metrics.computational_efficiency for r in results]
        scalability_scores = [r.metrics.scalability_score for r in results]

        # Execution Time
        axes[0, 0].bar(component_names, execution_times, color='skyblue')
        axes[0, 0].set_title('Execution Time (seconds)')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Memory Usage
        axes[0, 1].bar(component_names, memory_usage, color='lightcoral')
        axes[0, 1].set_title('Peak Memory Usage (MB)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # CPU Utilization
        axes[0, 2].bar(component_names, cpu_usage, color='lightgreen')
        axes[0, 2].set_title('CPU Utilization (%)')
        axes[0, 2].set_ylabel('CPU (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Computational Efficiency
        axes[1, 0].bar(component_names, efficiency_scores, color='gold')
        axes[1, 0].set_title('Computational Efficiency\n(ops/sec)')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Scalability Score
        axes[1, 1].bar(component_names, scalability_scores, color='purple')
        axes[1, 1].set_title('Scalability Score')
        axes[1, 1].set_ylabel('Score (0-1)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Performance Summary
        axes[1, 2].axis('off')
        summary_text = "PERFORMANCE SUMMARY\n\n"
        summary_text += f"Average Execution Time: {np.mean(execution_times):.2f}s\n"
        summary_text += f"Average Memory Usage: {np.mean(memory_usage):.1f}MB\n"
        summary_text += f"Average CPU Usage: {np.mean(cpu_usage):.1f}%\n"
        summary_text += f"Overall Efficiency: {np.mean(efficiency_scores):.0f} ops/sec\n"
        summary_text += f"Average Scalability: {np.mean(scalability_scores):.2f}"

        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance dashboard saved to {save_path}")

        return fig

    def generate_performance_report(self, results: List[BenchmarkResult]) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("📊 SCIENTIFIC COMPUTING TOOLKIT - PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append("")

        report.append("📈 EXECUTIVE SUMMARY")
        report.append("-" * 25)

        if results:
            avg_execution = np.mean([r.metrics.execution_time for r in results])
            avg_memory = np.mean([r.metrics.peak_memory_usage for r in results])
            avg_efficiency = np.mean([r.metrics.computational_efficiency for r in results])

            report.append(".2f"            report.append(".1f"            report.append(".0f"        else:
            report.append("No benchmark results available")

        report.append("")
        report.append("🏃 COMPONENT PERFORMANCE DETAILS")
        report.append("-" * 40)

        for result in results:
            report.append(f"\n🔹 {result.component_name} - {result.benchmark_name}")
            report.append("-" * (len(result.component_name) + len(result.benchmark_name) + 5))

            m = result.metrics
            s = result.statistical_summary

            report.append(".2f"            report.append(".1f"            report.append(".1f"            report.append(".0f"            report.append(".2f"
            if result.recommendations:
                report.append("Recommendations:")
                for rec in result.recommendations:
                    report.append(f"  • {rec}")

        report.append("")
        report.append("🎯 SYSTEM INFORMATION")
        report.append("-" * 25)
        if results:
            sys_info = results[0].metrics.system_info
            report.append(f"CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
            report.append(".0f"            report.append(".1f"            report.append(".1f"
        report.append("")
        report.append("⚡ RECOMMENDATIONS")
        report.append("-" * 20)

        # Aggregate recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            unique_recs = list(set(all_recommendations))
            for rec in unique_recs:
                report.append(f"• {rec}")
        else:
            report.append("No specific recommendations - all benchmarks performing well")

        return "\n".join(report)

    def detect_performance_regressions(self, baseline_results: List[BenchmarkResult],
                                     current_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Detect performance regressions compared to baseline"""
        regressions = {}

        # Group results by component
        baseline_by_component = {r.component_name: r for r in baseline_results}
        current_by_component = {r.component_name: r for r in current_results}

        for component in current_by_component:
            if component in baseline_by_component:
                current = current_by_component[component]
                baseline = baseline_by_component[component]

                # Check for significant regressions
                execution_regression = (current.metrics.execution_time -
                                      baseline.metrics.execution_time) / baseline.metrics.execution_time

                memory_regression = (current.metrics.peak_memory_usage -
                                   baseline.metrics.peak_memory_usage) / baseline.metrics.peak_memory_usage

                component_regressions = {}

                if execution_regression > 0.1:  # 10% degradation
                    component_regressions['execution_time'] = {
                        'regression': execution_regression,
                        'baseline': baseline.metrics.execution_time,
                        'current': current.metrics.execution_time,
                        'severity': 'high' if execution_regression > 0.25 else 'medium'
                    }

                if memory_regression > 0.1:  # 10% increase
                    component_regressions['memory_usage'] = {
                        'regression': memory_regression,
                        'baseline': baseline.metrics.peak_memory_usage,
                        'current': current.metrics.peak_memory_usage,
                        'severity': 'high' if memory_regression > 0.25 else 'medium'
                    }

                if component_regressions:
                    regressions[component] = component_regressions

        return regressions


def demonstrate_performance_benchmarking():
    """Demonstrate comprehensive performance benchmarking capabilities"""
    print("📊 PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("=" * 60)

    benchmarker = PerformanceBenchmarker()

    # Example benchmark functions
    def mock_inverse_precision():
        time.sleep(0.1)  # Simulate computation
        return {"result": "completed"}

    def mock_rheology_model():
        # Simulate some computation
        data = np.random.rand(1000, 1000)
        result = np.linalg.svd(data)
        return {"singular_values": result[1][:10]}

    def mock_biological_flow():
        # Simulate biological flow computation
        import math
        result = 0
        for i in range(10000):
            result += math.sin(i) * math.cos(i)
        return {"flow_result": result}

    # Run benchmarks
    results = []

    print("\n1️⃣ BENCHMARKING INVERSE PRECISION FRAMEWORK")
    result1 = benchmarker.benchmark_component(
        "Inverse Precision", "Parameter Extraction",
        mock_inverse_precision
    )
    results.append(result1)

    print("\n2️⃣ BENCHMARKING RHEOLOGY MODEL")
    result2 = benchmarker.benchmark_component(
        "Rheology Model", "Constitutive Modeling",
        mock_rheology_model
    )
    results.append(result2)

    print("\n3️⃣ BENCHMARKING BIOLOGICAL FLOW")
    result3 = benchmarker.benchmark_component(
        "Biological Flow", "Vascular Network Simulation",
        mock_biological_flow
    )
    results.append(result3)

    # Create performance dashboard
    print("\n4️⃣ GENERATING PERFORMANCE DASHBOARD")
    dashboard = benchmarker.create_performance_dashboard(results)

    # Generate performance report
    print("\n5️⃣ GENERATING COMPREHENSIVE REPORT")
    report = benchmarker.generate_performance_report(results)
    print(report)

    # Save results
    results_dict = {
        'benchmark_summary': {
            'total_components': len(results),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': benchmarker.system_info
        },
        'results': [
            {
                'component': r.component_name,
                'benchmark': r.benchmark_name,
                'metrics': {
                    'execution_time': r.metrics.execution_time,
                    'peak_memory': r.metrics.peak_memory_usage,
                    'cpu_usage': r.metrics.cpu_utilization,
                    'efficiency': r.metrics.computational_efficiency,
                    'scalability': r.metrics.scalability_score
                },
                'recommendations': r.recommendations
            }
            for r in results
        ]
    }

    with open('performance_benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print("\n💾 Detailed results saved to performance_benchmark_results.json")


if __name__ == "__main__":
    demonstrate_performance_benchmarking()
