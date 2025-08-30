#!/usr/bin/env python3
"""
Performance Benchmarking for Scientific Computing Toolkit

This script provides comprehensive performance benchmarking across all
integrated components, measuring latency, throughput, memory usage,
and CPU utilization.

Author: Scientific Computing Toolkit Team
Date: 2025
License: GPL-3.0-only
"""

import time
import psutil
import statistics
import numpy as np
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking utility."""

    def __init__(self):
        self.process = psutil.Process()
        self.results = []

    def benchmark_function(self, func, *args, iterations=100, name="test", **kwargs):
        """Benchmark a function and return performance metrics."""

        logger.info(f"Benchmarking {name} ({iterations} iterations)...")

        # Warmup
        for _ in range(10):
            func(*args, **kwargs)

        execution_times = []
        memory_usages = []
        cpu_usages = []

        for _ in range(iterations):
            mem_before = self.process.memory_info().rss / 1024 / 1024
            cpu_before = self.process.cpu_percent()

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            mem_after = self.process.memory_info().rss / 1024 / 1024
            cpu_after = self.process.cpu_percent()

            execution_times.append(end_time - start_time)
            memory_usages.append(mem_after - mem_before)
            cpu_usages.append(cpu_after)

        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        throughput = iterations / sum(execution_times)

        result_data = {
            'name': name,
            'iterations': iterations,
            'avg_latency_ms': avg_time * 1000,
            'throughput_ops_per_sec': throughput,
            'memory_delta_mb': statistics.mean(memory_usages),
            'cpu_usage_percent': statistics.mean(cpu_usages),
            'timestamp': datetime.now(timezone.utc).isoformat() + "Z"
        }

        self.results.append(result_data)
        return result_data


def benchmark_hybrid_uq():
    """Benchmark Hybrid UQ components."""
    print("\n🔬 Benchmarking Hybrid UQ Framework...")

    benchmark = PerformanceBenchmark()

    # Mock prediction function
    def mock_prediction():
        time.sleep(0.001)  # Simulate 1ms prediction
        return np.random.randn(10, 2)

    result = benchmark.benchmark_function(
        mock_prediction, iterations=100, name="hybrid_uq_prediction"
    )

    print(".2f")
    print(".1f")
    return result


def benchmark_communication():
    """Benchmark cross-framework communication."""
    print("\n📡 Benchmarking Cross-Framework Communication...")

    benchmark = PerformanceBenchmark()

    # Mock HTTP communication
    def mock_http_request():
        time.sleep(0.0005)  # Simulate 0.5ms HTTP request
        return {"status": "success"}

    result = benchmark.benchmark_function(
        mock_http_request, iterations=200, name="http_communication"
    )

    print(".2f")
    print(".1f")
    return result


def benchmark_data_processing():
    """Benchmark data processing pipeline."""
    print("\n🔄 Benchmarking Data Processing...")

    benchmark = PerformanceBenchmark()

    # Mock data processing
    def mock_data_processing():
        data = np.random.randn(1000, 10)
        result = np.fft.fft(data, axis=0)
        return np.mean(result, axis=0)

    result = benchmark.benchmark_function(
        mock_data_processing, iterations=50, name="data_processing"
    )

    print(".2f")
    print(".1f")
    return result


def benchmark_security():
    """Benchmark security framework."""
    print("\n🔒 Benchmarking Security Framework...")

    benchmark = PerformanceBenchmark()

    # Mock input validation
    def mock_validation():
        data = "test_input_" + str(np.random.randint(1000))
        time.sleep(0.0001)  # Simulate validation
        return len(data) > 0

    result = benchmark.benchmark_function(
        mock_validation, iterations=1000, name="input_validation"
    )

    print(".2f")
    print(".1f")
    return result


def run_all_benchmarks():
    """Run comprehensive performance benchmarking."""
    print("Scientific Computing Toolkit - Performance Benchmarking")
    print("="*60)

    results = []

    # Run component benchmarks
    results.append(benchmark_hybrid_uq())
    results.append(benchmark_communication())
    results.append(benchmark_data_processing())
    results.append(benchmark_security())

    # Generate summary report
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    total_tests = len(results)
    avg_latency = sum(r['avg_latency_ms'] for r in results) / total_tests
    total_throughput = sum(r['throughput_ops_per_sec'] for r in results)

    print(f"Total Benchmarks: {total_tests}")
    print(".2f")
    print(".1f")

    print("\nComponent Details:")
    for result in results:
        print("15"
              "6.1f"
              "6.1f"
              "6.1f")

    # Save results
    output_file = f"benchmark_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'benchmark_run': {
                'timestamp': datetime.now(timezone.utc).isoformat() + "Z",
                'total_benchmarks': total_tests,
                'summary': {
                    'avg_latency_ms': avg_latency,
                    'total_throughput_ops_per_sec': total_throughput
                }
            },
            'results': results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)

    return results


def run_quick_validation():
    """Quick performance validation for CI/CD."""
    print("Quick Performance Validation")
    print("="*30)

    benchmark = PerformanceBenchmark()

    def simple_test():
        data = np.random.randn(100, 100)
        result = np.linalg.inv(data @ data.T + np.eye(100))
        return result

    result = benchmark.benchmark_function(
        simple_test, iterations=10, name="quick_validation"
    )

    success = result['avg_latency_ms'] < 1000 and result['memory_delta_mb'] < 500

    if success:
        print("✅ Performance validation passed")
    else:
        print("⚠️ Performance validation failed")

    return success


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        results = run_all_benchmarks()
        print(f"\nBenchmarking completed successfully with {len(results)} test results")
