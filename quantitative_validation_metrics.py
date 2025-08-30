#!/usr/bin/env python3
"""
📈 QUANTITATIVE VALIDATION METRICS FRAMEWORK
==============================================

Advanced Quantitative Validation Framework for Scientific Computing Toolkit

This module provides comprehensive quantitative validation metrics, statistical
analysis, confidence intervals, and validation frameworks for all scientific
models and algorithms.

Features:
- Statistical validation metrics (MSE, RMSE, MAE, R², etc.)
- Confidence interval calculations
- Cross-validation frameworks
- Bootstrap validation methods
- Statistical significance testing
- Performance benchmarking metrics
- Validation report generation

Author: Scientific Computing Toolkit
License: GPL-3.0-only
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, normaltest


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics container"""

    # Basic error metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0

    # Correlation and fit metrics
    r_squared: float = 0.0
    pearson_r: float = 0.0
    spearman_r: float = 0.0

    # Statistical metrics
    mean_error: float = 0.0
    std_error: float = 0.0
    median_error: float = 0.0
    error_skewness: float = 0.0
    error_kurtosis: float = 0.0

    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Validation status
    validation_status: str = "UNKNOWN"  # "PASS", "FAIL", "WARNING"
    significance_level: float = 0.05

    # Additional metadata
    sample_size: int = 0
    degrees_of_freedom: int = 0
    computation_time: float = 0.0


@dataclass
class ValidationResult:
    """Complete validation result with statistical analysis"""

    model_name: str
    dataset_name: str
    metrics: ValidationMetrics
    statistical_tests: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    bootstrap_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = ""


class QuantitativeValidator:
    """
    Advanced quantitative validation framework

    Provides comprehensive statistical validation for scientific computing models
    with confidence intervals, cross-validation, and bootstrap analysis.
    """

    def __init__(self, confidence_level: float = 0.95, random_state: int = 42):
        """
        Initialize quantitative validator.

        Parameters
        ----------
        confidence_level : float
            Confidence level for statistical tests (default: 0.95)
        random_state : int
            Random state for reproducible results
        """
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.validation_history: List[ValidationResult] = []

        # Set random state for reproducibility
        np.random.seed(random_state)

    def comprehensive_validation(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Unknown Model",
                               dataset_name: str = "Unknown Dataset") -> ValidationResult:
        """
        Perform comprehensive quantitative validation of model predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Model predictions
        model_name : str
            Name of the model being validated
        dataset_name : str
            Name of the dataset used for validation

        Returns
        -------
        ValidationResult
            Complete validation results with statistical analysis
        """
        print(f"🔬 Performing comprehensive validation for {model_name} on {dataset_name}")

        start_time = datetime.now()

        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(y_true, y_pred)

        # Calculate statistical properties
        self._calculate_statistical_properties(metrics, y_true, y_pred)

        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(y_true, y_pred)

        # Cross-validation analysis
        cross_validation_results = self._perform_cross_validation_analysis(y_true, y_pred)

        # Bootstrap analysis
        bootstrap_analysis = self._perform_bootstrap_analysis(y_true, y_pred)

        # Generate recommendations
        recommendations = self._generate_validation_recommendations(metrics, statistical_tests)

        # Determine validation status
        metrics.validation_status = self._determine_validation_status(metrics, statistical_tests)

        # Set metadata
        metrics.sample_size = len(y_true)
        metrics.degrees_of_freedom = len(y_true) - 1
        metrics.computation_time = (datetime.now() - start_time).total_seconds()

        # Create validation result
        result = ValidationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            statistical_tests=statistical_tests,
            cross_validation_results=cross_validation_results,
            bootstrap_analysis=bootstrap_analysis,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

        # Store in history
        self.validation_history.append(result)

        print(".4f"        print(f"Status: {metrics.validation_status}")

        return result

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ValidationMetrics:
        """Calculate basic validation metrics"""
        # Error calculations
        errors = y_true - y_pred

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error) with handling for zero values
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
        else:
            mape = float('inf')

        # Correlation metrics
        r_squared = r2_score(y_true, y_pred)
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_r, _ = stats.spearmanr(y_true, y_pred)

        # Statistical properties of errors
        mean_error = np.mean(errors)
        std_error = np.std(errors, ddof=1)
        median_error = np.median(errors)
        error_skewness = stats.skew(errors)
        error_kurtosis = stats.kurtosis(errors)

        # Create metrics object
        metrics = ValidationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r_squared=r_squared,
            pearson_r=pearson_r,
            spearman_r=spearman_r,
            mean_error=mean_error,
            std_error=std_error,
            median_error=median_error,
            error_skewness=error_skewness,
            error_kurtosis=error_kurtosis
        )

        # Calculate confidence intervals
        metrics.confidence_intervals = self._calculate_confidence_intervals(
            errors, metrics, y_true, y_pred
        )

        return metrics

    def _calculate_statistical_properties(self, metrics: ValidationMetrics,
                                       y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate additional statistical properties"""
        errors = y_true - y_pred

        # Add confidence intervals for key metrics
        metrics.confidence_intervals.update({
            'rmse_ci': self._bootstrap_confidence_interval(
                lambda x, y: np.sqrt(mean_squared_error(x, y)), y_true, y_pred
            ),
            'mae_ci': self._bootstrap_confidence_interval(
                lambda x, y: mean_absolute_error(x, y), y_true, y_pred
            ),
            'r_squared_ci': self._bootstrap_confidence_interval(
                r2_score, y_true, y_pred
            )
        })

    def _calculate_confidence_intervals(self, errors: np.ndarray,
                                      metrics: ValidationMetrics,
                                      y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various metrics"""
        confidence_intervals = {}

        # Mean error confidence interval
        if len(errors) > 1:
            se_mean = stats.sem(errors)
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(errors) - 1)
            margin_error = t_value * se_mean
            confidence_intervals['mean_error'] = (
                metrics.mean_error - margin_error,
                metrics.mean_error + margin_error
            )

        # RMSE confidence interval using bootstrap
        confidence_intervals['rmse_bootstrap'] = self._bootstrap_confidence_interval(
            lambda x, y: np.sqrt(mean_squared_error(x, y)), y_true, y_pred
        )

        return confidence_intervals

    def _bootstrap_confidence_interval(self, metric_func: Callable,
                                     y_true: np.ndarray, y_pred: np.ndarray,
                                     n_boot: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a metric"""
        boot_metrics = []

        for _ in range(n_boot):
            # Bootstrap resampling
            indices = resample(np.arange(len(y_true)), n_samples=len(y_true),
                             random_state=self.random_state)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                metric_value = metric_func(y_true_boot, y_pred_boot)
                if np.isfinite(metric_value):
                    boot_metrics.append(metric_value)
            except:
                continue

        if not boot_metrics:
            return (0.0, 0.0)

        # Calculate confidence interval
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100

        return (
            np.percentile(boot_metrics, lower_percentile),
            np.percentile(boot_metrics, upper_percentile)
        )

    def _perform_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive statistical tests"""
        errors = y_true - y_pred

        statistical_tests = {}

        # Normality tests for errors
        try:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = shapiro(errors)
            statistical_tests['shapiro_normality'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        except:
            statistical_tests['shapiro_normality'] = {'error': 'Test failed'}

        # D'Agostino and Pearson's test
        try:
            dagostino_stat, dagostino_p = normaltest(errors)
            statistical_tests['dagostino_normality'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > 0.05
            }
        except:
            statistical_tests['dagostino_normality'] = {'error': 'Test failed'}

        # One-sample t-test (test if mean error is significantly different from 0)
        try:
            t_stat, t_p = stats.ttest_1samp(errors, 0)
            statistical_tests['one_sample_ttest'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'mean_significantly_different_from_zero': t_p < 0.05
            }
        except:
            statistical_tests['one_sample_ttest'] = {'error': 'Test failed'}

        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(errors)
            statistical_tests['wilcoxon_signed_rank'] = {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'median_significantly_different_from_zero': wilcoxon_p < 0.05
            }
        except:
            statistical_tests['wilcoxon_signed_rank'] = {'error': 'Test failed'}

        # Runs test for randomness
        try:
            statistical_tests['runs_test'] = self._runs_test(errors)
        except:
            statistical_tests['runs_test'] = {'error': 'Test failed'}

        return statistical_tests

    def _runs_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform runs test for randomness in residuals"""
        # Convert to binary sequence (positive/negative)
        binary_sequence = (data > np.median(data)).astype(int)

        # Count runs
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1

        # Expected runs and variance
        n1 = np.sum(binary_sequence)  # Number of positive values
        n2 = len(binary_sequence) - n1  # Number of negative values

        if n1 == 0 or n2 == 0:
            return {'error': 'Cannot perform runs test on constant sequence'}

        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))

        if variance_runs > 0:
            z_score = (runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

            return {
                'runs_observed': runs,
                'runs_expected': expected_runs,
                'z_score': z_score,
                'p_value': p_value,
                'is_random': p_value > 0.05
            }
        else:
            return {'error': 'Variance calculation failed'}

    def _perform_cross_validation_analysis(self, y_true: np.ndarray,
                                        y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation analysis"""
        # For cross-validation, we need to simulate having multiple folds
        # Since we only have predictions, we'll use bootstrap cross-validation

        n_folds = 5
        cv_scores = []

        for fold in range(n_folds):
            # Bootstrap sample
            indices = resample(np.arange(len(y_true)), n_samples=len(y_true),
                             random_state=self.random_state + fold)

            y_true_fold = y_true[indices]
            y_pred_fold = y_pred[indices]

            # Calculate R² for this fold
            try:
                fold_score = r2_score(y_true_fold, y_pred_fold)
                if np.isfinite(fold_score):
                    cv_scores.append(fold_score)
            except:
                continue

        if cv_scores:
            cv_results = {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_scores': cv_scores,
                'n_folds_successful': len(cv_scores),
                'n_folds_total': n_folds
            }
        else:
            cv_results = {'error': 'Cross-validation failed'}

        return cv_results

    def _perform_bootstrap_analysis(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform bootstrap analysis for robust statistics"""
        n_boot = 1000
        boot_statistics = {
            'r_squared': [],
            'rmse': [],
            'mae': [],
            'mean_error': []
        }

        for _ in range(n_boot):
            # Bootstrap resampling
            indices = resample(np.arange(len(y_true)), n_samples=len(y_true))

            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                # Calculate statistics for this bootstrap sample
                boot_statistics['r_squared'].append(r2_score(y_true_boot, y_pred_boot))
                boot_statistics['rmse'].append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
                boot_statistics['mae'].append(mean_absolute_error(y_true_boot, y_pred_boot))

                errors_boot = y_true_boot - y_pred_boot
                boot_statistics['mean_error'].append(np.mean(errors_boot))
            except:
                continue

        # Calculate bootstrap confidence intervals
        bootstrap_results = {}
        for stat_name, values in boot_statistics.items():
            if values:
                bootstrap_results[f'{stat_name}_mean'] = np.mean(values)
                bootstrap_results[f'{stat_name}_std'] = np.std(values)
                bootstrap_results[f'{stat_name}_ci'] = (
                    np.percentile(values, 2.5),
                    np.percentile(values, 97.5)
                )
            else:
                bootstrap_results[stat_name] = {'error': 'No valid bootstrap samples'}

        return bootstrap_results

    def _generate_validation_recommendations(self, metrics: ValidationMetrics,
                                           statistical_tests: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations based on results"""
        recommendations = []

        # R-squared recommendations
        if metrics.r_squared < 0.5:
            recommendations.append("Low R² indicates poor model fit - consider model improvement")
        elif metrics.r_squared < 0.8:
            recommendations.append("Moderate R² - model shows reasonable fit but may be improved")

        # Error magnitude recommendations
        if metrics.rmse > np.std(metrics.confidence_intervals.get('mean_error', (0, 0))):
            recommendations.append("RMSE is high relative to data variability - investigate outliers")

        # Normality recommendations
        normality_tests = [statistical_tests.get('shapiro_normality', {}),
                          statistical_tests.get('dagostino_normality', {})]

        normal_count = sum(1 for test in normality_tests if test.get('is_normal', False))

        if normal_count == 0:
            recommendations.append("Residuals are not normally distributed - consider transformation")
        elif normal_count == 1:
            recommendations.append("Mixed normality results - verify with additional tests")

        # Mean error recommendations
        if abs(metrics.mean_error) > metrics.std_error:
            recommendations.append("Mean error is significantly different from zero - investigate bias")

        # Sample size recommendations
        if metrics.sample_size < 30:
            recommendations.append("Small sample size - results may not be reliable")
        elif metrics.sample_size < 100:
            recommendations.append("Moderate sample size - consider collecting more data")

        return recommendations

    def _determine_validation_status(self, metrics: ValidationMetrics,
                                   statistical_tests: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        score = 0
        max_score = 5

        # R-squared score
        if metrics.r_squared > 0.8:
            score += 1
        elif metrics.r_squared > 0.6:
            score += 0.5

        # RMSE relative to data variability
        data_std = metrics.std_error
        if metrics.rmse < data_std:
            score += 1
        elif metrics.rmse < 2 * data_std:
            score += 0.5

        # Normality of residuals
        normality_tests = [statistical_tests.get('shapiro_normality', {}),
                          statistical_tests.get('dagostino_normality', {})]
        normal_count = sum(1 for test in normality_tests if test.get('is_normal', False))

        if normal_count >= 1:
            score += 1

        # Mean error significance
        ttest_result = statistical_tests.get('one_sample_ttest', {})
        if not ttest_result.get('mean_significantly_different_from_zero', True):
            score += 1

        # Sample size adequacy
        if metrics.sample_size >= 30:
            score += 0.5
        if metrics.sample_size >= 100:
            score += 0.5

        # Determine status based on score
        if score >= 4:
            return "PASS"
        elif score >= 2.5:
            return "WARNING"
        else:
            return "FAIL"

    def generate_validation_report(self, result: ValidationResult,
                                 output_format: str = "text") -> str:
        """Generate comprehensive validation report"""
        if output_format == "text":
            return self._generate_text_report(result)
        elif output_format == "json":
            return json.dumps(self._generate_json_report(result), indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate text-based validation report"""
        report = []
        report.append("📊 QUANTITATIVE VALIDATION REPORT")
        report.append("=" * 50)
        report.append("")

        report.append(f"Model: {result.model_name}")
        report.append(f"Dataset: {result.dataset_name}")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Validation Status: {result.metrics.validation_status}")
        report.append("")

        # Basic Metrics
        report.append("📈 BASIC METRICS")
        report.append("-" * 20)
        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append("")

        # Statistical Properties
        report.append("📊 STATISTICAL PROPERTIES")
        report.append("-" * 30)
        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append(".4f"        report.append("")

        # Confidence Intervals
        report.append("🎯 CONFIDENCE INTERVALS (95%)")
        report.append("-" * 35)
        for metric_name, ci in result.metrics.confidence_intervals.items():
            if isinstance(ci, tuple) and len(ci) == 2:
                report.append("15"        report.append("")

        # Statistical Tests
        report.append("🧪 STATISTICAL TESTS")
        report.append("-" * 25)
        for test_name, test_result in result.statistical_tests.items():
            if 'error' not in test_result:
                report.append(f"{test_name.upper()}:")
                for key, value in test_result.items():
                    if isinstance(value, float):
                        report.append(".4f"                    else:
                        report.append(f"  {key}: {value}")
                report.append("")

        # Cross-Validation Results
        report.append("🔄 CROSS-VALIDATION RESULTS")
        report.append("-" * 32)
        cv_results = result.cross_validation_results
        if 'error' not in cv_results:
            report.append(".4f"            report.append(".4f"            report.append(f"Successful Folds: {cv_results.get('n_folds_successful', 0)}/{cv_results.get('n_folds_total', 0)}")
        else:
            report.append("Cross-validation failed")
        report.append("")

        # Bootstrap Analysis
        report.append("🔁 BOOTSTRAP ANALYSIS")
        report.append("-" * 25)
        boot_results = result.bootstrap_analysis
        for key, value in boot_results.items():
            if isinstance(value, (int, float)):
                report.append(".4f"            elif isinstance(value, tuple) and len(value) == 2:
                report.append("15"        report.append("")

        # Recommendations
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 20)
        for rec in result.recommendations:
            report.append(f"• {rec}")

        return "\n".join(report)

    def _generate_json_report(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate JSON-based validation report"""
        return {
            'model_name': result.model_name,
            'dataset_name': result.dataset_name,
            'timestamp': result.timestamp,
            'validation_status': result.metrics.validation_status,
            'metrics': {
                'basic': {
                    'mse': result.metrics.mse,
                    'rmse': result.metrics.rmse,
                    'mae': result.metrics.mae,
                    'mape': result.metrics.mape,
                    'r_squared': result.metrics.r_squared,
                    'pearson_r': result.metrics.pearson_r,
                    'spearman_r': result.metrics.spearman_r
                },
                'statistical': {
                    'mean_error': result.metrics.mean_error,
                    'std_error': result.metrics.std_error,
                    'median_error': result.metrics.median_error,
                    'error_skewness': result.metrics.error_skewness,
                    'error_kurtosis': result.metrics.error_kurtosis
                },
                'metadata': {
                    'sample_size': result.metrics.sample_size,
                    'degrees_of_freedom': result.metrics.degrees_of_freedom,
                    'computation_time': result.metrics.computation_time
                }
            },
            'confidence_intervals': result.metrics.confidence_intervals,
            'statistical_tests': result.statistical_tests,
            'cross_validation': result.cross_validation_results,
            'bootstrap_analysis': result.bootstrap_analysis,
            'recommendations': result.recommendations
        }

    def create_validation_dashboard(self, results: List[ValidationResult],
                                  save_path: str = "validation_dashboard.png") -> plt.Figure:
        """Create comprehensive validation dashboard"""
        if not results:
            print("No validation results to visualize")
            return None

        n_results = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scientific Computing Toolkit - Validation Dashboard', fontsize=16)

        # Extract data
        model_names = [r.model_name for r in results]
        r_squared_values = [r.metrics.r_squared for r in results]
        rmse_values = [r.metrics.rmse for r in results]
        mae_values = [r.metrics.mae for r in results]
        validation_statuses = [r.metrics.validation_status for r in results]

        # R² Scores
        colors_r2 = ['green' if x > 0.8 else 'orange' if x > 0.6 else 'red' for x in r_squared_values]
        axes[0, 0].bar(model_names, r_squared_values, color=colors_r2)
        axes[0, 0].set_title('R² Scores')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # RMSE Values
        axes[0, 1].bar(model_names, rmse_values, color='lightcoral')
        axes[0, 1].set_title('RMSE Values')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # MAE Values
        axes[0, 2].bar(model_names, mae_values, color='lightblue')
        axes[0, 2].set_title('MAE Values')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Validation Status
        status_counts = {'PASS': 0, 'WARNING': 0, 'FAIL': 0}
        for status in validation_statuses:
            status_counts[status] += 1

        axes[1, 0].bar(status_counts.keys(), status_counts.values(),
                      color=['green', 'orange', 'red'])
        axes[1, 0].set_title('Validation Status Distribution')
        axes[1, 0].set_ylabel('Count')

        # Error Distribution (first result)
        if results:
            first_result = results[0]
            # This would require storing the original errors - simplified for now
            axes[1, 1].text(0.5, 0.5, 'Error Distribution\nAnalysis Available',
                           transform=axes[1, 1].transAxes, ha='center', va='center')
            axes[1, 1].set_title('Error Distribution')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)

        # Performance Summary
        axes[1, 2].axis('off')
        summary_text = "VALIDATION SUMMARY\n\n"
        summary_text += f"Total Models: {len(results)}\n"
        summary_text += f"Passed: {status_counts['PASS']}\n"
        summary_text += f"Warnings: {status_counts['WARNING']}\n"
        summary_text += f"Failed: {status_counts['FAIL']}\n"
        summary_text += ".3f"        summary_text += ".3f"        summary_text += ".3f"
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Validation dashboard saved to {save_path}")

        return fig


def demonstrate_quantitative_validation():
    """Demonstrate quantitative validation capabilities"""
    print("📈 QUANTITATIVE VALIDATION DEMONSTRATION")
    print("=" * 60)

    validator = QuantitativeValidator()

    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100

    # Test Case 1: Good model performance
    print("\n1️⃣ VALIDATING HIGH-PERFORMANCE MODEL")
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 1, n_samples)
    y_pred_good = 2.1 * x + 0.9  # Slight bias and slope error

    result_good = validator.comprehensive_validation(
        y_true, y_pred_good, "Linear Regression", "Synthetic Dataset A"
    )

    # Test Case 2: Poor model performance
    print("\n2️⃣ VALIDATING LOW-PERFORMANCE MODEL")
    y_pred_poor = np.random.normal(np.mean(y_true), np.std(y_true), n_samples)

    result_poor = validator.comprehensive_validation(
        y_true, y_pred_poor, "Random Predictor", "Synthetic Dataset A"
    )

    # Test Case 3: Perfect model
    print("\n3️⃣ VALIDATING PERFECT MODEL")
    result_perfect = validator.comprehensive_validation(
        y_true, y_true, "Perfect Model", "Synthetic Dataset A"
    )

    # Generate validation reports
    print("\n4️⃣ GENERATING VALIDATION REPORTS")

    # Text report for good model
    text_report = validator.generate_validation_report(result_good, "text")
    print("\n📄 HIGH-PERFORMANCE MODEL REPORT (excerpt):")
    print("-" * 50)
    lines = text_report.split('\n')
    for line in lines[:20]:  # First 20 lines
        print(line)

    # JSON report
    json_report = validator.generate_validation_report(result_good, "json")
    with open('validation_report_good.json', 'w') as f:
        f.write(json_report)

    # Create validation dashboard
    print("\n5️⃣ CREATING VALIDATION DASHBOARD")
    dashboard = validator.create_validation_dashboard([result_good, result_poor, result_perfect])

    # Summary statistics
    print("\n📊 VALIDATION SUMMARY")
    print("-" * 30)
    for result in [result_good, result_poor, result_perfect]:
        print(f"{result.model_name}:")
        print(".4f"        print(f"  Status: {result.metrics.validation_status}")
        print(f"  Sample Size: {result.metrics.sample_size}")
        print()

    print("💾 Detailed results saved to validation_report_good.json")
    print("📊 Dashboard saved to validation_dashboard.png")


if __name__ == "__main__":
    demonstrate_quantitative_validation()
