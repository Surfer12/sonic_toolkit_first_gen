"""
UOIF Toolkit Integration Module

This module provides seamless integration between the UOIF Ruleset framework
and the scientific computing toolkit's bootstrap analysis and uncertainty quantification.

Key Integrations:
- Bootstrap uncertainty quantification with UOIF scoring
- Confidence interval calculation using toolkit methods
- Performance monitoring integration with temporal decay
- Validation pipeline combining UOIF with toolkit optimization

Author: Scientific Computing Toolkit Team
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json

from .scoring_engine import ScoringEngine, ScoreComponents
from .allocation_system import AllocationSystem, AllocationResult


class BootstrapIntegrator:
    """
    Integrates UOIF scoring with toolkit's bootstrap analysis methods.

    This class provides comprehensive uncertainty quantification by combining
    UOIF's scoring components with bootstrap resampling techniques.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Initialize bootstrap integrator.

        Args:
            n_bootstrap: Number of bootstrap resamples
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.bootstrap_cache: Dict[str, Dict[str, Any]] = {}

    def bootstrap_score_uncertainty(self, claim_id: str, raw_components: Dict[str, float],
                                  claim_type: str, domain: str, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate uncertainty in UOIF scores using bootstrap resampling.

        Args:
            claim_id: Unique claim identifier
            raw_components: Raw component values with uncertainty
            claim_type: Type of claim
            domain: Research domain
            n_samples: Number of bootstrap samples (overrides default)

        Returns:
            Dict containing score statistics and confidence intervals
        """
        n_samples = n_samples or self.n_bootstrap

        # Generate bootstrap samples for each component
        component_samples = self._generate_component_bootstrap(raw_components, n_samples)

        # Calculate scores for each bootstrap sample
        bootstrap_scores = []
        for i in range(n_samples):
            sample_components = {comp: samples[i] for comp, samples in component_samples.items()}

            # Use scoring engine to calculate score for this sample
            scoring_engine = ScoringEngine()
            result = scoring_engine.score_claim(
                claim_id=f"{claim_id}_bootstrap_{i}",
                raw_components=sample_components,
                claim_type=claim_type,
                domain=domain
            )
            bootstrap_scores.append(result.total_score)

        # Calculate statistics
        bootstrap_scores = np.array(bootstrap_scores)
        score_stats = self._calculate_bootstrap_statistics(bootstrap_scores)

        # Cache results
        self.bootstrap_cache[claim_id] = {
            'score_stats': score_stats,
            'component_samples': component_samples,
            'n_samples': n_samples,
            'timestamp': datetime.now()
        }

        return score_stats

    def _generate_component_bootstrap(self, raw_components: Dict[str, float],
                                    n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate bootstrap samples for each scoring component.

        Args:
            raw_components: Raw component values
            n_samples: Number of bootstrap samples

        Returns:
            Dict of component bootstrap samples
        """
        component_samples = {}

        for component_name, raw_value in raw_components.items():
            if component_name == 'recurrence':
                # Citation counts - use Poisson distribution for uncertainty
                samples = np.random.poisson(lam=max(1, raw_value), size=n_samples)
                samples = np.clip(samples, 0, 10)  # Cap at reasonable maximum
            else:
                # Other components - use beta distribution for bounded uncertainty
                # Estimate parameters from value (assuming some uncertainty)
                uncertainty = 0.1  # Base uncertainty level
                alpha_param = raw_value * (1 - uncertainty) / uncertainty + 1
                beta_param = (1 - raw_value) * (1 - uncertainty) / uncertainty + 1

                samples = np.random.beta(alpha_param, beta_param, size=n_samples)
                samples = np.clip(samples, 0, 1)  # Ensure bounds

            component_samples[component_name] = samples

        return component_samples

    def _calculate_bootstrap_statistics(self, bootstrap_scores: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from bootstrap scores.

        Args:
            bootstrap_scores: Array of bootstrap score values

        Returns:
            Dict containing statistical measures
        """
        # Basic statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores, ddof=1)
        median_score = np.median(bootstrap_scores)

        # Confidence intervals
        ci_lower, ci_upper = stats.t.interval(
            self.confidence_level,
            len(bootstrap_scores) - 1,
            loc=mean_score,
            scale=stats.sem(bootstrap_scores)
        )

        # Percentiles for additional insights
        percentiles = np.percentile(bootstrap_scores, [5, 25, 75, 95])

        # Distribution characteristics
        skewness = stats.skew(bootstrap_scores)
        kurtosis = stats.kurtosis(bootstrap_scores)

        # Reliability assessment
        reliability_score = self._assess_reliability(bootstrap_scores)

        return {
            'mean': mean_score,
            'std': std_score,
            'median': median_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'percentiles': {
                '5th': percentiles[0],
                '25th': percentiles[1],
                '75th': percentiles[2],
                '95th': percentiles[3]
            },
            'distribution': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'range': bootstrap_scores.max() - bootstrap_scores.min()
            },
            'reliability': reliability_score,
            'confidence_level': self.confidence_level
        }

    def _assess_reliability(self, bootstrap_scores: np.ndarray) -> Dict[str, Any]:
        """
        Assess the reliability of bootstrap results.

        Args:
            bootstrap_scores: Bootstrap score samples

        Returns:
            Dict containing reliability metrics
        """
        # Coefficient of variation
        cv = np.std(bootstrap_scores, ddof=1) / np.mean(bootstrap_scores) if np.mean(bootstrap_scores) > 0 else float('inf')

        # Stability assessment (ratio of CI width to mean)
        ci_width = stats.t.interval(self.confidence_level, len(bootstrap_scores) - 1,
                                  loc=np.mean(bootstrap_scores), scale=stats.sem(bootstrap_scores))[1] - \
                  stats.t.interval(self.confidence_level, len(bootstrap_scores) - 1,
                                  loc=np.mean(bootstrap_scores), scale=stats.sem(bootstrap_scores))[0]

        stability_ratio = ci_width / np.mean(bootstrap_scores) if np.mean(bootstrap_scores) > 0 else float('inf')

        # Reliability classification
        if cv < 0.1 and stability_ratio < 0.2:
            reliability_class = 'high'
            reliability_score = 0.95
        elif cv < 0.2 and stability_ratio < 0.4:
            reliability_class = 'medium'
            reliability_score = 0.75
        elif cv < 0.3 and stability_ratio < 0.6:
            reliability_class = 'low'
            reliability_score = 0.50
        else:
            reliability_class = 'poor'
            reliability_score = 0.25

        return {
            'coefficient_of_variation': cv,
            'stability_ratio': stability_ratio,
            'reliability_class': reliability_class,
            'reliability_score': reliability_score
        }

    def get_bootstrap_cache(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached bootstrap results for a claim.

        Args:
            claim_id: Claim identifier

        Returns:
            Cached bootstrap results or None if not found
        """
        return self.bootstrap_cache.get(claim_id)

    def clear_cache(self, claim_id: Optional[str] = None):
        """
        Clear bootstrap cache.

        Args:
            claim_id: Specific claim ID to clear (None clears all)
        """
        if claim_id:
            self.bootstrap_cache.pop(claim_id, None)
        else:
            self.bootstrap_cache.clear()


class UncertaintyQuantificationIntegrator:
    """
    Integrates UOIF with comprehensive uncertainty quantification methods.

    This class combines multiple uncertainty quantification approaches:
    - Bootstrap analysis for empirical uncertainty
    - Analytical confidence intervals
    - Monte Carlo simulation for complex uncertainties
    - Bayesian credible intervals
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize uncertainty quantification integrator.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.bootstrap_integrator = BootstrapIntegrator(
            n_bootstrap=self.config['bootstrap_samples'],
            confidence_level=self.config['confidence_level']
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for uncertainty quantification."""
        return {
            'bootstrap_samples': 1000,
            'confidence_level': 0.95,
            'mc_samples': 5000,
            'methods': ['bootstrap', 'analytical', 'monte_carlo'],
            'performance_tracking': True
        }

    def comprehensive_uncertainty_analysis(self, claim_id: str,
                                        raw_components: Dict[str, float],
                                        claim_type: str, domain: str) -> Dict[str, Any]:
        """
        Perform comprehensive uncertainty analysis using multiple methods.

        Args:
            claim_id: Claim identifier
            raw_components: Raw component values
            claim_type: Type of claim
            domain: Research domain

        Returns:
            Dict containing results from all uncertainty methods
        """
        results = {
            'claim_id': claim_id,
            'timestamp': datetime.now(),
            'methods': {}
        }

        # Bootstrap analysis
        if 'bootstrap' in self.config['methods']:
            bootstrap_result = self.bootstrap_integrator.bootstrap_score_uncertainty(
                claim_id, raw_components, claim_type, domain
            )
            results['methods']['bootstrap'] = bootstrap_result

        # Analytical confidence intervals
        if 'analytical' in self.config['methods']:
            analytical_result = self._analytical_confidence_intervals(
                raw_components, claim_type, domain
            )
            results['methods']['analytical'] = analytical_result

        # Monte Carlo simulation
        if 'monte_carlo' in self.config['methods']:
            mc_result = self._monte_carlo_uncertainty(
                raw_components, claim_type, domain
            )
            results['methods']['monte_carlo'] = mc_result

        # Consensus analysis
        results['consensus'] = self._calculate_consensus(results['methods'])

        # Performance tracking
        if self.config['performance_tracking']:
            results['performance'] = self._track_performance_metrics(results)

        return results

    def _analytical_confidence_intervals(self, raw_components: Dict[str, float],
                                       claim_type: str, domain: str) -> Dict[str, Any]:
        """
        Calculate analytical confidence intervals for component uncertainties.

        Args:
            raw_components: Raw component values
            claim_type: Type of claim
            domain: Research domain

        Returns:
            Dict containing analytical confidence intervals
        """
        # Assume component uncertainties follow known distributions
        component_cis = {}

        for component_name, raw_value in raw_components.items():
            if component_name == 'recurrence':
                # Poisson confidence interval
                if raw_value > 0:
                    ci_lower = stats.poisson.ppf(0.025, raw_value)
                    ci_upper = stats.poisson.ppf(0.975, raw_value)
                else:
                    ci_lower, ci_upper = 0, 0
            else:
                # Beta distribution confidence interval
                uncertainty = 0.1  # Assumed uncertainty level
                alpha_param = raw_value * (1 - uncertainty) / uncertainty + 1
                beta_param = (1 - raw_value) * (1 - uncertainty) / uncertainty + 1

                ci_lower = stats.beta.ppf(0.025, alpha_param, beta_param)
                ci_upper = stats.beta.ppf(0.975, alpha_param, beta_param)

            component_cis[component_name] = {
                'mean': raw_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            }

        return {
            'component_confidence_intervals': component_cis,
            'method': 'analytical',
            'confidence_level': self.config['confidence_level']
        }

    def _monte_carlo_uncertainty(self, raw_components: Dict[str, float],
                               claim_type: str, domain: str) -> Dict[str, Any]:
        """
        Perform Monte Carlo simulation for uncertainty analysis.

        Args:
            raw_components: Raw component values
            claim_type: Type of claim
            domain: Research domain

        Returns:
            Dict containing Monte Carlo simulation results
        """
        n_mc = self.config['mc_samples']
        mc_scores = []

        # Generate Monte Carlo samples
        for _ in range(n_mc):
            # Sample each component with uncertainty
            mc_components = {}
            for component_name, raw_value in raw_components.items():
                if component_name == 'recurrence':
                    # Poisson sampling
                    mc_components[component_name] = np.random.poisson(max(1, raw_value))
                else:
                    # Beta distribution sampling with uncertainty
                    uncertainty = 0.1
                    alpha_param = raw_value * (1 - uncertainty) / uncertainty + 1
                    beta_param = (1 - raw_value) * (1 - uncertainty) / uncertainty + 1
                    mc_components[component_name] = np.random.beta(alpha_param, beta_param)

            # Calculate score for this Monte Carlo sample
            scoring_engine = ScoringEngine()
            result = scoring_engine.score_claim(
                claim_id=f"mc_sample_{_}",
                raw_components=mc_components,
                claim_type=claim_type,
                domain=domain
            )
            mc_scores.append(result.total_score)

        # Calculate Monte Carlo statistics
        mc_scores = np.array(mc_scores)
        mc_stats = self.bootstrap_integrator._calculate_bootstrap_statistics(mc_scores)
        mc_stats['method'] = 'monte_carlo'
        mc_stats['n_samples'] = n_mc

        return mc_stats

    def _calculate_consensus(self, method_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate consensus across multiple uncertainty methods.

        Args:
            method_results: Results from different uncertainty methods

        Returns:
            Dict containing consensus analysis
        """
        if not method_results:
            return {'consensus_score': 0.0, 'agreement_level': 'none'}

        # Extract mean scores from each method
        method_means = {}
        for method_name, result in method_results.items():
            if 'mean' in result:
                method_means[method_name] = result['mean']

        if not method_means:
            return {'consensus_score': 0.0, 'agreement_level': 'none'}

        # Calculate consensus statistics
        means = list(method_means.values())
        consensus_mean = np.mean(means)
        consensus_std = np.std(means, ddof=1) if len(means) > 1 else 0

        # Agreement assessment
        if len(means) == 1:
            agreement_level = 'single_method'
            agreement_score = 1.0
        else:
            # Coefficient of variation as agreement measure
            cv = consensus_std / consensus_mean if consensus_mean > 0 else float('inf')
            if cv < 0.05:
                agreement_level = 'excellent'
                agreement_score = 0.95
            elif cv < 0.10:
                agreement_level = 'good'
                agreement_score = 0.85
            elif cv < 0.20:
                agreement_level = 'fair'
                agreement_score = 0.70
            else:
                agreement_level = 'poor'
                agreement_score = 0.50

        return {
            'consensus_mean': consensus_mean,
            'consensus_std': consensus_std,
            'method_means': method_means,
            'agreement_level': agreement_level,
            'agreement_score': agreement_score,
            'coefficient_of_variation': cv if 'cv' in locals() else 0
        }

    def _track_performance_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track performance metrics for uncertainty analysis.

        Args:
            analysis_results: Complete analysis results

        Returns:
            Dict containing performance metrics
        """
        start_time = analysis_results.get('timestamp', datetime.now())
        current_time = datetime.now()
        execution_time = (current_time - start_time).total_seconds()

        # Estimate computational complexity
        total_samples = 0
        for method_result in analysis_results.get('methods', {}).values():
            total_samples += method_result.get('n_samples', 0)

        # Memory usage estimate (rough approximation)
        memory_estimate = total_samples * 8 * 6 / (1024 * 1024)  # 6 components, 8 bytes each

        return {
            'execution_time_seconds': execution_time,
            'total_samples': total_samples,
            'estimated_memory_mb': memory_estimate,
            'methods_executed': list(analysis_results.get('methods', {}).keys()),
            'consensus_achieved': analysis_results.get('consensus', {}).get('agreement_score', 0) > 0.7
        }


class UOIFTemporalIntegrator:
    """
    Integrates UOIF framework with temporal dynamics and performance monitoring.

    This class handles time-dependent reliability, performance regression detection,
    and temporal decay factors for claim validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temporal integrator.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_temporal_config()
        self.temporal_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _default_temporal_config(self) -> Dict[str, Any]:
        """Default configuration for temporal integration."""
        return {
            'decay_half_life_days': 30.0,
            'performance_window_days': 7.0,
            'regression_threshold': 0.95,
            'temporal_boost_max': 1.2,
            'temporal_penalty_min': 0.8,
            'update_frequency_hours': 24.0
        }

    def apply_temporal_decay(self, claim_id: str, base_score: float,
                           last_update: datetime) -> float:
        """
        Apply temporal decay to claim scores based on age.

        Args:
            claim_id: Claim identifier
            base_score: Original score
            last_update: Timestamp of last update

        Returns:
            Temporally adjusted score
        """
        time_since_update = datetime.now() - last_update
        days_since_update = time_since_update.total_seconds() / (24 * 3600)

        # Exponential decay with half-life
        half_life = self.config['decay_half_life_days']
        decay_rate = np.log(2) / half_life
        decay_factor = np.exp(-decay_rate * days_since_update)

        # Apply decay to score
        decayed_score = base_score * decay_factor

        # Cache temporal data
        if claim_id not in self.temporal_cache:
            self.temporal_cache[claim_id] = []

        self.temporal_cache[claim_id].append({
            'timestamp': datetime.now(),
            'days_since_update': days_since_update,
            'original_score': base_score,
            'decay_factor': decay_factor,
            'decayed_score': decayed_score
        })

        return decayed_score

    def detect_performance_regression(self, claim_id: str, current_score: float) -> Dict[str, Any]:
        """
        Detect performance regression in claim scores over time.

        Args:
            claim_id: Claim identifier
            current_score: Current score value

        Returns:
            Dict containing regression analysis
        """
        if claim_id not in self.temporal_cache:
            return {
                'regression_detected': False,
                'reason': 'insufficient_historical_data'
            }

        historical_scores = [entry['decayed_score'] for entry in self.temporal_cache[claim_id][-10:]]  # Last 10 entries

        if len(historical_scores) < 3:
            return {
                'regression_detected': False,
                'reason': 'insufficient_data_points'
            }

        # Calculate trend
        x = np.arange(len(historical_scores))
        y = np.array(historical_scores)

        # Linear regression to detect trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Regression detection
        recent_avg = np.mean(historical_scores[-3:])  # Last 3 scores
        overall_avg = np.mean(historical_scores)

        regression_ratio = recent_avg / overall_avg if overall_avg > 0 else 1.0
        regression_threshold = self.config['regression_threshold']

        regression_detected = regression_ratio < regression_threshold and slope < 0

        return {
            'regression_detected': regression_detected,
            'regression_ratio': regression_ratio,
            'slope': slope,
            'recent_avg': recent_avg,
            'overall_avg': overall_avg,
            'threshold': regression_threshold,
            'trend_strength': abs(slope),
            'confidence': r_value**2  # R-squared as confidence measure
        }

    def calculate_temporal_boost(self, claim_id: str, recent_performance: Dict[str, Any]) -> float:
        """
        Calculate temporal boost factor based on recent performance.

        Args:
            claim_id: Claim identifier
            recent_performance: Recent performance metrics

        Returns:
            Temporal boost factor (0.8 to 1.2)
        """
        boost_factor = 1.0

        # Performance-based boost
        if recent_performance.get('correlation', 0.9) >= 0.9987:  # Toolkit precision threshold
            boost_factor *= 1.1

        # Bootstrap sample boost
        if recent_performance.get('bootstrap_samples', 1000) >= 1000:
            boost_factor *= 1.05

        # Regression penalty
        if self.detect_performance_regression(claim_id, recent_performance.get('score', 0.8))['regression_detected']:
            boost_factor *= self.config['temporal_penalty_min']

        # Clamp to configured range
        boost_factor = max(self.config['temporal_penalty_min'],
                          min(self.config['temporal_boost_max'], boost_factor))

        return boost_factor

    def get_temporal_history(self, claim_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve temporal history for a claim.

        Args:
            claim_id: Claim identifier

        Returns:
            List of temporal history entries
        """
        return self.temporal_cache.get(claim_id, [])

    def clear_temporal_cache(self, claim_id: Optional[str] = None):
        """
        Clear temporal cache.

        Args:
            claim_id: Specific claim ID to clear (None clears all)
        """
        if claim_id:
            self.temporal_cache.pop(claim_id, None)
        else:
            self.temporal_cache.clear()
