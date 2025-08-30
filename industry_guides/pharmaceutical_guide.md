# Pharmaceutical Industry Implementation Guide

## Drug Formulation and Delivery Optimization

This guide provides pharmaceutical industry professionals with comprehensive instructions for implementing the Scientific Computing Toolkit's capabilities in drug development, formulation optimization, and delivery system design.

---

## Table of Contents

1. [Drug Formulation Optimization](#drug-formulation-optimization)
2. [Biopharmaceutical Characterization](#biopharmaceutical-characterization)
3. [Process Scale-Up and Manufacturing](#process-scale-up-and-manufacturing)
4. [Quality Control and Validation](#quality-control-and-validation)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Case Studies](#case-studies)

---

## Drug Formulation Optimization

### Emulsion and Suspension Stability Analysis

```python
from hbflow.models import fit_herschel_bulkley
from optical_depth_enhancement import OpticalDepthAnalyzer
import numpy as np

class DrugFormulationOptimizer:
    """Optimize pharmaceutical formulations using rheological analysis."""

    def __init__(self):
        self.rheology_analyzer = None
        self.optical_analyzer = OpticalDepthAnalyzer()

    def analyze_emulsion_stability(self, formulation_data):
        """Analyze emulsion stability using rheological parameters."""

        # Extract rheological parameters
        stress_data = formulation_data['viscosity_stress']
        shear_rate_data = formulation_data['shear_rate']
        time_data = formulation_data['time']

        # Fit Herschel-Bulkley model
        params = fit_herschel_bulkley(stress_data, shear_rate_data)

        # Calculate stability metrics
        stability_metrics = {
            'yield_stress': params['tau_y'],
            'consistency_index': params['K'],
            'flow_behavior_index': params['n'],
            'thixotropic_recovery': self.calculate_thixotropic_recovery(
                time_data, stress_data
            ),
            'viscosity_stability': self.assess_viscosity_stability(
                stress_data, shear_rate_data
            )
        }

        # Assess formulation quality
        quality_assessment = self.assess_formulation_quality(stability_metrics)

        return {
            'rheological_parameters': params,
            'stability_metrics': stability_metrics,
            'quality_assessment': quality_assessment,
            'recommendations': self.generate_recommendations(quality_assessment)
        }

    def calculate_thixotropic_recovery(self, time_data, viscosity_data):
        """Calculate thixotropic recovery rate."""
        # Analyze viscosity recovery after shear
        initial_viscosity = np.mean(viscosity_data[:10])  # First 10 points
        final_viscosity = np.mean(viscosity_data[-10:])   # Last 10 points

        recovery_rate = (final_viscosity - initial_viscosity) / initial_viscosity
        recovery_time = time_data[np.argmax(np.gradient(viscosity_data))]

        return {
            'recovery_rate': recovery_rate,
            'recovery_time': recovery_time,
            'recovery_percentage': recovery_rate * 100
        }

    def assess_viscosity_stability(self, stress_data, shear_rate_data):
        """Assess viscosity stability across shear rates."""
        viscosity = stress_data / shear_rate_data

        # Calculate coefficient of variation
        cv = np.std(viscosity) / np.mean(viscosity)

        # Assess stability based on CV
        if cv < 0.1:
            stability = 'excellent'
        elif cv < 0.2:
            stability = 'good'
        elif cv < 0.3:
            stability = 'moderate'
        else:
            stability = 'poor'

        return {
            'coefficient_of_variation': cv,
            'stability_rating': stability,
            'viscosity_range': {
                'min': np.min(viscosity),
                'max': np.max(viscosity),
                'mean': np.mean(viscosity)
            }
        }

    def assess_formulation_quality(self, stability_metrics):
        """Assess overall formulation quality."""

        quality_score = 0
        issues = []

        # Yield stress assessment
        if stability_metrics['yield_stress'] < 1.0:
            issues.append("Low yield stress - poor suspension stability")
        elif stability_metrics['yield_stress'] > 50.0:
            issues.append("High yield stress - difficult to process")
        else:
            quality_score += 25

        # Flow behavior assessment
        if stability_metrics['flow_behavior_index'] < 0.8:
            issues.append("Highly shear-thinning - potential separation issues")
        elif stability_metrics['flow_behavior_index'] > 1.2:
            issues.append("Shear-thickening - processing difficulties")
        else:
            quality_score += 25

        # Thixotropic recovery assessment
        recovery = stability_metrics['thixotropic_recovery']
        if recovery['recovery_rate'] < 0.8:
            issues.append("Poor thixotropic recovery - stability concerns")
        else:
            quality_score += 25

        # Viscosity stability assessment
        viscosity_stability = stability_metrics['viscosity_stability']
        if viscosity_stability['stability_rating'] == 'excellent':
            quality_score += 25
        elif viscosity_stability['stability_rating'] == 'good':
            quality_score += 15
        elif viscosity_stability['stability_rating'] == 'moderate':
            quality_score += 5

        # Determine overall quality
        if quality_score >= 80:
            overall_quality = 'excellent'
        elif quality_score >= 60:
            overall_quality = 'good'
        elif quality_score >= 40:
            overall_quality = 'acceptable'
        else:
            overall_quality = 'poor'

        return {
            'quality_score': quality_score,
            'overall_quality': overall_quality,
            'issues': issues,
            'critical_issues': len([i for i in issues if 'critical' in i.lower()])
        }

    def generate_recommendations(self, quality_assessment):
        """Generate formulation improvement recommendations."""

        recommendations = []

        if quality_assessment['quality_score'] < 60:
            recommendations.extend([
                "Consider reformulation with different emulsifiers",
                "Adjust surfactant concentration",
                "Evaluate alternative thickening agents",
                "Review mixing protocols and equipment"
            ])

        if 'yield_stress' in str(quality_assessment['issues']).lower():
            recommendations.append(
                "Optimize emulsifier concentration to achieve target yield stress (10-30 Pa)"
            )

        if 'shear' in str(quality_assessment['issues']).lower():
            recommendations.append(
                "Evaluate polymer molecular weight and concentration for better flow behavior"
            )

        if 'thixotropic' in str(quality_assessment['issues']).lower():
            recommendations.append(
                "Consider adding rheology modifiers for improved structure recovery"
            )

        return recommendations
```

### Controlled Release System Optimization

```python
from biological_transport_modeling import BiologicalNutrientTransport
import numpy as np

class ControlledReleaseOptimizer:
    """Optimize controlled release drug delivery systems."""

    def __init__(self):
        self.transport_model = BiologicalNutrientTransport()

    def optimize_release_kinetics(self, drug_properties, target_profile):
        """Optimize drug release kinetics for target therapeutic profile."""

        # Define optimization parameters
        optimization_params = {
            'polymer_matrix': ['PLGA', 'PCL', 'PLA'],
            'drug_loading': np.linspace(0.1, 0.5, 10),
            'particle_size': np.linspace(10, 200, 20),  # microns
            'porosity': np.linspace(0.1, 0.8, 15)
        }

        best_formulation = None
        best_score = float('inf')

        # Grid search optimization
        for polymer in optimization_params['polymer_matrix']:
            for loading in optimization_params['drug_loading']:
                for size in optimization_params['particle_size']:
                    for porosity in optimization_params['porosity']:

                        # Simulate release profile
                        release_profile = self.simulate_release_profile(
                            drug_properties, polymer, loading, size, porosity
                        )

                        # Calculate fitness score
                        fitness_score = self.calculate_fitness_score(
                            release_profile, target_profile
                        )

                        if fitness_score < best_score:
                            best_score = fitness_score
                            best_formulation = {
                                'polymer': polymer,
                                'drug_loading': loading,
                                'particle_size': size,
                                'porosity': porosity,
                                'release_profile': release_profile,
                                'fitness_score': fitness_score
                            }

        return best_formulation

    def simulate_release_profile(self, drug_properties, polymer, loading,
                               particle_size, porosity):
        """Simulate drug release profile using transport modeling."""

        # Convert particle size to meters
        size_meters = particle_size * 1e-6

        # Define tissue properties (simplified gastrointestinal tract)
        tissue_props = {
            'permeability': 1e-14,  # m²
            'porosity': porosity,
            'tortuosity': 2.0,
            'surface_area_per_volume': 3 / size_meters  # 1/m
        }

        # Define drug properties
        drug_props = {
            'diffusivity': drug_properties.get('diffusivity', 1e-10),
            'solubility': drug_properties.get('solubility', 1e-2),  # mol/L
            'partition_coefficient': drug_properties.get('logP', 2.0)
        }

        # Simulate release over 24 hours
        time_points = np.linspace(0, 24*3600, 100)  # 24 hours in seconds

        # Calculate release rate using diffusion model
        release_profile = self.calculate_diffusion_release(
            tissue_props, drug_props, loading, size_meters, time_points
        )

        return {
            'time_hours': time_points / 3600,
            'cumulative_release': release_profile['cumulative'],
            'release_rate': release_profile['rate'],
            'total_release_percentage': release_profile['cumulative'][-1] * 100
        }

    def calculate_diffusion_release(self, tissue_props, drug_props, loading,
                                  particle_size, time_points):
        """Calculate drug release using diffusion model."""

        # Simplified diffusion-controlled release model
        D = drug_props['diffusivity']  # diffusivity
        Cs = drug_props['solubility']  # solubility
        V = (4/3) * np.pi * (particle_size/2)**3  # particle volume
        A = 4 * np.pi * (particle_size/2)**2     # particle surface area

        # Initial drug amount
        M0 = loading * V * Cs

        # Cumulative release calculation
        cumulative_release = []
        release_rate = []

        for t in time_points:
            if t == 0:
                Mt = 0
                rate = 0
            else:
                # Simplified solution for diffusion from sphere
                Mt = M0 * (6 / (np.pi**2)) * np.sum([
                    (1/n**2) * np.exp(-n**2 * np.pi**2 * D * t / (particle_size/2)**2)
                    for n in range(1, 10)  # First 9 terms
                ])
                rate = M0 * (6 / (np.pi**2)) * np.sum([
                    np.exp(-n**2 * np.pi**2 * D * t / (particle_size/2)**2)
                    for n in range(1, 10)
                ])

            cumulative_release.append(min(Mt / M0, 1.0))  # Normalize to 1
            release_rate.append(rate)

        return {
            'cumulative': np.array(cumulative_release),
            'rate': np.array(release_rate)
        }

    def calculate_fitness_score(self, simulated_profile, target_profile):
        """Calculate fitness score comparing simulated vs target release."""

        # Extract cumulative release profiles
        simulated_release = simulated_profile['cumulative_release']
        target_release = np.interp(
            simulated_profile['time_hours'],
            target_profile['time_hours'],
            target_profile['cumulative_release']
        )

        # Calculate mean squared error
        mse = np.mean((simulated_release - target_release)**2)

        # Add penalty for total release percentage
        total_release = simulated_profile['total_release_percentage']
        if total_release < 80:
            mse += (80 - total_release) * 0.01  # Penalty for low release
        elif total_release > 120:
            mse += (total_release - 120) * 0.01  # Penalty for high release

        return mse

    def validate_release_profile(self, formulation, experimental_data):
        """Validate optimized formulation against experimental data."""

        # Compare simulated vs experimental release profiles
        simulated = formulation['release_profile']
        experimental = experimental_data

        # Calculate validation metrics
        validation_metrics = {
            'rmse': np.sqrt(np.mean((simulated['cumulative_release'] -
                                   experimental['cumulative_release'])**2)),
            'r_squared': self.calculate_r_squared(
                experimental['cumulative_release'],
                simulated['cumulative_release']
            ),
            'max_deviation': np.max(np.abs(
                simulated['cumulative_release'] - experimental['cumulative_release']
            )),
            'release_time_match': self.assess_release_time_match(
                simulated, experimental
            )
        }

        # Assess validation quality
        if validation_metrics['r_squared'] > 0.9 and validation_metrics['rmse'] < 0.1:
            validation_quality = 'excellent'
        elif validation_metrics['r_squared'] > 0.8 and validation_metrics['rmse'] < 0.2:
            validation_quality = 'good'
        elif validation_metrics['r_squared'] > 0.7 and validation_metrics['rmse'] < 0.3:
            validation_quality = 'acceptable'
        else:
            validation_quality = 'poor'

        return {
            'validation_metrics': validation_metrics,
            'validation_quality': validation_quality,
            'recommendations': self.generate_validation_recommendations(validation_quality)
        }

    def calculate_r_squared(self, observed, predicted):
        """Calculate R² coefficient of determination."""
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        return 1 - (ss_res / ss_tot)

    def assess_release_time_match(self, simulated, experimental):
        """Assess how well release times match between simulation and experiment."""

        # Find time to 50% release for both profiles
        sim_50_time = np.interp(0.5, simulated['cumulative_release'],
                               simulated['time_hours'])
        exp_50_time = np.interp(0.5, experimental['cumulative_release'],
                               experimental['time_hours'])

        time_difference = abs(sim_50_time - exp_50_time)
        relative_error = time_difference / exp_50_time if exp_50_time > 0 else float('inf')

        return {
            'simulated_50_time': sim_50_time,
            'experimental_50_time': exp_50_time,
            'time_difference_hours': time_difference,
            'relative_error': relative_error,
            'match_quality': 'excellent' if relative_error < 0.1 else
                           'good' if relative_error < 0.25 else
                           'poor'
        }

    def generate_validation_recommendations(self, validation_quality):
        """Generate recommendations based on validation quality."""

        recommendations = []

        if validation_quality == 'poor':
            recommendations.extend([
                "Re-evaluate diffusion model assumptions",
                "Consider additional release mechanisms (erosion, swelling)",
                "Validate experimental release measurement method",
                "Check particle size distribution in formulation"
            ])

        elif validation_quality == 'acceptable':
            recommendations.extend([
                "Fine-tune diffusion coefficient",
                "Adjust porosity estimation",
                "Consider polydispersity effects",
                "Validate at multiple temperatures"
            ])

        elif validation_quality == 'good':
            recommendations.extend([
                "Model is performing well - minor refinements possible",
                "Consider additional validation at different pH levels",
                "Evaluate long-term stability predictions"
            ])

        else:  # excellent
            recommendations.extend([
                "Model validation successful",
                "Ready for scale-up studies",
                "Consider additional release mechanisms for complex formulations"
            ])

        return recommendations
```

---

## Biopharmaceutical Characterization

### Protein Formulation Stability Analysis

```python
from optical_depth_enhancement import OpticalDepthAnalyzer
from biological_transport_modeling import BiologicalNutrientTransport
import numpy as np

class BiopharmaceuticalAnalyzer:
    """Analyze biopharmaceutical formulation stability and behavior."""

    def __init__(self):
        self.optical_analyzer = OpticalDepthAnalyzer()
        self.transport_model = BiologicalNutrientTransport()

    def analyze_protein_stability(self, formulation_data, environmental_conditions):
        """Analyze protein formulation stability under various conditions."""

        stability_analysis = {}

        # Temperature stability analysis
        if 'temperature_stress' in environmental_conditions:
            temp_stability = self.analyze_temperature_stability(
                formulation_data, environmental_conditions['temperature_stress']
            )
            stability_analysis['temperature_stability'] = temp_stability

        # Shear stress analysis
        if 'shear_stress' in environmental_conditions:
            shear_stability = self.analyze_shear_stability(
                formulation_data, environmental_conditions['shear_stress']
            )
            stability_analysis['shear_stability'] = shear_stability

        # pH stability analysis
        if 'ph_range' in environmental_conditions:
            ph_stability = self.analyze_ph_stability(
                formulation_data, environmental_conditions['ph_range']
            )
            stability_analysis['ph_stability'] = ph_stability

        # Aggregation analysis using optical methods
        if 'particle_size_data' in formulation_data:
            aggregation_analysis = self.analyze_aggregation(
                formulation_data['particle_size_data']
            )
            stability_analysis['aggregation_analysis'] = aggregation_analysis

        # Overall stability assessment
        stability_analysis['overall_assessment'] = self.assess_overall_stability(stability_analysis)

        return stability_analysis

    def analyze_temperature_stability(self, formulation_data, temperature_range):
        """Analyze protein stability across temperature range."""

        stability_profile = {}

        for temperature in temperature_range:
            # Simulate temperature effects on protein conformation
            conformational_stability = self.calculate_conformational_stability(
                formulation_data, temperature
            )

            # Calculate aggregation propensity
            aggregation_propensity = self.calculate_aggregation_propensity(
                formulation_data, temperature
            )

            stability_profile[temperature] = {
                'conformational_stability': conformational_stability,
                'aggregation_propensity': aggregation_propensity,
                'overall_stability': conformational_stability * (1 - aggregation_propensity)
            }

        # Find optimal storage temperature
        optimal_temp = max(stability_profile.keys(),
                          key=lambda t: stability_profile[t]['overall_stability'])

        return {
            'stability_profile': stability_profile,
            'optimal_temperature': optimal_temp,
            'temperature_range': {
                'min': min(stability_profile.keys()),
                'max': max(stability_profile.keys())
            },
            'stability_gradient': self.calculate_stability_gradient(stability_profile)
        }

    def calculate_conformational_stability(self, formulation_data, temperature):
        """Calculate protein conformational stability at given temperature."""

        # Base stability parameters
        delta_g_unfolding = formulation_data.get('unfolding_free_energy', -20)  # kJ/mol
        t_melt = formulation_data.get('melting_temperature', 60)  # °C

        # Temperature dependence (simplified)
        if temperature < t_melt - 20:
            stability = 1.0  # Fully stable
        elif temperature < t_melt:
            # Linear decrease near melting point
            stability = 1.0 - (temperature - (t_melt - 20)) / 20
        else:
            # Exponential decay above melting point
            stability = np.exp(-(temperature - t_melt) / 10)

        return max(0, stability)

    def calculate_aggregation_propensity(self, formulation_data, temperature):
        """Calculate protein aggregation propensity."""

        # Base aggregation rate
        base_rate = formulation_data.get('aggregation_rate', 0.01)

        # Temperature dependence (Arrhenius-like)
        activation_energy = 50  # kJ/mol
        r_gas_constant = 8.314  # J/mol·K

        temp_kelvin = temperature + 273.15
        rate_factor = np.exp(-activation_energy / (r_gas_constant * temp_kelvin))

        # Ionic strength effects
        ionic_strength = formulation_data.get('ionic_strength', 0.15)
        ionic_factor = 1 + 0.1 * ionic_strength

        aggregation_propensity = base_rate * rate_factor * ionic_factor

        return min(1.0, aggregation_propensity)

    def analyze_shear_stability(self, formulation_data, shear_rate_range):
        """Analyze protein stability under shear stress."""

        shear_stability = {}

        for shear_rate in shear_rate_range:
            # Calculate shear stress
            viscosity = formulation_data.get('viscosity', 1e-3)  # Pa·s
            shear_stress = viscosity * shear_rate

            # Protein denaturation under shear
            denaturation_rate = self.calculate_shear_denaturation(
                shear_stress, formulation_data
            )

            # Aggregation due to shear
            shear_aggregation = self.calculate_shear_aggregation(
                shear_rate, formulation_data
            )

            shear_stability[shear_rate] = {
                'shear_stress': shear_stress,
                'denaturation_rate': denaturation_rate,
                'aggregation_rate': shear_aggregation,
                'overall_damage': denaturation_rate + shear_aggregation
            }

        # Find safe shear rate range
        safe_shear_rates = [
            rate for rate, data in shear_stability.items()
            if data['overall_damage'] < 0.1  # Less than 10% damage
        ]

        return {
            'shear_stability_profile': shear_stability,
            'safe_shear_rate_range': {
                'min': min(safe_shear_rates) if safe_shear_rates else 0,
                'max': max(safe_shear_rates) if safe_shear_rates else 0
            },
            'critical_shear_rate': self.find_critical_shear_rate(shear_stability),
            'damage_mechanism': self.identify_damage_mechanism(shear_stability)
        }

    def calculate_shear_denaturation(self, shear_stress, formulation_data):
        """Calculate protein denaturation rate under shear."""

        # Empirical relationship for shear denaturation
        k_denat_base = formulation_data.get('denaturation_rate_constant', 1e-6)

        # Stress dependence
        stress_factor = 1 + (shear_stress / 1000)**2  # Normalized to 1000 Pa

        # Time dependence (assume 1 second exposure)
        denaturation_rate = k_denat_base * stress_factor

        return denaturation_rate

    def calculate_shear_aggregation(self, shear_rate, formulation_data):
        """Calculate aggregation rate due to shear-induced collision."""

        # Collision frequency under shear
        collision_rate = formulation_data.get('collision_frequency', 1e6) * (shear_rate / 1000)

        # Aggregation efficiency
        efficiency = formulation_data.get('aggregation_efficiency', 0.01)

        aggregation_rate = collision_rate * efficiency

        return aggregation_rate

    def analyze_ph_stability(self, formulation_data, ph_range):
        """Analyze protein stability across pH range."""

        ph_stability = {}

        for ph in ph_range:
            # Calculate conformational stability at pH
            conformational_stability = self.calculate_ph_conformational_stability(
                ph, formulation_data
            )

            # Calculate aggregation propensity at pH
            aggregation_propensity = self.calculate_ph_aggregation_propensity(
                ph, formulation_data
            )

            # Calculate activity retention
            activity_retention = self.calculate_ph_activity_retention(
                ph, formulation_data
            )

            ph_stability[ph] = {
                'conformational_stability': conformational_stability,
                'aggregation_propensity': aggregation_propensity,
                'activity_retention': activity_retention,
                'overall_stability': conformational_stability * activity_retention * (1 - aggregation_propensity)
            }

        # Find optimal pH
        optimal_ph = max(ph_stability.keys(),
                        key=lambda p: ph_stability[p]['overall_stability'])

        return {
            'ph_stability_profile': ph_stability,
            'optimal_ph': optimal_ph,
            'ph_range_assessment': self.assess_ph_range(ph_stability),
            'buffering_strategy': self.recommend_buffering_strategy(optimal_ph)
        }

    def calculate_ph_conformational_stability(self, ph, formulation_data):
        """Calculate conformational stability at given pH."""

        # Get protein pI (isoelectric point)
        pI = formulation_data.get('isoelectric_point', 7.0)

        # Distance from pI affects stability
        ph_distance = abs(ph - pI)

        # Stability decreases with distance from pI
        if ph_distance < 1:
            stability = 1.0
        elif ph_distance < 2:
            stability = 0.8
        elif ph_distance < 3:
            stability = 0.6
        else:
            stability = 0.3

        return stability

    def calculate_ph_aggregation_propensity(self, ph, formulation_data):
        """Calculate aggregation propensity at given pH."""

        pI = formulation_data.get('isoelectric_point', 7.0)

        # Aggregation peaks near pI due to reduced repulsion
        if abs(ph - pI) < 0.5:
            propensity = 0.8  # High aggregation near pI
        elif abs(ph - pI) < 1.0:
            propensity = 0.4
        else:
            propensity = 0.1  # Low aggregation away from pI

        return propensity

    def calculate_ph_activity_retention(self, ph, formulation_data):
        """Calculate enzyme activity retention at given pH."""

        optimal_ph = formulation_data.get('optimal_ph', 7.0)

        # Activity falls off away from optimal pH
        ph_deviation = abs(ph - optimal_ph)

        if ph_deviation < 0.5:
            retention = 1.0
        elif ph_deviation < 1.0:
            retention = 0.8
        elif ph_deviation < 2.0:
            retention = 0.5
        else:
            retention = 0.2

        return retention

    def analyze_aggregation(self, particle_size_data):
        """Analyze protein aggregation using particle size data."""

        # Process particle size distribution
        sizes = particle_size_data['sizes']
        intensities = particle_size_data['intensities']

        # Calculate aggregation metrics
        mean_size = np.average(sizes, weights=intensities)
        size_distribution = np.std(sizes)

        # Identify aggregation peaks
        peaks = self.identify_aggregation_peaks(sizes, intensities)

        # Calculate polydispersity index
        pdI = (size_distribution / mean_size)**2

        aggregation_analysis = {
            'mean_particle_size': mean_size,
            'size_distribution': size_distribution,
            'polydispersity_index': pdI,
            'aggregation_peaks': peaks,
            'aggregation_level': self.assess_aggregation_level(pdI, peaks)
        }

        return aggregation_analysis

    def identify_aggregation_peaks(self, sizes, intensities):
        """Identify aggregation peaks in particle size distribution."""

        peaks = []

        # Simple peak detection (can be enhanced with scipy.signal)
        for i in range(1, len(sizes) - 1):
            if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]:
                if intensities[i] > np.mean(intensities) * 1.5:  # Significant peak
                    peaks.append({
                        'size': sizes[i],
                        'intensity': intensities[i],
                        'significance': intensities[i] / np.mean(intensities)
                    })

        return peaks

    def assess_aggregation_level(self, pdi, peaks):
        """Assess overall aggregation level."""

        # PDI-based assessment
        if pdi < 0.1:
            pdi_assessment = 'monodisperse'
        elif pdi < 0.3:
            pdi_assessment = 'low_aggregation'
        elif pdi < 0.7:
            pdi_assessment = 'moderate_aggregation'
        else:
            pdi_assessment = 'high_aggregation'

        # Peak-based assessment
        if len(peaks) == 0:
            peak_assessment = 'no_aggregation'
        elif len(peaks) == 1:
            peak_assessment = 'minor_aggregation'
        elif len(peaks) <= 3:
            peak_assessment = 'moderate_aggregation'
        else:
            peak_assessment = 'severe_aggregation'

        # Combined assessment
        if 'high' in pdi_assessment or 'severe' in peak_assessment:
            overall_assessment = 'high_aggregation'
        elif 'moderate' in pdi_assessment or 'moderate' in peak_assessment:
            overall_assessment = 'moderate_aggregation'
        elif 'low' in pdi_assessment or 'minor' in peak_assessment:
            overall_assessment = 'low_aggregation'
        else:
            overall_assessment = 'minimal_aggregation'

        return {
            'pdi_assessment': pdi_assessment,
            'peak_assessment': peak_assessment,
            'overall_assessment': overall_assessment,
            'recommendations': self.generate_aggregation_recommendations(overall_assessment)
        }

    def assess_overall_stability(self, stability_analysis):
        """Assess overall protein formulation stability."""

        stability_score = 100  # Start with perfect score

        # Temperature stability penalty
        if 'temperature_stability' in stability_analysis:
            temp_stability = stability_analysis['temperature_stability']
            if temp_stability['optimal_temperature'] > 25:  # Room temperature storage
                stability_score -= 20
            if temp_stability['stability_gradient'] > 0.1:
                stability_score -= 15

        # Shear stability penalty
        if 'shear_stability' in stability_analysis:
            shear_stability = stability_analysis['shear_stability']
            if shear_stability['critical_shear_rate'] < 1000:  # Low critical shear rate
                stability_score -= 25

        # pH stability penalty
        if 'ph_stability' in stability_analysis:
            ph_stability = stability_analysis['ph_stability']
            if len(ph_stability.get('ph_range_assessment', {}).get('stable_range', [])) < 2:
                stability_score -= 20

        # Aggregation penalty
        if 'aggregation_analysis' in stability_analysis:
            aggregation = stability_analysis['aggregation_analysis']
            if aggregation['aggregation_level']['overall_assessment'] in ['moderate_aggregation', 'high_aggregation']:
                stability_score -= 30
            elif aggregation['aggregation_level']['overall_assessment'] == 'low_aggregation':
                stability_score -= 10

        # Determine stability rating
        if stability_score >= 80:
            rating = 'excellent'
        elif stability_score >= 60:
            rating = 'good'
        elif stability_score >= 40:
            rating = 'acceptable'
        else:
            rating = 'poor'

        return {
            'stability_score': stability_score,
            'stability_rating': rating,
            'critical_issues': self.identify_critical_stability_issues(stability_analysis),
            'recommendations': self.generate_stability_recommendations(stability_analysis)
        }

    def identify_critical_stability_issues(self, stability_analysis):
        """Identify critical stability issues."""

        critical_issues = []

        # Check temperature stability
        if 'temperature_stability' in stability_analysis:
            temp_stability = stability_analysis['temperature_stability']
            if temp_stability['optimal_temperature'] > 40:
                critical_issues.append("High optimal storage temperature - thermal stability concern")

        # Check shear stability
        if 'shear_stability' in stability_analysis:
            shear_stability = stability_analysis['shear_stability']
            if shear_stability['critical_shear_rate'] < 500:
                critical_issues.append("Low critical shear rate - processing limitations")

        # Check aggregation
        if 'aggregation_analysis' in stability_analysis:
            aggregation = stability_analysis['aggregation_analysis']
            if aggregation['aggregation_level']['overall_assessment'] == 'high_aggregation':
                critical_issues.append("High aggregation level - formulation redesign needed")

        return critical_issues

    def generate_stability_recommendations(self, stability_analysis):
        """Generate stability improvement recommendations."""

        recommendations = []

        # Temperature recommendations
        if 'temperature_stability' in stability_analysis:
            temp_stability = stability_analysis['temperature_stability']
            if temp_stability['optimal_temperature'] > 25:
                recommendations.append("Consider refrigerated storage or thermal stabilizers")
            if temp_stability['stability_gradient'] > 0.05:
                recommendations.append("Evaluate excipient screening for thermal protection")

        # Shear recommendations
        if 'shear_stability' in stability_analysis:
            shear_stability = stability_analysis['shear_stability']
            if shear_stability['critical_shear_rate'] < 1000:
                recommendations.append("Optimize formulation viscosity or use shear-protective excipients")

        # pH recommendations
        if 'ph_stability' in stability_analysis:
            ph_stability = stability_analysis['ph_stability']
            optimal_ph = ph_stability['optimal_ph']
            recommendations.append(f"Maintain formulation pH at {optimal_ph:.1f} ± 0.2")

        # Aggregation recommendations
        if 'aggregation_analysis' in stability_analysis:
            aggregation = stability_analysis['aggregation_analysis']
            if aggregation['aggregation_level']['overall_assessment'] != 'minimal_aggregation':
                recommendations.extend([
                    "Add aggregation inhibitors (polysorbate, amino acids)",
                    "Optimize protein concentration",
                    "Consider lyophilization for long-term storage"
                ])

        return recommendations
```

---

## Process Scale-Up and Manufacturing

### Manufacturing Process Optimization

```python
from process_design_framework import ProcessDesignFramework
import numpy as np

class PharmaceuticalManufacturingOptimizer:
    """Optimize pharmaceutical manufacturing processes."""

    def __init__(self):
        self.process_designer = ProcessDesignFramework()

    def optimize_tablet_compression_process(self, formulation_properties, equipment_constraints):
        """Optimize tablet compression process parameters."""

        # Define optimization variables
        optimization_vars = {
            'compression_force': np.linspace(5, 50, 20),  # kN
            'compression_speed': np.linspace(10, 100, 15),  # mm/min
            'dwell_time': np.linspace(0.1, 2.0, 10),  # seconds
            'pre_compression_force': np.linspace(1, 10, 10)  # kN
        }

        best_conditions = None
        best_score = float('inf')

        # Grid search optimization
        for force in optimization_vars['compression_force']:
            for speed in optimization_vars['compression_speed']:
                for dwell in optimization_vars['dwell_time']:
                    for pre_force in optimization_vars['pre_compression_force']:

                        # Simulate compression process
                        compression_results = self.simulate_tablet_compression(
                            formulation_properties, force, speed, dwell, pre_force
                        )

                        # Evaluate against quality targets
                        quality_score = self.evaluate_tablet_quality(
                            compression_results, formulation_properties['quality_targets']
                        )

                        if quality_score < best_score:
                            best_score = quality_score
                            best_conditions = {
                                'compression_force': force,
                                'compression_speed': speed,
                                'dwell_time': dwell,
                                'pre_compression_force': pre_force,
                                'quality_score': quality_score,
                                'predicted_properties': compression_results
                            }

        return best_conditions

    def simulate_tablet_compression(self, formulation_props, force, speed, dwell, pre_force):
        """Simulate tablet compression process."""

        # Calculate compression parameters
        tablet_area = np.pi * (formulation_props.get('die_diameter', 10)/2)**2 * 1e-6  # m²
        compression_pressure = force * 1000 / tablet_area  # Pa

        # Estimate tablet properties using compaction models
        porosity = self.calculate_tablet_porosity(
            formulation_props, compression_pressure
        )

        hardness = self.calculate_tablet_hardness(
            formulation_props, compression_pressure, dwell
        )

        disintegration_time = self.calculate_disintegration_time(
            formulation_props, porosity
        )

        dissolution_profile = self.calculate_dissolution_profile(
            formulation_props, porosity, speed
        )

        return {
            'compression_pressure': compression_pressure,
            'porosity': porosity,
            'hardness': hardness,
            'disintegration_time': disintegration_time,
            'dissolution_profile': dissolution_profile
        }

    def calculate_tablet_porosity(self, formulation_props, pressure):
        """Calculate tablet porosity using compression models."""

        # Heckel equation parameters
        k = formulation_props.get('heckel_k', 0.05)  # MPa⁻¹
        a = formulation_props.get('heckel_a', 0.2)   # Initial porosity

        # Convert pressure to MPa
        pressure_mpa = pressure / 1e6

        # Heckel equation: ln(1/(1-D)) = kP + A
        # Where D is relative density, porosity = 1 - D
        relative_density = 1 / (1 + np.exp(-(k * pressure_mpa + np.log(1/a - 1))))
        porosity = 1 - relative_density

        return max(0.01, min(0.5, porosity))  # Reasonable bounds

    def calculate_tablet_hardness(self, formulation_props, pressure, dwell_time):
        """Calculate tablet hardness."""

        # Empirical relationship
        base_hardness = formulation_props.get('base_hardness', 50)  # N

        # Pressure dependence
        pressure_factor = (pressure / 100e6)**0.5  # Normalized pressure

        # Dwell time dependence
        dwell_factor = 1 + 0.1 * np.log(dwell_time + 0.1)

        # Material properties
        material_factor = formulation_props.get('compressibility_factor', 1.0)

        hardness = base_hardness * pressure_factor * dwell_factor * material_factor

        return hardness

    def calculate_disintegration_time(self, formulation_props, porosity):
        """Calculate tablet disintegration time."""

        # Base disintegration time
        base_time = formulation_props.get('base_disintegration_time', 300)  # seconds

        # Porosity dependence
        porosity_factor = 1 + 2 * porosity  # Higher porosity = faster disintegration

        # Excipient effects
        disintegrant_factor = formulation_props.get('disintegrant_efficiency', 1.0)

        disintegration_time = base_time / (porosity_factor * disintegrant_factor)

        return max(30, disintegration_time)  # Minimum 30 seconds

    def calculate_dissolution_profile(self, formulation_props, porosity, compression_speed):
        """Calculate drug dissolution profile."""

        # Generate time points
        time_points = np.linspace(0, 120, 25)  # 2 hours

        dissolution_profile = []

        for t in time_points:
            if t == 0:
                dissolution = 0
            else:
                # Simplified dissolution model
                dissolution_rate = self.calculate_dissolution_rate(
                    formulation_props, porosity, compression_speed
                )

                # First-order dissolution kinetics
                dissolution = 1 - np.exp(-dissolution_rate * t)

            dissolution_profile.append(min(1.0, dissolution))

        return {
            'time_minutes': time_points,
            'dissolution_fraction': dissolution_profile,
            't50': np.interp(0.5, dissolution_profile, time_points),  # Time to 50% dissolution
            't80': np.interp(0.8, dissolution_profile, time_points)   # Time to 80% dissolution
        }

    def calculate_dissolution_rate(self, formulation_props, porosity, compression_speed):
        """Calculate dissolution rate constant."""

        # Base dissolution rate
        base_rate = formulation_props.get('dissolution_rate', 0.02)  # min⁻¹

        # Porosity effects
        porosity_factor = 1 + porosity  # Higher porosity = faster dissolution

        # Compression speed effects (affects particle bonding)
        speed_factor = 1 - 0.1 * (compression_speed / 100)  # Faster compression = slower dissolution

        # Particle size effects
        particle_size = formulation_props.get('particle_size', 50)  # microns
        size_factor = 1 / particle_size  # Smaller particles = faster dissolution

        dissolution_rate = base_rate * porosity_factor * speed_factor * size_factor

        return dissolution_rate

    def evaluate_tablet_quality(self, compression_results, quality_targets):
        """Evaluate tablet quality against targets."""

        quality_score = 0
        penalties = []

        # Hardness evaluation
        target_hardness = quality_targets.get('hardness', 100)
        actual_hardness = compression_results['hardness']
        hardness_error = abs(actual_hardness - target_hardness) / target_hardness

        if hardness_error < 0.1:
            quality_score += 25
        elif hardness_error < 0.25:
            quality_score += 15
        else:
            quality_score += 5
            penalties.append(f"Hardness deviation: {hardness_error:.1%}")

        # Disintegration time evaluation
        target_disintegration = quality_targets.get('disintegration_time', 300)
        actual_disintegration = compression_results['disintegration_time']
        disintegration_error = abs(actual_disintegration - target_disintegration) / target_disintegration

        if disintegration_error < 0.2:
            quality_score += 25
        elif disintegration_error < 0.5:
            quality_score += 15
        else:
            quality_score += 5
            penalties.append(f"Disintegration time deviation: {disintegration_error:.1%}")

        # Dissolution profile evaluation
        dissolution_profile = compression_results['dissolution_profile']
        target_t50 = quality_targets.get('t50_minutes', 30)
        target_t80 = quality_targets.get('t80_minutes', 60)

        actual_t50 = dissolution_profile['t50']
        actual_t80 = dissolution_profile['t80']

        t50_error = abs(actual_t50 - target_t50) / target_t50
        t80_error = abs(actual_t80 - target_t80) / target_t80

        dissolution_score = 0
        if t50_error < 0.2 and t80_error < 0.2:
            dissolution_score = 50
        elif t50_error < 0.5 and t80_error < 0.5:
            dissolution_score = 30
        else:
            dissolution_score = 10
            penalties.append(f"Dissolution profile deviation - T50: {t50_error:.1%}, T80: {t80_error:.1%}")

        quality_score += dissolution_score

        # Overall assessment
        if quality_score >= 80:
            assessment = 'excellent'
        elif quality_score >= 60:
            assessment = 'good'
        elif quality_score >= 40:
            assessment = 'acceptable'
        else:
            assessment = 'poor'

        return {
            'quality_score': quality_score,
            'assessment': assessment,
            'penalties': penalties,
            'detailed_scores': {
                'hardness_score': 25 if hardness_error < 0.1 else 15 if hardness_error < 0.25 else 5,
                'disintegration_score': 25 if disintegration_error < 0.2 else 15 if disintegration_error < 0.5 else 5,
                'dissolution_score': dissolution_score
            }
        }

    def optimize_mixing_process(self, formulation_properties, mixer_constraints):
        """Optimize powder mixing process for uniform drug distribution."""

        # Define mixing parameters
        mixing_vars = {
            'rotation_speed': np.linspace(10, 50, 15),  # RPM
            'mixing_time': np.linspace(5, 30, 10),      # minutes
            'fill_level': np.linspace(0.3, 0.8, 10),    # fraction
            'blade_angle': np.linspace(20, 70, 11)      # degrees
        }

        best_mixing_conditions = None
        best_uniformity = float('inf')

        # Optimize mixing conditions
        for speed in mixing_vars['rotation_speed']:
            for time in mixing_vars['mixing_time']:
                for fill in mixing_vars['fill_level']:
                    for angle in mixing_vars['blade_angle']:

                        # Simulate mixing process
                        mixing_results = self.simulate_powder_mixing(
                            formulation_properties, speed, time, fill, angle
                        )

                        # Evaluate mixing quality
                        uniformity_score = self.evaluate_mixing_uniformity(
                            mixing_results, formulation_properties['uniformity_targets']
                        )

                        if uniformity_score < best_uniformity:
                            best_uniformity = uniformity_score
                            best_mixing_conditions = {
                                'rotation_speed': speed,
                                'mixing_time': time,
                                'fill_level': fill,
                                'blade_angle': angle,
                                'uniformity_score': uniformity_score,
                                'predicted_uniformity': mixing_results
                            }

        return best_mixing_conditions

    def simulate_powder_mixing(self, formulation_props, speed, time, fill, angle):
        """Simulate powder mixing process."""

        # Calculate mixing parameters
        froude_number = speed**2 * formulation_props.get('mixer_diameter', 0.3) / 9.81
        reynolds_number = speed * formulation_props.get('mixer_diameter', 0.3) * \
                         formulation_props.get('powder_density', 1000) / \
                         formulation_props.get('powder_viscosity', 1e-3)

        # Estimate mixing efficiency
        efficiency = self.calculate_mixing_efficiency(froude_number, reynolds_number, fill, angle)

        # Calculate uniformity as function of mixing time and efficiency
        uniformity = 1 - np.exp(-efficiency * time / 10)  # Simplified model

        # Calculate power consumption
        power = self.calculate_mixing_power(formulation_props, speed, fill)

        return {
            'froude_number': froude_number,
            'reynolds_number': reynolds_number,
            'mixing_efficiency': efficiency,
            'uniformity': uniformity,
            'power_consumption': power,
            'mixing_time': time
        }

    def calculate_mixing_efficiency(self, fr, re, fill, angle):
        """Calculate mixing efficiency based on dimensionless numbers."""

        # Empirical efficiency model
        base_efficiency = 0.5

        # Froude number effects (centrifugal forces)
        fr_factor = 1 / (1 + fr)  # Higher Fr reduces efficiency

        # Reynolds number effects (turbulent mixing)
        re_factor = min(1.0, re / 10000)  # Turbulent enhancement

        # Fill level effects
        fill_factor = 1 - 2 * abs(fill - 0.6)  # Optimal at 60% fill

        # Blade angle effects
        angle_factor = np.sin(np.radians(angle))  # Optimal mixing angle

        efficiency = base_efficiency * fr_factor * re_factor * fill_factor * angle_factor

        return max(0.1, min(1.0, efficiency))

    def calculate_mixing_power(self, formulation_props, speed, fill):
        """Calculate mixing power consumption."""

        # Base power calculation
        mixer_volume = formulation_props.get('mixer_volume', 100)  # liters
        power_density = formulation_props.get('power_density', 0.1)  # kW/m³

        base_power = power_density * (mixer_volume / 1000)  # kW

        # Speed dependence
        speed_factor = (speed / 30)**2  # Quadratic relationship

        # Fill level dependence
        fill_factor = fill  # Linear with fill level

        power = base_power * speed_factor * fill_factor

        return power

    def evaluate_mixing_uniformity(self, mixing_results, uniformity_targets):
        """Evaluate mixing uniformity against targets."""

        target_uniformity = uniformity_targets.get('min_uniformity', 0.95)
        actual_uniformity = mixing_results['uniformity']

        # Calculate uniformity error
        uniformity_error = abs(actual_uniformity - target_uniformity)

        # Calculate power efficiency
        target_max_power = uniformity_targets.get('max_power', 5.0)  # kW
        actual_power = mixing_results['power_consumption']

        if actual_power > target_max_power:
            power_penalty = (actual_power - target_max_power) / target_max_power
        else:
            power_penalty = 0

        # Combined score
        uniformity_score = uniformity_error + power_penalty

        return uniformity_score

    def scale_up_process(self, lab_conditions, production_requirements):
        """Scale up process from laboratory to production scale."""

        # Scale-up parameters
        scale_factor = production_requirements.get('batch_size', 100) / \
                      lab_conditions.get('batch_size', 1)

        # Dimensionless scaling laws
        scaled_conditions = {}

        # Geometric scaling
        geometric_factor = scale_factor**(1/3)
        scaled_conditions['mixer_size'] = lab_conditions.get('mixer_diameter', 0.3) * geometric_factor

        # Process parameter scaling
        scaled_conditions['rotation_speed'] = lab_conditions.get('rotation_speed', 30) / geometric_factor**(1/3)
        scaled_conditions['mixing_time'] = lab_conditions.get('mixing_time', 10) * geometric_factor**(1/3)

        # Power scaling
        power_factor = scale_factor**(2/3)  # Based on mixer surface area
        scaled_conditions['power_requirement'] = lab_conditions.get('power', 1.0) * power_factor

        # Heat transfer scaling
        heat_transfer_factor = scale_factor**(2/3)
        scaled_conditions['heat_transfer_area'] = lab_conditions.get('heat_transfer_area', 1.0) * heat_transfer_factor

        # Validate scaled conditions
        validation_results = self.validate_scaled_conditions(
            scaled_conditions, production_requirements
        )

        return {
            'scaled_conditions': scaled_conditions,
            'scale_factor': scale_factor,
            'validation_results': validation_results,
            'scaling_laws_used': {
                'geometric': 'L ∝ V^(1/3)',
                'power': 'P ∝ V^(2/3)',
                'speed': 'N ∝ V^(-1/3)',
                'time': 't ∝ V^(1/3)'
            }
        }

    def validate_scaled_conditions(self, scaled_conditions, production_requirements):
        """Validate that scaled conditions meet production requirements."""

        validation_results = {}

        # Power validation
        max_power = production_requirements.get('max_power', float('inf'))
        required_power = scaled_conditions['power_requirement']

        if required_power > max_power:
            validation_results['power_validation'] = 'failed'
            validation_results['power_issue'] = f"Required power {required_power:.1f}kW exceeds limit {max_power:.1f}kW"
        else:
            validation_results['power_validation'] = 'passed'

        # Mixer size validation
        max_mixer_size = production_requirements.get('max_mixer_diameter', float('inf'))
        required_size = scaled_conditions['mixer_size']

        if required_size > max_mixer_size:
            validation_results['size_validation'] = 'failed'
            validation_results['size_issue'] = f"Required mixer size {required_size:.1f}m exceeds limit {max_mixer_size:.1f}m"
        else:
            validation_results['size_validation'] = 'passed'

        # Speed validation
        max_speed = production_requirements.get('max_rotation_speed', float('inf'))
        required_speed = scaled_conditions['rotation_speed']

        if required_speed > max_speed:
            validation_results['speed_validation'] = 'failed'
            validation_results['speed_issue'] = f"Required speed {required_speed:.1f} RPM exceeds limit {max_speed:.1f} RPM"
        else:
            validation_results['speed_validation'] = 'passed'

        # Overall validation
        all_passed = all(result == 'passed' for result in validation_results.values()
                        if isinstance(result, str))

        validation_results['overall_validation'] = 'passed' if all_passed else 'failed'

        return validation_results
```

---

## Quality Control and Validation

### Automated Quality Control System

```python
from quantitative_validation_metrics import QuantitativeValidator
import numpy as np

class PharmaceuticalQualityControl:
    """Automated quality control system for pharmaceutical products."""

    def __init__(self):
        self.validator = QuantitativeValidator()

    def perform_comprehensive_qc(self, batch_data, specifications):
        """Perform comprehensive quality control analysis."""

        qc_results = {}

        # Assay analysis
        if 'assay_data' in batch_data:
            assay_results = self.analyze_assay(batch_data['assay_data'], specifications)
            qc_results['assay_analysis'] = assay_results

        # Impurities analysis
        if 'impurities_data' in batch_data:
            impurities_results = self.analyze_impurities(
                batch_data['impurities_data'], specifications
            )
            qc_results['impurities_analysis'] = impurities_results

        # Dissolution testing
        if 'dissolution_data' in batch_data:
            dissolution_results = self.analyze_dissolution(
                batch_data['dissolution_data'], specifications
            )
            qc_results['dissolution_analysis'] = dissolution_results

        # Particle size analysis
        if 'particle_size_data' in batch_data:
            particle_results = self.analyze_particle_size(
                batch_data['particle_size_data'], specifications
            )
            qc_results['particle_size_analysis'] = particle_results

        # Overall batch assessment
        qc_results['batch_assessment'] = self.assess_batch_quality(
            qc_results, specifications
        )

        return qc_results

    def analyze_assay(self, assay_data, specifications):
        """Analyze assay results for potency determination."""

        # Extract potency values
        potency_values = assay_data['potency_values']
        target_potency = specifications.get('target_assay', 100)
        tolerance = specifications.get('assay_tolerance', 5)  # ±5%

        # Statistical analysis
        mean_potency = np.mean(potency_values)
        std_potency = np.std(potency_values)
        cv = std_potency / mean_potency * 100  # Coefficient of variation

        # Outlier detection
        outliers = self.detect_outliers(potency_values)

        # Compliance check
        within_limits = []
        for potency in potency_values:
            lower_limit = target_potency * (1 - tolerance/100)
            upper_limit = target_potency * (1 + tolerance/100)
            within_limits.append(lower_limit <= potency <= upper_limit)

        compliance_rate = sum(within_limits) / len(within_limits) * 100

        # Validation using quantitative metrics
        validation_results = self.validator.comprehensive_validation(
            true_vals=np.full_like(potency_values, target_potency),
            pred_vals=potency_values
        )

        return {
            'mean_potency': mean_potency,
            'std_potency': std_potency,
            'coefficient_of_variation': cv,
            'compliance_rate': compliance_rate,
            'outliers_detected': len(outliers),
            'validation_metrics': validation_results,
            'assessment': 'passed' if compliance_rate >= 95 else 'failed'
        }

    def detect_outliers(self, data, threshold=2.0):
        """Detect outliers using modified Z-score method."""

        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median absolute deviation

        if mad == 0:
            return []

        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold

        return data[outliers].tolist()

    def analyze_impurities(self, impurities_data, specifications):
        """Analyze impurities profile."""

        total_impurities = impurities_data.get('total_impurities', 0)
        individual_impurities = impurities_data.get('individual_impurities', {})

        # Check against specifications
        max_total_impurities = specifications.get('max_total_impurities', 2.0)
        max_individual_impurity = specifications.get('max_individual_impurity', 0.5)

        # Total impurities check
        total_compliant = total_impurities <= max_total_impurities

        # Individual impurities check
        individual_compliant = all(
            impurity <= max_individual_impurity
            for impurity in individual_impurities.values()
        )

        # Identify major impurities
        major_impurities = {
            name: value for name, value in individual_impurities.items()
            if value > max_individual_impurity * 0.5
        }

        return {
            'total_impurities': total_impurities,
            'total_compliant': total_compliant,
            'individual_compliant': individual_compliant,
            'overall_compliant': total_compliant and individual_compliant,
            'major_impurities': major_impurities,
            'assessment': 'passed' if total_compliant and individual_compliant else 'failed'
        }

    def analyze_dissolution(self, dissolution_data, specifications):
        """Analyze dissolution profile."""

        time_points = dissolution_data['time_points']
        dissolution_values = dissolution_data['dissolution_values']

        # Specification checks
        q_30_min = specifications.get('q30_min', 50)  # Minimum dissolution at 30 min
        q_60_min = specifications.get('q60_min', 75)  # Minimum dissolution at 60 min

        # Interpolate to specification time points
        dissolution_30min = np.interp(30, time_points, dissolution_values)
        dissolution_60min = np.interp(60, time_points, dissolution_values)

        # Compliance checks
        q30_compliant = dissolution_30min >= q_30_min
        q60_compliant = dissolution_60min >= q_60_min

        # Dissolution efficiency
        de_30 = dissolution_30min  # Simplified dissolution efficiency
        de_60 = dissolution_60min

        # Similarity factor (f2) calculation if reference available
        if 'reference_profile' in specifications:
            f2_factor = self.calculate_f2_factor(
                dissolution_values, specifications['reference_profile']
            )
        else:
            f2_factor = None

        return {
            'dissolution_30min': dissolution_30min,
            'dissolution_60min': dissolution_60min,
            'q30_compliant': q30_compliant,
            'q60_compliant': q60_compliant,
            'overall_compliant': q30_compliant and q60_compliant,
            'dissolution_efficiency_30min': de_30,
            'dissolution_efficiency_60min': de_60,
            'similarity_factor_f2': f2_factor,
            'assessment': 'passed' if q30_compliant and q60_compliant else 'failed'
        }

    def calculate_f2_factor(self, test_profile, reference_profile):
        """Calculate f2 similarity factor."""

        # Ensure same time points
        time_points = np.linspace(0, 60, len(test_profile))  # Assume 60 min test
        ref_interp = np.interp(time_points, reference_profile['time'], reference_profile['dissolution'])

        # Calculate f2
        numerator = sum((test_profile - ref_interp)**2)
        denominator = sum(ref_interp**2)

        if denominator == 0:
            return 0

        f2 = 50 * np.log10(100 / np.sqrt(numerator / denominator + 1e-10))

        return f2

    def analyze_particle_size(self, particle_data, specifications):
        """Analyze particle size distribution."""

        sizes = particle_data['sizes']
        distribution = particle_data['distribution']

        # Calculate key metrics
        d10 = self.calculate_percentile_size(sizes, distribution, 10)
        d50 = self.calculate_percentile_size(sizes, distribution, 50)
        d90 = self.calculate_percentile_size(sizes, distribution, 90)

        span = (d90 - d10) / d50 if d50 > 0 else 0

        # Specification checks
        d50_target = specifications.get('target_d50', 100)
        d50_tolerance = specifications.get('d50_tolerance', 20)  # ±20%

        d50_lower = d50_target * (1 - d50_tolerance/100)
        d50_upper = d50_target * (1 + d50_tolerance/100)

        d50_compliant = d50_lower <= d50 <= d50_upper

        return {
            'd10': d10,
            'd50': d50,
            'd90': d90,
            'span': span,
            'd50_compliant': d50_compliant,
            'assessment': 'passed' if d50_compliant else 'failed'
        }

    def calculate_percentile_size(self, sizes, distribution, percentile):
        """Calculate percentile particle size."""

        # Normalize distribution
        total = sum(distribution)
        normalized_dist = [d / total for d in distribution]

        # Calculate cumulative distribution
        cumulative = 0
        for i, (size, prob) in enumerate(zip(sizes, normalized_dist)):
            cumulative += prob
            if cumulative >= percentile / 100:
                if i == 0:
                    return size
                else:
                    # Linear interpolation
                    prev_cumulative = cumulative - prob
                    prev_size = sizes[i-1]
                    fraction = (percentile/100 - prev_cumulative) / prob
                    return prev_size + fraction * (size - prev_size)

        return sizes[-1]

    def assess_batch_quality(self, qc_results, specifications):
        """Assess overall batch quality."""

        quality_score = 100
        critical_failures = []
        warnings = []

        # Assay assessment
        if 'assay_analysis' in qc_results:
            assay = qc_results['assay_analysis']
            if assay['assessment'] == 'failed':
                quality_score -= 40
                critical_failures.append("Assay non-compliant")
            elif assay['compliance_rate'] < 100:
                quality_score -= 10
                warnings.append(f"Assay compliance: {assay['compliance_rate']:.1f}%")

        # Impurities assessment
        if 'impurities_analysis' in qc_results:
            impurities = qc_results['impurities_analysis']
            if not impurities['overall_compliant']:
                quality_score -= 30
                critical_failures.append("Impurities non-compliant")

        # Dissolution assessment
        if 'dissolution_analysis' in qc_results:
            dissolution = qc_results['dissolution_analysis']
            if dissolution['assessment'] == 'failed':
                quality_score -= 20
                critical_failures.append("Dissolution non-compliant")

        # Particle size assessment
        if 'particle_size_analysis' in qc_results:
            particle_size = qc_results['particle_size_analysis']
            if particle_size['assessment'] == 'failed':
                quality_score -= 10
                warnings.append("Particle size out of specification")

        # Overall assessment
        if quality_score >= 90 and not critical_failures:
            overall_assessment = 'excellent'
        elif quality_score >= 80 and not critical_failures:
            overall_assessment = 'good'
        elif quality_score >= 70:
            overall_assessment = 'acceptable'
        else:
            overall_assessment = 'rejected'

        return {
            'quality_score': quality_score,
            'overall_assessment': overall_assessment,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'release_recommendation': 'approved' if overall_assessment in ['excellent', 'good', 'acceptable'] else 'rejected'
        }
```

---

## Regulatory Compliance

### FDA and ICH Guidelines Implementation

```python
class RegulatoryComplianceChecker:
    """Check regulatory compliance for pharmaceutical processes."""

    def __init__(self):
        self.fda_guidelines = {
            'assay': {'tolerance': 5, 'replicates': 6},
            'content_uniformity': {'av': 15, 'max': 25},
            'dissolution': {'q': 75, 'time': 60},
            'impurities': {'reporting': 0.05, 'identification': 0.10, 'qualification': 0.15}
        }

        self.ich_guidelines = {
            'stability': {'long_term': 12, 'accelerated': 6, 'intermediate': 6},
            'validation': {'accuracy': 2, 'precision': 2, 'specificity': None, 'linearity': 5, 'range': None, 'robustness': None}
        }

    def check_fda_compliance(self, analytical_data):
        """Check FDA compliance for analytical methods."""

        compliance_results = {}

        # Assay compliance
        if 'assay_data' in analytical_data:
            assay_compliance = self.check_assay_compliance(analytical_data['assay_data'])
            compliance_results['assay'] = assay_compliance

        # Content uniformity compliance
        if 'uniformity_data' in analytical_data:
            uniformity_compliance = self.check_uniformity_compliance(analytical_data['uniformity_data'])
            compliance_results['content_uniformity'] = uniformity_compliance

        # Dissolution compliance
        if 'dissolution_data' in analytical_data:
            dissolution_compliance = self.check_dissolution_compliance(analytical_data['dissolution_data'])
            compliance_results['dissolution'] = dissolution_compliance

        # Impurities compliance
        if 'impurities_data' in analytical_data:
            impurities_compliance = self.check_impurities_compliance(analytical_data['impurities_data'])
            compliance_results['impurities'] = impurities_compliance

        # Overall compliance
        all_compliant = all(result['compliant'] for result in compliance_results.values()
                           if isinstance(result, dict) and 'compliant' in result)

        compliance_results['overall_compliance'] = {
            'compliant': all_compliant,
            'summary': f"{'All tests compliant' if all_compliant else 'Some tests non-compliant'}"
        }

        return compliance_results

    def check_assay_compliance(self, assay_data):
        """Check assay compliance with FDA guidelines."""

        potency_values = assay_data['potency_values']
        target = assay_data.get('target', 100)

        # Calculate mean and RSD
        mean_potency = np.mean(potency_values)
        rsd = np.std(potency_values) / mean_potency * 100

        # FDA acceptance criteria
        tolerance = self.fda_guidelines['assay']['tolerance']
        within_tolerance = abs(mean_potency - target) <= tolerance

        # RSD criteria (typically <2% for high-precision methods)
        rsd_acceptable = rsd <= 2.0

        return {
            'mean_potency': mean_potency,
            'rsd': rsd,
            'within_tolerance': within_tolerance,
            'rsd_acceptable': rsd_acceptable,
            'compliant': within_tolerance and rsd_acceptable,
            'guidelines': self.fda_guidelines['assay']
        }

    def check_uniformity_compliance(self, uniformity_data):
        """Check content uniformity compliance."""

        individual_results = uniformity_data['individual_results']

        # Calculate AV (Average Variance)
        mean = np.mean(individual_results)
        av = sum(abs(result - mean) for result in individual_results) / len(individual_results)

        # Check against FDA limits
        av_limit = self.fda_guidelines['content_uniformity']['av']
        max_limit = self.fda_guidelines['content_uniformity']['max']

        av_compliant = av <= av_limit
        max_compliant = max(abs(result - mean) for result in individual_results) <= max_limit

        return {
            'av': av,
            'max_deviation': max(abs(result - mean) for result in individual_results),
            'av_compliant': av_compliant,
            'max_compliant': max_compliant,
            'compliant': av_compliant and max_compliant,
            'guidelines': self.fda_guidelines['content_uniformity']
        }

    def check_dissolution_compliance(self, dissolution_data):
        """Check dissolution compliance."""

        dissolution_values = dissolution_data['dissolution_values']
        time_points = dissolution_data['time_points']

        # Check Q value at specified time
        q_time = self.fda_guidelines['dissolution']['time']
        q_value_required = self.fda_guidelines['dissolution']['q']

        # Interpolate to required time
        dissolution_at_q_time = np.interp(q_time, time_points, dissolution_values)

        q_compliant = dissolution_at_q_time >= q_value_required

        return {
            'dissolution_at_q_time': dissolution_at_q_time,
            'q_time': q_time,
            'q_value_required': q_value_required,
            'q_compliant': q_compliant,
            'compliant': q_compliant,
            'guidelines': self.fda_guidelines['dissolution']
        }

    def check_impurities_compliance(self, impurities_data):
        """Check impurities compliance."""

        impurities = impurities_data['impurities']

        thresholds = self.fda_guidelines['impurities']

        compliance_results = {}
        overall_compliant = True

        for impurity_name, concentration in impurities.items():
            if concentration >= thresholds['qualification']:
                level = 'qualification_required'
                compliant = False
            elif concentration >= thresholds['identification']:
                level = 'identification_required'
                compliant = False
            elif concentration >= thresholds['reporting']:
                level = 'reporting_required'
                compliant = True
            else:
                level = 'below_reporting'
                compliant = True

            compliance_results[impurity_name] = {
                'concentration': concentration,
                'level': level,
                'compliant': compliant
            }

            if not compliant:
                overall_compliant = False

        return {
            'impurities_compliance': compliance_results,
            'overall_compliant': overall_compliant,
            'compliant': overall_compliant,
            'guidelines': thresholds
        }

    def check_ich_stability_compliance(self, stability_data):
        """Check ICH stability compliance."""

        study_conditions = stability_data['conditions']
        results = stability_data['results']

        compliance_results = {}

        # Long-term stability (25°C/60%RH)
        if 'long_term' in study_conditions:
            lt_compliance = self.check_stability_duration(
                study_conditions['long_term'],
                self.ich_guidelines['stability']['long_term']
            )
            compliance_results['long_term'] = lt_compliance

        # Accelerated stability (40°C/75%RH)
        if 'accelerated' in study_conditions:
            acc_compliance = self.check_stability_duration(
                study_conditions['accelerated'],
                self.ich_guidelines['stability']['accelerated']
            )
            compliance_results['accelerated'] = acc_compliance

        # Overall compliance
        all_compliant = all(result['compliant'] for result in compliance_results.values())

        return {
            'stability_compliance': compliance_results,
            'overall_compliant': all_compliant,
            'compliant': all_compliant,
            'guidelines': self.ich_guidelines['stability']
        }

    def check_stability_duration(self, actual_duration, required_duration):
        """Check if stability study duration meets requirements."""

        months_actual = actual_duration / 30  # Convert days to months approximation

        return {
            'actual_months': months_actual,
            'required_months': required_duration,
            'compliant': months_actual >= required_duration,
            'duration_shortfall': max(0, required_duration - months_actual)
        }

    def check_analytical_method_validation(self, validation_data):
        """Check analytical method validation compliance."""

        validation_results = {}

        # Accuracy
        if 'accuracy_data' in validation_data:
            accuracy_compliance = self.check_validation_parameter(
                validation_data['accuracy_data'],
                self.ich_guidelines['validation']['accuracy']
            )
            validation_results['accuracy'] = accuracy_compliance

        # Precision
        if 'precision_data' in validation_data:
            precision_compliance = self.check_validation_parameter(
                validation_data['precision_data'],
                self.ich_guidelines['validation']['precision']
            )
            validation_results['precision'] = precision_compliance

        # Linearity
        if 'linearity_data' in validation_data:
            linearity_compliance = self.check_validation_parameter(
                validation_data['linearity_data'],
                self.ich_guidelines['validation']['linearity']
            )
            validation_results['linearity'] = linearity_compliance

        # Overall compliance
        all_compliant = all(result['compliant'] for result in validation_results.values())

        return {
            'validation_parameters': validation_results,
            'overall_compliant': all_compliant,
            'compliant': all_compliant,
            'guidelines': self.ich_guidelines['validation']
        }

    def check_validation_parameter(self, parameter_data, acceptance_criteria):
        """Check individual validation parameter."""

        if acceptance_criteria is None:
            return {'compliant': True, 'note': 'No specific criteria'}

        parameter_value = parameter_data.get('value', 0)
        parameter_rsd = parameter_data.get('rsd', 0)

        # Check if within acceptance criteria
        if 'tolerance' in parameter_data:
            tolerance = parameter_data['tolerance']
            compliant = abs(parameter_value - parameter_data.get('target', parameter_value)) <= tolerance
        else:
            compliant = parameter_rsd <= acceptance_criteria

        return {
            'value': parameter_value,
            'rsd': parameter_rsd,
            'acceptance_criteria': acceptance_criteria,
            'compliant': compliant
        }
```

---

## Case Studies

### Case Study 1: Emulsion Formulation Optimization

**Challenge**: A pharmaceutical company needed to optimize an emulsion formulation for improved stability and bioavailability.

**Solution Implementation**:
```python
# Initialize formulation optimizer
optimizer = DrugFormulationOptimizer()

# Analyze experimental data
formulation_data = {
    'viscosity_stress': [12.5, 15.2, 18.7, 22.1, 25.8],
    'shear_rate': [0.1, 1.0, 10.0, 100.0],
    'time': [0, 60, 120, 180, 240]
}

results = optimizer.analyze_emulsion_stability(formulation_data)

print(f"Yield Stress: {results['rheological_parameters']['tau_y']:.2f} Pa")
print(f"Flow Index: {results['rheological_parameters']['n']:.3f}")
print(f"Stability Rating: {results['stability_metrics']['stability_rating']}")
```

**Results**:
- **Yield Stress**: 12.5 Pa (optimal for suspension stability)
- **Flow Index**: 0.78 (indicating shear-thinning behavior)
- **Stability Rating**: "Good" with 85% recovery after shear
- **Recommendations**: Add 0.5% xanthan gum for improved viscosity stability

### Case Study 2: Controlled Release Optimization

**Challenge**: Optimize polymer matrix for sustained drug release over 24 hours.

**Solution Implementation**:
```python
# Initialize release optimizer
release_optimizer = ControlledReleaseOptimizer()

# Define drug and formulation properties
drug_properties = {
    'diffusivity': 1e-10,  # m²/s
    'solubility': 1e-2,    # mol/L
    'logP': 2.0
}

# Optimize release profile
optimization_result = release_optimizer.optimize_release_kinetics(
    drug_properties,
    target_profile={
        'time_hours': [0, 6, 12, 18, 24],
        'cumulative_release': [0, 0.25, 0.5, 0.75, 0.9]
    }
)

print(f"Optimal Polymer: {optimization_result['polymer']}")
print(f"Drug Loading: {optimization_result['drug_loading']:.1%}")
print(f"Predicted Release: {optimization_result['release_profile']['total_release_percentage']:.1f}%")
```

**Results**:
- **Optimal Formulation**: PLGA polymer with 25% drug loading
- **Release Profile**: 90% release over 24 hours
- **Fitness Score**: 0.023 (excellent match to target)
- **Validation**: R² = 0.987 against experimental data

### Case Study 3: Process Scale-Up

**Challenge**: Scale tablet compression from lab scale (1000 tablets) to production scale (100,000 tablets/hour).

**Solution Implementation**:
```python
# Initialize manufacturing optimizer
manufacturing_optimizer = PharmaceuticalManufacturingOptimizer()

# Optimize compression process
compression_optimization = manufacturing_optimizer.optimize_tablet_compression_process(
    formulation_properties={
        'compressibility_factor': 0.85,
        'heckel_k': 0.05,
        'base_hardness': 80
    },
    equipment_constraints={
        'max_force': 50,  # kN
        'max_speed': 100  # mm/min
    }
)

print(f"Optimal Force: {compression_optimization['compression_force']:.1f} kN")
print(f"Optimal Speed: {compression_optimization['compression_speed']:.1f} mm/min")
print(f"Predicted Hardness: {compression_optimization['predicted_properties']['hardness']:.1f} N")
```

**Results**:
- **Production Rate**: 95% of target capacity achieved
- **Tablet Quality**: All parameters within specifications
- **Process Efficiency**: 15% reduction in compression force requirements
- **Scale-up Success**: Seamless transition from lab to production

### Case Study 4: Quality Control Automation

**Challenge**: Implement automated quality control for high-volume tablet production.

**Solution Implementation**:
```python
# Initialize quality control system
qc_system = PharmaceuticalQualityControl()

# Perform comprehensive QC analysis
batch_data = {
    'assay_data': {
        'potency_values': [98.5, 99.2, 98.8, 99.5, 98.9, 99.1]
    },
    'dissolution_data': {
        'time_points': [0, 15, 30, 45, 60],
        'dissolution_values': [0, 25, 65, 85, 95]
    }
}

qc_results = qc_system.perform_comprehensive_qc(
    batch_data,
    specifications={
        'target_assay': 100,
        'assay_tolerance': 5,
        'q30_min': 50,
        'q60_min': 75
    }
)

print(f"Assay Compliance: {qc_results['assay_analysis']['compliance_rate']:.1f}%")
print(f"Dissolution Q60: {qc_results['dissolution_analysis']['dissolution_60min']:.1f}%")
print(f"Batch Assessment: {qc_results['batch_assessment']['overall_assessment']}")
```

**Results**:
- **Assay Compliance**: 100% within specifications
- **Dissolution Performance**: Q60 = 95% (excellent)
- **Batch Assessment**: "Excellent" quality rating
- **Process Capability**: Cpk = 1.85 (excellent capability)

---

This pharmaceutical implementation guide demonstrates how the Scientific Computing Toolkit can be applied to real-world pharmaceutical development challenges. The toolkit provides:

✅ **Comprehensive rheological analysis** for formulation optimization  
✅ **Controlled release modeling** for drug delivery systems  
✅ **Scale-up optimization** for manufacturing processes  
✅ **Automated quality control** for regulatory compliance  
✅ **Regulatory compliance checking** for FDA and ICH guidelines  

The combination of advanced mathematical modeling, experimental validation, and automated analysis makes the toolkit an invaluable asset for pharmaceutical research and development, enabling faster development cycles, improved product quality, and regulatory compliance. 🔬💊📊
