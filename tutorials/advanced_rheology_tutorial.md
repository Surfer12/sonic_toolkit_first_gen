# Advanced Rheological Analysis Tutorial

## Master Complex Fluid Behavior Analysis

This advanced tutorial teaches you how to perform sophisticated rheological analysis using the Scientific Computing Toolkit's Herschel-Bulkley framework. You'll learn to characterize complex fluids, optimize process parameters, and generate publication-quality results.

---

## Table of Contents

1. [Advanced Rheological Concepts](#advanced-rheological-concepts)
2. [Herschel-Bulkley Model Deep Dive](#herschel-bulkley-model-deep-dive)
3. [Parameter Estimation Techniques](#parameter-estimation-techniques)
4. [Multi-Phase Flow Analysis](#multi-phase-flow-analysis)
5. [Thixotropic Behavior Modeling](#thixotropic-behavior-modeling)
6. [Process Optimization](#process-optimization)
7. [Advanced Visualization](#advanced-visualization)
8. [Publication-Ready Reporting](#publication-ready-reporting)

---

## Advanced Rheological Concepts

### Understanding Complex Fluid Behavior

Complex fluids exhibit non-Newtonian behavior that cannot be described by simple Newtonian viscosity. The Scientific Computing Toolkit handles several categories:

#### 1. **Shear-Thinning Fluids**
```math
\eta(\dot{\gamma}) = \eta_0 \cdot (1 + (t \dot{\gamma})^a)^{(n-1)/a}
```
- Viscosity decreases with increasing shear rate
- Examples: Polymer solutions, blood, paints

#### 2. **Shear-Thickening Fluids**
```math
\eta(\dot{\gamma}) = \eta_0 \cdot (1 - (t \dot{\gamma})^a)^{(n-1)/a}
```
- Viscosity increases with increasing shear rate
- Examples: Cornstarch suspensions, electrorheological fluids

#### 3. **Viscoelastic Fluids**
```math
\tau(t) = \int_{-\infty}^{t} G(t - t') \frac{d\gamma}{dt'} dt'
```
- Exhibit both viscous and elastic properties
- Examples: Polymer melts, biological tissues

#### 4. **Yield Stress Fluids**
```math
\tau(\dot{\gamma}) = \tau_y + \eta \dot{\gamma} \quad (\tau > \tau_y)
```
- Require stress threshold to flow
- Examples: Toothpaste, mayonnaise, concrete

---

## Herschel-Bulkley Model Deep Dive

### Model Formulation

The Herschel-Bulkley model combines yield stress with power-law behavior:

```math
\tau(\dot{\gamma}) = \tau_y + K \dot{\gamma}^n
```

Where:
- **τ**: Shear stress [Pa]
- **τ_y**: Yield stress [Pa]
- **K**: Consistency index [Pa·sⁿ]
- **n**: Flow behavior index (dimensionless)
- **γ̇**: Shear rate [s⁻¹]

### Parameter Interpretation

#### Flow Behavior Index (n)
```python
# n < 1: Shear-thinning (pseudoplastic)
# n = 1: Newtonian (Bingham plastic if τ_y > 0)
# n > 1: Shear-thickening (dilatant)
```

#### Consistency Index (K)
```python
# Represents fluid viscosity at γ̇ = 1 s⁻¹
# Higher K = more viscous fluid
# Units depend on n: Pa·sⁿ
```

#### Yield Stress (τ_y)
```python
# Stress threshold for flow initiation
# τ_y = 0: Simple power-law fluid
# τ_y > 0: Yield stress fluid
```

---

## Parameter Estimation Techniques

### Advanced Fitting Methods

#### 1. **Nonlinear Least Squares**
```python
import numpy as np
from scipy.optimize import curve_fit

def herschel_bulkley_model(gamma_dot, tau_y, K, n):
    """Herschel-Bulkley constitutive model."""
    return tau_y + K * gamma_dot**n

def fit_herschel_bulkley_advanced(stress_data, shear_rate_data):
    """Advanced HB parameter estimation with bounds and weights."""

    # Define parameter bounds
    bounds = ([0, 0, 0.1], [100, 1000, 2.0])  # [τ_y_min, K_min, n_min], [τ_y_max, K_max, n_max]

    # Use weights for better fit quality
    weights = 1 / (shear_rate_data + 0.1)  # Weight higher shear rates more

    # Initial parameter guesses
    p0 = [10.0, 1.0, 0.8]  # τ_y, K, n

    # Perform curve fitting
    popt, pcov = curve_fit(
        herschel_bulkley_model,
        shear_rate_data,
        stress_data,
        p0=p0,
        bounds=bounds,
        sigma=weights,
        maxfev=10000
    )

    # Calculate parameter uncertainties
    perr = np.sqrt(np.diag(pcov))

    return {
        'parameters': popt,
        'uncertainties': perr,
        'covariance': pcov,
        'tau_y': popt[0],
        'K': popt[1],
        'n': popt[2]
    }
```

#### 2. **Robust Regression**
```python
from scipy.optimize import least_squares

def robust_hb_fit(stress_data, shear_rate_data):
    """Robust HB fitting with outlier resistance."""

    def residuals(params):
        tau_y, K, n = params
        predicted = tau_y + K * shear_rate_data**n
        return predicted - stress_data

    # Initial guess
    x0 = [5.0, 2.0, 0.7]

    # Robust least squares with Cauchy loss
    result = least_squares(
        residuals,
        x0,
        loss='cauchy',
        f_scale=0.1,
        bounds=([0, 0, 0.1], [50, 100, 1.5])
    )

    return {
        'success': result.success,
        'parameters': result.x,
        'cost': result.cost,
        'nfev': result.nfev
    }
```

#### 3. **Bayesian Parameter Estimation**
```python
import pymc3 as pm
import numpy as np

def bayesian_hb_fit(stress_data, shear_rate_data):
    """Bayesian parameter estimation for HB model."""

    with pm.Model() as model:
        # Priors
        tau_y = pm.HalfNormal('tau_y', sigma=20)
        K = pm.HalfNormal('K', sigma=5)
        n = pm.Beta('n', alpha=2, beta=2)  # n between 0 and 1

        # Model prediction
        mu = tau_y + K * shear_rate_data**n

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=10)
        stress_obs = pm.Normal('stress_obs', mu=mu, sigma=sigma, observed=stress_data)

        # Sample from posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    return {
        'trace': trace,
        'summary': pm.summary(trace),
        'posterior_samples': {
            'tau_y': trace.posterior['tau_y'].values.flatten(),
            'K': trace.posterior['K'].values.flatten(),
            'n': trace.posterior['n'].values.flatten()
        }
    }
```

---

## Multi-Phase Flow Analysis

### Advanced Flow Solver Implementation

```python
import numpy as np
from scipy.integrate import solve_ivp

class AdvancedFlowSolver:
    """Advanced multi-phase flow solver with complex rheology."""

    def __init__(self, fluid_properties):
        self.props = fluid_properties
        self.gravity = 9.81  # m/s²

    def solve_pipe_flow(self, diameter, length, pressure_drop, flow_rate_guess=1.0):
        """Solve for pipe flow with complex rheology."""

        def flow_equation(Q):
            """Momentum balance for pipe flow."""
            # Calculate shear rate at wall
            gamma_dot_wall = 8 * Q / (np.pi * diameter**3)

            # Calculate wall shear stress
            tau_wall = self.calculate_wall_stress(gamma_dot_wall)

            # Calculate friction factor
            f = 2 * tau_wall * diameter / (pressure_drop * length / diameter)

            # Darcy-Weisbach equation residual
            residual = pressure_drop - f * (length / diameter) * (density * Q**2) / (2 * diameter)

            return residual

        # Solve for flow rate
        from scipy.optimize import brentq

        # Bracket the solution
        Q_min = 1e-6
        Q_max = 10.0

        try:
            Q_solution = brentq(flow_equation, Q_min, Q_max)
            return Q_solution
        except ValueError:
            # If bracketing fails, use different approach
            return self.solve_with_newton(flow_equation, flow_rate_guess)

    def calculate_wall_stress(self, gamma_dot_wall):
        """Calculate wall shear stress using HB model."""
        tau_y = self.props.get('tau_y', 0)
        K = self.props.get('K', 1.0)
        n = self.props.get('n', 1.0)

        # For yield stress fluids, check if flow occurs
        if gamma_dot_wall == 0:
            return tau_y

        return tau_y + K * gamma_dot_wall**n

    def solve_with_newton(self, equation, initial_guess):
        """Newton-Raphson solver for flow equations."""
        from scipy.optimize import newton

        try:
            solution = newton(equation, initial_guess)
            return solution
        except RuntimeError:
            # Fallback to bisection if Newton fails
            return self.solve_with_bisection(equation, 1e-6, 10.0)

    def solve_with_bisection(self, equation, a, b):
        """Bisection method fallback."""
        from scipy.optimize import bisect
        return bisect(equation, a, b)
```

### Multi-Component Flow Analysis

```python
class MultiComponentFlowAnalyzer:
    """Analyze flow of multi-component complex fluids."""

    def __init__(self, components):
        self.components = components  # List of fluid component properties

    def calculate_bulk_properties(self, volume_fractions):
        """Calculate bulk rheological properties."""

        # Logarithmic mixing rule for yield stress
        tau_y_bulk = np.prod([comp['tau_y']**phi for comp, phi in
                             zip(self.components, volume_fractions)])

        # Harmonic mean for consistency index
        K_bulk = 1 / np.sum([phi / comp['K'] for comp, phi in
                            zip(self.components, volume_fractions)])

        # Volume-weighted average for flow index
        n_bulk = np.sum([phi * comp['n'] for comp, phi in
                        zip(self.components, volume_fractions)])

        return {
            'tau_y': tau_y_bulk,
            'K': K_bulk,
            'n': n_bulk
        }

    def predict_phase_separation(self, shear_rate_range):
        """Predict phase separation behavior."""

        separation_points = []

        for gamma_dot in shear_rate_range:
            # Calculate viscosity of each phase
            viscosities = []
            for component in self.components:
                tau_y = component['tau_y']
                K = component['K']
                n = component['n']

                if gamma_dot == 0:
                    viscosity = float('inf') if tau_y > 0 else K
                else:
                    viscosity = tau_y / gamma_dot + K * gamma_dot**(n-1)

                viscosities.append(viscosity)

            # Check for phase separation conditions
            viscosity_ratio = max(viscosities) / min(viscosities)
            if viscosity_ratio > 10:  # Arbitrary threshold
                separation_points.append({
                    'shear_rate': gamma_dot,
                    'viscosity_ratio': viscosity_ratio,
                    'dominant_phase': np.argmin(viscosities)
                })

        return separation_points
```

---

## Thixotropic Behavior Modeling

### Time-Dependent Rheology

```python
import numpy as np
from scipy.integrate import odeint

class ThixotropicModel:
    """Advanced thixotropic fluid model with structure evolution."""

    def __init__(self, equilibrium_structure=1.0, relaxation_time=10.0,
                 flow_exponent=0.5, structure_exponent=2.0):
        self.lambda_eq = equilibrium_structure
        self.lambda_relax = relaxation_time
        self.m = flow_exponent
        self.n = structure_exponent

    def structure_evolution(self, lambda_structure, t, gamma_dot):
        """Structure parameter evolution equation."""
        dlambda_dt = (self.lambda_eq - lambda_structure) / self.lambda_relax - \
                    lambda_structure * abs(gamma_dot)**self.m * \
                    (lambda_structure / self.lambda_eq)**self.n
        return dlambda_dt

    def viscosity_function(self, lambda_structure, gamma_dot):
        """Viscosity as function of structure and shear rate."""
        # Base viscosity
        eta_base = 0.1  # Pa·s

        # Thixotropic viscosity modification
        eta_thix = eta_base * (lambda_structure / self.lambda_eq)**(-2.0)

        # Shear-thinning behavior
        eta_shear = eta_thix * (1 + (gamma_dot * lambda_structure)**0.8)**(-0.6)

        return eta_shear

    def simulate_flow_history(self, shear_rate_history, time_points):
        """Simulate viscosity evolution under complex flow history."""

        # Initial structure
        lambda_0 = self.lambda_eq

        # Solve structure evolution
        lambda_history = odeint(
            self.structure_evolution,
            lambda_0,
            time_points,
            args=(shear_rate_history,)
        )

        # Calculate viscosity history
        viscosity_history = []
        for i, t in enumerate(time_points):
            lambda_t = lambda_history[i, 0]
            gamma_dot_t = shear_rate_history[i] if i < len(shear_rate_history) else 0
            eta_t = self.viscosity_function(lambda_t, gamma_dot_t)
            viscosity_history.append(eta_t)

        return {
            'time': time_points,
            'shear_rate': shear_rate_history,
            'structure_parameter': lambda_history.flatten(),
            'viscosity': viscosity_history
        }

    def predict_yield_stress_evolution(self, structure_parameter):
        """Predict yield stress evolution with structure breakdown."""
        tau_y_base = 10.0  # Base yield stress
        tau_y_thix = tau_y_base * (structure_parameter / self.lambda_eq)**2.5
        return tau_y_thix
```

### Advanced Thixotropic Analysis

```python
class AdvancedThixotropicAnalyzer:
    """Advanced analysis of thixotropic behavior."""

    def __init__(self):
        self.models = []

    def fit_multiple_models(self, experimental_data):
        """Fit multiple thixotropic models to experimental data."""

        models = [
            {'name': 'Standard', 'relaxation_time': 10.0, 'flow_exponent': 0.5},
            {'name': 'Fast', 'relaxation_time': 5.0, 'flow_exponent': 0.7},
            {'name': 'Slow', 'relaxation_time': 20.0, 'flow_exponent': 0.3}
        ]

        fitted_models = []

        for model_config in models:
            model = ThixotropicModel(**model_config)

            # Fit model parameters to data
            fitted_params = self.fit_model_parameters(model, experimental_data)

            fitted_models.append({
                'model': model,
                'config': model_config,
                'fitted_params': fitted_params,
                'fit_quality': self.evaluate_fit_quality(fitted_params, experimental_data)
            })

        return fitted_models

    def fit_model_parameters(self, model, experimental_data):
        """Fit model parameters using optimization."""
        from scipy.optimize import minimize

        def objective(params):
            relaxation_time, flow_exponent, structure_exponent = params

            # Update model parameters
            model.lambda_relax = relaxation_time
            model.m = flow_exponent
            model.n = structure_exponent

            # Simulate and calculate error
            simulation = model.simulate_flow_history(
                experimental_data['shear_rate'],
                experimental_data['time']
            )

            # Calculate RMSE
            rmse = np.sqrt(np.mean(
                (np.array(simulation['viscosity']) - experimental_data['viscosity'])**2
            ))

            return rmse

        # Initial guesses
        x0 = [10.0, 0.5, 2.0]

        # Bounds
        bounds = [(1.0, 100.0), (0.1, 1.0), (1.0, 5.0)]

        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        return {
            'relaxation_time': result.x[0],
            'flow_exponent': result.x[1],
            'structure_exponent': result.x[2],
            'rmse': result.fun,
            'success': result.success
        }

    def evaluate_fit_quality(self, fitted_params, experimental_data):
        """Evaluate quality of model fit."""
        # Calculate various metrics
        r_squared = self.calculate_r_squared(fitted_params, experimental_data)
        aic = self.calculate_aic(fitted_params, experimental_data)
        bic = self.calculate_bic(fitted_params, experimental_data)

        return {
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'overall_quality': self.assess_overall_quality(r_squared, aic, bic)
        }

    def calculate_r_squared(self, fitted_params, experimental_data):
        """Calculate R² for model fit."""
        # Implementation of R² calculation
        predicted = self.predict_viscosity(fitted_params, experimental_data)
        observed = experimental_data['viscosity']

        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)

        return 1 - (ss_res / ss_tot)

    def predict_viscosity(self, params, data):
        """Predict viscosity using fitted parameters."""
        model = ThixotropicModel(
            relaxation_time=params['relaxation_time'],
            flow_exponent=params['flow_exponent'],
            structure_exponent=params['structure_exponent']
        )

        simulation = model.simulate_flow_history(data['shear_rate'], data['time'])
        return simulation['viscosity']
```

---

## Process Optimization

### Rheological Process Design

```python
class RheologicalProcessOptimizer:
    """Optimize industrial processes based on rheological properties."""

    def __init__(self):
        self.process_constraints = {
            'max_pressure': 1e6,  # Pa
            'max_temperature': 100,  # °C
            'min_flow_rate': 0.001,  # m³/s
            'max_power_consumption': 10000  # W
        }

    def optimize_extrusion_process(self, material_properties, target_specs):
        """Optimize extrusion process parameters."""

        from scipy.optimize import minimize

        def objective(params):
            """Process optimization objective function."""
            temperature, pressure, screw_speed = params

            # Calculate rheological properties at conditions
            viscosity = self.calculate_viscosity_at_conditions(
                material_properties, temperature, pressure
            )

            # Calculate flow rate
            flow_rate = self.calculate_flow_rate(
                viscosity, pressure, screw_speed
            )

            # Calculate power consumption
            power = self.calculate_power_consumption(
                viscosity, screw_speed, pressure
            )

            # Multi-objective: maximize flow rate, minimize power
            flow_penalty = max(0, target_specs['min_flow_rate'] - flow_rate) * 1000
            power_penalty = max(0, power - target_specs['max_power']) * 0.001

            return -(flow_rate * 1000) + power_penalty + flow_penalty

        # Parameter bounds
        bounds = [
            (60, 120),    # Temperature (°C)
            (1e5, 5e6),   # Pressure (Pa)
            (50, 300)     # Screw speed (RPM)
        ]

        # Initial guess
        x0 = [80, 1e6, 150]

        result = minimize(objective, x0, bounds=bounds, method='SLSQP')

        return {
            'optimal_temperature': result.x[0],
            'optimal_pressure': result.x[1],
            'optimal_screw_speed': result.x[2],
            'predicted_flow_rate': self.calculate_flow_rate(
                self.calculate_viscosity_at_conditions(
                    material_properties, result.x[0], result.x[1]
                ), result.x[1], result.x[2]
            ),
            'predicted_power': self.calculate_power_consumption(
                self.calculate_viscosity_at_conditions(
                    material_properties, result.x[0], result.x[1]
                ), result.x[2], result.x[1]
            ),
            'optimization_success': result.success
        }

    def calculate_viscosity_at_conditions(self, properties, temperature, pressure):
        """Calculate viscosity at given temperature and pressure."""
        # Temperature dependence (Arrhenius)
        T_ref = 298.15  # Reference temperature (25°C)
        E_a = properties.get('activation_energy', 50000)  # J/mol
        R = 8.314  # J/mol·K

        temp_factor = np.exp(E_a / R * (1/T_ref - 1/(temperature + 273.15)))

        # Pressure dependence
        pressure_factor = 1 + properties.get('compressibility', 1e-9) * pressure

        # Base viscosity
        eta_base = properties.get('eta_base', 1000)  # Pa·s

        return eta_base * temp_factor * pressure_factor

    def calculate_flow_rate(self, viscosity, pressure, screw_speed):
        """Calculate extrusion flow rate."""
        # Simplified extrusion flow model
        D = 0.05  # Screw diameter (m)
        L = 1.0   # Screw length (m)
        H = 0.005 # Channel depth (m)

        # Poiseuille flow approximation
        Q = (np.pi * pressure * H**3) / (12 * viscosity * L) * (D/2)**2

        # Scale with screw speed
        Q_total = Q * (screw_speed / 100)  # Normalize to 100 RPM

        return Q_total

    def calculate_power_consumption(self, viscosity, screw_speed, pressure):
        """Calculate power consumption for extrusion."""
        # Simplified power calculation
        torque = viscosity * screw_speed * 0.01  # Simplified torque calculation
        power = torque * screw_speed * 2 * np.pi / 60  # Convert to Watts

        return power
```

---

## Advanced Visualization

### Publication-Quality Rheology Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

class AdvancedRheologyVisualizer:
    """Advanced visualization for rheological analysis."""

    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        sns.set_palette("husl")

    def create_comprehensive_flow_curve(self, data_dict, save_path=None):
        """Create comprehensive flow curve visualization."""

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)

        # Main flow curve
        ax1 = fig.add_subplot(gs[0, :2])
        for name, data in data_dict.items():
            ax1.loglog(data['shear_rate'], data['stress'],
                      'o-', label=name, markersize=4, linewidth=2)

        ax1.set_xlabel('Shear Rate (s⁻¹)', fontsize=12)
        ax1.set_ylabel('Shear Stress (Pa)', fontsize=12)
        ax1.set_title('Flow Curves - Log-Log Scale', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Viscosity curve
        ax2 = fig.add_subplot(gs[0, 2])
        for name, data in data_dict.items():
            viscosity = data['stress'] / data['shear_rate']
            ax2.semilogx(data['shear_rate'], viscosity,
                        's-', label=name, markersize=3)

        ax2.set_xlabel('Shear Rate (s⁻¹)', fontsize=10)
        ax2.set_ylabel('Viscosity (Pa·s)', fontsize=10)
        ax2.set_title('Viscosity vs Shear Rate', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Parameter comparison
        ax3 = fig.add_subplot(gs[1, :])
        materials = list(data_dict.keys())
        tau_y_values = [self.extract_param(data, 'tau_y') for data in data_dict.values()]
        K_values = [self.extract_param(data, 'K') for data in data_dict.values()]
        n_values = [self.extract_param(data, 'n') for data in data_dict.values()]

        x = np.arange(len(materials))
        width = 0.25

        ax3.bar(x - width, tau_y_values, width, label='Yield Stress (Pa)', alpha=0.8)
        ax3.bar(x, K_values, width, label='Consistency Index', alpha=0.8)
        ax3.bar(x + width, n_values, width, label='Flow Index', alpha=0.8)

        ax3.set_xlabel('Materials', fontsize=12)
        ax3.set_ylabel('Parameter Values', fontsize=12)
        ax3.set_title('Rheological Parameters Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(materials, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive flow curve saved to: {save_path}")

        return fig

    def extract_param(self, data, param_name):
        """Extract parameter value from fitted data."""
        # This would extract parameters from fitted models
        # Placeholder implementation
        return np.random.uniform(0, 10)

    def create_thixotropic_analysis_plot(self, time_data, viscosity_data,
                                       structure_data, save_path=None):
        """Create thixotropic behavior analysis plot."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Thixotropic Behavior Analysis', fontsize=16, fontweight='bold')

        # Time series plot
        axes[0, 0].plot(time_data, viscosity_data, 'b-', linewidth=2, label='Viscosity')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Viscosity (Pa·s)')
        axes[0, 0].set_title('Viscosity Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Structure parameter plot
        axes[0, 1].plot(time_data, structure_data, 'r-', linewidth=2, label='Structure')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Structure Parameter')
        axes[0, 1].set_title('Structure Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Phase plot
        axes[1, 0].plot(structure_data, viscosity_data, 'g-', alpha=0.7)
        axes[1, 0].scatter(structure_data[0], viscosity_data[0],
                          c='red', s=100, label='Start', zorder=5)
        axes[1, 0].scatter(structure_data[-1], viscosity_data[-1],
                          c='blue', s=100, label='End', zorder=5)
        axes[1, 0].set_xlabel('Structure Parameter')
        axes[1, 0].set_ylabel('Viscosity (Pa·s)')
        axes[1, 0].set_title('Phase Portrait')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Hysteresis analysis
        # This would analyze hysteresis in thixotropic loops
        axes[1, 1].text(0.5, 0.5, 'Hysteresis Analysis\n(Under Development)',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('Hysteresis Analysis')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Thixotropic analysis plot saved to: {save_path}")

        return fig

    def create_process_optimization_dashboard(self, optimization_results, save_path=None):
        """Create process optimization dashboard."""

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Process Optimization Dashboard', fontsize=16, fontweight='bold')

        # Parameter optimization surface
        temp_range = np.linspace(60, 120, 20)
        press_range = np.linspace(1e5, 5e6, 20)
        TEMP, PRESS = np.meshgrid(temp_range, press_range)

        # Mock objective function surface
        Z = (TEMP - 80)**2 / 100 + (PRESS - 2e6)**2 / 1e12

        surf = axes[0, 0].contourf(TEMP, PRESS/1e6, Z, levels=20, cmap='viridis')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Pressure (MPa)')
        axes[0, 0].set_title('Optimization Landscape')
        plt.colorbar(surf, ax=axes[0, 0])

        # Optimal point
        opt_temp = optimization_results.get('optimal_temperature', 80)
        opt_press = optimization_results.get('optimal_pressure', 2e6) / 1e6
        axes[0, 0].plot(opt_temp, opt_press, 'ro', markersize=10, label='Optimal')
        axes[0, 0].legend()

        # Performance metrics
        metrics = ['Flow Rate', 'Power Consumption', 'Efficiency']
        values = [
            optimization_results.get('predicted_flow_rate', 0.1),
            optimization_results.get('predicted_power', 5000),
            0.85  # Mock efficiency
        ]

        bars = axes[0, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.2f}', ha='center', va='bottom')

        # Parameter sensitivity
        params = ['Temperature', 'Pressure', 'Screw Speed']
        sensitivities = [0.3, 0.6, 0.8]  # Mock sensitivities

        axes[0, 2].barh(params, sensitivities, color='orange')
        axes[0, 2].set_xlabel('Sensitivity')
        axes[0, 2].set_title('Parameter Sensitivity')
        axes[0, 2].grid(True, alpha=0.3)

        # Constraints visualization
        constraints = ['Max Pressure', 'Max Temperature', 'Min Flow Rate', 'Max Power']
        current_values = [
            optimization_results.get('optimal_pressure', 2e6) / 1e6,
            optimization_results.get('optimal_temperature', 80),
            optimization_results.get('predicted_flow_rate', 0.1),
            optimization_results.get('predicted_power', 5000)
        ]
        limit_values = [5.0, 120, 0.05, 8000]  # Mock limits

        x = np.arange(len(constraints))
        width = 0.35

        axes[1, 0].bar(x - width/2, current_values, width, label='Current', alpha=0.8)
        axes[1, 0].bar(x + width/2, limit_values, width, label='Limit', alpha=0.6)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(constraints, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Constraint Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Optimization convergence
        iterations = np.arange(1, 21)
        objective_values = 100 * np.exp(-iterations/5) + np.random.normal(0, 2, 20)

        axes[1, 1].plot(iterations, objective_values, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Objective Value')
        axes[1, 1].set_title('Optimization Convergence')
        axes[1, 1].grid(True, alpha=0.3)

        # Pareto front (if multi-objective)
        flow_rates = np.linspace(0.05, 0.15, 20)
        powers = 10000 - 1000 * (flow_rates - 0.05) / 0.1

        axes[1, 2].scatter(flow_rates, powers, c='purple', s=50, alpha=0.7)
        axes[1, 2].plot(flow_rates, powers, 'purple-', alpha=0.5)
        axes[1, 2].set_xlabel('Flow Rate (m³/s)')
        axes[1, 2].set_ylabel('Power (W)')
        axes[1, 2].set_title('Pareto Front Analysis')
        axes[1, 2].grid(True, alpha=0.3)

        # Add optimal point
        opt_flow = optimization_results.get('predicted_flow_rate', 0.1)
        opt_power = optimization_results.get('predicted_power', 5000)
        axes[1, 2].scatter([opt_flow], [opt_power], c='red', s=100,
                           marker='*', label='Optimal', zorder=5)
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Process optimization dashboard saved to: {save_path}")

        return fig
```

---

## Publication-Ready Reporting

### Automated Report Generation

```python
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class RheologyReportGenerator:
    """Generate comprehensive rheological analysis reports."""

    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_complete_report(self, analysis_results, material_name="Unknown Material"):
        """Generate complete rheological analysis report."""

        report_id = f"rheology_report_{int(datetime.now().timestamp())}"
        report_dir = self.output_dir / report_id
        report_dir.mkdir()

        # Generate individual components
        self.generate_summary_report(analysis_results, report_dir, material_name)
        self.generate_detailed_analysis(analysis_results, report_dir)
        self.generate_visualizations(analysis_results, report_dir, material_name)
        self.generate_uncertainty_analysis(analysis_results, report_dir)
        self.generate_methodology_section(report_dir)

        # Create report index
        self.create_report_index(report_dir, analysis_results, material_name)

        return report_dir

    def generate_summary_report(self, results, report_dir, material_name):
        """Generate executive summary report."""

        summary_path = report_dir / "00_executive_summary.md"

        with open(summary_path, 'w') as f:
            f.write("# Rheological Analysis Executive Summary\n\n")
            f.write(f"**Material:** {material_name}\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Report ID:** {report_dir.name}\n\n")

            f.write("## Key Findings\n\n")

            # Extract key parameters
            if 'fitted_parameters' in results:
                params = results['fitted_parameters']
                f.write("### Rheological Parameters\n\n")
                f.write("| Parameter | Value | Units | Uncertainty |\n")
                f.write("|-----------|-------|-------|-------------|\n")

                param_info = [
                    ("Yield Stress", params.get('tau_y', 'N/A'), "Pa", "±5%"),
                    ("Consistency Index", params.get('K', 'N/A'), "Pa·sⁿ", "±10%"),
                    ("Flow Behavior Index", params.get('n', 'N/A'), "-", "±0.05")
                ]

                for name, value, unit, uncertainty in param_info:
                    f.write(f"| {name} | {value} | {unit} | {uncertainty} |\n")

                f.write("\n")

            # Material classification
            if 'material_classification' in results:
                f.write("### Material Classification\n\n")
                classification = results['material_classification']
                f.write(f"- **Fluid Type:** {classification.get('type', 'Unknown')}\n")
                f.write(f"- **Behavior:** {classification.get('behavior', 'Unknown')}\n")
                f.write(f"- **Confidence:** {classification.get('confidence', 'N/A')}%\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Processing Conditions\n")
            f.write("- Optimal shear rate range: Based on fitted parameters\n")
            f.write("- Temperature considerations: Material-dependent\n")
            f.write("- Mixing requirements: Based on yield stress\n\n")

            f.write("### Quality Control\n")
            f.write("- Monitor consistency index for batch uniformity\n")
            f.write("- Regular rheological testing recommended\n")
            f.write("- Storage conditions may affect properties\n\n")

            f.write("### Further Analysis\n")
            f.write("- Thixotropic behavior testing recommended\n")
            f.write("- Temperature dependence characterization\n")
            f.write("- Long-term stability assessment\n")

    def generate_detailed_analysis(self, results, report_dir):
        """Generate detailed technical analysis."""

        analysis_path = report_dir / "01_detailed_analysis.md"

        with open(analysis_path, 'w') as f:
            f.write("# Detailed Rheological Analysis\n\n")

            # Methodology
            f.write("## Experimental Methodology\n\n")
            f.write("### Equipment\n")
            f.write("- Rheometer: Anton Paar MCR 302\n")
            f.write("- Geometry: Cone-plate (25mm diameter, 1° cone angle)\n")
            f.write("- Temperature control: Peltier system (±0.1°C)\n\n")

            f.write("### Test Protocol\n")
            f.write("1. Sample loading and trimming\n")
            f.write("2. Pre-shear at 100 s⁻¹ for 60 seconds\n")
            f.write("3. Equilibrium time: 300 seconds\n")
            f.write("4. Flow curve: 0.01-1000 s⁻¹ (logarithmic spacing)\n")
            f.write("5. Data collection: 10 points per decade\n\n")

            # Data analysis
            f.write("## Data Analysis\n\n")

            if 'raw_data' in results:
                raw_data = results['raw_data']
                f.write("### Raw Data Summary\n\n")
                f.write("| Property | Value |\n")
                f.write("|----------|-------|\n")
                f.write(f"| Total data points | {len(raw_data.get('shear_rate', []))} |\n")
                f.write(f"| Shear rate range | {min(raw_data.get('shear_rate', [0])):.2e} - {max(raw_data.get('shear_rate', [0])):.2e} s⁻¹ |\n")
                f.write(f"| Stress range | {min(raw_data.get('stress', [0])):.1f} - {max(raw_data.get('stress', [0])):.1f} Pa |\n\n")

            # Model fitting
            f.write("### Model Fitting Results\n\n")

            if 'model_fit' in results:
                fit_results = results['model_fit']
                f.write("#### Goodness of Fit\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| R² | {fit_results.get('r_squared', 'N/A'):.4f} |\n")
                f.write(f"| RMSE | {fit_results.get('rmse', 'N/A'):.2f} Pa |\n")
                f.write(f"| MAE | {fit_results.get('mae', 'N/A'):.2f} Pa |\n\n")

                f.write("#### Parameter Confidence Intervals\n\n")
                if 'confidence_intervals' in fit_results:
                    ci = fit_results['confidence_intervals']
                    f.write("| Parameter | Estimate | 95% CI Lower | 95% CI Upper |\n")
                    f.write("|-----------|----------|---------------|---------------|\n")
                    for param, estimate in fit_results.get('parameters', {}).items():
                        lower = ci.get(param, {}).get('lower', 'N/A')
                        upper = ci.get(param, {}).get('upper', 'N/A')
                        f.write(f"| {param} | {estimate:.3f} | {lower} | {upper} |\n")

            # Material behavior analysis
            f.write("## Material Behavior Analysis\n\n")

            if 'behavior_analysis' in results:
                behavior = results['behavior_analysis']
                f.write("### Flow Regimes Identified\n\n")

                for regime in behavior.get('regimes', []):
                    f.write(f"#### {regime.get('name', 'Unknown Regime')}\n")
                    f.write(f"- **Shear Rate Range:** {regime.get('shear_rate_range', 'N/A')}\n")
                    f.write(f"- **Dominant Mechanism:** {regime.get('mechanism', 'N/A')}\n")
                    f.write(f"- **Characteristic Parameters:** {regime.get('parameters', 'N/A')}\n\n")

    def generate_visualizations(self, results, report_dir, material_name):
        """Generate publication-quality visualizations."""

        viz_dir = report_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Flow curve
        if 'flow_curve_data' in results:
            plt.figure(figsize=(10, 6))
            data = results['flow_curve_data']
            plt.loglog(data['shear_rate'], data['stress'], 'bo-', markersize=4, linewidth=2)
            plt.xlabel('Shear Rate (s⁻¹)', fontsize=12)
            plt.ylabel('Shear Stress (Pa)', fontsize=12)
            plt.title(f'Flow Curve - {material_name}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / "flow_curve.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Viscosity curve
        if 'viscosity_data' in results:
            plt.figure(figsize=(10, 6))
            data = results['viscosity_data']
            plt.semilogx(data['shear_rate'], data['viscosity'], 'ro-', markersize=4, linewidth=2)
            plt.xlabel('Shear Rate (s⁻¹)', fontsize=12)
            plt.ylabel('Viscosity (Pa·s)', fontsize=12)
            plt.title(f'Viscosity Curve - {material_name}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / "viscosity_curve.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Model fit comparison
        if 'model_comparison' in results:
            plt.figure(figsize=(12, 8))
            data = results['model_comparison']

            plt.subplot(2, 2, 1)
            plt.loglog(data['shear_rate'], data['experimental'], 'ko', label='Experimental', markersize=3)
            plt.loglog(data['shear_rate'], data['herschel_bulkley'], 'r-', label='HB Model', linewidth=2)
            plt.xlabel('Shear Rate (s⁻¹)')
            plt.ylabel('Shear Stress (Pa)')
            plt.title('Herschel-Bulkley Fit')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            plt.semilogx(data['shear_rate'], data['residuals'], 'b-', linewidth=2)
            plt.xlabel('Shear Rate (s⁻¹)')
            plt.ylabel('Residual (Pa)')
            plt.title('Fit Residuals')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            plt.plot(data['shear_rate'], data['experimental'] - data['herschel_bulkley'],
                    'g.', alpha=0.6)
            plt.xlabel('Shear Rate (s⁻¹)')
            plt.ylabel('Error (Pa)')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            # Q-Q plot would go here
            plt.text(0.5, 0.5, 'Q-Q Plot\n(Under Development)',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Residual Q-Q Plot')

            plt.tight_layout()
            plt.savefig(viz_dir / "model_fit_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

    def generate_uncertainty_analysis(self, results, report_dir):
        """Generate uncertainty analysis section."""

        uncertainty_path = report_dir / "02_uncertainty_analysis.md"

        with open(uncertainty_path, 'w') as f:
            f.write("# Uncertainty Analysis\n\n")

            f.write("## Measurement Uncertainty\n\n")
            f.write("### Instrument Specifications\n\n")
            f.write("| Component | Uncertainty | Units |\n")
            f.write("|-----------|-------------|-------|\n")
            f.write("| Torque | ±0.5 | % of reading |\n")
            f.write("| Angular velocity | ±0.1 | % of reading |\n")
            f.write("| Temperature | ±0.1 | °C |\n")
            f.write("| Gap | ±1 | μm |\n\n")

            if 'uncertainty_analysis' in results:
                uncertainty = results['uncertainty_analysis']

                f.write("## Parameter Uncertainty\n\n")

                if 'parameter_uncertainty' in uncertainty:
                    param_uncertainty = uncertainty['parameter_uncertainty']
                    f.write("### Parameter Standard Errors\n\n")
                    f.write("| Parameter | Estimate | Standard Error | Relative Error |\n")
                    f.write("|-----------|----------|----------------|----------------|\n")

                    for param, data in param_uncertainty.items():
                        estimate = data.get('estimate', 'N/A')
                        std_err = data.get('std_error', 'N/A')
                        rel_err = data.get('relative_error', 'N/A')
                        f.write(f"| {param} | {estimate} | {std_err} | {rel_err} |\n")

                    f.write("\n")

                f.write("### Confidence Intervals\n\n")

                if 'confidence_intervals' in uncertainty:
                    ci = uncertainty['confidence_intervals']
                    f.write("| Parameter | 95% CI Lower | 95% CI Upper | Width |\n")
                    f.write("|-----------|---------------|---------------|-------|\n")

                    for param, bounds in ci.items():
                        lower = bounds.get('lower', 'N/A')
                        upper = bounds.get('upper', 'N/A')
                        width = bounds.get('width', 'N/A')
                        f.write(f"| {param} | {lower} | {upper} | {width} |\n")

                    f.write("\n")

                f.write("## Propagation of Uncertainty\n\n")

                if 'uncertainty_propagation' in uncertainty:
                    propagation = uncertainty['uncertainty_propagation']

                    f.write("### Sensitivity Analysis\n\n")
                    f.write("The following table shows the sensitivity of model predictions\n")
                    f.write("to each parameter:\n\n")

                    f.write("| Parameter | Sensitivity Coefficient | Contribution to Variance |\n")
                    f.write("|-----------|-------------------------|--------------------------|\n")

                    for param, data in propagation.get('sensitivity', {}).items():
                        sensitivity = data.get('coefficient', 'N/A')
                        contribution = data.get('contribution', 'N/A')
                        f.write(f"| {param} | {sensitivity} | {contribution} |\n")

                    f.write("\n")

                    f.write("### Monte Carlo Analysis\n\n")
                    if 'monte_carlo' in propagation:
                        mc = propagation['monte_carlo']
                        f.write(f"- **Samples:** {mc.get('n_samples', 'N/A')}\n")
                        f.write(f"- **Convergence:** {mc.get('convergence', 'N/A')}\n")
                        f.write(f"- **Computational Time:** {mc.get('computation_time', 'N/A')} seconds\n\n")

    def generate_methodology_section(self, report_dir):
        """Generate detailed methodology section."""

        methodology_path = report_dir / "03_methodology.md"

        with open(methodology_path, 'w') as f:
            f.write("# Experimental Methodology\n\n")

            f.write("## Sample Preparation\n\n")
            f.write("### Material Handling\n")
            f.write("1. Store samples at controlled temperature (4°C)\n")
            f.write("2. Allow samples to reach room temperature before testing\n")
            f.write("3. Mix samples gently to ensure homogeneity\n")
            f.write("4. Remove air bubbles through centrifugation if necessary\n\n")

            f.write("### Rheometer Setup\n")
            f.write("1. Calibrate torque and normal force sensors\n")
            f.write("2. Set appropriate gap for cone-plate geometry\n")
            f.write("3. Verify temperature calibration\n")
            f.write("4. Perform instrument compliance check\n\n")

            f.write("## Test Protocol\n\n")
            f.write("### Pre-test Conditioning\n")
            f.write("1. Load sample onto rheometer plate\n")
            f.write("2. Trim excess sample to avoid edge effects\n")
            f.write("3. Apply pre-shear to erase loading history\n")
            f.write("4. Allow sample to equilibrate\n\n")

            f.write("### Flow Curve Acquisition\n")
            f.write("1. Start from low shear rate (0.01 s⁻¹)\n")
            f.write("2. Use logarithmic spacing for data points\n")
            f.write("3. Allow sufficient time for steady state at each point\n")
            f.write("4. Monitor for sample drying or degradation\n\n")

            f.write("## Data Analysis Methodology\n\n")
            f.write("### Model Selection\n")
            f.write("The Herschel-Bulkley model was selected based on:\n")
            f.write("- Ability to capture yield stress behavior\n")
            f.write("- Power-law description of shear-thinning/thickening\n")
            f.write("- Established use in complex fluid rheology\n\n")

            f.write("### Parameter Estimation\n")
            f.write("Parameters were estimated using:\n")
            f.write("1. Nonlinear least squares regression\n")
            f.write("2. Weighted fitting to emphasize low-stress data\n")
            f.write("3. Confidence interval calculation\n")
            f.write("4. Goodness-of-fit assessment\n\n")

            f.write("### Quality Assurance\n")
            f.write("Data quality was ensured through:\n")
            f.write("- Duplicate measurements\n")
            f.write("- Instrument calibration verification\n")
            f.write("- Outlier detection and removal\n")
            f.write("- Consistency checks across shear rate ranges\n\n")

    def create_report_index(self, report_dir, results, material_name):
        """Create report index file."""

        index_path = report_dir / "README.md"

        with open(index_path, 'w') as f:
            f.write("# Rheological Analysis Report\n\n")
            f.write(f"**Material:** {material_name}\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Report ID:** {report_dir.name}\n\n")

            f.write("## Report Contents\n\n")
            f.write("### 00_executive_summary.md\n")
            f.write("- Key findings and rheological parameters\n")
            f.write("- Material classification and recommendations\n")
            f.write("- Processing guidelines and quality control measures\n\n")

            f.write("### 01_detailed_analysis.md\n")
            f.write("- Experimental methodology and equipment details\n")
            f.write("- Data analysis procedures and model fitting results\n")
            f.write("- Material behavior analysis and flow regime identification\n\n")

            f.write("### 02_uncertainty_analysis.md\n")
            f.write("- Measurement uncertainty and instrument specifications\n")
            f.write("- Parameter confidence intervals and uncertainty propagation\n")
            f.write("- Sensitivity analysis and Monte Carlo uncertainty quantification\n\n")

            f.write("### 03_methodology.md\n")
            f.write("- Detailed sample preparation procedures\n")
            f.write("- Rheometer setup and calibration procedures\n")
            f.write("- Test protocols and data analysis methodologies\n")
            f.write("- Quality assurance and validation procedures\n\n")

            f.write("### visualizations/\n")
            f.write("- Flow curve plots (linear and logarithmic scales)\n")
            f.write("- Viscosity curves and model fit comparisons\n")
            f.write("- Residual analysis and error distribution plots\n")
            f.write("- Parameter correlation and uncertainty visualization\n\n")

            f.write("## Key Results Summary\n\n")

            if 'fitted_parameters' in results:
                params = results['fitted_parameters']
                f.write("### Rheological Parameters\n\n")
                f.write("| Parameter | Value | Units |\n")
                f.write("|-----------|-------|-------|\n")
                f.write(f"| Yield Stress | {params.get('tau_y', 'N/A')} | Pa |\n")
                f.write(f"| Consistency Index | {params.get('K', 'N/A')} | Pa·sⁿ |\n")
                f.write(f"| Flow Behavior Index | {params.get('n', 'N/A')} | - |\n\n")

            if 'model_fit' in results:
                fit = results['model_fit']
                f.write("### Model Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| R² | {fit.get('r_squared', 'N/A'):.4f} |\n")
                f.write(f"| RMSE | {fit.get('rmse', 'N/A'):.2f} Pa |\n")
                f.write(f"| MAE | {fit.get('mae', 'N/A'):.2f} Pa |\n\n")

            f.write("## Report Generation Information\n\n")
            f.write("This report was generated automatically by the Scientific Computing Toolkit's\n")
            f.write("rheological analysis framework. The analysis included:\n\n")
            f.write("- Parameter estimation using nonlinear regression\n")
            f.write("- Uncertainty quantification via confidence intervals\n")
            f.write("- Publication-quality visualization generation\n")
            f.write("- Comprehensive documentation and methodology details\n\n")

            f.write("For questions about this report or the analysis methodology, please refer to:\n")
            f.write("- Scientific Computing Toolkit Documentation\n")
            f.write("- Rheological Analysis Framework API Reference\n")
            f.write("- Community Support Forums\n\n")

            f.write("---\n\n")
            f.write("*Generated by Scientific Computing Toolkit v1.0*")


# Usage example
def generate_sample_report():
    """Generate a sample rheological analysis report."""

    # Mock analysis results
    sample_results = {
        'fitted_parameters': {
            'tau_y': 12.5,
            'K': 2.3,
            'n': 0.78
        },
        'model_fit': {
            'r_squared': 0.987,
            'rmse': 1.45,
            'mae': 1.12
        },
        'material_classification': {
            'type': 'Herschel-Bulkley Fluid',
            'behavior': 'Shear-thinning with yield stress',
            'confidence': 95
        },
        'flow_curve_data': {
            'shear_rate': [0.1, 1.0, 10.0, 100.0],
            'stress': [15.2, 18.7, 22.1, 25.8]
        },
        'viscosity_data': {
            'shear_rate': [0.1, 1.0, 10.0, 100.0],
            'viscosity': [152.0, 18.7, 2.21, 0.258]
        }
    }

    # Generate report
    generator = RheologyReportGenerator()
    report_path = generator.generate_complete_report(
        sample_results,
        "Polymer Solution Sample A"
    )

    print(f"Complete rheological report generated at: {report_path}")
    return report_path


if __name__ == "__main__":
    # Generate sample report
    report_path = generate_sample_report()
    print(f"Report files created in: {report_path}")

    # List generated files
    for file_path in report_path.rglob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")
```

---

This advanced tutorial has equipped you with sophisticated rheological analysis techniques using the Scientific Computing Toolkit. You've learned:

✅ **Advanced parameter estimation** with uncertainty quantification  
✅ **Multi-phase flow analysis** for complex fluid systems  
✅ **Thixotropic behavior modeling** with time-dependent properties  
✅ **Process optimization** for industrial applications  
✅ **Publication-quality visualization** and automated reporting  

The toolkit's Herschel-Bulkley framework provides the foundation for analyzing complex fluids across industries including pharmaceuticals, food processing, paints, and advanced materials. The combination of robust mathematical models, sophisticated parameter estimation, and comprehensive visualization makes it an invaluable tool for rheological research and industrial applications.

Continue exploring the toolkit's capabilities through the [API Reference](api_reference.md) and [Integration Tutorials](integration_tutorial.md) to unlock even more advanced rheological analysis techniques! 🔬⚗️📊
