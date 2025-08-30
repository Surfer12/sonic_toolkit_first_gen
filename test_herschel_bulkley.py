"""
Unit Tests for Herschel-Bulkley Model Implementation

This module contains comprehensive unit tests for the HB model including:
- Constitutive and inverse forms validation
- Parameter fitting accuracy
- Elliptical duct flow solver verification
- Edge cases and error handling
- Performance benchmarks

Author: Ryan David Oates
Date: August 26, 2025
License: GPL-3.0-only
"""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import warnings

# Import the HB model classes
from herschel_bulkley_model import (
    HBParameters, HerschelBulkleyModel, HBParameterFitter,
    EllipticalDuctFlowSolver, HBVisualizer, FitResult
)


class TestHBParameters(unittest.TestCase):
    """Test HBParameters dataclass."""

    def test_valid_parameters(self):
        """Test valid parameter creation."""
        params = HBParameters(tau_y=1.0, K=2.0, n=0.8)
        self.assertEqual(params.tau_y, 1.0)
        self.assertEqual(params.K, 2.0)
        self.assertEqual(params.n, 0.8)

    def test_invalid_tau_y(self):
        """Test negative yield stress."""
        with self.assertRaises(ValueError):
            HBParameters(tau_y=-1.0, K=1.0, n=1.0)

    def test_invalid_K(self):
        """Test non-positive consistency index."""
        with self.assertRaises(ValueError):
            HBParameters(tau_y=1.0, K=0.0, n=1.0)

    def test_invalid_n(self):
        """Test non-positive flow behavior index."""
        with self.assertRaises(ValueError):
            HBParameters(tau_y=1.0, K=1.0, n=0.0)

    def test_serialization(self):
        """Test parameter serialization."""
        params = HBParameters(tau_y=1.5, K=2.5, n=0.7)
        data = params.to_dict()

        self.assertEqual(data['tau_y'], 1.5)
        self.assertEqual(data['K'], 2.5)
        self.assertEqual(data['n'], 0.7)

        # Test reconstruction
        params2 = HBParameters.from_dict(data)
        self.assertEqual(params.tau_y, params2.tau_y)
        self.assertEqual(params.K, params2.K)
        self.assertEqual(params.n, params2.n)


class TestHerschelBulkleyModel(unittest.TestCase):
    """Test HerschelBulkleyModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.newtonian = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.0, n=1.0))
        self.bingham = HerschelBulkleyModel(HBParameters(tau_y=2.0, K=1.5, n=1.0))
        self.hb_shear_thinning = HerschelBulkleyModel(HBParameters(tau_y=1.0, K=2.0, n=0.6))
        self.power_law = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.8, n=0.8))

    def test_newtonian_constitutive(self):
        """Test Newtonian constitutive model."""
        gamma_dot = 2.0
        tau_expected = 2.0  # τ = μ·γ̇ = 1·2 = 2
        tau_actual = self.newtonian.constitutive_model(gamma_dot)
        self.assertAlmostEqual(tau_actual, tau_expected, places=6)

    def test_bingham_constitutive(self):
        """Test Bingham plastic constitutive model."""
        gamma_dot = 3.0
        tau_expected = 2.0 + 1.5 * 3.0  # τy + K·γ̇ = 2 + 4.5 = 6.5
        tau_actual = self.bingham.constitutive_model(gamma_dot)
        self.assertAlmostEqual(tau_actual, tau_expected, places=6)

    def test_hb_constitutive(self):
        """Test general HB constitutive model."""
        gamma_dot = 4.0
        tau_expected = 1.0 + 2.0 * (4.0**0.6)  # τy + K·γ̇^n
        tau_actual = self.hb_shear_thinning.constitutive_model(gamma_dot)
        self.assertAlmostEqual(tau_actual, tau_expected, places=6)

    def test_vectorized_constitutive(self):
        """Test vectorized constitutive model."""
        gamma_dot = np.array([1.0, 2.0, 3.0, 4.0])
        tau = self.bingham.constitutive_model(gamma_dot)

        expected = np.array([2.0 + 1.5*1, 2.0 + 1.5*2, 2.0 + 1.5*3, 2.0 + 1.5*4])
        np.testing.assert_array_almost_equal(tau, expected, decimal=6)

    def test_newtonian_inverse(self):
        """Test Newtonian inverse model."""
        tau = 3.0
        gamma_dot_expected = 3.0  # γ̇ = τ/μ = 3/1 = 3
        gamma_dot_actual = self.newtonian.inverse_model(tau)
        self.assertAlmostEqual(gamma_dot_actual, gamma_dot_expected, places=6)

    def test_bingham_inverse(self):
        """Test Bingham plastic inverse model."""
        # Above yield stress
        tau_above = 6.0
        gamma_dot_expected = (6.0 - 2.0) / 1.5  # (τ - τy)/K = 4/1.5 = 2.667
        gamma_dot_actual = self.bingham.inverse_model(tau_above)
        self.assertAlmostEqual(gamma_dot_actual, gamma_dot_expected, places=6)

        # Below yield stress
        tau_below = 1.5
        gamma_dot_actual = self.bingham.inverse_model(tau_below)
        self.assertEqual(gamma_dot_actual, 0.0)

    def test_hb_inverse(self):
        """Test general HB inverse model."""
        tau = 5.0
        gamma_dot_expected = ((5.0 - 1.0) / 2.0) ** (1.0/0.6)  # ((τ - τy)/K)^(1/n)
        gamma_dot_actual = self.hb_shear_thinning.inverse_model(tau)
        self.assertAlmostEqual(gamma_dot_actual, gamma_dot_expected, places=6)

    def test_vectorized_inverse(self):
        """Test vectorized inverse model."""
        tau = np.array([1.0, 3.0, 6.0, 8.0])
        gamma_dot = self.bingham.inverse_model(tau)

        # Below yield stress (τ=1.0): γ̇=0
        # Above yield stress: γ̇ = (τ - τy)/K
        expected = np.array([0.0, (3.0-2.0)/1.5, (6.0-2.0)/1.5, (8.0-2.0)/1.5])
        np.testing.assert_array_almost_equal(gamma_dot, expected, decimal=6)

    def test_inverse_clamping(self):
        """Test inverse model clamping behavior."""
        # Test without clamping
        gamma_dot_no_clamp = self.bingham.inverse_model(1.0, clamp_negative=False)
        self.assertLess(gamma_dot_no_clamp, 0)

        # Test with clamping (default)
        gamma_dot_clamped = self.bingham.inverse_model(1.0, clamp_negative=True)
        self.assertEqual(gamma_dot_clamped, 0.0)

    def test_apparent_viscosity(self):
        """Test apparent viscosity calculation."""
        gamma_dot = 2.0
        eta = self.bingham.apparent_viscosity(gamma_dot)

        tau = self.bingham.constitutive_model(gamma_dot)
        eta_expected = tau / gamma_dot

        self.assertAlmostEqual(eta, eta_expected, places=6)

    def test_zero_shear_rate_viscosity(self):
        """Test apparent viscosity at zero shear rate."""
        gamma_dot = 0.0
        eta = self.newtonian.apparent_viscosity(gamma_dot)
        self.assertEqual(eta, np.inf)

    def test_model_info(self):
        """Test model information classification."""
        info = self.newtonian.get_model_info()
        self.assertEqual(info['behavior'], 'Newtonian')
        self.assertTrue(info['is_newtonian'])
        self.assertFalse(info['is_power_law'])

        info = self.power_law.get_model_info()
        self.assertIn('Power-law', info['behavior'])
        self.assertFalse(info['is_newtonian'])
        self.assertTrue(info['is_power_law'])

        info = self.bingham.get_model_info()
        self.assertEqual(info['behavior'], 'Bingham plastic')
        self.assertFalse(info['is_newtonian'])
        self.assertFalse(info['is_power_law'])

        info = self.hb_shear_thinning.get_model_info()
        self.assertIn('Herschel-Bulkley', info['behavior'])

    def test_constitutive_inverse_consistency(self):
        """Test that constitutive and inverse models are consistent."""
        test_gamma_dot = np.logspace(-1, 2, 50)

        for gamma_dot in test_gamma_dot:
            # Forward then inverse
            tau = self.hb_shear_thinning.constitutive_model(gamma_dot)
            gamma_dot_recovered = self.hb_shear_thinning.inverse_model(tau)

            # Should recover original shear rate
            self.assertAlmostEqual(gamma_dot_recovered, gamma_dot, places=4)

    def test_domain_handling(self):
        """Test proper handling of domain boundaries."""
        # Test negative shear rates
        tau = self.newtonian.constitutive_model(-1.0)
        self.assertLess(tau, 0)  # Should be negative

        # Test zero shear rate
        tau = self.newtonian.constitutive_model(0.0)
        self.assertEqual(tau, 0.0)

        # Test very small shear rates
        gamma_dot = 1e-10
        tau = self.bingham.constitutive_model(gamma_dot)
        self.assertAlmostEqual(tau, self.bingham.parameters.tau_y, places=6)


class TestHBParameterFitter(unittest.TestCase):
    """Test parameter fitting functionality."""

    def setUp(self):
        """Set up test data."""
        self.fitter = HBParameterFitter()

        # Generate synthetic HB data
        np.random.seed(42)
        self.tau_y_true = 2.0
        self.K_true = 1.5
        self.n_true = 0.7

        gamma_dot = np.logspace(-1, 2, 50)
        tau = self.tau_y_true + self.K_true * np.power(gamma_dot, self.n_true)
        # Add some noise
        noise = np.random.normal(0, 0.1, len(tau))
        self.tau_data = tau + noise
        self.gamma_dot_data = gamma_dot

    def test_parameter_fitting(self):
        """Test parameter fitting on synthetic data."""
        result = self.fitter.fit_parameters(self.gamma_dot_data, self.tau_data)

        # Check that parameters are close to true values
        self.assertAlmostEqual(result.parameters.tau_y, self.tau_y_true, delta=0.5)
        self.assertAlmostEqual(result.parameters.K, self.K_true, delta=0.3)
        self.assertAlmostEqual(result.parameters.n, self.n_true, delta=0.2)

        # Check goodness of fit
        self.assertGreater(result.r_squared, 0.9)
        self.assertLess(result.rmse, 1.0)

    def test_fitting_insufficient_data(self):
        """Test fitting with insufficient data."""
        with self.assertRaises(ValueError):
            self.fitter.fit_parameters([1.0, 2.0], [1.0, 2.0])

    def test_fitting_invalid_data(self):
        """Test fitting with invalid data."""
        gamma_dot = np.array([1.0, 2.0, 3.0])
        tau = np.array([1.0, 2.0, np.nan])  # Invalid data

        with self.assertRaises(ValueError):
            self.fitter.fit_parameters(gamma_dot, tau)

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        cv_results = self.fitter.cross_validate(self.gamma_dot_data, self.tau_data, k_folds=3)

        self.assertGreater(cv_results['n_folds_successful'], 0)
        self.assertGreater(cv_results['cv_rmse_mean'], 0)
        self.assertTrue(np.isfinite(cv_results['cv_rmse_mean']))

    def test_fitting_newtonian(self):
        """Test fitting Newtonian fluid."""
        gamma_dot = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau = 1.2 * gamma_dot  # Newtonian: τ = μ·γ̇

        result = self.fitter.fit_parameters(gamma_dot, tau)

        # Should find τy ≈ 0, n ≈ 1
        self.assertLess(result.parameters.tau_y, 0.1)
        self.assertAlmostEqual(result.parameters.n, 1.0, delta=0.1)
        self.assertGreater(result.r_squared, 0.99)


class TestEllipticalDuctFlowSolver(unittest.TestCase):
    """Test elliptical duct flow solver."""

    def setUp(self):
        """Set up test fixtures."""
        self.newtonian = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.0, n=1.0))
        self.bingham = HerschelBulkleyModel(HBParameters(tau_y=2.0, K=1.5, n=1.0))

        # Elliptical duct geometry
        self.a = 0.01  # Semi-major axis [m]
        self.b = 0.005  # Semi-minor axis [m]

    def test_newtonian_solver(self):
        """Test flow solver with Newtonian fluid."""
        solver = EllipticalDuctFlowSolver(self.newtonian, self.a, self.b)
        dp_dx = -1000.0  # Pressure gradient [Pa/m]

        result = solver.calculate_flow_rate(dp_dx)

        # For Newtonian fluid, should have non-zero flow
        self.assertGreater(result['Q'], 0)
        self.assertEqual(result['yielded_fraction'], 0.0)  # Newtonian always yields

    def test_bingham_solver_above_yield(self):
        """Test Bingham fluid above yield stress."""
        solver = EllipticalDuctFlowSolver(self.bingham, self.a, self.b)
        dp_dx = -5000.0  # Large pressure gradient

        result = solver.calculate_flow_rate(dp_dx)

        self.assertGreater(result['Q'], 0)
        self.assertGreater(result['yielded_fraction'], 0)

    def test_bingham_solver_below_yield(self):
        """Test Bingham fluid below yield stress."""
        solver = EllipticalDuctFlowSolver(self.bingham, self.a, self.b)
        dp_dx = -10.0  # Small pressure gradient

        result = solver.calculate_flow_rate(dp_dx)

        self.assertEqual(result['Q'], 0.0)
        self.assertEqual(result['yielded_fraction'], 0.0)
        self.assertIn('no flow', result['warnings'][0])

    def test_zero_pressure_gradient(self):
        """Test zero pressure gradient."""
        solver = EllipticalDuctFlowSolver(self.newtonian, self.a, self.b)
        result = solver.calculate_flow_rate(0.0)

        self.assertEqual(result['Q'], 0.0)
        self.assertIn('Zero pressure gradient', result['warnings'][0])

    def test_invalid_geometry(self):
        """Test invalid duct geometry."""
        with self.assertRaises(ValueError):
            EllipticalDuctFlowSolver(self.newtonian, -1.0, 0.005)

        with self.assertRaises(ValueError):
            EllipticalDuctFlowSolver(self.newtonian, 0.01, 0.0)


class TestHBVisualizer(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = HerschelBulkleyModel(HBParameters(tau_y=1.5, K=2.0, n=0.8))
        self.visualizer = HBVisualizer()

    def test_rheogram_plot(self):
        """Test rheogram plotting."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig = self.visualizer.plot_rheogram(self.model, save_path=tmp_path)

            # Check that file was created
            self.assertTrue(Path(tmp_path).exists())
            self.assertGreater(Path(tmp_path).stat().st_size, 0)

            # Check figure properties
            self.assertEqual(len(fig.axes), 2)  # Two subplots

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_velocity_profile_plot(self):
        """Test velocity profile plotting."""
        # Create mock flow result
        flow_result = {
            'Q': 0.001,
            'velocity_profile': np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
            'yielded_fraction': 0.8,
            'wall_shear_stress': 5.0,
            'warnings': []
        }

        duct_geometry = {'a': 0.01, 'b': 0.005}

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig = self.visualizer.plot_velocity_profile(flow_result, duct_geometry, save_path=tmp_path)

            # Check that file was created
            self.assertTrue(Path(tmp_path).exists())
            self.assertGreater(Path(tmp_path).stat().st_size, 0)

        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface functionality."""

    def test_cli_import(self):
        """Test that CLI can be imported without errors."""
        try:
            from herschel_bulkley_model import create_cli_interface, main
            self.assertIsNotNone(create_cli_interface)
            self.assertIsNotNone(main)
        except ImportError as e:
            self.fail(f"CLI import failed: {e}")


class TestValidationLimits(unittest.TestCase):
    """Test validation against Newtonian and power-law limits."""

    def test_newtonian_limit(self):
        """Test that HB model reduces to Newtonian when τy=0, n=1."""
        hb_newtonian = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.2, n=1.0))

        gamma_dot = 2.5
        tau = hb_newtonian.constitutive_model(gamma_dot)
        gamma_dot_recovered = hb_newtonian.inverse_model(tau)

        # Should satisfy τ = μ·γ̇
        self.assertAlmostEqual(tau / gamma_dot, 1.2, places=6)
        self.assertAlmostEqual(gamma_dot_recovered, gamma_dot, places=6)

    def test_power_law_limit(self):
        """Test that HB model reduces to power-law when τy=0."""
        hb_power_law = HerschelBulkleyModel(HBParameters(tau_y=0.0, K=1.8, n=0.7))

        gamma_dot = 3.0
        tau = hb_power_law.constitutive_model(gamma_dot)
        gamma_dot_recovered = hb_power_law.inverse_model(tau)

        # Should satisfy τ = K·γ̇^n
        expected_tau = 1.8 * (3.0**0.7)
        self.assertAlmostEqual(tau, expected_tau, places=6)
        self.assertAlmostEqual(gamma_dot_recovered, gamma_dot, places=6)

    def test_bingham_limit(self):
        """Test Bingham plastic behavior."""
        hb_bingham = HerschelBulkleyModel(HBParameters(tau_y=2.5, K=1.3, n=1.0))

        # Test above yield stress
        gamma_dot = 2.0
        tau = hb_bingham.constitutive_model(gamma_dot)
        expected_tau = 2.5 + 1.3 * 2.0  # τy + K·γ̇
        self.assertAlmostEqual(tau, expected_tau, places=6)

        # Test below yield stress
        gamma_dot_below = hb_bingham.inverse_model(2.0)
        self.assertEqual(gamma_dot_below, 0.0)


class TestModelIntegration(unittest.TestCase):
    """Test integration across different model components."""

    def test_parameter_fitting_integration(self):
        """Test parameter fitting with flow calculations."""
        # Generate synthetic data
        np.random.seed(42)
        true_params = HBParameters(tau_y=1.5, K=2.0, n=0.8)
        model = HerschelBulkleyModel(true_params)

        gamma_dot = np.logspace(-1, 1, 20)
        tau = model.constitutive_model(gamma_dot)
        tau_noisy = tau + np.random.normal(0, 0.1, len(tau))

        # Fit parameters
        fitter = HBParameterFitter()
        fit_result = fitter.fit_parameters(gamma_dot, tau_noisy)

        # Create fitted model
        fitted_model = HerschelBulkleyModel(fit_result.parameters)

        # Test flow calculations with both models
        solver_true = EllipticalDuctFlowSolver(model, 0.01, 0.005)
        solver_fitted = EllipticalDuctFlowSolver(fitted_model, 0.01, 0.005)

        dp_dx = -1000.0
        result_true = solver_true.calculate_flow_rate(dp_dx)
        result_fitted = solver_fitted.calculate_flow_rate(dp_dx)

        # Results should be reasonably close (within 20% for this simplified model)
        if result_true['Q'] > 0 and result_fitted['Q'] > 0:
            ratio = result_fitted['Q'] / result_true['Q']
            self.assertGreater(ratio, 0.8, "Fitted model flow rate too different from true")
            self.assertLess(ratio, 1.25, "Fitted model flow rate too different from true")

    def test_model_serialization(self):
        """Test model serialization and reconstruction."""
        params = HBParameters(tau_y=2.5, K=1.8, n=0.9)
        model = HerschelBulkleyModel(params)

        # Serialize
        data = params.to_dict()

        # Reconstruct
        params_reconstructed = HBParameters.from_dict(data)
        model_reconstructed = HerschelBulkleyModel(params_reconstructed)

        # Test consistency
        test_gamma_dot = np.array([0.5, 1.0, 2.0])
        tau_original = model.constitutive_model(test_gamma_dot)
        tau_reconstructed = model_reconstructed.constitutive_model(test_gamma_dot)

        np.testing.assert_array_almost_equal(tau_original, tau_reconstructed, decimal=10)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small parameters
        small_params = HBParameters(tau_y=1e-6, K=1e-3, n=0.1)
        small_model = HerschelBulkleyModel(small_params)

        # Test with very large values
        large_params = HBParameters(tau_y=1e6, K=1e3, n=2.0)
        large_model = HerschelBulkleyModel(large_params)

        # Test edge cases
        test_values = np.array([1e-10, 1e-5, 1e0, 1e5, 1e10])

        for model_name, model in [("small", small_model), ("large", large_model)]:
            with self.subTest(model=model_name):
                # Should not raise exceptions
                tau = model.constitutive_model(test_values)
                gamma_dot = model.inverse_model(test_values)

                # Check for reasonable values (not NaN or inf)
                self.assertFalse(np.any(np.isnan(tau)), f"NaN in constitutive model for {model_name}")
                self.assertFalse(np.any(np.isinf(tau)), f"Inf in constitutive model for {model_name}")
                self.assertFalse(np.any(np.isnan(gamma_dot)), f"NaN in inverse model for {model_name}")
                self.assertFalse(np.any(np.isinf(gamma_dot)), f"Inf in inverse model for {model_name}")

    def test_units_consistency(self):
        """Test units consistency in calculations."""
        params = HBParameters(tau_y=1000.0, K=1000.0, n=1.0)  # Pa, Pa·s, -
        model = HerschelBulkleyModel(params)

        # Test with realistic values
        gamma_dot = 1.0  # 1/s
        tau = model.constitutive_model(gamma_dot)  # Should be in Pa

        # Bingham fluid: τ = τy + μ·γ̇, so μ = K = 1000 Pa·s
        expected_tau = 1000.0 + 1000.0 * 1.0  # 2000 Pa
        self.assertAlmostEqual(tau, expected_tau, places=6)

        # Inverse should recover shear rate
        gamma_dot_recovered = model.inverse_model(tau)
        self.assertAlmostEqual(gamma_dot_recovered, gamma_dot, places=6)


if __name__ == '__main__':
    # Set up test environment
    unittest.main(verbosity=2)
