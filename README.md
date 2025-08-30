# hbflow

Herschel–Bulkley utilities for constitutive modeling, inverse curve, parameter fitting, and elliptical-duct flow.

## Features
- Constitutive HB model and inverse γ̇(τ) with clamping
- Apparent viscosity and Papanastasiou regularization
- Robust parameter fitting (SciPy) with bootstrap uncertainty (optional)
- Elliptical-duct HB solver (variable-coefficient Poisson via Picard + SOR)
- CLI for prediction, fitting, and flow computation
- Plotting helpers and unit tests

## Install
```bash
pip install -r requirements.txt
# Optional for fitting/bootstrapping
pip install scipy
```

## Usage

### 1) Constitutive and inverse
```python
from hbflow.models import hb_tau_from_gamma, hb_gamma_from_tau

# Parameters
tau_y, K, n = 5.0, 2.0, 0.8

# Forward: τ(γ̇)
gamma = [0.1, 1.0, 10.0]
tau = hb_tau_from_gamma(gamma, tau_y, K, n)

# Inverse: γ̇(τ)
tau_vals = [5, 10, 50]
gamma_inv = hb_gamma_from_tau(tau_vals, tau_y, K, n)
```

### 2) Fit from CSV
CSV requires shear rate and stress columns (default: `gamma_dot`, `tau`).
```bash
python -m hbflow.cli fit --csv data.csv --bootstrap 200 --plot fit.png
```

### 3) Elliptical duct flow
```bash
python -m hbflow.cli flow --a 0.01 --b 0.02 --dpdz -100 --tau-y 5 --K 2 --n 0.8 --plot vel.png
```

### 4) CLI inverse prediction
```bash
python -m hbflow.cli predict --tau 5 10 50 --tau-y 5 --K 2 --n 0.8
```

## Notes
- Units: τ [Pa], γ̇ [1/s], K [Pa·s^n], lengths [m], ∂p/∂z [Pa/m].
- The elliptical solver uses Papanastasiou regularization for yield stress; convergence depends on `m_reg`, grid resolution, and `sor_omega`.
- For Newtonian fluids (τ_y=0, n=1), the solver approaches the analytical profile.
