# Hybrid UQ API Reference Guide

## Overview

This comprehensive API reference guide provides detailed documentation for the Hybrid Uncertainty Quantification (`hybrid_uq/`) framework within the Farmer scientific computing toolkit. The framework implements advanced uncertainty quantification with Ψ(x) confidence calibration and seamless integration with security frameworks.

---

## Table of Contents

1. [Core Classes](#core-classes)
2. [API Endpoints](#api-endpoints)
3. [Data Structures](#data-structures)
4. [Error Handling](#error-handling)
5. [Performance Optimization](#performance-optimization)
6. [Integration Patterns](#integration-patterns)

---

## Core Classes

### HybridModel

The main hybrid physics-informed neural network class with uncertainty quantification.

```python
class HybridModel(nn.Module):
    """Unified hybrid physics-neural model with Ψ(x) confidence quantification.

    This class implements a sophisticated hybrid framework that combines
    physics-based predictions with neural corrections and provides
    comprehensive uncertainty quantification.

    Attributes:
        phys (PhysicsInterpolator): Physics-based prediction component
        nn (ResidualNet): Neural residual correction component
        alpha (nn.Parameter): Hybrid weighting parameter
        lambdas (Dict[str, float]): Risk penalty weights
        beta (nn.Parameter): Posterior calibration parameter
    """

    def __init__(
        self,
        grid_metrics: Dict[str, float],
        in_ch: int = 2,
        out_ch: int = 2,
        residual_scale: float = 0.02,
        init_alpha: float = 0.5,
        lambda_cog: float = 0.5,
        lambda_eff: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        """Initialize hybrid model.

        Args:
            grid_metrics: Dictionary with 'dx', 'dy' grid spacing
            in_ch: Input channels (default: 2 for velocity fields)
            out_ch: Output channels (default: 2 for velocity fields)
            residual_scale: Neural residual scaling factor (default: 0.02)
            init_alpha: Initial hybrid weighting (default: 0.5)
            lambda_cog: Cognitive risk penalty weight (default: 0.5)
            lambda_eff: Efficiency risk penalty weight (default: 0.5)
            beta: Posterior calibration factor (default: 1.0)
        """
```

#### Methods

##### forward()

```python
def forward(
    self,
    x: torch.Tensor,
    external_P: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Forward pass computing hybrid predictions and uncertainty quantification.

    Args:
        x: Input tensor of shape [B, C, H, W]
        external_P: Optional external probability tensor

    Returns:
        Dictionary containing:
            - 'S': Physics-based predictions [B, C, H, W]
            - 'N': Neural corrections [B, C, H, W]
            - 'O': Hybrid predictions [B, C, H, W]
            - 'psi': Ψ(x) confidence values [B, C, H, W]
            - 'sigma_res': Residual uncertainties [B, C, H, W]
            - 'pen': Risk penalty values [B, 1, 1, 1]
            - 'R_cog': Cognitive risk values [B, 1, 1, 1]
            - 'R_eff': Efficiency risk values [B, 1, 1, 1]

    Raises:
        ValueError: If input tensor dimensions are invalid
        RuntimeError: If CUDA operations fail
    """
```

##### compute_penalties()

```python
def compute_penalties(self, diagnostics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cognitive and efficiency risk penalties.

    Args:
        diagnostics: Dictionary with vorticity and divergence fields

    Returns:
        Tuple of (R_cog, R_eff) penalty tensors

    Note:
        R_cog penalizes divergence and rotational smoothness
        R_eff penalizes gradient magnitude variations
    """
```

### PhysicsInterpolator

Domain-specific surface-to-sigma transformation with fluid dynamics diagnostics.

```python
class PhysicsInterpolator(nn.Module):
    """Physics-based interpolation from surface to sigma coordinates.

    This class implements domain-specific transformations with built-in
    fluid dynamics diagnostics including vorticity and divergence computation.

    Attributes:
        grid_metrics (Dict[str, float]): Grid spacing parameters
    """

    def __init__(self, grid_metrics: Dict[str, float]):
        """Initialize physics interpolator.

        Args:
            grid_metrics: Must contain 'dx' and 'dy' keys with grid spacing values
        """
```

#### Methods

##### forward()

```python
def forward(self, x_surface: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute physics-based interpolation and diagnostics.

    Args:
        x_surface: Surface field tensor [B, C, H, W]
                  Channels 0,1 treated as u,v velocity components

    Returns:
        Tuple of:
            - S: Sigma-space fields [B, C, H, W] (currently identity)
            - diagnostics: Dict with 'vorticity' and 'divergence' fields

    Note:
        Replace identity mapping with domain-specific operators for production use
    """
```

### ResidualNet

Heteroscedastic neural residual correction network.

```python
class ResidualNet(nn.Module):
    """Convolutional neural network for residual correction with uncertainty.

    Implements heteroscedastic uncertainty quantification where the network
    predicts both mean corrections and associated uncertainties.

    Attributes:
        conv1, conv2: Feature extraction convolutions
        conv_mu: Mean prediction head
        conv_log_sigma: Uncertainty prediction head
    """

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 128):
        """Initialize residual network.

        Args:
            in_ch: Input channels
            out_ch: Output channels
            hidden: Hidden layer size (default: 128)
        """
```

#### Methods

##### forward()

```python
def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residual corrections with heteroscedastic uncertainty.

    Args:
        feats: Input feature tensor [B, C, H, W]

    Returns:
        Tuple of:
            - mu: Mean residual predictions [B, out_ch, H, W]
            - sigma: Standard deviation predictions [B, out_ch, H, W]
            - log_sigma: Log standard deviation (clamped) [B, out_ch, H, W]

    Note:
        log_sigma is clamped to [-6.0, 3.0] for numerical stability
    """
```

### AlphaScheduler

Adaptive parameter optimization with variance-aware scheduling.

```python
class AlphaScheduler:
    """Variance-aware adaptive scheduling for hybrid parameters.

    Implements sophisticated parameter adaptation based on prediction
    variance and residual stability metrics.

    Attributes:
        alpha_min, alpha_max: Parameter bounds
        var_hi, var_lo: Variance thresholds
        k_up, k_dn: Adaptation rates
    """

    def __init__(
        self,
        alpha_min: float = 0.1,
        alpha_max: float = 0.95,
        var_hi: float = 0.02,
        var_lo: float = 0.005,
        k_up: float = 0.15,
        k_dn: float = 0.08,
    ):
        """Initialize alpha scheduler.

        Args:
            alpha_min: Minimum alpha value (default: 0.1)
            alpha_max: Maximum alpha value (default: 0.95)
            var_hi: High variance threshold (default: 0.02)
            var_lo: Low variance threshold (default: 0.005)
            k_up: Upward adaptation rate (default: 0.15)
            k_dn: Downward adaptation rate (default: 0.08)
        """
```

#### Methods

##### step()

```python
def step(
    self,
    model: HybridModel,
    pred_var: torch.Tensor,
    resid_stability: float,
    bifurcation_flag: bool = False,
) -> bool:
    """Update model parameters based on current conditions.

    Args:
        model: HybridModel instance to update
        pred_var: Prediction variance tensor
        resid_stability: Residual stability metric
        bifurcation_flag: Whether bifurcation detected

    Returns:
        bool: True if abstaining from prediction (high uncertainty)

    Note:
        Increases alpha when uncertainty is high (risky conditions)
        Decreases alpha when predictions are stable and accurate
        Adjusts penalty weights based on risk assessment
    """
```

### SplitConformal

Statistically guaranteed prediction intervals via conformal prediction.

```python
class SplitConformal:
    """Split conformal prediction for guaranteed coverage intervals.

    Implements quantile regression-based conformal prediction to provide
    statistically guaranteed prediction intervals.

    Attributes:
        quantile: Target quantile for interval computation
        q: Fitted quantile value (None until fit() called)
    """

    def __init__(self, quantile: float = 0.9):
        """Initialize conformal predictor.

        Args:
            quantile: Target quantile (default: 0.9 for 90% intervals)
        """
```

#### Methods

##### fit()

```python
def fit(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
    """Fit conformal predictor on calibration data.

    Args:
        preds: Model predictions [N]
        targets: True target values [N]

    Note:
        Must be called before intervals() method
        Computes quantile of absolute prediction errors
    """
```

##### intervals()

```python
def intervals(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute conformal prediction intervals.

    Args:
        preds: New predictions to intervalize

    Returns:
        Tuple of (lower_bounds, upper_bounds) with same shape as preds

    Raises:
        AssertionError: If fit() not called first
    """
```

---

## API Endpoints

### RESTful API Interface

#### POST /api/v1/hybrid_uq/predict

Main prediction endpoint for uncertainty quantification.

**Request Body:**
```json
{
  "inputs": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
  "grid_metrics": {"dx": 1.0, "dy": 1.0},
  "return_diagnostics": true,
  "conformal_calibration": true
}
```

**Response:**
```json
{
  "predictions": [[[0.15, 0.25], [0.35, 0.45]], [[0.55, 0.65], [0.75, 0.85]]],
  "uncertainty": [[[0.02, 0.03], [0.025, 0.035]], [[0.028, 0.032], [0.031, 0.037]]],
  "psi_confidence": [[[0.89, 0.92], [0.87, 0.91]], [[0.88, 0.93], [0.86, 0.90]]],
  "confidence_intervals": {
    "lower": [[[0.11, 0.19], [0.30, 0.38]], [[0.49, 0.60], [0.69, 0.78]]],
    "upper": [[[0.19, 0.31], [0.40, 0.52]], [[0.61, 0.70], [0.81, 0.92]]]
  },
  "diagnostics": {
    "vorticity": [[[0.05, -0.03], [0.02, 0.08]], [[-0.04, 0.06], [0.01, -0.02]]],
    "divergence": [[[0.12, 0.08], [0.15, 0.09]], [[0.11, 0.13], [0.07, 0.14]]],
    "risk_penalties": {
      "cognitive": 0.045,
      "efficiency": 0.032
    }
  },
  "processing_time": 0.0234,
  "confidence_level": 0.95
}
```

**Status Codes:**
- `200`: Successful prediction
- `400`: Invalid input format
- `422`: Model validation error
- `500`: Internal server error

#### GET /api/v1/hybrid_uq/health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_version": "hybrid_uq-v1.3.0",
  "gpu_available": true,
  "memory_usage": 45.6,
  "last_calibration": "2025-08-26T10:30:00Z",
  "uptime_seconds": 3600
}
```

#### POST /api/v1/hybrid_uq/calibrate

Online calibration endpoint for continuous learning.

**Request Body:**
```json
{
  "new_data": {
    "inputs": [...],
    "targets": [...]
  },
  "calibration_method": "online",
  "update_model": true
}
```

---

## Data Structures

### PredictionResult

Comprehensive prediction result structure.

```python
@dataclass
class PredictionResult:
    """Complete prediction result with uncertainty quantification.

    Attributes:
        predictions: Main model predictions
        uncertainty: Prediction uncertainties (standard deviations)
        psi_confidence: Ψ(x) confidence values
        confidence_intervals: Conformal prediction intervals
        diagnostics: Optional diagnostic information
        metadata: Processing metadata
    """

    predictions: np.ndarray
    uncertainty: np.ndarray
    psi_confidence: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    diagnostics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
```

### ModelConfig

Configuration structure for model initialization.

```python
@dataclass
class ModelConfig:
    """Configuration for HybridModel initialization.

    Attributes:
        grid_metrics: Grid spacing parameters
        in_channels: Input tensor channels
        out_channels: Output tensor channels
        residual_scale: Neural residual scaling
        alpha_init: Initial hybrid weighting
        lambda_cog: Cognitive risk penalty
        lambda_eff: Efficiency risk penalty
        beta: Posterior calibration factor
    """

    grid_metrics: Dict[str, float]
    in_channels: int = 2
    out_channels: int = 2
    residual_scale: float = 0.02
    alpha_init: float = 0.5
    lambda_cog: float = 0.5
    lambda_eff: float = 0.5
    beta: float = 1.0
```

### CalibrationData

Structure for conformal calibration data.

```python
@dataclass
class CalibrationData:
    """Data structure for conformal calibration.

    Attributes:
        predictions: Model predictions on calibration set
        targets: True target values
        quantile: Target quantile for intervals
    """

    predictions: torch.Tensor
    targets: torch.Tensor
    quantile: float = 0.9
```

---

## Error Handling

### Exception Hierarchy

```python
class HybridUQError(Exception):
    """Base exception for hybrid UQ framework."""
    pass

class ModelConfigurationError(HybridUQError):
    """Raised when model configuration is invalid."""
    pass

class InputValidationError(HybridUQError):
    """Raised when input data is malformed."""
    pass

class ComputationError(HybridUQError):
    """Raised when numerical computation fails."""
    pass

class GPUError(HybridUQError):
    """Raised when GPU operations fail."""
    pass
```

### Error Handling Patterns

```python
# Recommended error handling pattern
def safe_predict(model, inputs):
    """Safe prediction with comprehensive error handling."""
    try:
        # Validate inputs
        validate_inputs(inputs)

        # Perform prediction
        with torch.no_grad():
            outputs = model(inputs)

        # Validate outputs
        validate_outputs(outputs)

        return process_results(outputs)

    except InputValidationError as e:
        logger.error(f"Input validation failed: {e}")
        return fallback_prediction()

    except ComputationError as e:
        logger.error(f"Computation failed: {e}")
        return recovery_prediction()

    except GPUError as e:
        logger.warning(f"GPU error, falling back to CPU: {e}")
        return cpu_fallback_prediction()

    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        raise HybridUQError("Prediction failed") from e
```

---

## Performance Optimization

### GPU Acceleration

```python
def optimized_inference(model, inputs, device='cuda'):
    """GPU-optimized inference with memory management.

    Args:
        model: HybridModel instance
        inputs: Input tensor
        device: Target device ('cuda' or 'cpu')

    Returns:
        Dict of optimized results
    """
    model = model.to(device)
    inputs = inputs.to(device)

    # Use autocast for mixed precision
    with torch.cuda.amp.autocast(), torch.no_grad():
        outputs = model(inputs)

    # Efficient memory transfer
    results = {}
    for key, value in outputs.items():
        if key in ['psi', 'pen', 'R_cog', 'R_eff']:
            # Keep scalars on GPU
            results[key] = value
        else:
            # Transfer arrays to CPU
            results[key] = value.cpu()

    return results
```

### Memory Optimization

```python
def memory_efficient_batch_processing(model, data_loader, chunk_size=16):
    """Process large datasets with controlled memory usage.

    Args:
        model: HybridModel instance
        data_loader: PyTorch DataLoader
        chunk_size: Processing chunk size

    Yields:
        Processed results for each batch
    """
    for batch in data_loader:
        batch_results = []

        # Process in chunks
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]

            with torch.no_grad():
                chunk_outputs = model(chunk)

            batch_results.append(process_chunk(chunk_outputs))

            # Memory cleanup
            del chunk_outputs
            torch.cuda.empty_cache()

        yield batch_results
```

### Parallel Processing

```python
def parallel_inference(models, inputs, num_workers=4):
    """Parallel inference across multiple model instances.

    Args:
        models: List of HybridModel instances
        inputs: Input tensor to distribute
        num_workers: Number of parallel workers

    Returns:
        Aggregated results from all models
    """
    # Split inputs across models
    chunks = torch.chunk(inputs, num_workers)

    # Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(model, chunk)
            for model, chunk in zip(models, chunks)
        ]

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return aggregate_results(results)
```

---

## Integration Patterns

### Corpus/Qualia Security Integration

```java
// Java integration with hybrid UQ
public class HybridUQSecurityIntegration {

    private HybridUQClient uqClient;

    public HybridUQSecurityIntegration(String uqEndpoint) {
        this.uqClient = new HybridUQClient(uqEndpoint);
    }

    public SecurityAssessment assessWithUncertainty(
            double[] systemState,
            SecurityTest test) throws HybridUQException {

        // Get uncertainty quantification
        UncertaintyResult uqResult = uqClient.predict(systemState);

        // Apply to security assessment
        SecurityFinding finding = test.execute();
        finding.setConfidence(uqResult.getPsiConfidence());
        finding.setUncertaintyInterval(uqResult.getConfidenceInterval());

        return new SecurityAssessment(finding, uqResult);
    }
}
```

### Web Framework Integration

```python
# Flask integration example
from flask import Flask, request, jsonify
from hybrid_uq import HybridModel, ModelConfig

app = Flask(__name__)

# Initialize model
config = ModelConfig(grid_metrics={'dx': 1.0, 'dy': 1.0})
model = HybridModel(config)

@app.route('/predict', methods=['POST'])
def predict():
    """RESTful prediction endpoint."""
    data = request.get_json()

    # Convert to tensor
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)

    # Perform prediction
    with torch.no_grad():
        outputs = model(inputs)

    # Return results
    return jsonify({
        'predictions': outputs['O'].tolist(),
        'uncertainty': outputs['sigma_res'].tolist(),
        'psi_confidence': outputs['psi'].tolist(),
        'processing_time': time.time() - request.start_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Scientific Computing Integration

```python
# Integration with broader scientific framework
class ScientificComputingIntegration:

    def __init__(self, hybrid_model):
        self.hybrid_model = hybrid_model
        self.inverse_precision = InversePrecisionFramework()
        self.psi_framework = PsiConfidenceFramework()

    def comprehensive_analysis(self, data):
        """End-to-end scientific analysis with uncertainty."""

        # Step 1: Inverse problem solving
        parameters = self.inverse_precision.solve(data)

        # Step 2: Uncertainty quantification
        predictions, uncertainty = self.hybrid_model.predict_with_uq(parameters)

        # Step 3: Confidence assessment
        confidence = self.psi_framework.assess_confidence(predictions, uncertainty)

        return {
            'parameters': parameters,
            'predictions': predictions,
            'uncertainty': uncertainty,
            'confidence': confidence
        }
```

---

## Usage Examples

### Basic Prediction

```python
import torch
from hybrid_uq import HybridModel, ModelConfig

# Initialize model
config = ModelConfig(grid_metrics={'dx': 1.0, 'dy': 1.0})
model = HybridModel(config)

# Create sample input (velocity field)
inputs = torch.randn(1, 2, 64, 64)  # [B, C, H, W]

# Perform prediction
with torch.no_grad():
    outputs = model(inputs)

print(f"Predictions shape: {outputs['O'].shape}")
print(f"Uncertainty shape: {outputs['sigma_res'].shape}")
print(f"Ψ(x) confidence range: [{outputs['psi'].min():.3f}, {outputs['psi'].max():.3f}]")
```

### Advanced Configuration

```python
# Custom model configuration
advanced_config = ModelConfig(
    grid_metrics={'dx': 0.5, 'dy': 0.5},  # Finer grid
    residual_scale=0.01,                   # Smaller residuals
    lambda_cog=0.8,                        # Higher cognitive penalty
    lambda_eff=0.3,                        # Lower efficiency penalty
    beta=1.2                               # Higher calibration
)

model = HybridModel(advanced_config)

# Configure scheduler
scheduler = AlphaScheduler(
    alpha_min=0.2,     # Allow more physics
    alpha_max=0.8,     # Allow more neural
    var_hi=0.03,       # Higher variance threshold
    k_up=0.2,          # Faster adaptation
    k_dn=0.05          # Slower reduction
)
```

### Complete Analysis Pipeline

```python
def complete_analysis_pipeline(data):
    """End-to-end analysis with uncertainty quantification."""

    # 1. Data preprocessing
    processed_data = preprocess_data(data)

    # 2. Model prediction
    with torch.no_grad():
        predictions = model(processed_data)

    # 3. Uncertainty quantification
    uncertainty = predictions['sigma_res']

    # 4. Confidence assessment
    confidence = predictions['psi']

    # 5. Risk analysis
    cognitive_risk = predictions['R_cog']
    efficiency_risk = predictions['R_eff']

    # 6. Conformal intervals (if calibrated)
    if conformal_predictor.is_fitted():
        intervals = conformal_predictor.intervals(predictions['O'])
    else:
        intervals = None

    return {
        'predictions': predictions['O'],
        'uncertainty': uncertainty,
        'confidence': confidence,
        'risks': {
            'cognitive': cognitive_risk,
            'efficiency': efficiency_risk
        },
        'intervals': intervals,
        'diagnostics': predictions.get('diagnostics', {})
    }
```

---

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Solution: Use memory-efficient processing
def fix_memory_issues():
    """Memory optimization strategies."""

    # 1. Reduce batch size
    batch_size = max(1, original_batch_size // 4)

    # 2. Use gradient checkpointing
    model = torch.utils.checkpoint.checkpoint_wrapper(model)

    # 3. Enable memory efficient attention
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # 4. Use mixed precision
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
```

#### Numerical Instability
```python
# Solution: Add numerical stabilization
def fix_numerical_issues():
    """Numerical stability improvements."""

    # 1. Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 2. Better initialization
    for param in model.parameters():
        if len(param.shape) > 1:
            torch.nn.init.xavier_uniform_(param)

    # 3. Stable loss computation
    def stable_loss(pred, target):
        return torch.nn.functional.huber_loss(pred, target, delta=1.0)
```

#### Performance Issues
```python
# Solution: Performance optimization
def optimize_performance():
    """Performance improvement strategies."""

    # 1. JIT compilation
    model = torch.jit.script(model)

    # 2. TensorRT optimization
    # Convert to TensorRT engine for deployment

    # 3. Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 4. Asynchronous processing
    # Use asyncio for concurrent requests
```

---

## Version History

- **v1.3.0** (2025-08-26): Enhanced GPU optimization, improved memory efficiency
- **v1.2.0** (2025-07-15): Added conformal prediction, improved uncertainty quantification
- **v1.1.0** (2025-06-01): Ψ(x) framework integration, risk assessment improvements
- **v1.0.0** (2025-04-20): Initial release with core hybrid functionality

---

**This API reference guide provides comprehensive documentation for the Hybrid UQ framework, enabling developers to effectively integrate uncertainty quantification into their scientific computing applications.**
