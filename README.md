# svmix

**Ensemble Stochastic Volatility Filter with Bayesian Model Averaging**

A high-performance C library with Python bindings for real-time volatility estimation using mixtures of stochastic volatility models. Designed for production deployment in quantitative finance applications.

---

## Overview

svmix implements online Bayesian inference for stochastic volatility using an ensemble of particle filters with adaptive model weighting. The library maintains K independent SV models, each running its own particle filter, and continuously updates model weights based on predictive likelihood.

### Key Features

- **Bayesian Model Averaging**: Automatically adapts to changing market regimes without manual re-estimation
- **High Performance**: C core with OpenMP parallelization, processing 500+ observations per second
- **Production Ready**: Deterministic execution, checkpoint/restore, comprehensive test coverage
- **Python Integration**: Clean Pythonic API with NumPy support
- **Validated**: Outperforms GARCH(1,1) by 3-12% in predictive log-likelihood on real market data

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         svmix Ensemble                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Model 1 │  │ Model 2 │  │ Model 3 │  ...  │ Model K │       │
│  │  (θ₁)   │  │  (θ₂)   │  │  (θ₃)   │       │  (θₖ)   │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                 │             │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐       ┌────▼────┐       │
│  │   PF    │  │   PF    │  │   PF    │  ...  │   PF    │       │
│  │(N part.)│  │(N part.)│  │(N part.)│       │(N part.)│       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                 │             │
│       └────────────┴─────┬──────┴─────────────────┘             │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │  Mixture  │  Weighted average of posteriors│
│                    │  Belief   │  w₁p(h|M₁) + ... + wₖp(h|Mₖ)  │
│                    └───────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- GCC or Clang with C99 support
- Python 3.8+ (for Python bindings)
- NumPy 1.20+ (for Python bindings)
- OpenMP (optional, for parallel execution)

### Building from Source

```bash
# Clone repository with submodules
git clone --recursive https://github.com/TiaanViviers/svmix.git
cd svmix

# Build C library and run tests
make test

# Build Python bindings
make pylib

# Install Python package (development mode)
make pyinstall
```

### Verifying Installation

```bash
# Run C tests
make test

# Run Python tests
make pytest
```

---

## Documentation

**Python API Documentation**: Comprehensive Sphinx documentation is available in `python/docs/`:

```bash
cd python/docs
pip install -r requirements.txt  # Install Sphinx and theme
make view                        # Build and open in browser
```

The documentation includes:
- **Quick Start Guide**: Step-by-step tutorial with examples
- **API Reference**: Complete documentation of all classes and methods
- **Theory**: Mathematical foundations and algorithm details

**C API Documentation**: See `include/svmix.h` for the complete C API with inline documentation.

---

## Quick Start

### Python

```python
from svmix import Svmix, SvmixConfig, SvParams, Spec

# Configure the ensemble filter
config = SvmixConfig(
    spec=Spec.VOL,
    num_models=50,           # K: number of models in ensemble
    num_particles=500,       # N: particles per model
    lambda_=0.995,           # Exponential forgetting factor
    beta=0.8,                # Softmax temperature
    epsilon=0.05,            # Anti-starvation weight floor
    seed=42                  # For reproducibility
)

# Generate parameter grid spanning the model space
params = SvParams.linspace(
    num_models=50,
    phi=(0.90, 0.99),        # Persistence range
    sigma=(0.10, 0.30),      # Vol-of-vol range
    nu=10.0,                 # Degrees of freedom
    mu=-0.5                  # Long-run mean of log-variance
)

# Create filter
svmix = Svmix(config, params)

# Process observations (e.g., 1-minute log returns)
for return_t in returns:
    svmix.step(return_t)
    
    # Get filtered volatility estimate
    belief = svmix.get_belief()
    volatility = belief.mean_sigma
    
    # Get model weights (for diagnostics)
    weights = svmix.get_weights()

# Cleanup
svmix.free()
```

### C

```c
#include "svmix.h"

int main(void) {
    // Configure ensemble
    svmix_cfg_t cfg = {
        .num_models = 50,
        .num_particles = 500,
        .spec = SVMIX_SPEC_VOL,
        .ensemble = {
            .lambda = 0.995,
            .beta = 0.8,
            .epsilon = 0.05,
            .num_threads = 0  // Auto-detect
        }
    };

    // Define model parameters (example: 3 models)
    svmix_sv_params_t models[3] = {
        {.mu_h = -0.5, .phi_h = 0.96, .sigma_h = 0.15, .nu = 10.0},
        {.mu_h = -0.5, .phi_h = 0.98, .sigma_h = 0.20, .nu = 10.0},
        {.mu_h = -0.5, .phi_h = 0.99, .sigma_h = 0.25, .nu = 10.0}
    };
    unsigned long seeds[3] = {42, 43, 44};

    // Create filter
    svmix_t* sv = svmix_create(&cfg, models, seeds);
    if (!sv) return 1;

    // Process observations
    for (int t = 0; t < T; t++) {
        svmix_step(sv, returns[t]);
        
        svmix_belief_t belief;
        svmix_get_belief(sv, &belief);
        printf("t=%d: volatility=%.6f\n", t, belief.mean_sigma);
    }

    svmix_free(sv);
    return 0;
}
```

---

## Model Specification

svmix implements the standard stochastic volatility model with Student-t observations:

**State Equation (Log-Volatility Process)**

$$h_t = \mu_h + \phi_h (h_{t-1} - \mu_h) + \sigma_h \eta_t, \quad \eta_t \sim N(0,1)$$

**Observation Equation**

$$r_t \sim \text{Student-t}(\nu, 0, \exp(h_t/2))$$

### Parameters

| Parameter | Symbol | Description | Constraints |
|-----------|--------|-------------|-------------|
| `mu_h` | μ | Long-run mean of log-variance | Any real value |
| `phi_h` | φ | Persistence of log-variance | 0 < φ < 1 |
| `sigma_h` | σ | Volatility of volatility | σ > 0 |
| `nu` | ν | Degrees of freedom (tail fatness) | ν > 2 |

### Ensemble Weighting

Model weights are updated via exponential forgetting and tempered softmax:

1. **Score Update**: $S_i^{(t)} = \lambda \cdot S_i^{(t-1)} + \log p(y_t \mid y_{1:t-1}, M_i)$
2. **Softmax**: $w_i \propto \exp(\beta \cdot S_i)$
3. **Anti-starvation**: $w_i \leftarrow (1-\epsilon) w_i + \epsilon/K$

---

## API Reference

### Python API

#### `SvmixConfig`

Configuration dataclass for filter creation.

```python
SvmixConfig(
    spec: Spec,              # Model specification (Spec.VOL)
    num_models: int,         # K: ensemble size
    num_particles: int,      # N: particles per model
    lambda_: float,          # Forgetting factor (0 < λ ≤ 1)
    beta: float,             # Softmax temperature (β > 0)
    epsilon: float,          # Anti-starvation (0 ≤ ε < 1)
    num_threads: int = 0,    # OpenMP threads (0 = auto)
    seed: int = 0            # RNG seed (0 = random)
)
```

#### `SvParams`

Model parameter container with factory methods.

```python
# Create single parameter set
params = SvParams(phi=0.97, sigma=0.2, nu=10, mu=-0.5)

# Generate linearly-spaced grid
params = SvParams.linspace(
    num_models=50,
    phi=(0.90, 0.99),     # Range or single value
    sigma=(0.10, 0.30),
    nu=10,
    mu=-0.5
)

# Generate Cartesian product grid
params = SvParams.grid(
    phi=[0.95, 0.97, 0.99],
    sigma=[0.1, 0.2, 0.3],
    nu=[8, 10, 12],
    mu=-0.5
)
```

#### `Svmix`

Main filter class.

| Method | Description |
|--------|-------------|
| `Svmix(config, params)` | Create filter instance |
| `step(observation)` | Process one observation |
| `get_belief()` | Get current belief state (mean_h, var_h, mean_sigma) |
| `get_weights()` | Get model weights (K-dimensional array) |
| `get_last_log_likelihood()` | Get predictive log-likelihood from last step |
| `save_checkpoint(path)` | Save complete state to file |
| `load_checkpoint(path)` | Restore state from file (class method) |
| `free()` | Release resources |

#### `Belief`

Posterior belief summary.

| Attribute | Description |
|-----------|-------------|
| `mean_h` | Posterior mean of log-variance |
| `var_h` | Posterior variance of log-variance |
| `mean_sigma` | Approximate volatility: exp(mean_h/2) |
| `valid` | Whether belief is valid |

For more examples on how to use the Python API, please see the `perf/` 
directory.

### C API

See `include/svmix.h` for complete documentation. Key functions:

```c
// Lifecycle
svmix_t* svmix_create(const svmix_cfg_t* cfg, 
                      const svmix_sv_params_t* models,
                      const unsigned long* seeds);
void svmix_free(svmix_t* svmix);

// Filtering
svmix_status_t svmix_step(svmix_t* svmix, double observation);
svmix_status_t svmix_get_belief(const svmix_t* svmix, svmix_belief_t* belief);
svmix_status_t svmix_get_weights(const svmix_t* svmix, double* weights, size_t K);
double svmix_get_last_log_likelihood(const svmix_t* svmix);

// Checkpointing
svmix_status_t svmix_save_checkpoint(const svmix_t* svmix, const char* filepath);
svmix_t* svmix_load_checkpoint(const char* filepath, svmix_status_t* status);
```

---

## Configuration Guide

### Recommended Configurations

**Real-time Trading (Low Latency)**
```python
config = SvmixConfig(
    spec=Spec.VOL,
    num_models=20,
    num_particles=250,
    lambda_=0.99,
    beta=0.8,
    epsilon=0.02
)
```
- Throughput: 2000+ obs/s
- Use case: HFT, real-time risk

**Production (Balanced)**
```python
config = SvmixConfig(
    spec=Spec.VOL,
    num_models=100,
    num_particles=500,
    lambda_=0.995,
    beta=0.8,
    epsilon=0.05
)
```
- Throughput: 500+ obs/s
- Use case: Daily operations, risk management

**Research (Maximum Accuracy)**
```python
config = SvmixConfig(
    spec=Spec.VOL,
    num_models=150,
    num_particles=500,
    lambda_=0.995,
    beta=0.8,
    epsilon=0.05
)
params = SvParams.linspace(
    150,
    phi=(0.85, 0.995),
    sigma=(0.10, 0.50),
    nu=(6.0, 14.0),
    mu=-0.5
)
```
- Throughput: 500+ obs/s
- Use case: Backtesting, model validation

### Parameter Guidelines

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| K (num_models) | More models = better regime coverage | 20-150 |
| N (num_particles) | Diminishing returns beyond 500 | 250-500 |
| λ (lambda_) | Higher = slower adaptation | 0.99-0.999 |
| β (beta) | Higher = more aggressive weighting | 0.5-1.0 |
| ε (epsilon) | Prevents model starvation | 0.02-0.10 |

---

## Performance

### Benchmarks

Tested on Intel Core i7-12700H (14 cores), Ubuntu 22.04:

| Configuration | Throughput | Latency |
|--------------|------------|---------|
| K=20, N=250 | 2,100 obs/s | 0.48 ms |
| K=50, N=500 | 850 obs/s | 1.18 ms |
| K=100, N=500 | 680 obs/s | 1.92 ms |
| K=150, N=500 | 520 obs/s | 1.92 ms |

### Validation Results

Evaluated on US30 (Dow Jones) 1-minute data across multiple market regimes (2008-2025):

| Metric | svmix (K=150) | GARCH-Normal | GARCH-t |
|--------|---------------|--------------|---------|
| Mean PLL | **6.36** | 6.17 | 5.69 |
| Coverage (95% CI) | **96.6%** | 93.9% | 94.1% |
| VaR Violations | **3.0%** | 4.8% | 4.6% |

**Long-term stability test** (1.7M observations, 5 years):
- No re-estimation required
- Coverage stable at 97.1%
- No significant performance drift

---

## Checkpointing

svmix supports complete state serialization for crash recovery and deployment:

```python
# Save state
svmix.save_checkpoint("model_state.svmix")

# Restore and continue
svmix = Svmix.load_checkpoint("model_state.svmix")
svmix.step(next_observation)  # Continue filtering
```

The checkpoint format includes:
- All particle states and weights
- RNG state (for deterministic continuation)
- Model parameters and configuration
- Score accumulators

---

## Project Structure

```
svmix/
├── include/
│   └── svmix.h              # Public C API
├── src/
│   ├── svmix.c              # API implementation
│   ├── ensemble.c           # Ensemble management
│   ├── model_sv.c           # SV model callbacks
│   ├── checkpoint.c         # Serialization
│   └── util.c               # Numerical utilities
├── python/
│   └── svmix/
│       ├── __init__.py      # Public Python API
│       ├── core.py          # Main Svmix class
│       ├── config.py        # Configuration
│       ├── params.py        # Parameter generation
│       └── types.py         # Data types
├── third_party/
│   └── fastpf/              # Particle filter engine (submodule)
├── tests/                   # C test suite
├── perf/                    # Benchmarks and validation
├── examples/                # C usage examples
└── Makefile                 # Build system
```

---

## Building

### Make Targets

| Target | Description |
|--------|-------------|
| `make test` | Build and run all C tests |
| `make pylib` | Build shared library for Python |
| `make pyinstall` | Install Python package (editable) |
| `make pytest` | Run Python test suite |
| `make examples` | Build C examples |
| `make clean` | Remove build artifacts |

### Build Options

```bash
# Enable OpenMP (recommended for production)
make OPENMP=1 pylib

# Build with debug symbols
make CFLAGS="-g -O0" test

# Run with sanitizers
make sanitizer
```

---

## Testing

### C Tests

```bash
# All tests
make test

# Individual test suites
make test-unit          # Unit tests
make test-integration   # Integration tests
make valgrind           # Memory leak detection
make sanitizer          # Address/UB sanitizers
```

### Python Tests

```bash
# Run all Python tests
make pytest

# With coverage
cd python && pytest --cov=svmix tests/
```

---

## References

The methodology is based on established literature in stochastic volatility modeling and Bayesian filtering:

- Jacquier, E., Polson, N. G., & Rossi, P. E. (1994). "Bayesian Analysis of Stochastic Volatility Models." *Journal of Business & Economic Statistics*, 12(4), 371-389.

- Kim, S., Shephard, N., & Chib, S. (1998). "Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models." *Review of Economic Studies*, 65(3), 361-393.

- Pitt, M. K., & Shephard, N. (1999). "Filtering via Simulation: Auxiliary Particle Filters." *Journal of the American Statistical Association*, 94(446), 590-599.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/TiaanViviers/svmix/issues).
