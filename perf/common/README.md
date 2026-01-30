# Accuracy Metrics

This module provides standard metrics for evaluating svmix filter performance.

## Metrics Overview

### 1. RMSE (Root Mean Squared Error)
**Requires:** Ground truth (synthetic data only)

Measures the typical magnitude of estimation error between true states and filter estimates.

```python
from metrics import rmse

rmse_h = rmse(h_true, h_estimated)
# Typical values: 0.05-0.20 for log-volatility
```

**Interpretation:**
- Lower is better
- Scale-dependent (compare within same state variable only)
- Sensitive to outliers

**Use case:** Compare different configurations (K, N) on synthetic data

---

### 2. PLL (Predictive Log-Likelihood)
**Requires:** Observations only (works on real data)

Measures how well the filter predicts future observations. Does NOT require ground truth.

```python
from metrics import pll_sequence, mean_pll

# Total PLL (scales with sequence length)
total_pll = pll_sequence(observations, log_likelihoods)

# Mean PLL (normalized, comparable across lengths)
avg_pll = mean_pll(observations, log_likelihoods)
```

**Interpretation:**
- Higher is better (less negative)
- Relative metric: only useful for comparing models
- Proper scoring rule (incentivizes honest forecasts)
- Difference of 10+ is strong evidence

**Use case:** Model selection, compare K values, compare specifications

---

### 3. Coverage (95% Credible Interval Coverage)
**Requires:** Ground truth + posterior variances

Checks what percentage of true values fall within the filter's 95% credible intervals. Tests calibration.

```python
from metrics import coverage_95, validate_coverage_calibration

cov = coverage_95(h_true, estimated_means, estimated_variances)
status = validate_coverage_calibration(cov)
# status: 'well-calibrated', 'overconfident', or 'underconfident'
```

**Interpretation:**
- Target: ~95%
- < 90%: Overconfident (underestimates uncertainty)
- 90-98%: Well-calibrated
- > 98%: Underconfident (overestimates uncertainty)

**Use case:** Check if posterior uncertainty estimates are reliable

---

## Usage Example

### Synthetic Data (All Metrics)

```python
import sys
sys.path.insert(0, 'python')
sys.path.insert(0, 'perf/common')

from svmix import Svmix, SvmixConfig, SvParamsVol, Spec
from metrics import compute_all_metrics, format_metrics

# Create filter
config = SvmixConfig(spec=Spec.VOL, num_models=50, num_particles=1000)
params = SvParamsVol.linspace(50, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
svmix = Svmix(config, params)

# Run filter and collect data
h_true = [...]  # Ground truth from synthetic generator
h_estimated = []
h_variances = []
log_likelihoods = []

for obs in observations:
    svmix.step(obs)
    belief = svmix.get_belief()
    h_estimated.append(belief.mean_h)
    h_variances.append(belief.var_h)
    log_likelihoods.append(svmix.get_last_log_likelihood())

# Compute all metrics
metrics = compute_all_metrics(
    observations=observations,
    log_likelihoods=log_likelihoods,
    true_states=h_true,
    estimated_means=h_estimated,
    estimated_variances=h_variances
)

print(format_metrics(metrics))
# Output:
# PLL (total): -125.4000
# PLL (mean):  -0.5016
# RMSE:        0.0845
# Coverage:    96.0%
```

### Real Data (PLL Only)

```python
# Run filter
log_likelihoods = []
for obs in observations:
    svmix.step(obs)
    log_likelihoods.append(svmix.get_last_log_likelihood())

# Compute PLL
metrics = compute_all_metrics(observations, log_likelihoods)
print(f"Mean PLL: {metrics['pll_mean']:.4f}")
```

---

## Guidelines

### Comparing Configurations

**K (number of models):**
- More models → Better RMSE, Better PLL, Same coverage
- Diminishing returns after K=50-100

**N (number of particles):**
- More particles → Better RMSE, Better coverage
- PLL relatively insensitive (averaged over models)
- Diminishing returns after N=1000-5000

**Threads:**
- No effect on accuracy (only speed)
- Always verify determinism with same seed

### Expected Values (Synthetic SV Data)

Typical performance on synthetic data (T=1000):

| Configuration | RMSE | Coverage | Mean PLL |
|---------------|------|----------|----------|
| K=5, N=250    | 0.18 | 92%      | -0.55    |
| K=20, N=1000  | 0.10 | 94%      | -0.52    |
| K=100, N=5000 | 0.06 | 96%      | -0.51    |

*(Values approximate, depend on true parameters)*

---

## Implementation Notes

- All functions work with plain Python lists (no numpy required)
- `compute_all_metrics()` automatically computes applicable metrics based on available data
- `format_metrics()` provides human-readable output
- Self-tests included: run `python3 metrics.py` to verify
