# Synthetic Accuracy Benchmarks

This directory contains benchmarks that evaluate the **accuracy** of the svmix particle filter using synthetic data with known ground truth.

## Overview

These benchmarks explore the **fundamental tradeoffs** in particle filter and stochastic 
volatility design:
- **K (ensemble size)**: How many models?
- **N (particles per model)**: How many particles?
- **Grid width**: How broad should parameter coverage be?
- **K × Width frontier**: How do these interact?

**Key insight:** There are optimal configurations that balance precision (narrow grids) with robustness (wide grids).

## Key Findings

### 1. **Ensemble Size (K) Matters More Than Particles (N)**

**From `ensemble_accuracy_scaling.py` and `particle_accuracy_scaling.py`:**

- **K scaling**: Dramatic improvement from K=1 → K=5 (10-21% RMSE reduction)
- **N scaling**: Minimal improvement from N=100 → N=10,000 (<1% RMSE change)

**Insight:** Ensemble diversity (different models) >> Particle count (sampling within a model)

**Recommendation:** Use K=5-20 for most applications, N=100-250 is sufficient.

---

### 2. **Grid Width Creates a Precision vs Robustness Tradeoff**

**From `grid_width_tradeoff.py`:**

| Grid Width | Stable Performance | Regime Change Performance |
|------------|-------------------|---------------------------|
| **Ultra-narrow** | Excellent (RMSE=0.51) | Poor (Coverage=81%) |
| **Medium** | Good (RMSE=0.53) | Good (Coverage=95%) |
| **Wide** | Fair (RMSE=0.54) | Excellent (Coverage=96%) |

**Insight:** Narrow grids optimize for known regimes but fail during shifts. Wide grids sacrifice 5% precision for 15% better coverage during crises.

**Recommendation:** 
- Use **narrow grids** if regime is stable and known
- Use **medium/wide grids** for production systems that must handle regime changes

---

### 3. **The K × Width Frontier: Wider Grids Benefit from More Models**

**From `grid_width_vs_k_frontier.py`:**

**Narrow grids plateau quickly:**
- K=20: Excellent performance
- K=100: No improvement (already dense coverage)

**Wide grids improve with K:**
- K=20: Sparse coverage, underperforms
- K=150: Dense coverage, matches narrow grid precision while maintaining robustness

**Insight:** Wide grids need more models to "fill in" the sparse parameter space. The optimal K depends on grid width.

**Frontier relationship:**
```
Narrow grid:  Optimal K = 20-50
Medium grid:  Optimal K = 50-100  
Wide grid:    Optimal K = 100-150
Ultra-wide:   Optimal K = 150-200
```

---

## Production Recommendations

Based on extensive benchmarking across stable scenarios, regime changes, and frontier analysis:

### **For Known Stable Regimes:**
```python
K = 20-50        # Ensemble size
N = 250          # Particles (more is wasteful)
Width = Narrow   # φ ∈ [0.91, 0.98], σ ∈ [0.15, 0.30]
```

### **For Multi-Year Production (No Intervention):**
```python
K = 100-150      # Larger ensemble for coverage
N = 500          # Conservative safety margin
Width = Wide     # φ ∈ [0.85, 0.995], σ ∈ [0.10, 0.50]
```
- Handles 2× volatility jumps
- 96%+ coverage during regime changes
- ~5s runtime per 3000 observations

### **Key Tradeoff:**
- Narrow grids: 5% better precision in stable conditions
- Wide grids: 15% better coverage during crises

**For long-running systems, robustness > precision.**

---

---

## Benchmark Scripts

### Core Benchmarks (Run These First)

#### 1. `ensemble_accuracy_scaling.py`
Tests how ensemble size (K) affects accuracy.

```bash
python ensemble_accuracy_scaling.py --threads 8
```

**Tests:** K ∈ [1, 5, 20, 50, 100] across 4 scenarios  
**Key Result:** K=5-10 gives best accuracy/cost tradeoff for narrow grids

---

#### 2. `particle_accuracy_scaling.py`
Tests how particle count (N) affects accuracy.

```bash
python particle_accuracy_scaling.py --threads 8 --models 10
```

**Tests:** N ∈ [100, 250, 500, 1000, 2500, 5000, 10000]  
**Key Result:** N=100-250 is sufficient; more particles give <1% improvement

---

#### 3. `grid_width_tradeoff.py`
Tests precision vs robustness tradeoff across grid widths.

```bash
python grid_width_tradeoff.py --threads 8
```

**Tests:** 5 grid widths (ultra-narrow → ultra-wide) on stable scenarios + regime changes  
**Key Result:** Medium grid best for known regimes, wide grid best for production robustness

---

#### 4. `grid_width_vs_k_frontier.py` 
**Most comprehensive benchmark.** Tests K × Width frontier to find optimal configurations.

```bash
python grid_width_vs_k_frontier.py --threads 8
```

**Tests:** 5 widths × 5 K values = 25 configurations  
**Key Result:** Wide grids need K=100-150 to achieve dense coverage; narrow grids plateau at K=20-50

---

## Metrics

All benchmarks use:

- **RMSE**: Root mean squared error between true and estimated log-volatility
- **95% Coverage**: Percentage of true values inside 95% credible intervals (target: 95%)
- **Mean PLL**: Average predictive log-likelihood (higher is better)

**Coverage interpretation:**
- <90%: Overconfident (intervals too narrow)
- 93-97%: Well-calibrated
- >98%: Underconfident (intervals too wide)

---

## Scenarios

All benchmarks test on:

- **baseline**: Moderate persistence (φ=0.95), moderate volatility (σ=0.20)
- **high_persistence**: Very high persistence (φ=0.98), near random walk
- **high_volatility**: High volatility of volatility (σ=0.35)
- **heavy_tails**: Heavy-tailed shocks (ν=5.0)

Plus **regime change tests** that shift parameters mid-sequence to test adaptation.

**Key findings:**
- Varying σ is essential for high-volatility scenarios (9% RMSE improvement, 12% coverage improvement)
- Narrow φ×σ grid best overall (φ∈[0.93,0.99], σ∈[0.10,0.40])
- K=20 gives 99% of K=100's accuracy at 20% of the cost
- Multi-dimensional grids (φ×σ×ν) excel for heavy-tailed data

---

### `grid_width_tradeoff.py`

Tests the fundamental tradeoff: **Does a wider parameter grid hurt accuracy in stable conditions, or is wider always better?**

This benchmark answers the key question for long-running production systems: Should we optimize for specific regimes (narrow grid) or maximize robustness (wide grid)?

**Grid widths tested (all K=50, N=500):**
1. **Ultra-Narrow**: φ∈[0.94,0.98], σ∈[0.15,0.25] - Optimized for baseline
2. **Narrow**: φ∈[0.92,0.99], σ∈[0.12,0.35] - Typical conditions
3. **Medium**: φ∈[0.88,0.995], σ∈[0.10,0.45] - Broad coverage
4. **Wide**: φ∈[0.85,0.998], σ∈[0.08,0.55] - Very broad
5. **Ultra-Wide**: φ∈[0.75,0.998], σ∈[0.06,0.70] - Extreme coverage

**Test scenarios:**
- **Stable**: Test on baseline, high_persistence, high_volatility with fixed parameters
- **Regime Changes**: Test adaptation when parameters shift mid-sequence
  - Baseline → High Volatility
  - High Persistence → Baseline  
  - Baseline → Heavy Tails

**Metrics:**
- Overall RMSE and coverage
- Per-regime RMSE and coverage
- Adaptation speed (RMSE in first 100 steps after regime change)

**Usage:**
```bash
# Run full benchmark
python grid_width_tradeoff.py --threads 8

# Different seed
python grid_width_tradeoff.py --seed 123
```

**Output:**
Three tables comparing grid widths:
1. Stable scenario performance (does wide grid hurt?)
2. Regime change overall + per-regime RMSE
3. Coverage during regime changes

**Key questions answered:**
- Is there a "too wide" point where accuracy degrades?
- How much does grid width help during regime changes?
- What's the optimal width for multi-year production systems?

---

### `particle_accuracy_scaling.py`

Tests how particle count (N) affects filter accuracy across different scenarios.

**Configuration:**
- N values: [100, 250, 500, 1000, 2500, 5000, 10000]
- Scenarios: baseline, high_persistence, high_volatility, heavy_tails
- Fixed: K=10 models, T=5000
- Metrics: RMSE, mean PLL, 95% coverage

**Usage:**
```bash
# Run with default settings (K=10, T=5000)
python particle_accuracy_scaling.py

# Customize ensemble size and threads
python particle_accuracy_scaling.py --models 20 --threads 8

# Set random seed for reproducibility
python particle_accuracy_scaling.py --seed 123
```

**Output:**
Formatted tables showing:
1. RMSE vs N for each scenario
2. Mean PLL vs N for each scenario
3. 95% coverage vs N for each scenario
4. Runtime vs N for each scenario

**Expected patterns:**
- RMSE should decrease as N increases (better state estimation)
- PLL should improve with N (better predictive distributions)
- Coverage should approach 95% as N increases (better uncertainty quantification)
- Diminishing returns beyond N=1000-2500 in most scenarios
- High volatility scenarios may benefit from larger N

---

## Quick Start

**Run the full benchmark suite:**
```bash
cd perf/synthetic

# 1. Test ensemble size (K) scaling (~80s)
python ensemble_accuracy_scaling.py --threads 8

# 2. Test particle count (N) scaling (~100s)
python particle_accuracy_scaling.py --threads 8

# 3. Test grid width tradeoffs (~45s)
python grid_width_tradeoff.py --threads 8

# 4. Test K × Width frontier (~220s)
python grid_width_vs_k_frontier.py --threads 8
```

---

## When to Use Each Benchmark

| Question | Run This Benchmark |
|----------|-------------------|
| "How many models do I need?" | `ensemble_accuracy_scaling.py` |
| "How many particles do I need?" | `particle_accuracy_scaling.py` |
| "Should I use wide or narrow grids?" | `grid_width_tradeoff.py` |
| "What's the optimal (K, width) combo?" | `grid_width_vs_k_frontier.py` |

---

