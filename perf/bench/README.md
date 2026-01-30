# Performance Benchmarks

Performance evaluation of the svmix particle filter across thread counts, ensemble sizes, and particle counts.

## Overview

```
bench/
├── bench_comprehensive.py       # Full parameter grid (threads × K × N)
├── bench_ensemble_scaling.py    # Model count (K) scaling
├── bench_particle_scaling.py    # Particle count (N) scaling
├── plot_performance.py          # 3D visualization module
└── performance_3d.html          # Interactive performance landscape
```

## Prerequisites

Build the library with OpenMP support from project root:
```bash
make clean && make pylib
```

Generate test data in `perf/bench/common` (if not present):
```bash
python generate_synthetic.py --T 500
```

Install visualization dependencies (optional):
```bash
pip install plotly pandas
```

## Running Benchmarks

### Comprehensive Benchmark

Tests all parameter combinations:
```bash
python bench_comprehensive.py
```

Outputs: `performance_3d.html` (interactive visualization)

### Focused Benchmarks

Quick analysis of individual dimensions:

```bash
python bench_ensemble_scaling.py

python bench_particle_scaling.py
```

## Interpreting Results

### Metrics

- **Throughput**: Observations per second (obs/s)
- **Time per observation**: Milliseconds per step (ms/obs)
- **Efficiency**: Ratio of theoretical to actual scaling (%)

### Performance Categories

| Category | Throughput | Latency |
|----------|-----------|---------|
| Real-time | >100 obs/s | <10 ms/obs |
| Interactive | >50 obs/s | <20 ms/obs |
| Batch | >10 obs/s | <100 ms/obs |

### Visualization

Open `performance_3d.html` in a browser to explore the performance landscape interactively. Each point represents a (threads, K, N) configuration colored by throughput.

## Notes

- Results are hardware-dependent
- All benchmarks use a fixed random seed for reproducibility
- OpenMP threading can be disabled at runtime (set threads=1)
