"""
Benchmark: Particle count (N) scaling analysis.

Tests how performance scales with the number of particles per model.
Higher N = better accuracy but more computation.
Goal: Find diminishing returns point.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import config_model, speed_test, read_data


# Benchmark parameters
PARTICLE_COUNTS = [100, 250, 500, 1000, 2000, 5000, 10000]
MODEL_COUNT = 20      # Fixed
THREAD_COUNT = 8      # Fixed (best from thread benchmark)


def main():
    """Run particle scaling benchmark."""
    print("=" * 80)
    print("Particle Scaling Benchmark (N)")
    print("=" * 80)
    print(f"  Model count (K):      {MODEL_COUNT} (fixed)")
    print(f"  Thread count:         {THREAD_COUNT} (fixed)")
    print(f"  Particle counts (N):  {PARTICLE_COUNTS}")
    
    observations = read_data()
    num_obs = len(observations)
    print(f"  Observations:         {num_obs}")
    print("=" * 80)
    print()
    
    # Run benchmarks
    results = []
    for particle_count in PARTICLE_COUNTS:
        model = config_model(MODEL_COUNT, particle_count, THREAD_COUNT)
        elapsed = speed_test(model, observations)
        model.free()
        
        results.append({
            'N': particle_count,
            'time': elapsed,
            'time_per_obs': elapsed / num_obs
        })
        
        print_result(
            {'N': particle_count},
            elapsed,
            num_obs
        )
    
    # Summary
    print()
    print("=" * 80)
    print_summary(results, num_obs)
    print("=" * 80)


def print_summary(results, num_obs):
    """Print particle scaling summary."""
    print("Summary:")
    print()
    
    # Scaling analysis
    print("  Scaling analysis:")
    print(f"  {'N (particles)':<15} {'Time (s)':<12} {'Time/obs (ms)':<15} {'Throughput (obs/s)':<20}")
    print("  " + "-" * 62)
    
    for r in results:
        throughput = num_obs / r['time']
        print(f"  {r['N']:<15} {r['time']:<12.3f} {r['time_per_obs']*1000:<15.2f} {throughput:<20.1f}")
    
    # Linearity check (expect O(N) or slightly worse due to resampling)
    print()
    print("  Scaling efficiency (time ratio vs N ratio):")
    print("  " + "-" * 45)
    
    baseline = results[0]
    for r in results[1:]:
        n_ratio = r['N'] / baseline['N']
        time_ratio = r['time'] / baseline['time']
        efficiency = (n_ratio / time_ratio) * 100  # Perfect linear = 100%
        
        print(f"  N={baseline['N']:5d} → N={r['N']:5d}: "
              f"Time ratio = {time_ratio:5.2f}x, N ratio = {n_ratio:5.2f}x, "
              f"Efficiency = {efficiency:5.1f}%")
    
    # Cost per particle
    print()
    print("  Incremental cost:")
    print("  " + "-" * 45)
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        delta_n = curr['N'] - prev['N']
        delta_time = curr['time'] - prev['time']
        cost_per_particle = (delta_time / num_obs / delta_n) * 1e6  # microseconds
        
        print(f"  N={prev['N']:5d} → N={curr['N']:5d}: "
              f"+{delta_n:5d} particles = +{delta_time:.3f}s "
              f"({cost_per_particle:.2f} µs/particle/obs)")


def print_result(config_params, elapsed_time, num_obs):
    """Print a single benchmark result."""
    parts = [f"{k}={v}" for k, v in config_params.items()]
    param_str = ", ".join(parts)
    
    time_per_obs = elapsed_time / num_obs
    throughput = num_obs / elapsed_time
    
    print(f"{param_str:<20} Total: {elapsed_time:6.3f} s   "
          f"Per obs: {time_per_obs*1000:5.2f} ms   "
          f"Throughput: {throughput:6.1f} obs/s")


if __name__ == "__main__":
    main()
