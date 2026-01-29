"""
Benchmark: Ensemble size (K) scaling analysis.

Tests how performance scales with the number of models in the ensemble.
Theoretical expectation: O(K) linear scaling.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import config_model, speed_test, read_data

MODEL_COUNTS = [1, 5, 10, 20, 50, 100, 200]
PARTICLE_COUNT = 1000
THREAD_COUNT = 8


def main():
    """Run ensemble scaling benchmark."""
    print("=" * 80)
    print("Ensemble Scaling Benchmark (K)")
    print("=" * 80)
    print(f"  Particle count (N): {PARTICLE_COUNT} (fixed)")
    print(f"  Thread count:       {THREAD_COUNT} (fixed)")
    print(f"  Model counts (K):   {MODEL_COUNTS}")
    
    observations = read_data()
    num_obs = len(observations)
    print(f"  Observations:       {num_obs}")
    print("=" * 80)
    print()
    
    # Run benchmarks
    results = []
    for model_count in MODEL_COUNTS:
        model = config_model(model_count, PARTICLE_COUNT, THREAD_COUNT)
        elapsed = speed_test(model, observations)
        model.free()
        
        results.append({
            'K': model_count,
            'time': elapsed,
            'time_per_obs': elapsed / num_obs
        })
        
        print_result(
            {'K': model_count},
            elapsed,
            num_obs
        )
    
    # Summary
    print()
    print("=" * 80)
    print_summary(results, num_obs)
    print("=" * 80)


def print_summary(results, num_obs):
    """Print ensemble scaling summary."""
    print("Summary:")
    print()
    
    # Scaling analysis
    print("  Scaling analysis:")
    print(f"  {'K (models)':<12} {'Time (s)':<12} {'Time/obs (ms)':<15} {'Throughput (obs/s)':<20}")
    print("  " + "-" * 59)
    
    for r in results:
        throughput = num_obs / r['time']
        print(f"  {r['K']:<12} {r['time']:<12.3f} {r['time_per_obs']*1000:<15.2f} {throughput:<20.1f}")
    
    # Linearity check
    print()
    print("  Linearity (time ratio vs K ratio):")
    print("  " + "-" * 40)
    
    baseline = results[0]
    for r in results[1:]:
        k_ratio = r['K'] / baseline['K']
        time_ratio = r['time'] / baseline['time']
        efficiency = (k_ratio / time_ratio) * 100  # Perfect linear = 100%
        
        print(f"  K={baseline['K']:3d} → K={r['K']:3d}: "
              f"Time ratio = {time_ratio:5.2f}x, K ratio = {k_ratio:5.2f}x, "
              f"Efficiency = {efficiency:5.1f}%")
        
        
def print_result(config_params, elapsed_time, num_obs):
    """
    Print a single benchmark result.
    
    Parameters:
    -----------
    config_params : dict
        Configuration parameters (K, N, threads, etc.)
    elapsed_time : float
        Total elapsed time in seconds
    num_obs : int
        Number of observations processed
    """
    # Build parameter string
    parts = [f"{k}={v}" for k, v in config_params.items()]
    param_str = ", ".join(parts)
    
    time_per_obs = elapsed_time / num_obs
    throughput = num_obs / elapsed_time
    
    print(f"{param_str} Total: {elapsed_time:6.3f} s   "
          f"Per obs: {time_per_obs*1000:5.2f} ms   "
          f"Throughput: {throughput:6.1f} obs/s")


if __name__ == "__main__":
    main()
