"""
Comprehensive performance benchmark across all dimensions.

Systematically tests performance across:
- Thread counts (OpenMP parallelization)
- Model counts (ensemble size K)
- Particle counts (accuracy N)

Generates CSV results and interactive visualizations.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import svmix
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import config_model, speed_test, read_data
from plot_performance import plot_3d_scatter



# Benchmark grid
THREAD_COUNTS = [1, 4, 8, 12, 16]
MODEL_COUNTS = [5, 10, 50, 100, 200]
PARTICLE_COUNTS = [250, 1000, 5000, 10_000]


def main():
    """Run comprehensive benchmark across all dimensions."""
    print("=" * 80)
    print("Comprehensive Performance Benchmark")
    print("=" * 80)
    print(f"  Thread counts:    {THREAD_COUNTS}")
    print(f"  Model counts (K): {MODEL_COUNTS}")
    print(f"  Particle counts:  {PARTICLE_COUNTS}")
    print(f"  Total configs:    {len(THREAD_COUNTS) * len(MODEL_COUNTS) * len(PARTICLE_COUNTS)}")
    
    observations = read_data()
    num_obs = len(observations)
    print(f"  Observations:     {num_obs}")
    print("=" * 80)
    print()
    
    # Estimate runtime
    total_configs = len(THREAD_COUNTS) * len(MODEL_COUNTS) * len(PARTICLE_COUNTS)
    print(f"Running {total_configs} configurations...")
    print()
    
    # Run benchmarks
    results = []
    config_num = 0
    
    for threads in THREAD_COUNTS:
        for K in MODEL_COUNTS:
            for N in PARTICLE_COUNTS:
                config_num += 1
                
                # Run benchmark
                model = config_model(K, N, threads)
                elapsed = speed_test(model, observations)
                model.free()
                
                # Calculate metrics
                time_per_obs = elapsed / num_obs
                throughput = num_obs / elapsed
                
                results.append({
                    'threads': threads,
                    'K': K,
                    'N': N,
                    'time': elapsed,
                    'time_per_obs': time_per_obs,
                    'throughput': throughput
                })
                
                # Progress indicator
                print(f"[{config_num:3d}/{total_configs}] "
                        f"threads={threads:2d}, K={K:3d}, N={N:5d}: "
                        f"{throughput:7.1f} obs/s")
    
    print()
    print("=" * 80)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print_summary(df, num_obs)
    
    
    print("\nGenerating visualizations...")
    try:
        plot_3d_scatter(df)
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")
    
    print("=" * 80)


def print_summary(df, num_obs):
    """Print comprehensive summary statistics."""
    best = df.loc[df['throughput'].idxmax()]
    print(f"  Best overall throughput:")
    print(f"    {best['throughput']:.1f} obs/s")
    print(f"    (threads={int(best['threads'])}, K={int(best['K'])}, N={int(best['N'])})")
    print()
    
    

if __name__ == "__main__":
    main()
