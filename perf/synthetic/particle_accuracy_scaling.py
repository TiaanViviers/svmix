#!/usr/bin/env python3
"""
Particle Accuracy Scaling Benchmark
====================================

Tests how particle count (N) affects filter accuracy across different scenarios.

Evaluates:
- N values: [100, 250, 500, 1000, 2500, 5000]
- Scenarios: baseline, high_persistence, high_volatility, heavy_tails
- Metrics: RMSE, mean PLL, 95% coverage
- Fixed: K=10 (moderate ensemble size), T=5000

Usage:
    python particle_accuracy_scaling.py [--models K] [--threads T] [--seed S]

Output:
    Formatted table showing accuracy metrics vs N for each scenario
"""

import sys
import argparse
import time
from pathlib import Path

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from svmix import Svmix, SvmixConfig, SvParams, Spec
from synthetic import SyntheticDataGenerator
from metrics import rmse, coverage_95
import numpy as np


def run_accuracy_test(observations, true_states, K, N, threads, seed):
    """
    Run filter on synthetic data and compute accuracy metrics.
    
    Parameters:
    -----------
    observations : np.ndarray
        Observed time series
    true_states : np.ndarray
        True latent log-volatility states
    K : int
        Number of ensemble members
    N : int
        Number of particles per model
    threads : int
        Number of OpenMP threads
    seed : int
        Random seed
        
    Returns:
    --------
    dict : Computed metrics (rmse, pll, coverage, runtime)
    """
    # Configure model
    config = SvmixConfig(
        spec=Spec.VOL,  # Estimate volatility only
        num_models=K,
        num_particles=N,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        num_threads=threads,
        seed=seed
    )
    
    # Create parameter grid centered around typical SV values
    # This creates K models with evenly spaced phi values
    params = SvParams.linspace(
        num_models=K,
        phi=(0.90, 0.99),  # Wide range to capture different persistence levels
        sigma=0.20,
        nu=10.0,
        mu=-0.5
    )
    
    # Initialize filter
    svmix = Svmix(config, params)
    
    # Run filter and collect beliefs + PLL
    T = len(observations)
    estimated_states = []
    estimated_variances = []
    pll_values = []
    
    start = time.time()
    for obs in observations:
        svmix.step(obs)
        belief = svmix.get_belief()
        
        # Collect mean log-volatility estimate
        estimated_states.append(belief.mean_h)
        
        # Collect variance (ensure non-negative)
        # Note: var_h should be non-negative, but numerical issues can occur
        var = max(belief.var_h, 1e-10)  # Floor at small positive value
        estimated_variances.append(var)
        
        # Collect predictive log-likelihood
        pll_values.append(svmix.get_last_log_likelihood())
    
    runtime = time.time() - start
    
    # Clean up
    svmix.free()
    
    # Convert to arrays
    estimated_states = np.array(estimated_states)
    estimated_variances = np.array(estimated_variances)
    
    # Compute accuracy metrics
    metrics = {
        'rmse': rmse(true_states, estimated_states),
        'mean_pll': np.mean(pll_values),  # Direct average of PLL values
        'coverage_95': coverage_95(true_states, estimated_states, estimated_variances),
        'runtime': runtime
    }
    
    return metrics


def run_scenario(scenario_name, N_values, K, threads, T, seed):
    """
    Run accuracy tests for a single scenario across all N values.
    
    Parameters:
    -----------
    scenario_name : str
        Name of the scenario to test
    N_values : list of int
        Particle counts to test
    K : int
        Number of ensemble members (fixed)
    threads : int
        Number of OpenMP threads
    T : int
        Sequence length
    seed : int
        Random seed
        
    Returns:
    --------
    dict : Results for each N value
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")
    
    # Generate synthetic data with ground truth
    generator = SyntheticDataGenerator()
    
    # Get standard parameters for this scenario
    params = generator.get_standard_params(scenario_name)
    
    # Remove description field and override T
    params = {k: v for k, v in params.items() if k != 'description'}
    params['T'] = T  # Override with our desired sequence length
    
    print(f"Generating T={T:,} observations...")
    data = generator.generate(**params, seed=seed)
    
    observations = data['observations']
    true_states = data['states']  # True log-volatility h_t
    
    print(f"True parameters: phi={data['params']['phi']:.3f}, "
          f"sigma={data['params']['sigma']:.3f}, "
          f"nu={data['params']['nu']:.1f}, "
          f"mu={data['params']['mu']:.3f}")
    
    # Test each N value
    results = {}
    for N in N_values:
        print(f"\nTesting N={N:4d} (K={K}, threads={threads})...", end=' ', flush=True)
        
        # Run filter and compute metrics
        metrics = run_accuracy_test(
            observations=observations,
            true_states=true_states,
            K=K,
            N=N,
            threads=threads,
            seed=seed + N  # Different seed per N
        )
        
        results[N] = metrics
        
        print(f"RMSE={metrics['rmse']:.4f}, "
              f"PLL={metrics['mean_pll']:.4f}, "
              f"Cov={metrics['coverage_95']:.1%}, "
              f"Time={metrics['runtime']:.2f}s")
    
    return results


def format_results_table(all_results, scenarios, N_values):
    """
    Format results as a nice table.
    
    Parameters:
    -----------
    all_results : dict
        Nested dict: {scenario: {N: metrics}}
    scenarios : list of str
        Scenario names
    N_values : list of int
        N values tested
    """
    print(f"\n\n{'='*100}")
    print("PARTICLE ACCURACY SCALING RESULTS")
    print(f"{'='*100}\n")
    
    # RMSE table
    print("ROOT MEAN SQUARED ERROR (RMSE)")
    print("-" * 100)
    print(f"{'Scenario':<20} " + " ".join(f"N={N:>4d}" for N in N_values))
    print("-" * 100)
    
    for scenario in scenarios:
        values = " ".join(f"{all_results[scenario][N]['rmse']:>8.4f}" for N in N_values)
        print(f"{scenario:<20} {values}")
    
    # Mean PLL table
    print(f"\n{'MEAN PREDICTIVE LOG-LIKELIHOOD (higher is better)'}")
    print("-" * 100)
    print(f"{'Scenario':<20} " + " ".join(f"N={N:>4d}" for N in N_values))
    print("-" * 100)
    
    for scenario in scenarios:
        values = " ".join(f"{all_results[scenario][N]['mean_pll']:>8.4f}" for N in N_values)
        print(f"{scenario:<20} {values}")
    
    # Coverage table
    print(f"\n{'95% CREDIBLE INTERVAL COVERAGE'}")
    print("-" * 100)
    print(f"{'Scenario':<20} " + " ".join(f"N={N:>4d}" for N in N_values))
    print("-" * 100)
    
    for scenario in scenarios:
        values = " ".join(f"{all_results[scenario][N]['coverage_95']:>7.1%}" for N in N_values)
        print(f"{scenario:<20} {values}")
    
    # Runtime table
    print(f"\n{'RUNTIME (seconds)'}")
    print("-" * 100)
    print(f"{'Scenario':<20} " + " ".join(f"N={N:>4d}" for N in N_values))
    print("-" * 100)
    
    for scenario in scenarios:
        values = " ".join(f"{all_results[scenario][N]['runtime']:>8.2f}" for N in N_values)
        print(f"{scenario:<20} {values}")
    
    print("=" * 100)


def main():
    """Main benchmark routine."""
    parser = argparse.ArgumentParser(
        description='Benchmark particle count vs accuracy on synthetic data'
    )
    parser.add_argument('--models', '-K', type=int, default=10,
                       help='Number of ensemble models (default: 10)')
    parser.add_argument('--threads', '-t', type=int, default=8,
                       help='Number of OpenMP threads (default: 8)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Configuration
    N_values = [100, 250, 500, 1000, 2500, 5000, 10_000]
    scenarios = ['baseline', 'high_persistence', 'high_volatility', 'heavy_tails']
    T = 5000  # Long sequence for statistical reliability
    
    print("PARTICLE ACCURACY SCALING BENCHMARK")
    print("=" * 70)
    print(f"N values:  {N_values}")
    print(f"Scenarios: {scenarios}")
    print(f"Models:    K={args.models}")
    print(f"Threads:   {args.threads}")
    print(f"Sequence:  T={T:,}")
    print(f"Seed:      {args.seed}")
    
    # Run all scenarios
    all_results = {}
    total_start = time.time()
    
    for scenario in scenarios:
        results = run_scenario(
            scenario_name=scenario,
            N_values=N_values,
            K=args.models,
            threads=args.threads,
            T=T,
            seed=args.seed
        )
        all_results[scenario] = results
    
    total_time = time.time() - total_start
    
    # Display results
    format_results_table(all_results, scenarios, N_values)
    
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    print(f"Average per configuration: {total_time / (len(scenarios) * len(N_values)):.1f}s")


if __name__ == '__main__':
    main()
