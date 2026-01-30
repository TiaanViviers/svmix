#!/usr/bin/env python3
"""
Grid Width vs K Frontier Analysis
==================================

Tests the interaction between grid width and ensemble size (K).

Hypothesis: Wider grids need MORE models to achieve same density.
- Narrow grid: K=20 might be sufficient (small parameter space)
- Wide grid: K=100+ needed to cover larger parameter space densely

This explores the WIDTH × K frontier to find optimal configurations.

Key Questions:
1. Does increasing K help wide grids catch up to narrow grids?
2. Is there a K threshold where wide grids become superior?
3. What's the optimal (width, K) pair for production?

Test Matrix:
- Grid widths: ultra_narrow, narrow, medium, wide, ultra_wide
- K values: [20, 50, 100, 150, 200]
- N: 500 (sufficient from particle scaling)
- Scenarios: Stable + Regime changes

Usage:
    python grid_width_vs_k_frontier.py [--threads T] [--seed S]

Output:
    Heatmaps showing RMSE and coverage as function of (width, K)
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from svmix import Svmix, SvmixConfig, SvParamsVol, Spec
from synthetic import SyntheticDataGenerator
from metrics import rmse, coverage_95


def create_grid(width_type, K):
    """Create parameter grid with specified width and K models.
    
    Maintains same WIDTH but varies DENSITY via K.
    """
    if width_type == 'ultra_narrow':
        phi_range = (0.93, 0.97)
        sigma_range = (0.18, 0.22)
        nu_values = [10.0]
        
    elif width_type == 'narrow':
        phi_range = (0.91, 0.98)
        sigma_range = (0.15, 0.30)
        nu_values = [8.0, 12.0]
        
    elif width_type == 'medium':
        phi_range = (0.88, 0.99)
        sigma_range = (0.12, 0.40)
        nu_values = [6.0, 14.0]
        
    elif width_type == 'wide':
        phi_range = (0.85, 0.995)
        sigma_range = (0.10, 0.50)
        nu_values = [5.0, 10.0, 18.0]
        
    else:  # ultra_wide
        phi_range = (0.80, 0.998)
        sigma_range = (0.08, 0.60)
        nu_values = [4.0, 8.0, 15.0, 25.0]
    
    # Distribute K models across the grid
    # Use 2D grid for phi×sigma, then replicate for each nu
    n_nu = len(nu_values)
    models_per_nu = K // n_nu
    
    # Calculate phi and sigma counts for roughly square grid
    n_phi = int(np.sqrt(models_per_nu * 1.2))  # Slightly more phi than sigma
    n_sigma = models_per_nu // n_phi
    
    # Ensure we have at least minimum coverage
    n_phi = max(n_phi, 3)
    n_sigma = max(n_sigma, 3)
    
    phi_values = np.linspace(phi_range[0], phi_range[1], n_phi)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_sigma)
    
    params = []
    for nu in nu_values:
        for sigma in sigma_values:
            for phi in phi_values:
                params.append(SvParamsVol(phi=phi, sigma=sigma, nu=nu, mu=-0.5))
                if len(params) >= K:
                    break
            if len(params) >= K:
                break
        if len(params) >= K:
            break
    
    # Return exactly K models
    return params[:K]


def run_test(observations, true_states, width_type, K, N, threads, seed):
    """Run single test with given grid width and K."""
    params = create_grid(width_type, K)
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=len(params),
        num_particles=N,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        num_threads=threads,
        seed=seed
    )
    
    svmix = Svmix(config, params)
    
    # Run filter
    estimated_states = []
    estimated_variances = []
    
    start = time.time()
    for obs in observations:
        svmix.step(obs)
        belief = svmix.get_belief()
        estimated_states.append(belief.mean_h)
        estimated_variances.append(max(belief.var_h, 1e-10))
    
    runtime = time.time() - start
    svmix.free()
    
    # Compute metrics
    estimated_states = np.array(estimated_states)
    estimated_variances = np.array(estimated_variances)
    
    return {
        'rmse': rmse(true_states, estimated_states),
        'coverage': coverage_95(true_states, estimated_states, estimated_variances),
        'runtime': runtime
    }


def run_stable_scenario(scenario_name, width_types, K_values, N, threads, T, seed):
    """Test all (width, K) combinations for a stable scenario."""
    print(f"\n{'='*80}")
    print(f"STABLE SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    # Generate data
    generator = SyntheticDataGenerator()
    params = generator.get_standard_params(scenario_name)
    params = {k: v for k, v in params.items() if k != 'description'}
    params['T'] = T
    
    data = generator.generate(**params, seed=seed)
    observations = data['observations']
    true_states = data['states']
    
    print(f"True params: phi={data['params']['phi']:.3f}, "
          f"sigma={data['params']['sigma']:.3f}, nu={data['params']['nu']:.1f}")
    
    # Test all combinations
    results = {}
    for width in width_types:
        results[width] = {}
        print(f"\n{width.upper().replace('_', ' ')}:")
        for K in K_values:
            print(f"  K={K:3d}...", end=' ', flush=True)
            metrics = run_test(observations, true_states, width, K, N, threads, seed + K)
            results[width][K] = metrics
            print(f"RMSE={metrics['rmse']:.4f}, Cov={metrics['coverage']:.1%}, "
                  f"Time={metrics['runtime']:.2f}s")
    
    return results


def run_regime_change(name, regime1_params, regime2_params, width_types, K_values, 
                      N, threads, T_per_regime, seed):
    """Test regime change for all (width, K) combinations."""
    print(f"\n{'='*80}")
    print(f"REGIME CHANGE: {name}")
    print(f"{'='*80}")
    
    generator = SyntheticDataGenerator()
    
    # Generate regime 1
    data1 = generator.generate(**regime1_params, T=T_per_regime, seed=seed)
    
    # Generate regime 2
    data2 = generator.generate(**regime2_params, T=T_per_regime, seed=seed + 10000)
    
    # Concatenate
    observations = np.concatenate([data1['observations'], data2['observations']])
    true_states = np.concatenate([data1['states'], data2['states']])
    
    print(f"Regime 1: phi={regime1_params['phi']:.3f}, sigma={regime1_params['sigma']:.3f}")
    print(f"Regime 2: phi={regime2_params['phi']:.3f}, sigma={regime2_params['sigma']:.3f}")
    print(f"Change point: T={T_per_regime}")
    
    # Test all combinations
    results = {}
    for width in width_types:
        results[width] = {}
        print(f"\n{width.upper().replace('_', ' ')}:")
        for K in K_values:
            print(f"  K={K:3d}...", end=' ', flush=True)
            metrics = run_test(observations, true_states, width, K, N, threads, seed + K)
            
            # Also compute regime-specific metrics
            split_point = T_per_regime
            regime1_rmse = rmse(true_states[:split_point], 
                               np.array([metrics['rmse']] * split_point))  # Placeholder
            regime2_rmse = rmse(true_states[split_point:],
                               np.array([metrics['rmse']] * (len(true_states) - split_point)))
            
            results[width][K] = metrics
            print(f"RMSE={metrics['rmse']:.4f}, Cov={metrics['coverage']:.1%}, "
                  f"Time={metrics['runtime']:.2f}s")
    
    return results


def format_results_matrix(all_results, scenarios, width_types, K_values):
    """Format results as width × K matrices."""
    print(f"\n\n{'='*100}")
    print("GRID WIDTH × K FRONTIER ANALYSIS")
    print(f"{'='*100}\n")
    
    for scenario in scenarios:
        results = all_results[scenario]
        
        print(f"\n{scenario.upper().replace('_', ' ')}")
        print("=" * 100)
        
        # RMSE matrix
        print(f"\nRMSE (lower is better)")
        print("-" * 100)
        print(f"{'Width':<15} " + " ".join(f"K={K:>3d}" for K in K_values))
        print("-" * 100)
        
        for width in width_types:
            values = " ".join(f"{results[width][K]['rmse']:>7.4f}" for K in K_values)
            print(f"{width:<15} {values}")
        
        # Coverage matrix
        print(f"\n95% Coverage (target: 95%)")
        print("-" * 100)
        print(f"{'Width':<15} " + " ".join(f"K={K:>3d}" for K in K_values))
        print("-" * 100)
        
        for width in width_types:
            values = " ".join(f"{results[width][K]['coverage']:>6.1%}" for K in K_values)
            print(f"{width:<15} {values}")
        
        # Runtime matrix
        print(f"\nRuntime (seconds)")
        print("-" * 100)
        print(f"{'Width':<15} " + " ".join(f"K={K:>3d}" for K in K_values))
        print("-" * 100)
        
        for width in width_types:
            values = " ".join(f"{results[width][K]['runtime']:>7.2f}" for K in K_values)
            print(f"{width:<15} {values}")
        
        print("\n")


def analyze_frontier(all_results, scenarios, width_types, K_values):
    """Find optimal (width, K) points on the Pareto frontier."""
    print(f"\n{'='*100}")
    print("FRONTIER ANALYSIS - Optimal (Width, K) Configurations")
    print(f"{'='*100}\n")
    
    for scenario in scenarios:
        results = all_results[scenario]
        
        print(f"\n{scenario.upper().replace('_', ' ')}")
        print("-" * 100)
        
        # Find best RMSE for each K
        print(f"{'K':<6} {'Best Width':<15} {'RMSE':<10} {'Coverage':<10} {'Runtime'}")
        print("-" * 100)
        
        for K in K_values:
            best_width = min(width_types, key=lambda w: results[w][K]['rmse'])
            metrics = results[best_width][K]
            print(f"{K:<6} {best_width:<15} {metrics['rmse']:<10.4f} "
                  f"{metrics['coverage']:<10.1%} {metrics['runtime']:.2f}s")
        
        # Find best overall
        print("\nBest Overall Configuration:")
        best_config = min(
            [(w, K, results[w][K]) for w in width_types for K in K_values],
            key=lambda x: x[2]['rmse']
        )
        print(f"  Width: {best_config[0]}")
        print(f"  K: {best_config[1]}")
        print(f"  RMSE: {best_config[2]['rmse']:.4f}")
        print(f"  Coverage: {best_config[2]['coverage']:.1%}")
        print(f"  Runtime: {best_config[2]['runtime']:.2f}s")


def main():
    """Main benchmark routine."""
    parser = argparse.ArgumentParser(
        description='Analyze grid width vs K frontier'
    )
    parser.add_argument('--threads', '-t', type=int, default=0,
                       help='Number of OpenMP threads (default: 0=auto)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Configuration
    width_types = ['ultra_narrow', 'narrow', 'medium', 'wide', 'ultra_wide']
    K_values = [20, 50, 100, 150, 200]
    N = 500
    
    print("GRID WIDTH × K FRONTIER BENCHMARK")
    print("=" * 80)
    print(f"Width types: {width_types}")
    print(f"K values:    {K_values}")
    print(f"Particles:   N={N}")
    print(f"Threads:     {args.threads} (0=auto)")
    print(f"Seed:        {args.seed}")
    
    # Test scenarios
    scenarios = ['baseline', 'high_volatility']
    all_results = {}
    
    total_start = time.time()
    
    # Stable scenarios
    for scenario in scenarios:
        results = run_stable_scenario(
            scenario_name=scenario,
            width_types=width_types,
            K_values=K_values,
            N=N,
            threads=args.threads,
            T=2500,
            seed=args.seed
        )
        all_results[scenario] = results
    
    # Regime change
    regime_results = run_regime_change(
        name='baseline→high_volatility',
        regime1_params={'phi': 0.95, 'sigma': 0.20, 'nu': 10, 'mu': -0.5},
        regime2_params={'phi': 0.95, 'sigma': 0.35, 'nu': 10, 'mu': -0.5},
        width_types=width_types,
        K_values=K_values,
        N=N,
        threads=args.threads,
        T_per_regime=1500,
        seed=args.seed
    )
    all_results['regime_change'] = regime_results
    
    total_time = time.time() - total_start
    
    # Display results
    format_results_matrix(all_results, 
                         ['baseline', 'high_volatility', 'regime_change'],
                         width_types, K_values)
    
    analyze_frontier(all_results,
                    ['baseline', 'high_volatility', 'regime_change'],
                    width_types, K_values)
    
    print(f"\n{'='*100}")
    print(f"Total benchmark time: {total_time:.1f}s")
    num_configs = len(scenarios) * len(width_types) * len(K_values) + len(width_types) * len(K_values)
    print(f"Average per configuration: {total_time / num_configs:.1f}s")


if __name__ == '__main__':
    main()
