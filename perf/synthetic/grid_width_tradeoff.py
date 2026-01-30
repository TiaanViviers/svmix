#!/usr/bin/env python3
"""
Grid Width vs Accuracy Benchmark
=================================

Tests the tradeoff between grid width and accuracy. Answers:
1. Does a wider grid hurt accuracy in stable regimes?
2. Does a wider grid help during regime changes?
3. Is there an optimal grid width, or is wider always better?

Grid Strategies (K=50 fixed, N=500):
1. Ultra-Narrow: φ∈[0.94,0.98], σ∈[0.15,0.25] - Optimized for baseline
2. Narrow:       φ∈[0.92,0.99], σ∈[0.12,0.35] - Good for typical conditions
3. Medium:       φ∈[0.88,0.995], σ∈[0.10,0.45] - Broad coverage
4. Wide:         φ∈[0.85,0.998], σ∈[0.08,0.55] - Very wide
5. Ultra-Wide:   φ∈[0.75,0.998], σ∈[0.06,0.70] - Extreme coverage

Test Scenarios:
A. Stable Regimes: Test on baseline, high_persistence, high_volatility
   → Does ultra-narrow beat ultra-wide when parameters are stable?

B. Regime Change: Generate data that SHIFTS parameters mid-sequence
   → Baseline → High Volatility
   → High Persistence → Baseline
   → Baseline → Crisis (σ=0.60, ν=3)

Metrics:
- RMSE (overall and per-regime)
- Coverage (overall and per-regime)
- Adaptation speed (RMSE in first 100 steps after regime change)

Usage:
    python grid_width_tradeoff.py [--threads T] [--seed S]
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from svmix import Svmix, SvmixConfig, SvParams, SvParamsVol, Spec
from synthetic import SyntheticDataGenerator
from metrics import rmse, coverage_95


def create_grid(width_type, K=50):
    """Create parameter grids of different widths.
    
    All grids have same K (50 models), but different coverage ranges.
    Uses φ×σ grids with nu=10 fixed (focus on φ and σ diversity).
    """
    grids = {
        'ultra_narrow': {
            'phi_range': (0.94, 0.98),
            'sigma_range': (0.15, 0.25),
            'description': 'Ultra-Narrow (optimized for baseline)',
            'n_phi': 10, 'n_sigma': 5
        },
        'narrow': {
            'phi_range': (0.92, 0.99),
            'sigma_range': (0.12, 0.35),
            'description': 'Narrow (typical conditions)',
            'n_phi': 10, 'n_sigma': 5
        },
        'medium': {
            'phi_range': (0.88, 0.995),
            'sigma_range': (0.10, 0.45),
            'description': 'Medium (broad coverage)',
            'n_phi': 10, 'n_sigma': 5
        },
        'wide': {
            'phi_range': (0.85, 0.998),
            'sigma_range': (0.08, 0.55),
            'description': 'Wide (very broad)',
            'n_phi': 10, 'n_sigma': 5
        },
        'ultra_wide': {
            'phi_range': (0.75, 0.998),
            'sigma_range': (0.06, 0.70),
            'description': 'Ultra-Wide (extreme coverage)',
            'n_phi': 10, 'n_sigma': 5
        }
    }
    
    grid_spec = grids[width_type]
    
    # Create grid
    n_phi = grid_spec['n_phi']
    n_sigma = grid_spec['n_sigma']
    
    phi_values = np.linspace(grid_spec['phi_range'][0], grid_spec['phi_range'][1], n_phi)
    sigma_values = np.linspace(grid_spec['sigma_range'][0], grid_spec['sigma_range'][1], n_sigma)
    
    params = []
    for sigma in sigma_values:
        for phi in phi_values:
            params.append(SvParamsVol(phi=phi, sigma=sigma, nu=10.0, mu=-0.5))
            if len(params) >= K:
                break
        if len(params) >= K:
            break
    
    return params[:K], grid_spec['description']


def generate_stable_scenario(scenario_name, T=2500, seed=42):
    """Generate stable scenario with fixed parameters."""
    generator = SyntheticDataGenerator()
    params = generator.get_standard_params(scenario_name)
    params = {k: v for k, v in params.items() if k != 'description'}
    params['T'] = T
    
    data = generator.generate(**params, seed=seed)
    
    return {
        'observations': data['observations'],
        'states': data['states'],
        'params': data['params'],
        'regime_name': scenario_name,
        'regime_changes': []  # No regime changes
    }


def generate_regime_change(regime1, regime2, T_per_regime=1500, seed=42):
    """Generate data with regime change in the middle.
    
    First T_per_regime steps: regime1 parameters
    Last T_per_regime steps: regime2 parameters
    """
    generator = SyntheticDataGenerator()
    
    # Generate first regime
    params1 = generator.get_standard_params(regime1)
    params1 = {k: v for k, v in params1.items() if k != 'description'}
    params1['T'] = T_per_regime
    data1 = generator.generate(**params1, seed=seed)
    
    # Generate second regime (continuing from last state)
    params2 = generator.get_standard_params(regime2)
    params2 = {k: v for k, v in params2.items() if k != 'description'}
    params2['T'] = T_per_regime
    data2 = generator.generate(**params2, seed=seed + 1000)
    
    # Concatenate
    observations = np.concatenate([data1['observations'], data2['observations']])
    states = np.concatenate([data1['states'], data2['states']])
    
    return {
        'observations': observations,
        'states': states,
        'params1': data1['params'],
        'params2': data2['params'],
        'regime_name': f"{regime1}→{regime2}",
        'regime_changes': [T_per_regime],  # Change point
        'T_per_regime': T_per_regime
    }


def run_filter(observations, grid_params, N, threads, seed):
    """Run filter and collect beliefs."""
    K = len(grid_params)
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=K,
        num_particles=N,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        num_threads=threads,
        seed=seed
    )
    
    svmix = Svmix(config, grid_params)
    
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
    
    return np.array(estimated_states), np.array(estimated_variances), runtime


def compute_regime_metrics(true_states, estimated_states, estimated_variances, regime_changes):
    """Compute metrics overall and per regime."""
    T = len(true_states)
    
    # Overall metrics
    overall_rmse = rmse(true_states, estimated_states)
    overall_cov = coverage_95(true_states, estimated_states, estimated_variances)
    
    if not regime_changes:
        # No regime changes - single regime
        return {
            'overall_rmse': overall_rmse,
            'overall_coverage': overall_cov,
            'regime_metrics': []
        }
    
    # Per-regime metrics
    regime_metrics = []
    
    # First regime: 0 to first change point
    start_idx = 0
    for i, change_point in enumerate(regime_changes + [T]):
        end_idx = change_point
        
        regime_true = true_states[start_idx:end_idx]
        regime_est = estimated_states[start_idx:end_idx]
        regime_var = estimated_variances[start_idx:end_idx]
        
        regime_rmse = rmse(regime_true, regime_est)
        regime_cov = coverage_95(regime_true, regime_est, regime_var)
        
        # Adaptation speed: RMSE in first 100 steps after regime change
        if start_idx > 0:  # Not the first regime
            adaptation_window = min(100, end_idx - start_idx)
            adapt_true = true_states[start_idx:start_idx + adaptation_window]
            adapt_est = estimated_states[start_idx:start_idx + adaptation_window]
            adapt_rmse = rmse(adapt_true, adapt_est)
        else:
            adapt_rmse = None  # No adaptation needed for first regime
        
        regime_metrics.append({
            'regime_num': i + 1,
            'rmse': regime_rmse,
            'coverage': regime_cov,
            'adaptation_rmse': adapt_rmse
        })
        
        start_idx = end_idx
    
    return {
        'overall_rmse': overall_rmse,
        'overall_coverage': overall_cov,
        'regime_metrics': regime_metrics
    }


def test_stable_scenario(scenario_name, grid_types, K, N, threads, T, seed):
    """Test all grid widths on a stable scenario."""
    print(f"\n{'='*80}")
    print(f"STABLE: {scenario_name}")
    print(f"{'='*80}")
    
    # Generate data
    data = generate_stable_scenario(scenario_name, T=T, seed=seed)
    print(f"True parameters: phi={data['params']['phi']:.3f}, "
          f"sigma={data['params']['sigma']:.3f}, nu={data['params']['nu']:.1f}")
    
    results = {}
    
    for grid_type in grid_types:
        grid_params, description = create_grid(grid_type, K=K)
        
        print(f"\n  {description}...", end=' ', flush=True)
        
        est_states, est_vars, runtime = run_filter(
            data['observations'], grid_params, N, threads, seed + hash(grid_type) % 1000
        )
        
        metrics = compute_regime_metrics(
            data['states'], est_states, est_vars, data['regime_changes']
        )
        
        metrics['runtime'] = runtime
        metrics['grid_type'] = grid_type
        results[grid_type] = metrics
        
        print(f"RMSE={metrics['overall_rmse']:.4f}, "
              f"Cov={metrics['overall_coverage']:.1%}, "
              f"Time={runtime:.2f}s")
    
    return results


def test_regime_change(regime1, regime2, grid_types, K, N, threads, T_per_regime, seed):
    """Test all grid widths on a regime change scenario."""
    print(f"\n{'='*80}")
    print(f"REGIME CHANGE: {regime1} → {regime2}")
    print(f"{'='*80}")
    
    # Generate data
    data = generate_regime_change(regime1, regime2, T_per_regime=T_per_regime, seed=seed)
    print(f"Regime 1: phi={data['params1']['phi']:.3f}, sigma={data['params1']['sigma']:.3f}")
    print(f"Regime 2: phi={data['params2']['phi']:.3f}, sigma={data['params2']['sigma']:.3f}")
    print(f"Change point: T={T_per_regime}")
    
    results = {}
    
    for grid_type in grid_types:
        grid_params, description = create_grid(grid_type, K=K)
        
        print(f"\n  {description}...", end=' ', flush=True)
        
        est_states, est_vars, runtime = run_filter(
            data['observations'], grid_params, N, threads, seed + hash(grid_type) % 1000
        )
        
        metrics = compute_regime_metrics(
            data['states'], est_states, est_vars, data['regime_changes']
        )
        
        metrics['runtime'] = runtime
        metrics['grid_type'] = grid_type
        results[grid_type] = metrics
        
        # Print summary
        r1 = metrics['regime_metrics'][0]
        r2 = metrics['regime_metrics'][1]
        print(f"Overall RMSE={metrics['overall_rmse']:.4f}, Cov={metrics['overall_coverage']:.1%}")
        print(f"    Regime1: RMSE={r1['rmse']:.4f}, Cov={r1['coverage']:.1%}")
        print(f"    Regime2: RMSE={r2['rmse']:.4f}, Cov={r2['coverage']:.1%}, "
              f"Adapt={r2['adaptation_rmse']:.4f}")
    
    return results


def format_results(stable_results, regime_change_results, grid_types):
    """Format comprehensive results tables."""
    print(f"\n\n{'='*100}")
    print("GRID WIDTH TRADEOFF ANALYSIS")
    print(f"{'='*100}\n")
    
    # Table 1: Stable Scenario Performance
    print("="*100)
    print("STABLE SCENARIOS - Does wider grid hurt accuracy?")
    print("="*100)
    print(f"{'Scenario':<25} {'Grid Width':<20} {'RMSE':<10} {'Coverage':<10} {'Runtime':<10}")
    print("-"*100)
    
    for scenario_name, results in stable_results.items():
        for i, grid_type in enumerate(grid_types):
            scenario_label = scenario_name if i == 0 else ""
            metrics = results[grid_type]
            print(f"{scenario_label:<25} {grid_type:<20} "
                  f"{metrics['overall_rmse']:<10.4f} "
                  f"{metrics['overall_coverage']:<10.1%} "
                  f"{metrics['runtime']:<10.2f}")
        print()
    
    # Table 2: Regime Change Performance
    print("\n" + "="*100)
    print("REGIME CHANGES - Does wider grid help adaptation?")
    print("="*100)
    print(f"{'Scenario':<25} {'Grid Width':<20} {'Overall':<12} {'Regime1':<12} "
          f"{'Regime2':<12} {'Adapt':<10}")
    print("-"*100)
    
    for scenario_name, results in regime_change_results.items():
        for i, grid_type in enumerate(grid_types):
            scenario_label = scenario_name if i == 0 else ""
            metrics = results[grid_type]
            r1 = metrics['regime_metrics'][0]
            r2 = metrics['regime_metrics'][1]
            
            print(f"{scenario_label:<25} {grid_type:<20} "
                  f"{metrics['overall_rmse']:<12.4f} "
                  f"{r1['rmse']:<12.4f} "
                  f"{r2['rmse']:<12.4f} "
                  f"{r2['adaptation_rmse']:<10.4f}")
        print()
    
    # Table 3: Coverage in Regime Changes
    print("\n" + "="*100)
    print("COVERAGE DURING REGIME CHANGES")
    print("="*100)
    print(f"{'Scenario':<25} {'Grid Width':<20} {'Overall':<12} {'Regime1':<12} {'Regime2':<12}")
    print("-"*100)
    
    for scenario_name, results in regime_change_results.items():
        for i, grid_type in enumerate(grid_types):
            scenario_label = scenario_name if i == 0 else ""
            metrics = results[grid_type]
            r1 = metrics['regime_metrics'][0]
            r2 = metrics['regime_metrics'][1]
            
            print(f"{scenario_label:<25} {grid_type:<20} "
                  f"{metrics['overall_coverage']:<12.1%} "
                  f"{r1['coverage']:<12.1%} "
                  f"{r2['coverage']:<12.1%}")
        print()
    
    print("="*100)


def main():
    """Main benchmark routine."""
    parser = argparse.ArgumentParser(
        description='Test tradeoff between grid width and accuracy'
    )
    parser.add_argument('--threads', '-t', type=int, default=0,
                       help='Number of OpenMP threads (default: 0=auto)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Configuration
    grid_types = ['ultra_narrow', 'narrow', 'medium', 'wide', 'ultra_wide']
    K = 50
    N = 500
    T_stable = 2500
    T_per_regime = 1500
    
    print("GRID WIDTH vs ACCURACY TRADEOFF BENCHMARK")
    print("=" * 80)
    print(f"Grid types: {grid_types}")
    print(f"Models:     K={K}")
    print(f"Particles:  N={N}")
    print(f"Threads:    {args.threads} (0=auto)")
    print(f"Seed:       {args.seed}")
    print(f"\nStable scenarios: T={T_stable}")
    print(f"Regime changes:   T={T_per_regime} per regime")
    
    total_start = time.time()
    
    # Test 1: Stable scenarios
    print("\n" + "="*80)
    print("PART 1: STABLE SCENARIOS")
    print("="*80)
    
    stable_results = {}
    for scenario in ['baseline', 'high_persistence', 'high_volatility']:
        results = test_stable_scenario(
            scenario, grid_types, K, N, args.threads, T_stable, args.seed
        )
        stable_results[scenario] = results
    
    # Test 2: Regime changes
    print("\n" + "="*80)
    print("PART 2: REGIME CHANGES")
    print("="*80)
    
    regime_change_results = {}
    
    # Baseline → High Volatility
    results = test_regime_change(
        'baseline', 'high_volatility', grid_types, K, N, args.threads, 
        T_per_regime, args.seed
    )
    regime_change_results['baseline→high_volatility'] = results
    
    # High Persistence → Baseline
    results = test_regime_change(
        'high_persistence', 'baseline', grid_types, K, N, args.threads,
        T_per_regime, args.seed + 1000
    )
    regime_change_results['high_persistence→baseline'] = results
    
    # Baseline → Heavy Tails
    results = test_regime_change(
        'baseline', 'heavy_tails', grid_types, K, N, args.threads,
        T_per_regime, args.seed + 2000
    )
    regime_change_results['baseline→heavy_tails'] = results
    
    total_time = time.time() - total_start
    
    # Format comprehensive results
    format_results(stable_results, regime_change_results, grid_types)
    
    print(f"\nTotal benchmark time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
