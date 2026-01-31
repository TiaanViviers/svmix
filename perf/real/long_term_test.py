"""
Long-term persistence test: Multi-year continuous operation.

Tests Svmix Wide (K=150) on full US30 dataset (2008-2025) without intervention.
Monitors ensemble health and calibration stability over extended periods.

Key questions:
1. Does coverage remain stable over multiple years?
2. Do ensemble weights concentrate (lose diversity)?
3. Does PLL degrade over time?
4. Are there periods requiring intervention?

This test validates the "no re-estimation needed" claim.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
import warnings
from scipy.stats import linregress
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from utils import create_filter_config, compute_metrics, summarize_period
from svmix import Svmix


def analyze_ensemble_health(svmix_filter) -> dict:
    """Analyze ensemble diversity and concentration.
    
    Computes metrics to detect if ensemble is collapsing to single model
    or maintaining healthy diversity.
    
    Returns:
        Dictionary with ensemble health metrics:
        - mean_sigma: Current volatility estimate
        - var_h: Uncertainty in log-volatility
        - mean_h: Mean log-volatility
        - effective_num_models: Diversity metric (1=collapsed, K=uniform)
        - max_weight: Largest model weight (>0.9 indicates dominance)
        - weight_entropy: Shannon entropy of weights (higher=more diverse)
    """
    belief = svmix_filter.get_belief()
    weights = svmix_filter.get_weights()
    
    # Effective number of models (inverse Simpson's index)
    effective_K = 1.0 / np.sum(weights ** 2)
    
    # Maximum weight (high value = one model dominates)
    max_weight = float(np.max(weights))
    
    # Shannon entropy (higher = more diverse)
    # H = -sum(w_i * log(w_i))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    weight_entropy = -float(np.sum(weights * np.log(weights + epsilon)))
    
    return {
        'mean_sigma': belief.mean_sigma,
        'var_h': belief.var_h,
        'mean_h': belief.mean_h,
        'effective_num_models': effective_K,
        'max_weight': max_weight,
        'weight_entropy': weight_entropy
    }


def rolling_window_metrics(
    returns: np.ndarray,
    predicted_vols: np.ndarray,
    pll_values: np.ndarray,
    window_size: int = 5000
) -> pd.DataFrame:
    """Compute metrics over rolling windows to detect drift.
    
    Args:
        returns: Full return series
        predicted_vols: Volatility predictions
        pll_values: PLL values
        window_size: Window size in observations
        
    Returns:
        DataFrame with rolling metrics
    """
    n = len(returns)
    results = []
    
    for i in range(window_size, n, window_size):
        window_returns = returns[i-window_size:i]
        window_vols = predicted_vols[i-window_size:i]
        window_pll = pll_values[i-window_size:i]
        
        metrics = compute_metrics(window_returns, window_vols)
        
        results.append({
            'end_obs': i,
            'pll_mean': np.mean(window_pll),
            'coverage': metrics['coverage'],
            'var_rate': metrics['violation_rate'],
            'mean_vol': np.mean(window_vols)
        })
    
    return pd.DataFrame(results)


def run_long_term_test(
    data_file: str,
    K: int = 150,
    N: int = 500,
    grid_width: str = 'wide',
    threads: int = 0,
    seed: int = 42
):
    """Run long-term persistence test on full dataset.
    
    Args:
        data_file: Path to full US30 CSV file
        K: Number of models
        N: Number of particles
        grid_width: Parameter grid width
        threads: OpenMP threads
        seed: Random seed
    """
    print("="*80)
    print("LONG-TERM PERSISTENCE TEST")
    print("="*80)
    print(f"\nConfiguration: K={K}, N={N}, grid={grid_width}")
    print(f"Data source: {data_file}")
    print("\n" + "="*80)
    
    # Load data
    print("\nLoading full dataset...")
    df = pd.read_csv(data_file, parse_dates=[0], names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.iloc[1:].reset_index(drop=True)
    
    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    returns = df['returns'].values
    T = len(returns)
    
    # Create filter
    print("\nInitializing filter...")
    config, params = create_filter_config(K=K, N=N, grid_width=grid_width, threads=threads, seed=seed)
    svmix = Svmix(config, params)
    
    # Storage
    predicted_vols = []
    pll_values = []
    ensemble_health = []
    
    burn_in = 100
    
    print(f"\nRunning filter on {T:,} observations...")
    print("Progress: ", end='', flush=True)
    
    checkpoint_interval = 100000  # Report every 100K observations
    
    # Burn-in
    for t in range(burn_in):
        svmix.step(returns[t])
    
    # Main loop

    start_time = time.time()
    
    for t in range(burn_in, T):
        # Get prediction before update
        belief_prev = svmix.get_belief()
        predicted_vol = belief_prev.mean_sigma
        
        # Update
        svmix.step(returns[t])
        pll = svmix.get_last_log_likelihood()
        
        # Store
        predicted_vols.append(predicted_vol)
        pll_values.append(pll)
        
        # Checkpoint
        if (t - burn_in) % checkpoint_interval == 0 and t > burn_in:
            elapsed = time.time() - start_time
            rate = (t - burn_in) / elapsed
            remaining = (T - t) / rate
            
            print(f"\n  {t:,}/{T:,} ({t/T*100:.1f}%) - {rate:.0f} obs/s - ETA: {remaining/60:.1f}min", end='', flush=True)
            
            # Record ensemble health
            health = analyze_ensemble_health(svmix)
            health['observation'] = t
            ensemble_health.append(health)
    
    total_time = time.time() - start_time
    print(f"\n\nCompleted in {total_time:.1f}s ({(T-burn_in)/total_time:.1f} obs/s)")
    
    # Clean up
    svmix.free()
    
    # Compute metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    eval_returns = returns[burn_in:]
    predicted_vols = np.array(predicted_vols)
    pll_values = np.array(pll_values)
    
    overall_metrics = compute_metrics(eval_returns, predicted_vols)
    
    print(f"\nTotal observations evaluated: {len(eval_returns):,}")
    print(f"Mean PLL: {np.mean(pll_values):.4f}")
    print(f"Coverage: {overall_metrics['coverage']:.1%}")
    print(f"VaR violations: {overall_metrics['violation_rate']:.1%}")
    print(f"Mean volatility: {np.mean(predicted_vols)*100:.3f}%")
    
    # Rolling window analysis
    print("\n" + "="*80)
    print("TEMPORAL STABILITY ANALYSIS")
    print("="*80)
    
    window_size = 5000  # ~1 day of 1-min data
    rolling_metrics = rolling_window_metrics(eval_returns, predicted_vols, pll_values, window_size)
    
    print(f"\nRolling window metrics ({window_size} obs windows):")
    print(f"Number of windows: {len(rolling_metrics)}")
    print("\nPLL stability:")
    print(f"  Mean: {rolling_metrics['pll_mean'].mean():.4f}")
    print(f"  Std: {rolling_metrics['pll_mean'].std():.4f}")
    print(f"  Min: {rolling_metrics['pll_mean'].min():.4f}")
    print(f"  Max: {rolling_metrics['pll_mean'].max():.4f}")
    
    print("\nCoverage stability:")
    print(f"  Mean: {rolling_metrics['coverage'].mean():.1%}")
    print(f"  Std: {rolling_metrics['coverage'].std()*100:.1f}pp")
    print(f"  Min: {rolling_metrics['coverage'].min():.1%}")
    print(f"  Max: {rolling_metrics['coverage'].max():.1%}")
    
    # Detect periods with poor calibration
    poor_coverage = rolling_metrics[
        (rolling_metrics['coverage'] < 0.90) | (rolling_metrics['coverage'] > 0.98)
    ]
    
    if len(poor_coverage) > 0:
        print(f"\nWARNING: {len(poor_coverage)} windows with coverage outside [90%, 98%]:")
        print(poor_coverage[['end_obs', 'coverage', 'pll_mean']].to_string(index=False))
    else:
        print("\nNo periods with poor calibration detected.")
    
    # Time series analysis
    print("\n" + "="*80)
    print("DRIFT ANALYSIS")
    print("="*80)
    
    # Test for drift in PLL over time
    time_index = np.arange(len(rolling_metrics))
    pll_trend = linregress(time_index, rolling_metrics['pll_mean'].values)
    cov_trend = linregress(time_index, rolling_metrics['coverage'].values)
    
    print(f"\nLinear trend analysis:")
    print(f"PLL slope: {pll_trend.slope:.6f} per window (p={pll_trend.pvalue:.4f})")
    print(f"Coverage slope: {cov_trend.slope:.6f} per window (p={cov_trend.pvalue:.4f})")
    
    if abs(pll_trend.pvalue) < 0.05:
        direction = "improving" if pll_trend.slope > 0 else "degrading"
        print(f"WARNING: Statistically significant PLL trend detected ({direction})")
    else:
        print("No significant PLL drift detected.")
    
    if abs(cov_trend.pvalue) < 0.05:
        direction = "increasing" if cov_trend.slope > 0 else "decreasing"
        print(f"WARNING: Statistically significant coverage trend detected ({direction})")
    else:
        print("No significant coverage drift detected.")
    
    # Ensemble health analysis
    print("\n" + "="*80)
    print("ENSEMBLE HEALTH ANALYSIS")
    print("="*80)
    
    if ensemble_health:
        health_df = pd.DataFrame(ensemble_health)
        
        print(f"\nMonitored at {len(health_df)} checkpoints (every 100K obs)")
        print("\nEffective number of models:")
        print(f"  Mean: {health_df['effective_num_models'].mean():.2f}")
        print(f"  Min: {health_df['effective_num_models'].min():.2f}")
        print(f"  Max: {health_df['effective_num_models'].max():.2f}")
        
        print("\nMaximum model weight:")
        print(f"  Mean: {health_df['max_weight'].mean():.3f}")
        print(f"  Max: {health_df['max_weight'].max():.3f}")
        
        print("\nWeight entropy:")
        print(f"  Mean: {health_df['weight_entropy'].mean():.3f}")
        print(f"  Min: {health_df['weight_entropy'].min():.3f}")
        
        # Check for ensemble collapse
        collapsed_checkpoints = health_df[health_df['effective_num_models'] < 2.0]
        if len(collapsed_checkpoints) > 0:
            print(f"\nWARNING: Ensemble collapsed (eff_K < 2.0) at {len(collapsed_checkpoints)} checkpoints:")
            print(f"  This indicates strong dominance by a single model")
        else:
            print("\nNo ensemble collapse detected (eff_K stayed above 2.0)")
        
        # Check for single model dominance
        dominant_checkpoints = health_df[health_df['max_weight'] > 0.8]
        if len(dominant_checkpoints) > 0:
            print(f"\nWARNING: Single model dominated (weight > 0.8) at {len(dominant_checkpoints)} checkpoints")
        else:
            print("\nNo single-model dominance detected (max weight stayed below 0.8)")
    else:
        print("\nNo ensemble health data collected (run time too short)")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path(__file__).parent
    
    # Save rolling metrics
    rolling_file = output_dir / 'long_term_rolling_metrics.csv'
    rolling_metrics.to_csv(rolling_file, index=False)
    print(f"\nRolling metrics saved to: {rolling_file}")
    
    # Save ensemble health
    health_df = None  # Initialize for later use
    if ensemble_health:
        health_df = pd.DataFrame(ensemble_health)
        health_file = output_dir / 'long_term_ensemble_health.csv'
        health_df.to_csv(health_file, index=False)
        print(f"Ensemble health saved to: {health_file}")
    
    # Summary report
    summary = {
        'total_observations': T,
        'evaluated_observations': len(eval_returns),
        'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        'mean_pll': float(np.mean(pll_values)),
        'pll_std': float(np.std(pll_values)),
        'coverage': float(overall_metrics['coverage']),
        'var_violations': float(overall_metrics['violation_rate']),
        'pll_drift_slope': float(pll_trend.slope),
        'pll_drift_pvalue': float(pll_trend.pvalue),
        'coverage_drift_slope': float(cov_trend.slope),
        'coverage_drift_pvalue': float(cov_trend.pvalue),
        'runtime_seconds': float(total_time),
        'processing_rate': float((T-burn_in)/total_time)
    }
    
    # Add ensemble health summary if available
    if health_df is not None:
        summary.update({
            'ensemble_mean_effective_K': float(health_df['effective_num_models'].mean()),
            'ensemble_min_effective_K': float(health_df['effective_num_models'].min()),
            'ensemble_mean_max_weight': float(health_df['max_weight'].mean()),
            'ensemble_max_weight_ever': float(health_df['max_weight'].max()),
            'ensemble_mean_entropy': float(health_df['weight_entropy'].mean())
        })
    
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / 'long_term_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Check for issues requiring intervention
    ensemble_collapsed = False
    if health_df is not None:
        # Ensemble collapsed if mean effective K < 2 or any checkpoint shows severe collapse
        ensemble_collapsed = (
            health_df['effective_num_models'].mean() < 2.0 or
            health_df['effective_num_models'].min() < 1.2
        )
    
    needs_intervention = (
        (abs(pll_trend.pvalue) < 0.05 and pll_trend.slope < -0.001) or
        (abs(cov_trend.pvalue) < 0.05 and abs(cov_trend.slope) > 0.001) or
        len(poor_coverage) > len(rolling_metrics) * 0.1 or
        ensemble_collapsed
    )
    
    if needs_intervention:
        print("\nRESULT: Manual intervention may be required.")
        print("Evidence:")
        if abs(pll_trend.pvalue) < 0.05 and pll_trend.slope < -0.001:
            print("  - Significant PLL degradation over time")
        if abs(cov_trend.pvalue) < 0.05:
            print("  - Significant coverage drift over time")
        if len(poor_coverage) > len(rolling_metrics) * 0.1:
            print(f"  - {len(poor_coverage)/len(rolling_metrics)*100:.1f}% of windows have poor calibration")
        if ensemble_collapsed:
            print("  - Ensemble collapsed (single model dominance)")
    else:
        print("\nRESULT: No intervention required.")
        print("The filter maintains stable performance over the full period.")
        print("Safe for long-term autonomous operation.")
    
    print("\n" + "="*80)
    
    return summary


if __name__ == '__main__':
    """Run long-term persistence test."""    
    parser = argparse.ArgumentParser(description='Long-term persistence test')
    parser.add_argument('--data', type=str, required=True, help='Path to full US30 CSV file')
    parser.add_argument('--K', type=int, default=150, help='Number of models')
    parser.add_argument('--N', type=int, default=500, help='Number of particles')
    parser.add_argument('--grid', type=str, default='wide', help='Grid width')
    parser.add_argument('--threads', type=int, default=8, help='OpenMP threads')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_long_term_test(
        data_file=args.data,
        K=args.K,
        N=args.N,
        grid_width=args.grid,
        threads=args.threads,
        seed=args.seed
    )
