"""
Basic validation: Run production config on all real market periods.

This is Phase 1 of real data validation:
- Load all 6 market periods (2008 crisis through 2021)
- Run Wide K=150, N=500 production configuration
- Compute PLL, coverage, VaR violations
- Verify filter produces sensible results on messy real data

Expected outcomes:
- Coverage should be 90-96% (within 5% of nominal 95%)
- PLL should be positive and consistent across periods
- No catastrophic failures or numerical issues
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_all_periods,
    run_filter_with_metrics,
    compute_var_violations,
    summarize_period
)


def run_basic_validation(
    K: int = 150,
    N: int = 500,
    grid_width: str = 'wide',
    threads: int = 0,
    seed: int = 42
):
    """Run basic validation on all periods.
    
    Args:
        K: Number of ensemble models (default: 150 for production)
        N: Number of particles (default: 500 for production)
        grid_width: Parameter grid width (default: 'wide' for production)
        threads: OpenMP threads (0=auto)
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("BASIC VALIDATION: Production Config on Real Market Data")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  K (models): {K}")
    print(f"  N (particles): {N}")
    print(f"  Grid width: {grid_width}")
    print(f"  Threads: {threads if threads > 0 else 'auto'}")
    print(f"  Seed: {seed}")
    print("\n" + "="*80)
    
    # Load all periods
    print("\nLoading market periods...")
    periods = load_all_periods()
    print(f"Loaded {len(periods)} periods\n")
    
    # Results storage
    results = []
    
    # Process each period
    for period_name, df in sorted(periods.items()):
        print(f"\n{'='*80}")
        print(f"Period: {period_name}")
        print(f"{'='*80}")
        
        # Period summary
        summary = summarize_period(df)
        print(f"\nData summary:")
        print(f"  Observations: {summary['n_obs']:,}")
        print(f"  Annualized volatility: {summary['volatility_annualized']:.1f}%")
        print(f"  Mean return: {summary['mean_return']*1e4:.2f} bps")
        print(f"  Skewness: {summary['skewness']:.2f}")
        print(f"  Kurtosis: {summary['kurtosis']:.1f}")
        print(f"  Min/Max return: {summary['min_return']*100:.2f}% / {summary['max_return']*100:.2f}%")
        
        # Run filter
        print(f"\nRunning filter...")
        returns = df['returns'].values
        
        try:
            metrics = run_filter_with_metrics(
                returns,
                K=K,
                N=N,
                grid_width=grid_width,
                burn_in=100,
                threads=threads,
                seed=seed
            )
            
            # VaR violations
            var_results = compute_var_violations(
                metrics['eval_returns'],
                metrics['predicted_vols'],
                confidence=0.95
            )
            
            # Print results
            print(f"\nResults:")
            print(f"  Total PLL: {metrics['pll_total']:.2f}")
            print(f"  Mean PLL: {metrics['pll_mean']:.4f}")
            print(f"  Coverage (95%): {metrics['coverage_95']:.1%}")
            print(f"  VaR violations: {var_results['violations']} ({var_results['violation_rate']:.1%})")
            print(f"  Expected violations: {var_results['expected_rate']:.1%}")
            print(f"  Mean volatility: {metrics['predicted_vols'].mean()*100:.2f}%")
            print(f"  Runtime: {metrics['runtime']:.2f}s ({metrics['runtime']/metrics['T_eval']*1000:.2f}ms/obs)")
            
            # Store results
            results.append({
                'period': period_name,
                'n_obs': summary['n_obs'],
                'ann_vol': summary['volatility_annualized'],
                'skew': summary['skewness'],
                'kurtosis': summary['kurtosis'],
                'pll_total': metrics['pll_total'],
                'pll_mean': metrics['pll_mean'],
                'coverage': metrics['coverage_95'],
                'var_violations': var_results['violations'],
                'var_rate': var_results['violation_rate'],
                'mean_vol': metrics['predicted_vols'].mean(),
                'runtime': metrics['runtime'],
                'ms_per_obs': metrics['runtime'] / metrics['T_eval'] * 1000
            })
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'period': period_name,
                'n_obs': summary['n_obs'],
                'ann_vol': summary['volatility_annualized'],
                'error': str(e)
            })
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: All Periods")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    # Check for errors
    if 'error' in df_results.columns and df_results['error'].notna().any():
        print("\n  Some periods had errors:")
        error_df = df_results[df_results['error'].notna()][['period', 'error']]
        print(error_df.to_string(index=False))
        print()
        # Success table (exclude error rows)
        success_df = df_results[df_results['error'].isna()]
    else:
        # All succeeded
        success_df = df_results
    
    if len(success_df) > 0:
        print("\nPerformance Metrics:")
        print("-"*80)
        
        # Create display table
        display_df = success_df[['period', 'n_obs', 'ann_vol', 'pll_mean', 'coverage', 'var_rate', 'ms_per_obs']].copy()
        display_df.columns = ['Period', 'N', 'Vol%', 'PLL', 'Cov', 'VaR', 'ms/obs']
        
        # Format
        display_df['N'] = display_df['N'].apply(lambda x: f"{x:,}")
        display_df['Vol%'] = display_df['Vol%'].apply(lambda x: f"{x:.1f}")
        display_df['PLL'] = display_df['PLL'].apply(lambda x: f"{x:.4f}")
        display_df['Cov'] = display_df['Cov'].apply(lambda x: f"{x:.1%}")
        display_df['VaR'] = display_df['VaR'].apply(lambda x: f"{x:.1%}")
        display_df['ms/obs'] = display_df['ms/obs'].apply(lambda x: f"{x:.2f}")
        
        print(display_df.to_string(index=False))
        
        # Overall statistics
        print("\n" + "-"*80)
        print("Aggregate Statistics:")
        print(f"  Mean PLL: {success_df['pll_mean'].mean():.4f} ± {success_df['pll_mean'].std():.4f}")
        print(f"  Mean Coverage: {success_df['coverage'].mean():.1%} (target: 95.0%)")
        print(f"  Coverage range: [{success_df['coverage'].min():.1%}, {success_df['coverage'].max():.1%}]")
        print(f"  Mean VaR violations: {success_df['var_rate'].mean():.1%} (expected: 5.0%)")
        print(f"  Total runtime: {success_df['runtime'].sum():.1f}s")
        print(f"  Mean speed: {success_df['ms_per_obs'].mean():.2f}ms/observation")
        
        # Check calibration (industry standard: 90-98% acceptable)
        print("\n" + "-"*80)
        print("Calibration Assessment:")
        
        coverage_ok = success_df['coverage'].between(0.90, 0.98).all()
        var_ok = success_df['var_rate'].between(0.01, 0.10).all()
        
        # Check if slightly conservative (96-98%) - this is actually preferred!
        conservative = success_df['coverage'].between(0.96, 0.98).all()
    
    print("\n" + "="*80)
    
    return df_results


if __name__ == '__main__':
    """Run basic validation with production config."""
    
    # Production configuration (from synthetic benchmarks)
    # - K=150: Optimal for wide grids
    # - N=500: More than sufficient
    # - Wide grid: Maximum robustness for regime changes
    
    results = run_basic_validation(
        K=150,
        N=500,
        grid_width='wide',
        threads=8,  # Adjust based on your CPU
        seed=42
    )
    
    # Save results
    #output_dir = Path(__file__).parent
    #results_file = output_dir / 'basic_validation_results.csv'
    #results.to_csv(results_file, index=False)
    #print(f"\n💾 Results saved to: {results_file}")
