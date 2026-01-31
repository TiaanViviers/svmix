"""
Config comparison: Test narrow vs medium vs wide grids on real data.

This is Phase 2 of real data validation:
- Test 3 configurations (narrow, medium, wide) on all 6 periods
- Compare PLL, coverage, VaR violations
- Validate synthetic benchmark findings on real data

Key hypothesis to test:
- Narrow grids should excel in calm markets (2014, 2021)
- Wide grids should excel in crisis markets (2008, 2020)
- Medium grids should be balanced

Expected outcomes:
- Narrow: Best PLL in calm, degraded coverage in crises
- Wide: Consistent coverage everywhere, slightly lower PLL in calm
- This matches our synthetic findings!
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_all_periods,
    run_filter_with_metrics,
    compute_var_violations,
    summarize_period
)


def run_config_comparison(
    threads: int = 0,
    seed: int = 42
):
    """Compare narrow, medium, and wide configurations on all periods.
    
    Tests 3 configs × 6 periods = 18 total runs.
    
    Args:
        threads: OpenMP threads (0=auto)
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("CONFIG COMPARISON: Narrow vs Medium vs Wide on Real Data")
    print("="*80)
    print("\nConfigurations to test:")
    print("  1. Narrow (K=50):  Optimized for typical conditions")
    print("     - φ ∈ [0.91, 0.98], σ ∈ [0.15, 0.30]")
    print("     - Best for: Calm markets, high precision")
    print()
    print("  2. Medium (K=100): Broad coverage, balanced")
    print("     - φ ∈ [0.88, 0.99], σ ∈ [0.12, 0.40]")
    print("     - Best for: Mixed regimes")
    print()
    print("  3. Wide (K=150):   Maximum robustness")
    print("     - φ ∈ [0.85, 0.995], σ ∈ [0.10, 0.50]")
    print("     - Best for: All regimes, especially crises")
    print()
    print("  4. Ultra-Wide (K=200): EXPERIMENTAL - Finding the frontier")
    print("     - φ ∈ [0.80, 0.998], σ ∈ [0.05, 0.70]")
    print("     - Goal: Find PLL maximum / optimal calibration")
    print()
    print("  Threads:", threads if threads > 0 else 'auto')
    print("  Seed:", seed)
    print("\n" + "="*80)
    
    # Load all periods
    print("\nLoading market periods...")
    periods = load_all_periods()
    print(f"Loaded {len(periods)} periods")
    
    # Configurations to test
    configs = [
        {'name': 'Narrow', 'K': 50, 'N': 500, 'grid_width': 'narrow'},
        {'name': 'Medium', 'K': 100, 'N': 500, 'grid_width': 'medium'},
        {'name': 'Wide', 'K': 150, 'N': 500, 'grid_width': 'wide'},
        {'name': 'UltraWide', 'K': 200, 'N': 500, 'grid_width': 'ultra_wide'},
    ]
    
    # Results storage
    all_results = []
    
    # Process each period × config combination
    for period_name, df in sorted(periods.items()):
        print(f"\n{'='*80}")
        print(f"Period: {period_name}")
        print(f"{'='*80}")
        
        # Period summary
        summary = summarize_period(df)
        print(f"\nData: {summary['n_obs']:,} obs, {summary['volatility_annualized']:.1f}% vol")
        
        returns = df['returns'].values
        
        # Test each configuration
        for config in configs:
            print(f"\n  Testing {config['name']} (K={config['K']})...", end=' ', flush=True)
            
            try:
                metrics = run_filter_with_metrics(
                    returns,
                    K=config['K'],
                    N=config['N'],
                    grid_width=config['grid_width'],
                    burn_in=100,
                    threads=threads,
                    seed=seed
                )
                
                var_results = compute_var_violations(
                    metrics['eval_returns'],
                    metrics['predicted_vols'],
                    confidence=0.95
                )
                
                print(f"PLL={metrics['pll_mean']:.4f}, Cov={metrics['coverage_95']:.1%}, " +
                      f"VaR={var_results['violation_rate']:.1%}, {metrics['runtime']:.1f}s")
                
                # Store results
                all_results.append({
                    'period': period_name,
                    'config': config['name'],
                    'K': config['K'],
                    'N': config['N'],
                    'grid_width': config['grid_width'],
                    'n_obs': summary['n_obs'],
                    'ann_vol': summary['volatility_annualized'],
                    'pll_mean': metrics['pll_mean'],
                    'coverage': metrics['coverage_95'],
                    'var_rate': var_results['violation_rate'],
                    'mean_vol': metrics['predicted_vols'].mean(),
                    'runtime': metrics['runtime'],
                    'ms_per_obs': metrics['runtime'] / metrics['T_eval'] * 1000
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    'period': period_name,
                    'config': config['name'],
                    'error': str(e)
                })
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results)
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY: Performance by Configuration")
    print("="*80)
    
    # Check for errors
    if 'error' in df_results.columns and df_results['error'].notna().any():
        print("\n⚠️  Some runs had errors:")
        error_df = df_results[df_results['error'].notna()][['period', 'config', 'error']]
        print(error_df.to_string(index=False))
        print()
        success_df = df_results[df_results['error'].isna()]
    else:
        success_df = df_results
    
    # Overall statistics by config
    print("\nOverall Performance by Config:")
    print("-"*80)
    
    config_summary = success_df.groupby('config').agg({
        'pll_mean': ['mean', 'std'],
        'coverage': ['mean', 'min', 'max'],
        'var_rate': 'mean',
        'runtime': 'sum'
    }).round(4)
    
    print(config_summary)
    
    # Performance by volatility regime
    print("\n\nPerformance by Volatility Regime:")
    print("-"*80)
    
    # Categorize periods by volatility
    success_df['regime'] = success_df['ann_vol'].apply(
        lambda x: 'Calm' if x < 10 else ('Normal' if x < 30 else 'Crisis')
    )
    
    regime_summary = success_df.groupby(['regime', 'config']).agg({
        'pll_mean': 'mean',
        'coverage': 'mean',
        'var_rate': 'mean'
    }).round(4)
    
    print(regime_summary)
    
    # Detailed comparison table
    print("\n\nDetailed Results (all periods × configs):")
    print("-"*80)
    
    # Pivot table for easy comparison
    pivot_pll = success_df.pivot(index='period', columns='config', values='pll_mean')
    pivot_cov = success_df.pivot(index='period', columns='config', values='coverage')
    pivot_var = success_df.pivot(index='period', columns='config', values='var_rate')
    
    # Add volatility info
    vol_info = success_df.drop_duplicates('period').set_index('period')['ann_vol']
    
    print("\nPLL (Predictive Log-Likelihood) - Higher is Better:")
    pll_display = pivot_pll.copy()
    pll_display.insert(0, 'Vol%', vol_info)
    print(pll_display.round(4).to_string())
    
    print("\nCoverage (%) - Target: 95%:")
    cov_display = pivot_cov.copy()
    cov_display.insert(0, 'Vol%', vol_info)
    print((cov_display * 100).round(1).to_string())
    
    print("\nVaR Violations (%) - Expected: 5%:")
    var_display = pivot_var.copy()
    var_display.insert(0, 'Vol%', vol_info)
    print((var_display * 100).round(1).to_string())
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 1. PLL comparison
    calm_periods = success_df[success_df['regime'] == 'Calm']
    crisis_periods = success_df[success_df['regime'] == 'Crisis']
    
    if len(calm_periods) > 0:
        calm_by_config = calm_periods.groupby('config')['pll_mean'].mean()
        print(f"\n1. PLL in Calm Markets (Vol < 10%):")
        for config in ['Narrow', 'Medium', 'Wide', 'UltraWide']:
            if config in calm_by_config:
                print(f"   {config:10s}: {calm_by_config[config]:.4f}")
        best_calm = calm_by_config.idxmax()
        print(f"   → Winner: {best_calm}")
    
    if len(crisis_periods) > 0:
        crisis_by_config = crisis_periods.groupby('config')['pll_mean'].mean()
        print(f"\n2. PLL in Crisis Markets (Vol > 30%):")
        for config in ['Narrow', 'Medium', 'Wide', 'UltraWide']:
            if config in crisis_by_config:
                print(f"   {config:10s}: {crisis_by_config[config]:.4f}")
        best_crisis = crisis_by_config.idxmax()
        print(f"   → Winner: {best_crisis}")
    
    # 2. Coverage stability
    print(f"\n3. Coverage Stability (range across periods):")
    for config in ['Narrow', 'Medium', 'Wide', 'UltraWide']:
        config_data = success_df[success_df['config'] == config]
        if len(config_data) > 0:
            cov_range = config_data['coverage'].max() - config_data['coverage'].min()
            cov_mean = config_data['coverage'].mean()
            print(f"   {config:10s}: {cov_mean:.1%} ± {cov_range*100:.1f}pp")
    
    # 3. Crisis robustness
    if len(crisis_periods) > 0:
        print(f"\n4. Crisis Robustness (Coverage in Vol > 30%):")
        crisis_by_config = crisis_periods.groupby('config')['coverage'].mean()
        for config in ['Narrow', 'Medium', 'Wide', 'UltraWide']:
            if config in crisis_by_config:
                print(f"   {config:10s}: {crisis_by_config[config]:.1%}")
        best_crisis_cov = crisis_by_config.idxmax()
        print(f"   → Winner: {best_crisis_cov}")
    
    # 4. Speed comparison
    print(f"\n5. Runtime Comparison (total across all periods):")
    runtime_by_config = success_df.groupby('config')['runtime'].sum()
    for config in ['Narrow', 'Medium', 'Wide', 'UltraWide']:
        if config in runtime_by_config:
            print(f"   {config:10s}: {runtime_by_config[config]:.1f}s")
    
    
    return df_results


if __name__ == '__main__':
    """Run configuration comparison."""
    
    results = run_config_comparison(
        threads=8,  # Adjust for your CPU
        seed=42
    )
    
    # Save results
    #output_dir = Path(__file__).parent
    #results_file = output_dir / 'config_comparison_results.csv'
    #results.to_csv(results_file, index=False)
    #print(f"\nResults saved to: {results_file}")
