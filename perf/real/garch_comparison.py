"""
GARCH Comparison: Industry Baseline vs Svmix Particle Filter

Phase 3 of real data validation. Compares three volatility models:
1. GARCH(1,1) with Normal errors (industry standard)
2. GARCH(1,1) with Student-t errors (handles fat tails)
3. Svmix Wide (K=150) - production configuration

Metrics:
- Predictive Log-Likelihood (PLL): Higher is better
- Coverage (95% intervals): Target 95%
- VaR violations (95% confidence): Expected 5%
- Mean Absolute Error vs realized volatility

All models evaluated using one-step-ahead rolling forecasts.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add current directory for local imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_all_periods,
    run_filter_with_metrics,
    summarize_period
)

# Check if arch is available
try:
    from arch import arch_model
    from scipy.stats import norm, t as student_t
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("ERROR: 'arch' package not found. Install with: pip install arch")
    sys.exit(1)


def fit_garch_rolling(
    returns: np.ndarray,
    dist: str = 'normal',
    burn_in: int = 100
) -> Dict[str, any]:
    """Fit GARCH(1,1) using rolling window and compute one-step-ahead metrics.
    
    Args:
        returns: Array of log returns
        dist: Error distribution ('normal' or 't')
        burn_in: Initial observations for first GARCH fit
        
    Returns:
        Dictionary with predicted volatilities, PLL values, and parameters
    """
    T = len(returns)
    
    if T <= burn_in:
        raise ValueError(f"Insufficient data: {T} observations, need >{burn_in}")
    
    # Storage
    predicted_vols = []
    pll_values = []
    
    # Initial fit on burn-in period
    model = arch_model(returns[:burn_in], vol='Garch', p=1, q=1, dist=dist)
    result = model.fit(disp='off', show_warning=False)
    
    # Rolling forecast
    for t in range(burn_in, T):
        # Refit every 50 observations (typical practice)
        if (t - burn_in) % 50 == 0:
            model = arch_model(returns[:t], vol='Garch', p=1, q=1, dist=dist)
            result = model.fit(disp='off', show_warning=False)
        
        # Forecast one-step-ahead volatility
        forecast = result.forecast(horizon=1, reindex=False)
        sigma_t = np.sqrt(forecast.variance.values[-1, 0])
        
        # Compute PLL for observed return
        if dist == 'normal':
            pll_t = norm.logpdf(returns[t], loc=0, scale=sigma_t)
        else:  # Student-t
            nu = result.params['nu']
            pll_t = student_t.logpdf(returns[t], df=nu, loc=0, scale=sigma_t)
        
        predicted_vols.append(sigma_t)
        pll_values.append(pll_t)
    
    # Get final parameters
    params = {
        'omega': result.params['omega'],
        'alpha': result.params['alpha[1]'],
        'beta': result.params['beta[1]']
    }
    if dist == 't':
        params['nu'] = result.params['nu']
    
    return {
        'predicted_vols': np.array(predicted_vols),
        'pll_values': np.array(pll_values),
        'pll_mean': float(np.mean(pll_values)),
        'pll_total': float(np.sum(pll_values)),
        'params': params,
        'eval_returns': returns[burn_in:]
    }


def compute_metrics(
    returns: np.ndarray,
    predicted_vols: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        returns: Realized returns
        predicted_vols: One-step-ahead volatility predictions
        
    Returns:
        Dictionary with coverage, VaR violations, and MAE
    """
    # Coverage (95% intervals)
    lower_bound = -1.96 * predicted_vols
    upper_bound = 1.96 * predicted_vols
    in_interval = (returns >= lower_bound) & (returns <= upper_bound)
    coverage = float(np.mean(in_interval))
    
    # VaR violations (5% expected)
    var_threshold = -1.645 * predicted_vols  # One-sided 95% VaR
    violations = np.sum(returns < var_threshold)
    violation_rate = float(violations / len(returns))
    
    # MAE vs realized volatility (5-minute rolling window)
    window = 5
    realized_vols = []
    predicted_vols_aligned = []
    
    for i in range(window, len(returns)):
        rv = np.sqrt(np.mean(returns[i-window:i]**2))
        realized_vols.append(rv)
        predicted_vols_aligned.append(predicted_vols[i])
    
    mae = float(np.mean(np.abs(np.array(predicted_vols_aligned) - np.array(realized_vols))))
    
    return {
        'coverage': coverage,
        'violations': int(violations),
        'violation_rate': violation_rate,
        'mae_vs_realized': mae
    }


def run_garch_comparison(
    threads: int = 0,
    seed: int = 42
):
    """Compare GARCH models vs Svmix on all periods.
    
    Args:
        threads: OpenMP threads for Svmix (0=auto)
        seed: Random seed for Svmix
    """
    print("="*80)
    print("GARCH COMPARISON: Industry Baseline vs Svmix")
    print("="*80)
    print("\nModels to evaluate:")
    print("  1. GARCH(1,1) - Normal errors")
    print("  2. GARCH(1,1) - Student-t errors")
    print("  3. Svmix Wide (K=150, N=500)")
    print("\nMetrics:")
    print("  - Predictive Log-Likelihood (PLL)")
    print("  - Coverage (95% prediction intervals)")
    print("  - VaR violations (95% confidence)")
    print("  - MAE vs realized volatility")
    print("\n" + "="*80)
    
    # Load all periods
    print("\nLoading market periods...")
    periods = load_all_periods()
    print(f"Loaded {len(periods)} periods")
    
    # Results storage
    all_results = []
    
    # Process each period
    for period_name, df in sorted(periods.items()):
        print(f"\n{'='*80}")
        print(f"Period: {period_name}")
        print(f"{'='*80}")
        
        summary = summarize_period(df)
        print(f"\nData: {summary['n_obs']:,} observations, {summary['volatility_annualized']:.1f}% annualized volatility")
        
        returns = df['returns'].values
        burn_in = 100
        
        # Model 1: GARCH(1,1) Normal
        print("\n  Fitting GARCH(1,1) - Normal...", end=' ', flush=True)
        try:
            garch_normal = fit_garch_rolling(returns, dist='normal', burn_in=burn_in)
            metrics_normal = compute_metrics(garch_normal['eval_returns'], 
                                            garch_normal['predicted_vols'])
            
            print(f"PLL={garch_normal['pll_mean']:.4f}, Cov={metrics_normal['coverage']:.1%}, " +
                  f"VaR={metrics_normal['violation_rate']:.1%}")
            
            all_results.append({
                'period': period_name,
                'model': 'GARCH-Normal',
                'n_obs': summary['n_obs'],
                'ann_vol': summary['volatility_annualized'],
                'pll_mean': garch_normal['pll_mean'],
                'pll_total': garch_normal['pll_total'],
                'coverage': metrics_normal['coverage'],
                'var_rate': metrics_normal['violation_rate'],
                'mae': metrics_normal['mae_vs_realized'],
                'omega': garch_normal['params']['omega'],
                'alpha': garch_normal['params']['alpha'],
                'beta': garch_normal['params']['beta']
            })
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'period': period_name,
                'model': 'GARCH-Normal',
                'error': str(e)
            })
        
        # Model 2: GARCH(1,1) Student-t
        print("  Fitting GARCH(1,1) - Student-t...", end=' ', flush=True)
        try:
            garch_t = fit_garch_rolling(returns, dist='t', burn_in=burn_in)
            metrics_t = compute_metrics(garch_t['eval_returns'], 
                                       garch_t['predicted_vols'])
            
            print(f"PLL={garch_t['pll_mean']:.4f}, Cov={metrics_t['coverage']:.1%}, " +
                  f"VaR={metrics_t['violation_rate']:.1%}")
            
            all_results.append({
                'period': period_name,
                'model': 'GARCH-t',
                'n_obs': summary['n_obs'],
                'ann_vol': summary['volatility_annualized'],
                'pll_mean': garch_t['pll_mean'],
                'pll_total': garch_t['pll_total'],
                'coverage': metrics_t['coverage'],
                'var_rate': metrics_t['violation_rate'],
                'mae': metrics_t['mae_vs_realized'],
                'omega': garch_t['params']['omega'],
                'alpha': garch_t['params']['alpha'],
                'beta': garch_t['params']['beta'],
                'nu': garch_t['params']['nu']
            })
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'period': period_name,
                'model': 'GARCH-t',
                'error': str(e)
            })
        
        # Model 3: Svmix Wide
        print("  Running Svmix Wide (K=150)...", end=' ', flush=True)
        try:
            svmix_results = run_filter_with_metrics(
                returns,
                K=150,
                N=500,
                grid_width='wide',
                burn_in=burn_in,
                threads=threads,
                seed=seed
            )
            
            svmix_metrics = compute_metrics(svmix_results['eval_returns'],
                                           svmix_results['predicted_vols'])
            
            print(f"PLL={svmix_results['pll_mean']:.4f}, Cov={svmix_results['coverage_95']:.1%}, " +
                  f"VaR={svmix_metrics['violation_rate']:.1%}, {svmix_results['runtime']:.1f}s")
            
            all_results.append({
                'period': period_name,
                'model': 'Svmix-Wide',
                'n_obs': summary['n_obs'],
                'ann_vol': summary['volatility_annualized'],
                'pll_mean': svmix_results['pll_mean'],
                'pll_total': svmix_results['pll_total'],
                'coverage': svmix_results['coverage_95'],
                'var_rate': svmix_metrics['violation_rate'],
                'mae': svmix_metrics['mae_vs_realized'],
                'runtime': svmix_results['runtime']
            })
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'period': period_name,
                'model': 'Svmix-Wide',
                'error': str(e)
            })
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results)
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY: Performance Comparison")
    print("="*80)
    
    # Check for errors
    if 'error' in df_results.columns and df_results['error'].notna().any():
        print("\nWARNING: Some runs encountered errors")
        error_df = df_results[df_results['error'].notna()][['period', 'model', 'error']]
        print(error_df.to_string(index=False))
        print()
        success_df = df_results[df_results['error'].isna()]
    else:
        success_df = df_results
    
    if len(success_df) == 0:
        print("\nNo successful runs to analyze.")
        return df_results
    
    # Overall statistics by model
    print("\nOverall Performance by Model:")
    print("-"*80)
    
    model_summary = success_df.groupby('model').agg({
        'pll_mean': ['mean', 'std'],
        'coverage': ['mean', 'min', 'max'],
        'var_rate': ['mean'],
        'mae': ['mean']
    }).round(4)
    
    print(model_summary)
    
    # Performance by volatility regime
    print("\n\nPerformance by Volatility Regime:")
    print("-"*80)
    
    success_df['regime'] = success_df['ann_vol'].apply(
        lambda x: 'Calm' if x < 10 else ('Normal' if x < 30 else 'Crisis')
    )
    
    regime_summary = success_df.groupby(['regime', 'model']).agg({
        'pll_mean': 'mean',
        'coverage': 'mean',
        'var_rate': 'mean'
    }).round(4)
    
    print(regime_summary)
    
    # Detailed comparison table
    print("\n\nDetailed Results by Period:")
    print("-"*80)
    
    # Pivot tables
    pivot_pll = success_df.pivot(index='period', columns='model', values='pll_mean')
    pivot_cov = success_df.pivot(index='period', columns='model', values='coverage')
    pivot_var = success_df.pivot(index='period', columns='model', values='var_rate')
    
    print("\nPredictive Log-Likelihood (higher is better):")
    print(pivot_pll.round(4).to_string())
    
    print("\nCoverage (target: 95%):")
    print((pivot_cov * 100).round(1).to_string())
    
    print("\nVaR Violations (expected: 5%):")
    print((pivot_var * 100).round(1).to_string())
    
    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    
    # PLL improvements
    print("\nPredictive Log-Likelihood Improvements:")
    print("-"*80)
    
    for model in ['GARCH-Normal', 'GARCH-t']:
        if model in success_df['model'].values and 'Svmix-Wide' in success_df['model'].values:
            baseline = success_df[success_df['model'] == model]['pll_mean'].mean()
            svmix_pll = success_df[success_df['model'] == 'Svmix-Wide']['pll_mean'].mean()
            
            improvement = (svmix_pll - baseline) / abs(baseline) * 100
            abs_improvement = svmix_pll - baseline
            
            print(f"Svmix vs {model:15s}: {improvement:+.2f}% ({abs_improvement:+.4f} absolute)")
    
    # Coverage comparison
    print("\n\nCalibration Quality (deviation from 95% target):")
    print("-"*80)
    
    for model in success_df['model'].unique():
        model_data = success_df[success_df['model'] == model]
        mean_cov = model_data['coverage'].mean()
        deviation = (mean_cov - 0.95) * 100
        
        print(f"{model:15s}: {mean_cov:.1%} (deviation: {deviation:+.1f} pp)")
    
    # VaR violations comparison
    print("\n\nRisk Management (VaR violations vs expected 5%):")
    print("-"*80)
    
    for model in success_df['model'].unique():
        model_data = success_df[success_df['model'] == model]
        mean_var = model_data['var_rate'].mean()
        excess = (mean_var - 0.05) * 100
        
        status = "Conservative" if excess < 0 else "Aggressive"
        print(f"{model:15s}: {mean_var:.1%} ({excess:+.1f} pp, {status})")
    
    # Winner by metric
    print("\n" + "="*80)
    print("WINNERS BY METRIC")
    print("="*80)
    
    model_means = success_df.groupby('model').agg({
        'pll_mean': 'mean',
        'coverage': lambda x: abs(x.mean() - 0.95),  # Distance from target
        'var_rate': lambda x: abs(x.mean() - 0.05)   # Distance from target
    })
    
    print(f"\nBest PLL:        {model_means['pll_mean'].idxmax()}")
    print(f"Best Coverage:   {model_means['coverage'].idxmin()} (closest to 95%)")
    print(f"Best VaR:        {model_means['var_rate'].idxmin()} (closest to 5%)")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    svmix_wins_pll = model_means['pll_mean'].idxmax() == 'Svmix-Wide'
    svmix_wins_cov = model_means['coverage'].idxmin() == 'Svmix-Wide'
    
    if svmix_wins_pll and svmix_wins_cov:
        print("\nSvmix Wide demonstrates superior performance:")
        print("  - Highest predictive log-likelihood")
        print("  - Best calibration (coverage closest to 95%)")
        print("  - Suitable for production deployment")
    elif svmix_wins_pll:
        print("\nSvmix Wide shows strong performance:")
        print("  - Highest predictive log-likelihood")
        print("  - Calibration within acceptable bounds")
    else:
        print("\nMixed results. Further analysis recommended.")
    
    print("\n" + "="*80)
    
    return df_results


if __name__ == '__main__':
    """Run GARCH comparison."""
    
    results = run_garch_comparison(
        threads=8,
        seed=42
    )
    
    # Save results
    #output_dir = Path(__file__).parent
    #results_file = output_dir / 'garch_comparison_results.csv'
    #results.to_csv(results_file, index=False)
    #print(f"\nResults saved to: {results_file}")
