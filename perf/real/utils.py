"""
Utilities for real data validation.

Implements industry-standard best practices for volatility filter evaluation:
1. One-step-ahead prediction coverage (out-of-sample)
2. Proper PLL calculation
3. Close-to-close log returns
4. Burn-in period handling
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

from svmix import Svmix, SvmixConfig, SvParams, SvParamsVol, Spec


# Constants
DATA_DIR = Path(__file__).parent.parent / 'data'
BURN_IN = 100  # Industry standard: use first 100 obs for filter initialization


def load_period(period_name: str) -> pd.DataFrame:
    """Load a specific market period from CSV.
    
    Args:
        period_name: Name of period (e.g., 'us30_2008_crisis')
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, returns
        
    Raises:
        FileNotFoundError: If period file doesn't exist
    """
    filepath = DATA_DIR / f"{period_name}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Period data not found: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    # Compute close-to-close log returns (industry standard)
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Drop first row (NaN return)
    df = df.iloc[1:].reset_index(drop=True)
    
    return df


def load_all_periods() -> Dict[str, pd.DataFrame]:
    """Load all available market periods.
    
    Returns:
        Dictionary mapping period name to DataFrame
    """
    # Read metadata to get period names
    metadata_path = DATA_DIR / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    
    periods = {}
    for _, row in metadata.iterrows():
        period_name = row['name']
        try:
            periods[period_name] = load_period(period_name)
        except Exception as e:
            print(f"Warning: Could not load {period_name}: {e}")
    
    return periods


def create_filter_config(
    K: int = 150,
    N: int = 500,
    grid_width: str = 'wide',
    threads: int = 0,
    seed: int = 42
) -> Tuple[SvmixConfig, List[SvParamsVol]]:
    """Create filter configuration with specified parameters.
    
    Uses SvParams.linspace() to generate parameter grid automatically.
    
    Args:
        K: Number of ensemble models
        N: Number of particles per model
        grid_width: 'narrow', 'medium', or 'wide'
        threads: Number of OpenMP threads (0=auto)
        seed: Random seed
        
    Returns:
        (config, params) tuple for Svmix initialization
    """
    # Configuration
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=K,
        num_particles=N,
        lambda_=0.995,  # Long-term stability for real data
        beta=0.8,
        epsilon=0.05,  # Anti-starvation
        num_threads=threads,
        seed=seed
    )
    
    # Parameter grids (from synthetic benchmarks)
    # Using linspace to automatically generate K models
    if grid_width == 'narrow':
        # Narrow: Optimized for typical conditions
        phi_range = (0.91, 0.98)
        sigma_range = (0.15, 0.30)
    
    elif grid_width == 'medium':
        # Medium: Broad coverage, good for stable + regime changes
        phi_range = (0.88, 0.99)
        sigma_range = (0.12, 0.40)
    
    elif grid_width == 'wide':
        # Wide: Maximum robustness for production
        phi_range = (0.85, 0.995)
        sigma_range = (0.10, 0.50)
    
    elif grid_width == 'ultra_wide':
        # Ultra-Wide: EXPERIMENTAL - Finding the optimal frontier
        # Even broader parameter coverage to maximize PLL
        phi_range = (0.80, 0.998)  # Very low to very high persistence
        sigma_range = (0.05, 0.70)  # Very low to very high volatility-of-volatility
    
    else:
        raise ValueError(f"Unknown grid_width: {grid_width}. Use 'narrow', 'medium', 'wide', or 'ultra_wide'.")
    
    # Use linspace to generate exactly K parameter sets
    # Each parameter can be a single value or a (min, max) tuple
    params = SvParams.linspace(
        K,
        phi=phi_range,      # (min, max) tuple
        sigma=sigma_range,  # (min, max) tuple
        nu=(6.0, 14.0),     # (light, heavy) tails
        mu=-0.5             # Fixed log-vol mean
    )
    
    return config, params


def run_filter_with_metrics(
    returns: np.ndarray,
    K: int = 150,
    N: int = 500,
    grid_width: str = 'wide',
    burn_in: int = BURN_IN,
    threads: int = 0,
    seed: int = 42
) -> Dict[str, any]:
    """Run filter on returns and compute evaluation metrics.
    
    Implements industry-standard one-step-ahead prediction evaluation:
    1. Burn-in period for filter initialization
    2. One-step-ahead coverage (out-of-sample)
    3. Predictive log-likelihood
    
    Args:
        returns: Array of log returns
        K: Number of models
        N: Number of particles
        grid_width: 'narrow', 'medium', or 'wide'
        burn_in: Number of initial observations for filter warm-up
        threads: OpenMP threads
        seed: Random seed
        
    Returns:
        Dictionary with:
            - pll_total: Total predictive log-likelihood
            - pll_mean: Mean PLL per observation
            - coverage_95: One-step-ahead 95% coverage rate
            - vol_estimates: Array of volatility estimates
            - predicted_vols: Array of one-step-ahead vol predictions
            - eval_returns: Returns used for evaluation (post burn-in)
            - runtime: Execution time in seconds
    """
    import time
    
    T = len(returns)
    
    if T <= burn_in:
        raise ValueError(f"Not enough data: {T} observations, need >{burn_in}")
    
    # Create filter
    config, params = create_filter_config(K, N, grid_width, threads, seed)
    svmix = Svmix(config, params)
    
    # Storage for metrics
    pll_values = []
    predicted_vols = []  # One-step-ahead predictions
    vol_estimates = []   # Current volatility estimates
    
    start = time.time()
    
    # Burn-in period (initialize filter, don't evaluate)
    for t in range(burn_in):
        svmix.step(returns[t])
    
    # Evaluation period (compute metrics)
    for t in range(burn_in, T):
        # CRITICAL: Get prediction BEFORE seeing new data
        # This is the one-step-ahead prediction for y_t
        belief_prev = svmix.get_belief()
        predicted_vol = belief_prev.mean_sigma  # E[σ_t | y_{1:t-1}]
        
        # Update with new observation
        svmix.step(returns[t])
        
        # Get PLL for this step (already one-step-ahead by design)
        pll = svmix.get_last_log_likelihood()
        
        # Get updated volatility estimate
        belief_curr = svmix.get_belief()
        vol_estimate = belief_curr.mean_sigma
        
        # Store for evaluation
        predicted_vols.append(predicted_vol)
        pll_values.append(pll)
        vol_estimates.append(vol_estimate)
    
    runtime = time.time() - start
    
    # Clean up
    svmix.free()
    
    # Compute coverage (one-step-ahead)
    eval_returns = returns[burn_in:]
    predicted_vols = np.array(predicted_vols)
    
    # Check if returns fall in predicted 95% intervals
    # For Student-t with ν degrees of freedom, use 1.96 for large ν
    # (could be more precise with scipy.stats.t.ppf, but 1.96 is standard)
    lower_bound = -1.96 * predicted_vols
    upper_bound = 1.96 * predicted_vols
    
    in_interval = (eval_returns >= lower_bound) & (eval_returns <= upper_bound)
    coverage_95 = float(np.mean(in_interval))
    
    # Compute PLL statistics
    pll_total = float(np.sum(pll_values))
    pll_mean = float(np.mean(pll_values))
    
    return {
        'pll_total': pll_total,
        'pll_mean': pll_mean,
        'coverage_95': coverage_95,
        'vol_estimates': np.array(vol_estimates),
        'predicted_vols': predicted_vols,
        'eval_returns': eval_returns,
        'runtime': runtime,
        'T_total': T,
        'T_eval': T - burn_in,
        'burn_in': burn_in
    }


def compute_metrics(
    returns: np.ndarray,
    predicted_vols: np.ndarray
) -> Dict[str, float]:
    """Compute coverage and VaR metrics for model evaluation.
    
    Args:
        returns: Realized returns
        predicted_vols: One-step-ahead volatility predictions
        
    Returns:
        Dictionary with coverage and VaR violation metrics
    """
    # 95% coverage (two-sided)
    lower_bound = -1.96 * predicted_vols
    upper_bound = 1.96 * predicted_vols
    in_interval = (returns >= lower_bound) & (returns <= upper_bound)
    coverage = float(np.mean(in_interval))
    
    # VaR violations (one-sided)
    var_metrics = compute_var_violations(returns, predicted_vols, confidence=0.95)
    
    return {
        'coverage': coverage,
        'violation_rate': var_metrics['violation_rate'],
        'violations': var_metrics['violations']
    }


def compute_var_violations(
    returns: np.ndarray,
    predicted_vols: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """Compute VaR violations for backtesting.
    
    Value at Risk (VaR) backtesting is standard in risk management.
    A well-calibrated model should have violations ≈ (1 - confidence).
    
    Args:
        returns: Realized returns
        predicted_vols: One-step-ahead volatility predictions
        confidence: VaR confidence level (default 95%)
        
    Returns:
        Dictionary with:
            - violations: Number of VaR breaches
            - violation_rate: Proportion of violations
            - expected_rate: Expected violation rate (1 - confidence)
            - excess_violations: violations - expected
    """
    from scipy.stats import norm
    
    # VaR quantile (e.g., -1.645 for 95% one-sided)
    z = norm.ppf(1 - confidence)
    
    # VaR threshold (negative because we care about losses)
    var_threshold = z * predicted_vols
    
    # Count violations (returns below VaR threshold)
    violations = np.sum(returns < var_threshold)
    violation_rate = violations / len(returns)
    expected_rate = 1 - confidence
    
    return {
        'violations': int(violations),
        'violation_rate': float(violation_rate),
        'expected_rate': float(expected_rate),
        'excess_violations': violations - len(returns) * expected_rate
    }


def summarize_period(df: pd.DataFrame) -> Dict[str, any]:
    """Compute summary statistics for a market period.
    
    Args:
        df: DataFrame with 'returns' column
        
    Returns:
        Dictionary with summary statistics
    """
    returns = df['returns'].dropna()
    
    # Annualization factor (390 mins per trading day, 252 days per year)
    annual_factor = np.sqrt(390 * 252)
    
    return {
        'n_obs': len(returns),
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'skewness': float(returns.skew()),
        'kurtosis': float(returns.kurtosis()),
        'min_return': float(returns.min()),
        'max_return': float(returns.max()),
        'volatility_annualized': float(returns.std() * annual_factor * 100)  # %
    }


if __name__ == '__main__':
    """Test utilities by loading data and running basic checks."""
    print("Testing real data utilities...")
    print("="*70)
    
    # Test 1: Load all periods
    print("\n1. Loading all periods...")
    periods = load_all_periods()
    print(f"   Loaded {len(periods)} periods:")
    for name, df in periods.items():
        print(f"   - {name}: {len(df):,} observations")
    
    # Test 2: Load and summarize one period
    print("\n2. Testing period summary...")
    test_period = 'us30_2008_crisis'
    df = load_period(test_period)
    summary = summarize_period(df)
    
    print(f"   Period: {test_period}")
    print(f"   Observations: {summary['n_obs']:,}")
    print(f"   Annualized vol: {summary['volatility_annualized']:.1f}%")
    print(f"   Mean return: {summary['mean_return']*100:.4f}%")
    print(f"   Skewness: {summary['skewness']:.2f}")
    print(f"   Kurtosis: {summary['kurtosis']:.2f}")
    
    # Test 3: Create filter configs
    print("\n3. Testing filter configuration...")
    for width in ['narrow', 'medium', 'wide']:
        config, params = create_filter_config(K=50, N=250, grid_width=width)
        print(f"   {width:8s}: K={len(params)}, phi∈[{params[0].phi:.2f},{params[-1].phi:.3f}]")
    
    # Test 4: Run filter on small sample
    print("\n4. Testing filter execution...")
    returns = df['returns'].values
    results = run_filter_with_metrics(
        returns[:1000],  # Small sample for quick test
        K=20,
        N=250,
        grid_width='wide',
        burn_in=50,
        threads=8
    )
    
    print(f"   Evaluated {results['T_eval']} observations (after {results['burn_in']} burn-in)")
    print(f"   Mean PLL: {results['pll_mean']:.4f}")
    print(f"   Coverage: {results['coverage_95']:.1%}")
    print(f"   Runtime: {results['runtime']:.2f}s")
    
    print("\n" + "="*70)
    print("✓ All tests passed! Utilities are ready.")
