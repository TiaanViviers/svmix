"""
Accuracy metrics for evaluating svmix filter performance.

This module provides standard metrics for comparing filter estimates
against ground truth (synthetic data) or evaluating predictive quality
(real data).

Metrics:
    - RMSE: Root mean squared error (requires ground truth)
    - PLL: Predictive log-likelihood (model comparison, no ground truth needed)
    - Coverage: 95% credible interval coverage (calibration check)

Usage:
    >>> from metrics import rmse, coverage_95, pll_sequence
    >>> 
    >>> # With ground truth (synthetic data)
    >>> rmse_h = rmse(h_true, h_estimated)
    >>> cov = coverage_95(h_true, belief_means, belief_vars)
    >>> 
    >>> # Without ground truth (real data)
    >>> total_pll = pll_sequence(svmix, observations)
"""

import math
from typing import List, Sequence, Optional


def rmse(true_values: Sequence[float], estimated_values: Sequence[float]) -> float:
    """Compute root mean squared error.
    
    RMSE measures the typical magnitude of estimation error. Lower is better.
    Requires ground truth, so only applicable to synthetic data.
    
    Args:
        true_values: Ground truth values (e.g., true log-volatility h_t)
        estimated_values: Filter estimates (e.g., belief.mean_h)
        
    Returns:
        RMSE value (same units as inputs)
        
    Raises:
        ValueError: If sequences have different lengths or are empty
        
    Example:
        >>> h_true = [0.0, 0.1, 0.2, 0.15, 0.1]
        >>> h_est = [0.05, 0.12, 0.18, 0.17, 0.09]
        >>> error = rmse(h_true, h_est)
        >>> print(f"RMSE: {error:.4f}")
        RMSE: 0.0346
        
    Note:
        - Sensitive to outliers (squared error)
        - Scale-dependent (not comparable across different state variables)
        - For log-volatility h, typical RMSE: 0.05-0.20 depending on N, K
    """
    if len(true_values) != len(estimated_values):
        raise ValueError(
            f"Length mismatch: true_values={len(true_values)}, "
            f"estimated_values={len(estimated_values)}"
        )
    
    if len(true_values) == 0:
        raise ValueError("Cannot compute RMSE on empty sequences")
    
    squared_errors = [(t - e)**2 for t, e in zip(true_values, estimated_values)]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    
    return math.sqrt(mean_squared_error)


def coverage_95(
    true_values: Sequence[float],
    estimated_means: Sequence[float],
    estimated_variances: Sequence[float]
) -> float:
    """Compute 95% credible interval coverage rate.
    
    Checks what percentage of true values fall within the filter's 95%
    credible intervals. Well-calibrated filters should achieve ~95% coverage.
    
    The 95% interval is computed as: mean ± 1.96 * sqrt(variance)
    
    Args:
        true_values: Ground truth values
        estimated_means: Filter posterior means
        estimated_variances: Filter posterior variances
        
    Returns:
        Coverage percentage (0.0 to 1.0)
        
    Raises:
        ValueError: If sequences have different lengths or are empty
        
    Example:
        >>> h_true = [0.0, 0.1, 0.2, 0.15, 0.1]
        >>> means = [0.05, 0.12, 0.18, 0.17, 0.09]
        >>> variances = [0.01, 0.01, 0.01, 0.01, 0.01]
        >>> cov = coverage_95(h_true, means, variances)
        >>> print(f"Coverage: {cov:.1%}")
        Coverage: 100.0%
        
    Interpretation:
        - Coverage < 0.90: Overconfident (variances too small)
        - Coverage ≈ 0.95: Well-calibrated
        - Coverage > 0.98: Underconfident (variances too large)
        
    Note:
        For particle filters, coverage often slightly exceeds 95% due to
        heavy-tailed posterior distributions.
    """
    if not (len(true_values) == len(estimated_means) == len(estimated_variances)):
        raise ValueError(
            f"Length mismatch: true={len(true_values)}, "
            f"means={len(estimated_means)}, vars={len(estimated_variances)}"
        )
    
    if len(true_values) == 0:
        raise ValueError("Cannot compute coverage on empty sequences")
    
    # 95% interval: mean ± 1.96 * std
    z_score = 1.96
    in_interval_count = 0
    
    for true_val, mean, variance in zip(true_values, estimated_means, estimated_variances):
        if variance < 0:
            raise ValueError(f"Negative variance encountered: {variance}")
        
        std = math.sqrt(variance)
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        if lower <= true_val <= upper:
            in_interval_count += 1
    
    coverage = in_interval_count / len(true_values)
    return coverage


def pll_sequence(observations: Sequence[float], log_likelihoods: Sequence[float]) -> float:
    """Compute total predictive log-likelihood over a sequence.
    
    PLL (Predictive Log-Likelihood) measures how well the filter predicts
    observations. Higher is better. Does NOT require ground truth.
    
    This is simply the sum of per-step log p(y_t | y_{1:t-1}) values.
    
    Args:
        observations: Observation sequence (for length validation only)
        log_likelihoods: Per-step log-likelihoods from svmix.get_last_log_likelihood()
        
    Returns:
        Total PLL (sum of log-likelihoods)
        
    Raises:
        ValueError: If sequences have different lengths or contain invalid values
        
    Example:
        >>> observations = [0.01, -0.02, 0.015, -0.005]
        >>> log_liks = [-0.52, -0.48, -0.51, -0.49]
        >>> total_pll = pll_sequence(observations, log_liks)
        >>> print(f"Total PLL: {total_pll:.2f}")
        Total PLL: -2.00
        
    Usage with svmix:
        >>> svmix = Svmix(config, params)
        >>> log_liks = []
        >>> for obs in observations:
        ...     svmix.step(obs)
        ...     log_liks.append(svmix.get_last_log_likelihood())
        >>> total_pll = pll_sequence(observations, log_liks)
        
    Interpretation:
        - PLL is RELATIVE: only useful for comparing models
        - More negative = worse fit
        - Difference of 10 in PLL is strong evidence for better model
        - PLL scales with sequence length (longer = more negative)
        
    Note:
        - Not affected by lambda (exponential forgetting)
        - Proper scoring rule (incentivizes honest probability forecasts)
        - Can be normalized: PLL / T gives average per-step log-likelihood
    """
    if len(observations) != len(log_likelihoods):
        raise ValueError(
            f"Length mismatch: observations={len(observations)}, "
            f"log_likelihoods={len(log_likelihoods)}"
        )
    
    if len(observations) == 0:
        raise ValueError("Cannot compute PLL on empty sequences")
    
    # Check for invalid log-likelihoods
    for i, ll in enumerate(log_likelihoods):
        if not math.isfinite(ll):
            raise ValueError(
                f"Non-finite log-likelihood at index {i}: {ll}. "
                "Make sure to call svmix.step() before get_last_log_likelihood()."
            )
    
    total_pll = sum(log_likelihoods)
    return total_pll


def mean_pll(observations: Sequence[float], log_likelihoods: Sequence[float]) -> float:
    """Compute mean predictive log-likelihood (normalized by sequence length).
    
    This is PLL / T, giving the average log-likelihood per observation.
    Useful for comparing across different sequence lengths.
    
    Args:
        observations: Observation sequence
        log_likelihoods: Per-step log-likelihoods
        
    Returns:
        Mean PLL (average per-step log-likelihood)
        
    Example:
        >>> observations = [0.01, -0.02, 0.015, -0.005]
        >>> log_liks = [-0.52, -0.48, -0.51, -0.49]
        >>> avg_pll = mean_pll(observations, log_liks)
        >>> print(f"Mean PLL: {avg_pll:.4f}")
        Mean PLL: -0.5000
    """
    total = pll_sequence(observations, log_likelihoods)
    return total / len(observations)


# ==============================================================================
# Summary Statistics
# ==============================================================================

def compute_all_metrics(
    observations: Sequence[float],
    log_likelihoods: Sequence[float],
    true_states: Optional[Sequence[float]] = None,
    estimated_means: Optional[Sequence[float]] = None,
    estimated_variances: Optional[Sequence[float]] = None
) -> dict:
    """Compute all applicable metrics and return as dictionary.
    
    Args:
        observations: Observation sequence
        log_likelihoods: Per-step log-likelihoods from filter
        true_states: Ground truth states (optional, for synthetic data)
        estimated_means: Filter posterior means (optional, needed if true_states given)
        estimated_variances: Filter posterior variances (optional, for coverage)
        
    Returns:
        Dictionary with computed metrics. Keys present depend on available data:
            - 'pll_total': Total predictive log-likelihood (always)
            - 'pll_mean': Mean predictive log-likelihood per step (always)
            - 'rmse': Root mean squared error (if true_states given)
            - 'coverage_95': 95% credible interval coverage (if true_states + variances given)
            
    Example:
        >>> # Synthetic data (all metrics)
        >>> metrics = compute_all_metrics(
        ...     observations=obs,
        ...     log_likelihoods=lls,
        ...     true_states=h_true,
        ...     estimated_means=h_est,
        ...     estimated_variances=h_var
        ... )
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
        >>> print(f"Coverage: {metrics['coverage_95']:.1%}")
        >>> print(f"Mean PLL: {metrics['pll_mean']:.4f}")
        
        >>> # Real data (PLL only)
        >>> metrics = compute_all_metrics(
        ...     observations=obs,
        ...     log_likelihoods=lls
        ... )
        >>> print(f"Mean PLL: {metrics['pll_mean']:.4f}")
    """
    results = {}
    
    # PLL metrics (always available)
    results['pll_total'] = pll_sequence(observations, log_likelihoods)
    results['pll_mean'] = results['pll_total'] / len(observations)
    
    # RMSE (requires ground truth)
    if true_states is not None:
        if estimated_means is None:
            raise ValueError("estimated_means required when true_states provided")
        results['rmse'] = rmse(true_states, estimated_means)
    
    # Coverage (requires ground truth and variances)
    if true_states is not None and estimated_variances is not None:
        if estimated_means is None:
            raise ValueError("estimated_means required when computing coverage")
        results['coverage_95'] = coverage_95(true_states, estimated_means, estimated_variances)
    
    return results


def format_metrics(metrics: dict, precision: int = 4) -> str:
    """Format metrics dictionary as human-readable string.
    
    Args:
        metrics: Dictionary from compute_all_metrics()
        precision: Number of decimal places for numerical values
        
    Returns:
        Formatted string
        
    Example:
        >>> metrics = {'pll_total': -125.4, 'pll_mean': -0.5016, 'rmse': 0.0845, 'coverage_95': 0.96}
        >>> print(format_metrics(metrics))
        PLL (total): -125.4000
        PLL (mean):  -0.5016
        RMSE:        0.0845
        Coverage:    96.0%
    """
    lines = []
    
    if 'pll_total' in metrics:
        lines.append(f"PLL (total): {metrics['pll_total']:.{precision}f}")
    
    if 'pll_mean' in metrics:
        lines.append(f"PLL (mean):  {metrics['pll_mean']:.{precision}f}")
    
    if 'rmse' in metrics:
        lines.append(f"RMSE:        {metrics['rmse']:.{precision}f}")
    
    if 'coverage_95' in metrics:
        lines.append(f"Coverage:    {metrics['coverage_95']*100:.1f}%")
    
    return '\n'.join(lines)


# ==============================================================================
# Validation helpers
# ==============================================================================

def validate_coverage_calibration(coverage: float, tolerance: float = 0.05) -> str:
    """Assess whether coverage indicates well-calibrated uncertainty estimates.
    
    Args:
        coverage: Coverage value from coverage_95()
        tolerance: Acceptable deviation from 0.95
        
    Returns:
        Assessment string: 'well-calibrated', 'overconfident', or 'underconfident'
        
    Example:
        >>> cov = 0.87
        >>> print(validate_coverage_calibration(cov))
        overconfident
    """
    target = 0.95
    
    if abs(coverage - target) <= tolerance:
        return 'well-calibrated'
    elif coverage < target:
        return 'overconfident'
    else:
        return 'underconfident'


if __name__ == '__main__':
    print("Testing metrics module...")
    print("=" * 60)
    
    # Test RMSE
    true = [0.0, 0.1, 0.2, 0.15, 0.1]
    est = [0.05, 0.12, 0.18, 0.17, 0.09]
    r = rmse(true, est)
    print(f"RMSE test: {r:.4f} (expected ~0.0276)")
    assert 0.02 < r < 0.04, f"RMSE test failed: {r}"
    
    # Test coverage
    means = [0.0, 0.1, 0.2, 0.15, 0.1]
    vars = [0.01] * 5
    cov = coverage_95(true, means, vars)
    print(f"Coverage test: {cov:.1%} (expected 100%)")
    assert cov == 1.0, f"Coverage test failed: {cov}"
    
    # Test PLL
    obs = [0.01, -0.02, 0.015, -0.005]
    lls = [-0.52, -0.48, -0.51, -0.49]
    pll = pll_sequence(obs, lls)
    print(f"PLL test: {pll:.2f} (expected -2.00)")
    assert abs(pll - (-2.0)) < 0.01, f"PLL test failed: {pll}"
    
    # Test mean PLL
    mpll = mean_pll(obs, lls)
    print(f"Mean PLL test: {mpll:.4f} (expected -0.5000)")
    assert abs(mpll - (-0.5)) < 0.01, f"Mean PLL test failed: {mpll}"
    
    # Test compute_all_metrics
    metrics = compute_all_metrics(
        observations=obs,
        log_likelihoods=lls,
        true_states=true[:4],
        estimated_means=est[:4],
        estimated_variances=vars[:4]
    )
    print("\nAll metrics:")
    print(format_metrics(metrics))
    
    # Test calibration
    cal = validate_coverage_calibration(0.96)
    print(f"\nCalibration (0.96): {cal}")
    assert cal == 'well-calibrated', f"Calibration test failed: {cal}"
    
    print("\n" + "=" * 60)
    print("✓ All self-tests passed!")
