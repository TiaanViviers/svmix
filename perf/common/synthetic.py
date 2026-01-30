"""
Synthetic data generation for svmix testing and benchmarking.

This module provides programmatic generation of synthetic stochastic volatility
data with ground truth states. Designed for accuracy testing and benchmarking.

Key Features:
    - Programmatic API (no CLI required)
    - Returns ground truth states for RMSE/coverage metrics
    - Reproducible with seed control
    - Standard test scenarios included
    - Lightweight (numpy only)

Usage:
    >>> from perf.common.synthetic import SyntheticDataGenerator
    >>> 
    >>> gen = SyntheticDataGenerator()
    >>> data = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=1000, seed=42)
    >>> 
    >>> observations = data['observations']
    >>> true_states = data['states']
    >>> true_volatility = data['volatility']
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SVParams:
    """Stochastic volatility model parameters."""
    phi: float      # Persistence (0 < phi < 1)
    sigma: float    # Volatility of volatility (> 0)
    nu: float       # Degrees of freedom for Student-t (> 2)
    mu: float       # Long-run mean of log-volatility
    
    def validate(self) -> None:
        """Validate parameters are in valid ranges."""
        if not 0 < self.phi < 1:
            raise ValueError(f"phi must be in (0, 1), got {self.phi}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.nu <= 2:
            raise ValueError(f"nu must be > 2, got {self.nu}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SyntheticDataGenerator:
    """Generate synthetic stochastic volatility data with ground truth.
    
    This generator implements the standard SV model:
        h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t,  eta_t ~ t(nu)
        y_t = exp(h_t / 2) * epsilon_t,  epsilon_t ~ t(nu)
    
    where h_t is log-volatility and y_t is the observed return.
    
    Example:
        >>> gen = SyntheticDataGenerator()
        >>> 
        >>> # Single scenario
        >>> data = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=1000)
        >>> print(f"Generated {len(data['observations'])} observations")
        >>> 
        >>> # Multiple scenarios
        >>> scenarios = gen.generate_batch([
        ...     {'phi': 0.90, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
        ...     {'phi': 0.95, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
        ...     {'phi': 0.98, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
        ... ])
        >>> 
        >>> # Standard test scenarios
        >>> baseline = gen.baseline()
        >>> high_persistence = gen.high_persistence()
    """
    
    def __init__(self, base_seed: int = 42):
        """Initialize generator.
        
        Args:
            base_seed: Base random seed for reproducibility
        """
        self.base_seed = base_seed
    
    def generate(
        self,
        phi: float,
        sigma: float,
        nu: float,
        mu: float,
        T: int,
        seed: Optional[int] = None,
        return_volatility: bool = True
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic SV data with ground truth.
        
        Args:
            phi: Persistence parameter (0 < phi < 1)
            sigma: Volatility of volatility (> 0)
            nu: Degrees of freedom for Student-t (> 2)
            mu: Long-run mean of log-volatility
            T: Number of time steps
            seed: Random seed (uses base_seed if None)
            return_volatility: If True, include exp(h/2) in output
            
        Returns:
            Dictionary with keys:
                - 'observations': np.ndarray of shape (T,) - observed returns y_t
                - 'states': np.ndarray of shape (T,) - true log-volatility h_t
                - 'volatility': np.ndarray of shape (T,) - true volatility exp(h_t/2)
                - 'params': dict - parameter values used
                - 'T': int - sequence length
                - 'seed': int - seed used
                
        Raises:
            ValueError: If parameters are invalid
            
        Example:
            >>> gen = SyntheticDataGenerator()
            >>> data = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=1000)
            >>> 
            >>> # Access components
            >>> y = data['observations']  # What the filter sees
            >>> h_true = data['states']   # Ground truth for metrics
            >>> vol_true = data['volatility']  # exp(h/2)
            >>> params = data['params']    # Parameters used
        """
        # Validate parameters
        params = SVParams(phi=phi, sigma=sigma, nu=nu, mu=mu)
        params.validate()
        
        # Set seed
        if seed is None:
            seed = self.base_seed
        np.random.seed(seed)
        
        # Generate log-volatility process h_t
        h = np.zeros(T)
        h[0] = mu  # Initialize at long-run mean
        
        for t in range(1, T):
            # h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t
            eta_t = np.random.standard_t(nu)
            h[t] = mu + phi * (h[t-1] - mu) + sigma * eta_t
        
        # Compute volatility
        volatility = np.exp(h / 2.0)
        
        # Generate observations y_t
        observations = np.zeros(T)
        for t in range(T):
            # y_t = exp(h_t / 2) * epsilon_t
            epsilon_t = np.random.standard_t(nu)
            observations[t] = volatility[t] * epsilon_t
        
        # Build result dictionary
        result = {
            'observations': observations,
            'states': h,
            'params': params.to_dict(),
            'T': T,
            'seed': seed
        }
        
        if return_volatility:
            result['volatility'] = volatility
        
        return result
    
    def generate_batch(
        self,
        scenarios: List[Dict],
        base_seed: Optional[int] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Generate multiple scenarios from parameter specifications.
        
        Args:
            scenarios: List of parameter dictionaries, each containing
                      phi, sigma, nu, mu, T, and optionally seed
            base_seed: Base seed for reproducibility (incremented per scenario)
            
        Returns:
            List of data dictionaries (same format as generate())
            
        Example:
            >>> gen = SyntheticDataGenerator()
            >>> scenarios = [
            ...     {'phi': 0.90, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
            ...     {'phi': 0.95, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
            ...     {'phi': 0.98, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 1000},
            ... ]
            >>> batch = gen.generate_batch(scenarios)
            >>> print(f"Generated {len(batch)} scenarios")
        """
        if base_seed is None:
            base_seed = self.base_seed
        
        results = []
        for i, scenario in enumerate(scenarios):
            # Use per-scenario seed if not specified
            if 'seed' not in scenario:
                scenario = dict(scenario)  # Copy to avoid mutating input
                scenario['seed'] = base_seed + i
            
            data = self.generate(**scenario)
            results.append(data)
        
        return results
    
    # =========================================================================
    # Standard Test Scenarios
    # =========================================================================
    
    @classmethod
    def get_standard_params(cls, scenario: str) -> Dict:
        """Get parameter dict for a standard test scenario.
        
        Args:
            scenario: One of 'baseline', 'high_persistence', 'high_volatility',
                     'heavy_tails', 'short', 'long'
                     
        Returns:
            Dictionary with phi, sigma, nu, mu, T
            
        Raises:
            ValueError: If scenario name unknown
        """
        scenarios = {
            'baseline': {
                'phi': 0.95,
                'sigma': 0.20,
                'nu': 10.0,
                'mu': -0.5,
                'T': 1000,
                'description': 'Typical SV parameters'
            },
            'high_persistence': {
                'phi': 0.98,
                'sigma': 0.15,
                'nu': 10.0,
                'mu': -0.5,
                'T': 1000,
                'description': 'Slow-moving volatility'
            },
            'high_volatility': {
                'phi': 0.95,
                'sigma': 0.35,
                'nu': 10.0,
                'mu': -0.5,
                'T': 1000,
                'description': 'Rapid volatility changes'
            },
            'heavy_tails': {
                'phi': 0.95,
                'sigma': 0.20,
                'nu': 5.0,
                'mu': -0.5,
                'T': 1000,
                'description': 'Heavy-tailed shocks (robust test)'
            },
            'short': {
                'phi': 0.95,
                'sigma': 0.20,
                'nu': 10.0,
                'mu': -0.5,
                'T': 250,
                'description': 'Short sequence (initialization test)'
            },
            'long': {
                'phi': 0.95,
                'sigma': 0.20,
                'nu': 10.0,
                'mu': -0.5,
                'T': 5000,
                'description': 'Long sequence (stability test)'
            },
        }
        
        if scenario not in scenarios:
            valid = ', '.join(scenarios.keys())
            raise ValueError(f"Unknown scenario '{scenario}'. Valid: {valid}")
        
        return scenarios[scenario]
    
    def baseline(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate baseline scenario (typical SV parameters).
        
        Parameters: phi=0.95, sigma=0.20, nu=10, mu=-0.5, T=1000
        """
        params = self.get_standard_params('baseline')
        # Remove description field before passing to generate()
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    def high_persistence(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate high persistence scenario (slow-moving volatility).
        
        Parameters: phi=0.98, sigma=0.15, nu=10, mu=-0.5, T=1000
        """
        params = self.get_standard_params('high_persistence')
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    def high_volatility(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate high volatility scenario (rapid changes).
        
        Parameters: phi=0.95, sigma=0.35, nu=10, mu=-0.5, T=1000
        """
        params = self.get_standard_params('high_volatility')
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    def heavy_tails(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate heavy tails scenario (robust test).
        
        Parameters: phi=0.95, sigma=0.20, nu=5, mu=-0.5, T=1000
        """
        params = self.get_standard_params('heavy_tails')
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    def short_sequence(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate short sequence scenario (initialization test).
        
        Parameters: phi=0.95, sigma=0.20, nu=10, mu=-0.5, T=250
        """
        params = self.get_standard_params('short')
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    def long_sequence(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate long sequence scenario (stability test).
        
        Parameters: phi=0.95, sigma=0.20, nu=10, mu=-0.5, T=5000
        """
        params = self.get_standard_params('long')
        params = {k: v for k, v in params.items() if k != 'description'}
        return self.generate(**params, seed=seed)
    
    @classmethod
    def list_standard_scenarios(cls) -> List[str]:
        """List all available standard scenario names."""
        return [
            'baseline',
            'high_persistence', 
            'high_volatility',
            'heavy_tails',
            'short',
            'long'
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_quick(
    T: int = 1000,
    seed: int = 42,
    phi: float = 0.95,
    sigma: float = 0.20,
    nu: float = 10.0,
    mu: float = -0.5
) -> np.ndarray:
    """Quick helper to generate observations only (for benchmarking).
    
    Returns just the observation array, not ground truth.
    Useful when you only care about performance, not accuracy.
    
    Args:
        T: Number of observations
        seed: Random seed
        phi, sigma, nu, mu: SV parameters (defaults are typical)
        
    Returns:
        np.ndarray of observations (shape: (T,))
        
    Example:
        >>> observations = generate_quick(T=1000, seed=42)
        >>> # Use in benchmark
        >>> svmix = Svmix(config, params)
        >>> for obs in observations:
        ...     svmix.step(obs)
    """
    gen = SyntheticDataGenerator(base_seed=seed)
    data = gen.generate(phi=phi, sigma=sigma, nu=nu, mu=mu, T=T, seed=seed)
    return data['observations']


def generate_with_truth(
    phi: float = 0.95,
    sigma: float = 0.20,
    nu: float = 10.0,
    mu: float = -0.5,
    T: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate synthetic data with ground truth (convenience wrapper).
    
    Args:
        phi, sigma, nu, mu: SV parameters
        T: Number of time steps
        seed: Random seed
        
    Returns:
        Tuple of (observations, true_states, params_dict)
        
    Example:
        >>> obs, states, params = generate_with_truth(phi=0.95, sigma=0.2, T=1000)
        >>> # Run filter...
        >>> rmse_val = rmse(states, estimated_states)
    """
    gen = SyntheticDataGenerator(base_seed=seed)
    data = gen.generate(phi=phi, sigma=sigma, nu=nu, mu=mu, T=T, seed=seed)
    return data['observations'], data['states'], data['params']


if __name__ == '__main__':
    print("Testing synthetic data generator...")
    print("=" * 70)
    
    # Test basic generation
    gen = SyntheticDataGenerator()
    data = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=100, seed=42)
    
    print(f"✓ Basic generation: T={len(data['observations'])}")
    assert len(data['observations']) == 100
    assert len(data['states']) == 100
    assert len(data['volatility']) == 100
    print(f"  Mean return: {data['observations'].mean():.6f}")
    print(f"  Std return: {data['observations'].std():.6f}")
    print(f"  Mean log-vol: {data['states'].mean():.6f}")
    
    # Test reproducibility
    data2 = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=100, seed=42)
    assert np.allclose(data['observations'], data2['observations'])
    print(f"✓ Reproducibility: Same seed → same data")
    
    # Test different seed
    data3 = gen.generate(phi=0.95, sigma=0.2, nu=10, mu=-0.5, T=100, seed=999)
    assert not np.allclose(data['observations'], data3['observations'])
    print(f"✓ Different seed → different data")
    
    # Test standard scenarios
    print(f"\n✓ Standard scenarios:")
    scenario_methods = {
        'baseline': gen.baseline,
        'high_persistence': gen.high_persistence,
        'high_volatility': gen.high_volatility,
        'heavy_tails': gen.heavy_tails,
        'short': gen.short_sequence,
        'long': gen.long_sequence,
    }
    for name in gen.list_standard_scenarios():
        scenario_data = scenario_methods[name]()
        print(f"  - {name}: T={scenario_data['T']}, phi={scenario_data['params']['phi']}")
    
    # Test batch generation
    scenarios = [
        {'phi': 0.90, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 50},
        {'phi': 0.95, 'sigma': 0.2, 'nu': 10, 'mu': -0.5, 'T': 50},
    ]
    batch = gen.generate_batch(scenarios)
    assert len(batch) == 2
    print(f"\n✓ Batch generation: {len(batch)} scenarios")
    
    # Test convenience functions
    obs_quick = generate_quick(T=50, seed=42)
    assert len(obs_quick) == 50
    print(f"✓ generate_quick(): T={len(obs_quick)}")
    
    obs, states, params = generate_with_truth(T=50, seed=42)
    assert len(obs) == len(states) == 50
    print(f"✓ generate_with_truth(): obs={len(obs)}, states={len(states)}")
    
    # Test parameter validation
    try:
        gen.generate(phi=1.5, sigma=0.2, nu=10, mu=-0.5, T=10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Parameter validation: {e}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
