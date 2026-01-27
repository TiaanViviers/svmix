"""
Parameter generation utilities for svmix.

Provides convenient ways to create parameter grids for model ensembles.
"""

from dataclasses import dataclass
from typing import List, Union, Tuple
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("numpy not found, linspace/grid functions will be limited")


def _to_array(val: Union[float, Tuple[float, float]], n: int) -> List[float]:
    """Convert value or range to array.
    
    Args:
        val: Single value or (min, max) tuple
        n: Number of points
        
    Returns:
        List of n values
    """
    if isinstance(val, tuple):
        if not HAS_NUMPY:
            # Fallback without numpy
            if n == 1:
                return [(val[0] + val[1]) / 2]
            step = (val[1] - val[0]) / (n - 1)
            return [val[0] + i * step for i in range(n)]
        return np.linspace(val[0], val[1], n).tolist()
    else:
        return [val] * n


def _to_list(val: Union[float, List[float]]) -> List[float]:
    """Convert value or list to list."""
    if isinstance(val, list):
        return val
    else:
        return [val]


@dataclass
class SvParamsVol:
    """Parameters for SPEC_VOL (V1: Stochastic Volatility).
    
    Attributes:
        phi: Mean reversion rate (0 < phi < 1)
             Higher = slower mean reversion
        sigma: Volatility of volatility (sigma > 0)
               Controls volatility clustering strength
        nu: Student-t degrees of freedom (nu > 2)
            Lower = fatter tails
        mu: Long-run mean of log-volatility (any real)
            Typically negative (e.g., -0.5)
    
    Example:
        >>> params = SvParamsVol(phi=0.97, sigma=0.2, nu=10, mu=-0.5)
    """
    phi: float
    sigma: float
    nu: float
    mu: float
    
    def validate(self):
        """Validate parameter constraints.
        
        Raises:
            ValueError: If parameters violate constraints
        """
        if not (0 < self.phi < 1):
            raise ValueError(f"phi must be in (0, 1), got {self.phi}")
        
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        
        if self.nu <= 2:
            raise ValueError(
                f"nu must be > 2 for finite variance, got {self.nu}"
            )
        
        # mu can be any value, no constraint
    
    @classmethod
    def linspace(cls,
                 num_models: int,
                 phi: Union[float, Tuple[float, float]],
                 sigma: Union[float, Tuple[float, float]],
                 nu: Union[float, Tuple[float, float]],
                 mu: Union[float, Tuple[float, float]]) -> List['SvParamsVol']:
        """Generate parameters with evenly spaced values.
        
        For each parameter:
        - If float: use same value for all models
        - If (min, max) tuple: linearly space num_models points
        
        Args:
            num_models: Number of parameter sets to generate
            phi: Single value or (min, max) range
            sigma: Single value or (min, max) range
            nu: Single value or (min, max) range
            mu: Single value or (min, max) range
            
        Returns:
            List of SvParamsVol instances
            
        Example:
            >>> # 50 models varying only phi
            >>> params = SvParamsVol.linspace(
            ...     num_models=50,
            ...     phi=(0.90, 0.99),
            ...     sigma=0.2,
            ...     nu=10,
            ...     mu=-0.5
            ... )
            
            >>> # 20 models varying phi and sigma
            >>> params = SvParamsVol.linspace(
            ...     num_models=20,
            ...     phi=(0.90, 0.99),
            ...     sigma=(0.1, 0.3),
            ...     nu=10,
            ...     mu=-0.5
            ... )
        """
        if num_models <= 0:
            raise ValueError(f"num_models must be > 0, got {num_models}")
        
        # Generate arrays for each parameter
        phi_vals = _to_array(phi, num_models)
        sigma_vals = _to_array(sigma, num_models)
        nu_vals = _to_array(nu, num_models)
        mu_vals = _to_array(mu, num_models)
        
        # Create parameter instances
        params = [
            cls(phi=phi_vals[i], sigma=sigma_vals[i],
                nu=nu_vals[i], mu=mu_vals[i])
            for i in range(num_models)
        ]
        
        # Validate all
        for i, p in enumerate(params):
            try:
                p.validate()
            except ValueError as e:
                raise ValueError(f"Model {i}: {e}")
        
        return params
    
    @classmethod
    def grid(cls,
             phi: Union[float, List[float]],
             sigma: Union[float, List[float]],
             nu: Union[float, List[float]],
             mu: Union[float, List[float]]) -> List['SvParamsVol']:
        """Generate full Cartesian product grid of parameters.
        
        Creates all combinations of the provided parameter values.
        Number of models = len(phi) × len(sigma) × len(nu) × len(mu)
        
        Args:
            phi: Single value or list of values
            sigma: Single value or list of values
            nu: Single value or list of values
            mu: Single value or list of values
            
        Returns:
            List of all parameter combinations
            
        Example:
            >>> # 3×4×5×1 = 60 models
            >>> params = SvParamsVol.grid(
            ...     phi=[0.95, 0.97, 0.99],
            ...     sigma=[0.1, 0.2, 0.3, 0.4],
            ...     nu=[5, 10, 15, 20, 25],
            ...     mu=-0.5
            ... )
        """
        phi_list = _to_list(phi)
        sigma_list = _to_list(sigma)
        nu_list = _to_list(nu)
        mu_list = _to_list(mu)
        
        params = []
        for p in phi_list:
            for s in sigma_list:
                for n in nu_list:
                    for m in mu_list:
                        param = cls(phi=p, sigma=s, nu=n, mu=m)
                        param.validate()
                        params.append(param)
        
        return params
    
    def __repr__(self):
        return (f"SvParams(φ={self.phi:.3f}, σ={self.sigma:.3f}, "
                f"ν={self.nu:.1f}, μ={self.mu:.3f})")


# Alias for backward compatibility and shorter name
SvParams = SvParamsVol
