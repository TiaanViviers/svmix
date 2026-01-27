"""
Configuration classes for svmix.

Defines SvmixConfig for filter configuration.
"""

from dataclasses import dataclass
from typing import Optional

from .types import Spec


@dataclass
class SvmixConfig:
    """Configuration for svmix filter creation.
    
    Attributes:
        spec: Model specification (VOL, DRIFT, VOL_DRIFT)
        num_models: Number of models in ensemble (K)
        num_particles: Number of particles per model (N)
        lambda_: Exponential forgetting factor (0 < lambda <= 1)
                 Recommended: 0.99-0.999 for minute-frequency data
        beta: Softmax temperature for model weighting (beta > 0)
              Recommended: 0.5-1.0 to prevent premature convergence
        epsilon: Anti-starvation mixing weight (0 <= epsilon < 1)
                 Each model gets at least epsilon/K weight
                 Recommended: 0.01-0.05
        num_threads: OpenMP thread count (0=auto, >0=explicit)
                     Only used if compiled with OpenMP support
        seed: Random seed for reproducibility (0=random)
    
    Example:
        >>> config = SvmixConfig(
        ...     spec=Spec.VOL,
        ...     num_models=50,
        ...     num_particles=1000,
        ...     lambda_=0.99,
        ...     beta=0.8,
        ...     epsilon=0.02
        ... )
    """
    spec: Spec
    num_models: int
    num_particles: int
    lambda_: float
    beta: float
    epsilon: float
    num_threads: int = 0
    seed: int = 0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_models <= 0:
            raise ValueError(f"num_models must be > 0, got {self.num_models}")
        
        if self.num_particles <= 0:
            raise ValueError(f"num_particles must be > 0, got {self.num_particles}")
        
        if not (0 < self.lambda_ <= 1):
            raise ValueError(f"lambda_ must be in (0, 1], got {self.lambda_}")
        
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        
        if not (0 <= self.epsilon < 1):
            raise ValueError(f"epsilon must be in [0, 1), got {self.epsilon}")
        
        if self.num_threads < 0:
            raise ValueError(f"num_threads must be >= 0, got {self.num_threads}")
        
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
    
    def __repr__(self):
        return (
            f"SvmixConfig(spec={self.spec.name}, K={self.num_models}, "
            f"N={self.num_particles}, λ={self.lambda_}, β={self.beta}, "
            f"ε={self.epsilon})"
        )
