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

    Args:
        spec: Model specification (currently only Spec.VOL supported).
        num_models: Number of models in ensemble (K). Typical: 20-150.
        num_particles: Number of particles per model (N). Typical: 250-500.
        lambda_: Exponential forgetting factor in (0, 1]. Typical: 0.99-0.999.
        beta: Softmax temperature (> 0). Typical: 0.5-1.0.
        epsilon: Anti-starvation weight floor in [0, 1). Typical: 0.02-0.10.
        num_threads: OpenMP thread count. 0=auto, >0=explicit.
        seed: Random seed for reproducibility. 0=random.

    Example::

        config = SvmixConfig(
            spec=Spec.VOL,
            num_models=50,
            num_particles=500,
            lambda_=0.995,
            beta=0.8,
            epsilon=0.05
        )
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
