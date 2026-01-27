"""
svmix - Stochastic Volatility Model Mixture for Bayesian Filtering

A high-performance C library with Python bindings for real-time volatility
filtering using mixtures of stochastic volatility models.

Example:
    >>> from svmix import Svmix, SvmixConfig, SvParams, Spec
    >>> 
    >>> config = SvmixConfig(
    ...     spec=Spec.VOL,
    ...     num_models=50,
    ...     num_particles=1000,
    ...     lambda_=0.99,
    ...     beta=0.8,
    ...     epsilon=0.02
    ... )
    >>> 
    >>> params = SvParams.linspace(
    ...     num_models=50,
    ...     phi=(0.90, 0.99),
    ...     sigma=0.2,
    ...     nu=10,
    ...     mu=-0.5
    ... )
    >>> 
    >>> svmix = Svmix(config, params)
    >>> 
    >>> for return_t in returns:
    ...     svmix.step(return_t)
    ...     belief = svmix.get_belief()
    ...     weights = svmix.get_weights()
    >>> 
    >>> svmix.free()
"""

__version__ = "1.0.0"
__author__ = "Tiaan Viviers"

# Import public API
from .core import Svmix, version
from .config import SvmixConfig
from .params import SvParams, SvParamsVol
from .types import (
    Belief,
    Spec,
    Status,
    SvmixError,
    SvmixNullPointerError,
    SvmixInvalidParamError,
    SvmixAllocError,
    SvmixInternalError,
    SvmixFileIOError,
    SvmixCheckpointCorruptError,
    SvmixVersionMismatchError,
)

# Public API
__all__ = [
    # Main class
    "Svmix",
    
    # Configuration
    "SvmixConfig",
    
    # Parameters
    "SvParams",
    "SvParamsVol",
    
    # Types
    "Belief",
    "Spec",
    "Status",
    
    # Exceptions
    "SvmixError",
    "SvmixNullPointerError",
    "SvmixInvalidParamError",
    "SvmixAllocError",
    "SvmixInternalError",
    "SvmixFileIOError",
    "SvmixCheckpointCorruptError",
    "SvmixVersionMismatchError",
    
    # Utilities
    "version",
]
