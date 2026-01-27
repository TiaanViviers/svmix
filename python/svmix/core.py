"""
High-level Python API for svmix.

This is the main user-facing interface. It wraps the low-level ctypes
bindings with a clean, Pythonic API.
"""

from typing import List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from . import _native
from .config import SvmixConfig
from .params import SvParamsVol
from .types import Belief, Status, check_status, SvmixError


class Svmix:
    """High-level interface to svmix filter.
    
    This class manages a mixture of stochastic volatility models for
    Bayesian filtering of volatility from return observations.
    
    Memory management:
        Call free() explicitly when done, or rely on __del__ as fallback.
    
    Example:
        >>> from svmix import Svmix, SvmixConfig, SvParams, Spec
        >>> 
        >>> # Configure filter
        >>> config = SvmixConfig(
        ...     spec=Spec.VOL,
        ...     num_models=50,
        ...     num_particles=1000,
        ...     lambda_=0.99,
        ...     beta=0.8,
        ...     epsilon=0.02
        ... )
        >>> 
        >>> # Generate parameter grid
        >>> params = SvParams.linspace(
        ...     num_models=50,
        ...     phi=(0.90, 0.99),
        ...     sigma=0.2,
        ...     nu=10,
        ...     mu=-0.5
        ... )
        >>> 
        >>> # Create filter
        >>> svmix = Svmix(config, params)
        >>> 
        >>> # Process observations
        >>> for return_t in returns:
        ...     svmix.step(return_t)
        ...     belief = svmix.get_belief()
        ...     weights = svmix.get_weights()
        >>> 
        >>> # Cleanup
        >>> svmix.free()
    """
    
    def __init__(self, config: SvmixConfig, sv_params: List[SvParamsVol]):
        """Create svmix filter.
        
        Args:
            config: Filter configuration
            sv_params: List of SV parameters (one per model)
            
        Raises:
            ValueError: If num_models doesn't match len(sv_params)
            SvmixError: If creation fails
        """
        # Initialize state early to prevent __del__ errors
        self._handle = 0
        self._freed = True
        self._num_models = 0
        
        if len(sv_params) != config.num_models:
            raise ValueError(
                f"Config specifies {config.num_models} models but "
                f"{len(sv_params)} parameter sets provided"
            )
        
        # Convert to C structures
        ens_cfg = _native.CSvmixEnsembleCfg(
            lambda_=config.lambda_,
            beta=config.beta,
            epsilon=config.epsilon,
            num_threads=config.num_threads
        )
        
        c_config = _native.CSvmixCfg(
            num_models=config.num_models,
            num_particles=config.num_particles,
            spec=int(config.spec),
            ensemble=ens_cfg
        )
        
        c_params = [
            _native.CSvParams(
                mu_h=p.mu,      # Map Python names to C names
                phi_h=p.phi,
                sigma_h=p.sigma,
                nu=p.nu
            )
            for p in sv_params
        ]
        
        # Generate seeds (one per model, derived from config.seed)
        if config.seed == 0:
            import random
            seeds = [random.randint(1, 2**32-1) for _ in range(config.num_models)]
        else:
            # Deterministic seed generation
            seeds = [config.seed + i for i in range(config.num_models)]
        
        # Create C instance
        handle = _native.create(c_config, c_params, seeds)
        if not handle:  # NULL pointer returned
            raise SvmixError("Failed to create svmix (C function returned NULL)")
        
        self._handle = handle
        self._freed = False
        self._num_models = config.num_models
    
    def free(self):
        """Free C resources explicitly.
        
        After calling this, the instance cannot be used.
        Safe to call multiple times.
        """
        if not self._freed and self._handle is not None:
            _native.free(self._handle)
            self._handle = None
            self._freed = True
    
    def __del__(self):
        """Destructor - automatic cleanup fallback.
        
        Prefer explicit free() for deterministic resource management.
        """
        if not self._freed:
            try:
                self.free()
            except:
                # Library might be unloaded, ignore errors
                pass
    
    def _check_freed(self):
        """Raise error if instance has been freed."""
        if self._freed:
            raise ValueError(
                "Svmix instance has been freed and cannot be used. "
                "Create a new instance or load from checkpoint."
            )
    
    def step(self, observation: float):
        """Update filter with new observation.
        
        Args:
            observation: Return observation (y_t)
            
        Raises:
            SvmixError: If update fails
            
        Example:
            >>> svmix.step(0.01)  # 1% return
        """
        self._check_freed()
        
        status = _native.step(self._handle, float(observation))
        check_status(status, "Failed to step filter")
    
    def get_belief(self) -> Belief:
        """Get current belief state.
        
        Returns:
            Belief object with mean_h, var_h, mean_sigma, valid
            
        Raises:
            SvmixError: If retrieval fails
            
        Example:
            >>> belief = svmix.get_belief()
            >>> print(f"Volatility: {belief.mean_sigma:.4f}")
        """
        self._check_freed()
        
        status, c_belief = _native.get_belief(self._handle)
        check_status(status, "Failed to get belief")
        
        return Belief(
            mean_h=c_belief.mean_h,
            var_h=c_belief.var_h,
            mean_sigma=c_belief.mean_sigma,
            valid=bool(c_belief.valid)
        )
    
    def get_weights(self) -> 'np.ndarray | List[float]':
        """Get current model weights.
        
        Returns:
            numpy array if numpy available, otherwise list
            
        Raises:
            SvmixError: If retrieval fails
            
        Example:
            >>> weights = svmix.get_weights()
            >>> dominant_model = np.argmax(weights)
        """
        self._check_freed()
        
        status, weights = _native.get_weights(self._handle, self._num_models)
        check_status(status, "Failed to get weights")
        
        if HAS_NUMPY:
            return np.array(weights)
        return weights
    
    def save_checkpoint(self, filepath: str):
        """Save complete filter state to file.
        
        Saves configuration, parameters, particle states, and RNG state
        for exact resumption.
        
        Args:
            filepath: Path to checkpoint file (typically .svmix extension)
            
        Raises:
            SvmixFileIOError: If save fails
            
        Example:
            >>> svmix.save_checkpoint("state_t1000.svmix")
        """
        self._check_freed()
        
        status = _native.save_checkpoint(self._handle, filepath)
        check_status(status, f"Failed to save checkpoint to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'Svmix':
        """Load filter from checkpoint file.
        
        Creates a new Svmix instance from saved state.
        
        **IMPORTANT:** After loading, belief will be invalid until the next step().
        This is expected behavior - checkpoints are for continuing filtering,
        not for inspecting historical belief.
        
        Typical usage:
            >>> svmix = Svmix.load_checkpoint("state.svmix")
            >>> svmix.step(next_observation)  # Always step immediately!
            >>> belief = svmix.get_belief()   # Now valid
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            New Svmix instance with restored state
            
        Raises:
            SvmixFileIOError: If file cannot be read
            SvmixCheckpointCorruptError: If file is corrupted
            SvmixVersionMismatchError: If version incompatible
            
        Note:
            Model weights ARE fully restored and filtering continues correctly.
            Only the belief summary is unavailable until next observation.
        """
        handle, status = _native.load_checkpoint(filepath)
        if not handle:
            check_status(status, f"Failed to load checkpoint from {filepath}")
        
        # Create instance without going through __init__
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._freed = False
        instance._num_models = _native.get_num_models(handle)
        
        return instance
    
    @property
    def num_models(self) -> int:
        """Get number of models in ensemble."""
        return self._num_models
    
    def __repr__(self):
        if self._freed:
            return "Svmix(freed)"
        # Handle could be 0 (valid), so check _freed instead
        handle_str = hex(self._handle) if self._handle else "0x0"
        return f"Svmix(K={self._num_models}, handle={handle_str})"


def version() -> str:
    """Get svmix version string.
    
    Returns:
        Version string (e.g., "1.0.0")
        
    Example:
        >>> import svmix
        >>> print(svmix.version())
        1.0.0
    """
    return _native.version()
