"""
Core svmix filter implementation.

Provides the main :class:`Svmix` class for Bayesian volatility filtering.
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
    """Ensemble stochastic volatility filter.

    Maintains K independent SV models with particle filters, combining their
    predictions via Bayesian model averaging.

    Args:
        config: Filter configuration (:class:`SvmixConfig`).
        sv_params: List of K parameter sets (:class:`SvParams`).

    Example::

        config = SvmixConfig(
            spec=Spec.VOL, num_models=50, num_particles=500,
            lambda_=0.995, beta=0.8, epsilon=0.05
        )
        params = SvParams.linspace(50, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)

        svmix = Svmix(config, params)
        for r in returns:
            svmix.step(r)
            vol = svmix.get_belief().mean_sigma
        svmix.free()

    Note:
        Call :meth:`free` explicitly when done, or use as context manager.
    """
    
    def __init__(self, config: SvmixConfig, sv_params: List[SvParamsVol]):
        """Create svmix filter instance.

        Raises:
            ValueError: If len(sv_params) != config.num_models.
            SvmixError: If C library initialization fails.
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
        """Release C resources. Safe to call multiple times."""
        if not self._freed and self._handle is not None:
            _native.free(self._handle)
            self._handle = None
            self._freed = True
    
    def __del__(self):
        """Destructor. Prefer explicit :meth:`free` for deterministic cleanup."""
        if not self._freed:
            try:
                self.free()
            except:
                # Library might be unloaded, ignore errors
                pass
    
    def _check_freed(self):
        """Raise if instance has been freed."""
        if self._freed:
            raise ValueError(
                "Svmix instance has been freed and cannot be used. "
                "Create a new instance or load from checkpoint."
            )
    
    def step(self, observation: float):
        """Process one observation through the filter.

        Args:
            observation: Return value (typically log return).

        Raises:
            SvmixError: If the update fails.
        """
        self._check_freed()
        
        status = _native.step(self._handle, float(observation))
        check_status(status, "Failed to step filter")
    
    def get_belief(self) -> Belief:
        """Get current volatility belief state.

        Returns:
            Belief with mean_h, var_h, mean_sigma, and valid flag.

        Raises:
            SvmixError: If retrieval fails.
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
            Array of K weights summing to 1. NumPy array if available.

        Raises:
            SvmixError: If retrieval fails.
        """
        self._check_freed()
        
        status, weights = _native.get_weights(self._handle, self._num_models)
        check_status(status, "Failed to get weights")
        
        if HAS_NUMPY:
            return np.array(weights)
        return weights
    
    def get_last_log_likelihood(self) -> float:
        """Get predictive log-likelihood from the last step.

        Returns the one-step-ahead log p(y_t | y_{1:t-1}) from the mixture.
        Not affected by the forgetting parameter lambda.

        Returns:
            Log-likelihood value, or -inf if no observations processed.
        """
        self._check_freed()
        return _native.get_last_log_likelihood(self._handle)
    
    def save_checkpoint(self, filepath: str):
        """Save complete filter state to file.

        Serializes configuration, particle states, and RNG state for
        deterministic resumption.

        Args:
            filepath: Destination path (recommended extension: .svmix).

        Raises:
            SvmixFileIOError: If write fails.
        """
        self._check_freed()
        
        status = _native.save_checkpoint(self._handle, filepath)
        check_status(status, f"Failed to save checkpoint to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'Svmix':
        """Load filter state from checkpoint file.

        Creates a new Svmix instance with restored state. Call :meth:`step`
        immediately after loading to resume filtering.

        Args:
            filepath: Path to checkpoint file.

        Returns:
            Restored Svmix instance.

        Raises:
            SvmixFileIOError: If file cannot be read.
            SvmixCheckpointCorruptError: If file is corrupted.
            SvmixVersionMismatchError: If checkpoint version is incompatible.
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
    
    @property
    def timestep(self) -> int:
        """Number of observations processed (step calls)."""
        self._check_freed()
        return _native.get_timestep(self._handle)
    
    @property
    def effective_num_models(self) -> float:
        """Effective number of active models (inverse Simpson index).

        Measures ensemble diversity: 1 = collapsed to single model,
        K = uniform weights. Values below 2 may indicate convergence issues.
        """
        self._check_freed()
        
        weights = self.get_weights()
        
        if HAS_NUMPY:
            return float(1.0 / np.sum(weights ** 2))
        else:
            return 1.0 / sum(w**2 for w in weights)
    
    def __repr__(self):
        if self._freed:
            return "Svmix(freed)"
        # Handle could be 0 (valid), so check _freed instead
        handle_str = hex(self._handle) if self._handle else "0x0"
        return f"Svmix(K={self._num_models}, handle={handle_str})"


def version() -> str:
    """Get svmix library version string."""
    return _native.version()
