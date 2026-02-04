"""
Data types for svmix Python interface.

This module defines the data structures returned by svmix operations.
Kept separate for easy extension and documentation.
"""

from dataclasses import dataclass
from enum import IntEnum


class Spec(IntEnum):
    """Model specification types.

    Specifies which stochastic volatility model variant to use.

    Attributes:
        VOL: Standard SV model with AR(1) log-volatility and Student-t
            observations. This is the only currently supported specification.

    Note:
        DRIFT and VOL_DRIFT are reserved for future extensions.

    Example::

        config = SvmixConfig(spec=Spec.VOL, ...)
    """
    VOL = 1        # V1: Stochastic volatility only
    DRIFT = 2      # V2: + Drift in returns (future)
    VOL_DRIFT = 3  # V3: + Stochastic volatility of drift (future)


class Status(IntEnum):
    """Status codes returned by svmix operations.

    Matches svmix_status_t enum in C.
    """
    OK = 0
    ERR_NULL_POINTER = -1
    ERR_INVALID_PARAM = -2
    ERR_ALLOC_FAILED = -3
    ERR_FILE_IO = -4
    ERR_CHECKPOINT_CORRUPT = -5
    ERR_VERSION_MISMATCH = -6
    ERR_INTERNAL = -99


@dataclass
class Belief:
    """Filtered volatility estimate and posterior state.

    Contains the model-averaged posterior distribution over the current
    volatility state after observing all data up to time t.

    Attributes:
        mean_h: Posterior mean of log-volatility h_t (E[h_t | data]).
            Volatility = exp(h_t / 2). For most uses, prefer the vol property.

        var_h: Posterior variance of log-volatility. Indicates uncertainty.
            High values (> 0.5) suggest insufficient data, regime changes, or
            poor parameter specification.

        mean_sigma: Mean volatility exp(mean_h / 2). Equivalent to vol property.

        valid: Whether belief is valid. False before first observation or if
            filter encountered errors. Always check this before using.

    Properties:
        vol: Current volatility estimate (standard deviation of returns).
            **Primary output for applications**. Computed as exp(mean_h / 2).

        log_vol: Log-volatility estimate (mean_h / 2). For diagnostics.

    Example::

        belief = svmix.get_belief()
        if belief.valid:
            # Primary volatility estimate
            current_vol = belief.vol
            
            # Annualize if using daily returns
            annual_vol = current_vol * np.sqrt(252)
            
            # Check uncertainty
            if belief.var_h > 0.5:
                print(\"High uncertainty\")
            
            # 95% prediction interval for next return
            lower = -1.96 * current_vol
            upper = +1.96 * current_vol

    See Also:
        To inspect which parameters the data supports, use
        :meth:`~svmix.core.Svmix.get_weights` to examine model probabilities.
    """
    mean_h: float
    var_h: float
    mean_sigma: float
    valid: bool

    @property
    def vol(self) -> float:
        """Annualized volatility in percentage terms (convenience property).
        
        Returns volatility as annual percentage (e.g., 18.2 = 18.2%).
        For more control over units, use :meth:`get_vol` instead.
        
        Equivalent to: ``get_vol(annualize=True, as_percentage=True)``
        """
        return self.get_vol(annualize=True, as_percentage=True)
    
    def get_vol(self, annualize: bool = True, as_percentage: bool = True) -> float:
        """Get volatility estimate with configurable units.
        
        Computes exp(mean_h/2) with optional annualization and percentage conversion.
        
        Args:
            annualize: If True, scale daily vol to annual by multiplying by sqrt(252).
                      If False, return daily volatility in same units as returns.
            as_percentage: If True, multiply by 100 to get percentage (e.g., 18.2%).
                          If False, return as decimal (e.g., 0.182).
        
        Returns:
            Volatility in requested units. Returns 0.0 if belief invalid.
        
        Examples:
            >>> belief = svmix.get_belief()
            
            >>> # Annualized percentage (standard finance) - DEFAULT
            >>> vol = belief.get_vol(annualize=True, as_percentage=True)
            >>> # e.g., 18.2 means "18.2% annual volatility"
            
            >>> # Daily percentage (matches your return scale)
            >>> vol = belief.get_vol(annualize=False, as_percentage=True)
            >>> # e.g., 1.15 means "1.15% daily moves"
            
            >>> # Daily decimal (for position sizing calculations)
            >>> vol = belief.get_vol(annualize=False, as_percentage=False)
            >>> # e.g., 0.0115 means "1.15% daily moves"
            >>> position_size = capital * (target_risk / vol)
            
            >>> # Annualized decimal (for Sharpe ratio, etc.)
            >>> vol = belief.get_vol(annualize=True, as_percentage=False)
            >>> # e.g., 0.182 means "18.2% annual volatility"
            >>> sharpe = annual_return / vol
        
        Note:
            - Daily vol is exp(mean_h/2), matches log-return scale
            - Annualization multiplies by sqrt(252) trading days
            - Percentage multiplies by 100 for readability
        """
        import math
        if not self.valid:
            return 0.0
        
        # Base: daily volatility (exp(mean_h/2))
        daily_vol = math.exp(self.mean_h / 2.0)
        
        # Optional annualization
        vol = daily_vol * math.sqrt(252) if annualize else daily_vol
        
        # Optional percentage conversion
        vol = vol * 100 if as_percentage else vol
        
        return vol

    @property
    def log_vol(self) -> float:
        """Log-volatility estimate (mean_h/2)."""
        return self.mean_h / 2.0 if self.valid else 0.0

    def __repr__(self):
        if not self.valid:
            return "Belief(valid=False)"
        return (f"Belief(vol={self.vol:.4f}, mean_h={self.mean_h:.4f}, "
                f"var_h={self.var_h:.4f})")


class SvmixError(Exception):
    """Base exception for svmix errors."""
    pass


class SvmixNullPointerError(SvmixError):
    """Raised when a null pointer is passed to C function."""
    pass


class SvmixInvalidParamError(SvmixError):
    """Raised when invalid parameters are provided."""
    pass


class SvmixAllocError(SvmixError):
    """Raised when memory allocation fails."""
    pass


class SvmixInternalError(SvmixError):
    """Raised when an internal error occurs."""
    pass


class SvmixFileIOError(SvmixError):
    """Raised when file I/O fails."""
    pass


class SvmixCheckpointCorruptError(SvmixError):
    """Raised when checkpoint file is corrupted."""
    pass


class SvmixVersionMismatchError(SvmixError):
    """Raised when checkpoint version doesn't match."""
    pass


_STATUS_TO_EXCEPTION = {
    Status.ERR_NULL_POINTER: SvmixNullPointerError,
    Status.ERR_INVALID_PARAM: SvmixInvalidParamError,
    Status.ERR_ALLOC_FAILED: SvmixAllocError,
    Status.ERR_FILE_IO: SvmixFileIOError,
    Status.ERR_CHECKPOINT_CORRUPT: SvmixCheckpointCorruptError,
    Status.ERR_VERSION_MISMATCH: SvmixVersionMismatchError,
    Status.ERR_INTERNAL: SvmixInternalError,
}


def check_status(status_code: int, message: str = "Operation failed") -> None:
    """Check status code and raise appropriate exception if error.
    
    Args:
        status_code: Status code from C function
        message: Error message prefix
        
    Raises:
        SvmixError: Appropriate subclass based on status code
    """
    if status_code == Status.OK:
        return
    
    exception_cls = _STATUS_TO_EXCEPTION.get(status_code, SvmixError)
    raise exception_cls(f"{message}: status={status_code}")
