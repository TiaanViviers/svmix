"""
Data types for svmix Python interface.

This module defines the data structures returned by svmix operations.
Kept separate for easy extension and documentation.
"""

from dataclasses import dataclass
from enum import IntEnum


class Spec(IntEnum):
    """Model specification types.
    
    Determines which SV model variant is used and which parameters are required.
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
    ERR_INTERNAL = -4
    ERR_FILE_IO = -5
    ERR_CHECKPOINT_CORRUPT = -6
    ERR_VERSION_MISMATCH = -7


@dataclass
class Belief:
    """Belief state from the filter.
    
    Represents the filtered estimate of the latent volatility state.
    
    Attributes:
        mean_h: Mean log-volatility E[h_t | y_{1:t}]
        var_h: Variance of log-volatility Var[h_t | y_{1:t}]
        mean_sigma: Mean volatility E[exp(h_t/2) | y_{1:t}]
        valid: Whether belief is valid (False before first observation)
    """
    mean_h: float
    var_h: float
    mean_sigma: float
    valid: bool
    
    def __repr__(self):
        if not self.valid:
            return "Belief(valid=False)"
        return (f"Belief(mean_h={self.mean_h:.4f}, var_h={self.var_h:.4f}, "
                f"mean_sigma={self.mean_sigma:.4f})")


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


# Map status codes to exceptions
_STATUS_TO_EXCEPTION = {
    Status.ERR_NULL_POINTER: SvmixNullPointerError,
    Status.ERR_INVALID_PARAM: SvmixInvalidParamError,
    Status.ERR_ALLOC_FAILED: SvmixAllocError,
    Status.ERR_INTERNAL: SvmixInternalError,
    Status.ERR_FILE_IO: SvmixFileIOError,
    Status.ERR_CHECKPOINT_CORRUPT: SvmixCheckpointCorruptError,
    Status.ERR_VERSION_MISMATCH: SvmixVersionMismatchError,
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
