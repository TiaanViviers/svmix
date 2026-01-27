"""
Low-level ctypes bindings to libsvmix.so.

This module provides the raw C interface. Users should not use this directly;
use the high-level API in core.py instead.

Design: Isolated, easy to replace with Cython/pybind11 if needed.
"""

import ctypes
import os
import platform
from pathlib import Path
from typing import Optional

# Version info
SVMIX_VERSION_MAJOR = 1
SVMIX_VERSION_MINOR = 0
SVMIX_VERSION_PATCH = 0


def _get_lib_name() -> str:
    """Get library name for current platform."""
    system = platform.system()
    if system == "Linux":
        return "libsvmix.so"
    elif system == "Darwin":  # macOS
        return "libsvmix.dylib"
    elif system == "Windows":
        return "libsvmix.dll"
    else:
        raise OSError(f"Unsupported platform: {system}")


def _find_library() -> str:
    """Find libsvmix library.
    
    Search order:
    1. Bundled with Python package (python/svmix/lib/)
    2. Environment variable SVMIX_LIB_PATH
    3. System library path
    
    Returns:
        Path to library
        
    Raises:
        FileNotFoundError: If library cannot be found
    """
    lib_name = _get_lib_name()
    
    # 1. Bundled with package
    package_dir = Path(__file__).parent
    bundled = package_dir / "lib" / lib_name
    if bundled.exists():
        return str(bundled)
    
    # 2. Environment variable
    if "SVMIX_LIB_PATH" in os.environ:
        custom = Path(os.environ["SVMIX_LIB_PATH"])
        if custom.exists():
            return str(custom)
    
    # 3. Try system path (will raise if not found)
    try:
        lib = ctypes.util.find_library("svmix")
        if lib:
            return lib
    except Exception:
        pass
    
    raise FileNotFoundError(
        f"Could not find {lib_name}. "
        "Make sure to run 'make python-lib' before using the Python package, "
        "or set SVMIX_LIB_PATH environment variable."
    )


# Load library
_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path)


# ============================================================================
# C struct definitions (matching C code)
# ============================================================================

class CSvmixEnsembleCfg(ctypes.Structure):
    """Matches svmix_ensemble_cfg_t in C."""
    _fields_ = [
        ("lambda_", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("num_threads", ctypes.c_int),
    ]


class CSvmixCfg(ctypes.Structure):
    """Matches svmix_cfg_t in C."""
    _fields_ = [
        ("num_models", ctypes.c_size_t),
        ("num_particles", ctypes.c_size_t),
        ("spec", ctypes.c_int),  # svmix_spec_t
        ("ensemble", CSvmixEnsembleCfg),
    ]


class CSvParams(ctypes.Structure):
    """Matches svmix_sv_params_t in C."""
    _fields_ = [
        ("mu_h", ctypes.c_double),
        ("phi_h", ctypes.c_double),
        ("sigma_h", ctypes.c_double),
        ("nu", ctypes.c_double),
    ]


class CSvmixBelief(ctypes.Structure):
    """Matches svmix_belief_t in C."""
    _fields_ = [
        ("mean_h", ctypes.c_double),
        ("var_h", ctypes.c_double),
        ("mean_sigma", ctypes.c_double),
        ("valid", ctypes.c_int),
    ]


# ============================================================================
# Function signatures (all svmix_ functions)
# ============================================================================

# svmix_create - returns pointer directly, takes seeds array
_lib.svmix_create.argtypes = [
    ctypes.POINTER(CSvmixCfg),
    ctypes.POINTER(CSvParams),
    ctypes.POINTER(ctypes.c_ulong)  # seeds array
]
_lib.svmix_create.restype = ctypes.c_void_p  # Returns svmix_t* directly

# svmix_free
_lib.svmix_free.argtypes = [ctypes.c_void_p]
_lib.svmix_free.restype = None

# svmix_step
_lib.svmix_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.svmix_step.restype = ctypes.c_int

# svmix_get_belief
_lib.svmix_get_belief.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(CSvmixBelief)
]
_lib.svmix_get_belief.restype = ctypes.c_int

# svmix_get_weights
_lib.svmix_get_weights.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
_lib.svmix_get_weights.restype = ctypes.c_int

# svmix_get_num_models
_lib.svmix_get_num_models.argtypes = [ctypes.c_void_p]
_lib.svmix_get_num_models.restype = ctypes.c_size_t

# svmix_save_checkpoint
_lib.svmix_save_checkpoint.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p
]
_lib.svmix_save_checkpoint.restype = ctypes.c_int

# svmix_load_checkpoint - returns pointer, writes status to output param
_lib.svmix_load_checkpoint.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int)
]
_lib.svmix_load_checkpoint.restype = ctypes.c_void_p

# svmix_version
_lib.svmix_version.argtypes = None
_lib.svmix_version.restype = ctypes.c_char_p

# svmix_status_string
_lib.svmix_status_string.argtypes = [ctypes.c_int]
_lib.svmix_status_string.restype = ctypes.c_char_p


# ============================================================================
# Wrapper functions (thin layer over C, handles basic types)
# ============================================================================

def create(config: CSvmixCfg, params: list, seeds: list) -> int:
    """Create svmix instance.
    
    Args:
        config: Configuration structure
        params: List of CSvParams
        seeds: List of seeds (one per model)
        
    Returns:
        handle (pointer value, 0/NULL on error)
    """
    # Convert params list to C array
    params_array = (CSvParams * len(params))(*params)
    
    # Convert seeds list to C array
    seeds_array = (ctypes.c_ulong * len(seeds))(*seeds)
    
    handle = _lib.svmix_create(
        ctypes.byref(config),
        params_array,
        seeds_array
    )
    
    return handle


def free(handle: int) -> None:
    """Free svmix instance."""
    _lib.svmix_free(ctypes.c_void_p(handle))


def step(handle: int, observation: float) -> int:
    """Update filter with observation."""
    return _lib.svmix_step(ctypes.c_void_p(handle), observation)


def get_belief(handle: int) -> tuple:
    """Get belief state.
    
    Returns:
        (status_code, CSvmixBelief) tuple
    """
    belief = CSvmixBelief()
    status = _lib.svmix_get_belief(
        ctypes.c_void_p(handle),
        ctypes.byref(belief)
    )
    return status, belief


def get_weights(handle: int, num_models: int) -> tuple:
    """Get model weights.
    
    Returns:
        (status_code, list_of_weights) tuple
    """
    weights = (ctypes.c_double * num_models)()
    status = _lib.svmix_get_weights(
        ctypes.c_void_p(handle),
        weights,
        num_models
    )
    return status, list(weights)


def get_num_models(handle: int) -> int:
    """Get number of models."""
    return _lib.svmix_get_num_models(ctypes.c_void_p(handle))


def save_checkpoint(handle: int, filepath: str) -> int:
    """Save checkpoint to file."""
    return _lib.svmix_save_checkpoint(
        ctypes.c_void_p(handle),
        filepath.encode('utf-8')
    )


def load_checkpoint(filepath: str) -> tuple:
    """Load checkpoint from file.
    
    Returns:
        (handle, status_code) tuple - handle is 0 if load failed
    """
    status = ctypes.c_int()
    handle = _lib.svmix_load_checkpoint(
        filepath.encode('utf-8'),
        ctypes.byref(status)
    )
    return handle, status.value


def version() -> str:
    """Get svmix version string."""
    return _lib.svmix_version().decode('utf-8')


def status_string(status_code: int) -> str:
    """Get human-readable status string."""
    return _lib.svmix_status_string(status_code).decode('utf-8')
