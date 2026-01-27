"""
Unit tests for svmix types, enums, and data structures.

No C library required for these tests.
"""

import pytest


def test_imports():
    """Test that all public API can be imported."""
    from svmix import (
        Svmix,
        SvmixConfig,
        SvParams,
        SvParamsVol,
        Belief,
        Spec,
        Status,
        version,
    )
    assert version() is not None


def test_spec_enum():
    """Test Spec enum values."""
    from svmix import Spec
    
    assert Spec.VOL == 1
    assert Spec.DRIFT == 2
    assert Spec.VOL_DRIFT == 3


def test_status_enum():
    """Test Status enum values."""
    from svmix import Status
    
    assert Status.OK == 0
    assert Status.ERR_INVALID_PARAM < 0
    assert Status.ERR_NULL_POINTER < 0


def test_belief_dataclass():
    """Test Belief dataclass creation."""
    from svmix import Belief
    
    belief = Belief(mean_h=-1.5, var_h=0.5, mean_sigma=0.2, valid=True)
    assert belief.mean_h == -1.5
    assert belief.var_h == 0.5
    assert belief.mean_sigma == 0.2
    assert belief.valid is True
    
    # Test repr
    repr_str = repr(belief)
    assert "mean_h" in repr_str
    assert "0.2" in repr_str  # mean_sigma formatted


def test_exceptions():
    """Test exception hierarchy."""
    from svmix import (
        SvmixError,
        SvmixNullPointerError,
        SvmixInvalidParamError,
        SvmixAllocError,
        SvmixInternalError,
        SvmixFileIOError,
        SvmixCheckpointCorruptError,
        SvmixVersionMismatchError
    )
    
    # All should inherit from SvmixError
    assert issubclass(SvmixNullPointerError, SvmixError)
    assert issubclass(SvmixInvalidParamError, SvmixError)
    assert issubclass(SvmixAllocError, SvmixError)
    assert issubclass(SvmixInternalError, SvmixError)
    assert issubclass(SvmixFileIOError, SvmixError)
    assert issubclass(SvmixCheckpointCorruptError, SvmixError)
    assert issubclass(SvmixVersionMismatchError, SvmixError)
