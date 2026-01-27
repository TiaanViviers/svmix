"""
Pytest configuration and fixtures for svmix tests.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "requires_lib: mark test as requiring the C library to be built"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle library requirements."""
    # Check if library is available
    try:
        import svmix._native
        lib_available = True
    except (ImportError, OSError):
        lib_available = False
    
    if not lib_available:
        skip_lib = pytest.mark.skip(reason="C library not built - run 'make python-lib' first")
        for item in items:
            if "requires_lib" in item.keywords:
                item.add_marker(skip_lib)


@pytest.fixture
def temp_checkpoint_path(tmp_path):
    """Provide a temporary checkpoint file path."""
    return str(tmp_path / "test_checkpoint.svmix")


@pytest.fixture
def simple_config():
    """Provide a simple test configuration."""
    from svmix import SvmixConfig, Spec
    
    return SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )


@pytest.fixture
def simple_params():
    """Provide simple test parameters."""
    from svmix import SvParams
    
    return SvParams.linspace(
        num_models=5,
        phi=(0.90, 0.98),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )


@pytest.fixture
def svmix_instance(simple_config, simple_params):
    """Provide a ready-to-use svmix instance.
    
    Note: Caller is responsible for calling .free()
    """
    pytest.importorskip("svmix._native")
    from svmix import Svmix
    
    return Svmix(simple_config, simple_params)
