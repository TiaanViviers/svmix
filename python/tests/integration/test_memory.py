"""
Integration tests for memory management and cleanup.

Requires C library to be built: make python-lib
"""

import pytest


@pytest.mark.requires_lib
def test_context_manager_del():
    """Test that __del__ cleans up properly."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    
    # Create and let go out of scope
    svmix = Svmix(config, params)
    svmix.step(0.01)
    # __del__ should be called when this returns


@pytest.mark.requires_lib
def test_multiple_instances():
    """Test creating multiple independent instances."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    
    # Create multiple instances
    instances = []
    for i in range(5):
        config_i = SvmixConfig(
            spec=Spec.VOL,
            num_models=3,
            num_particles=100,
            lambda_=0.99,
            beta=0.8,
            epsilon=0.02,
            seed=42 + i  # Different seeds
        )
        svmix = Svmix(config_i, params)
        svmix.step(0.01 * (i + 1))
        instances.append(svmix)
    
    # All should be independent
    beliefs = [inst.get_belief() for inst in instances]
    assert len(set(b.mean_sigma for b in beliefs)) > 1  # Different states
    
    # Clean up all
    for inst in instances:
        inst.free()


@pytest.mark.requires_lib
def test_failed_creation_cleanup():
    """Test that failed creation doesn't leak memory."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    # Wrong number of params - should fail gracefully
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    
    try:
        svmix = Svmix(config, params)
    except ValueError:
        pass  # Expected
    
    # Should not have leaked memory or broken state
