"""
Integration tests for edge cases and boundary conditions.

Requires C library to be built: make python-lib
"""

import pytest


@pytest.mark.requires_lib
def test_single_model():
    """Test with K=1 (single model)."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=1,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = [SvParams(phi=0.97, sigma=0.2, nu=10, mu=-0.5)]
    svmix = Svmix(config, params)
    
    svmix.step(0.01)
    belief = svmix.get_belief()
    assert belief.valid
    
    weights = svmix.get_weights()
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    assert len(weights) == 1
    assert weights[0] == pytest.approx(1.0)
    
    svmix.free()


@pytest.mark.requires_lib
def test_small_particle_count():
    """Test with very small particle count."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=10,  # Very small
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Should still work, just noisier
    svmix.step(0.01)
    belief = svmix.get_belief()
    assert belief.valid
    
    svmix.free()


@pytest.mark.requires_lib
def test_double_free_protection():
    """Test that double free is prevented."""
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
    svmix = Svmix(config, params)
    
    svmix.free()
    
    # Second free should be safe (no-op)
    svmix.free()  # Should not crash
    
    # Operations after free should error
    with pytest.raises(ValueError, match="freed"):
        svmix.step(0.01)


@pytest.mark.requires_lib
def test_very_long_sequence():
    """Test processing long sequence of observations."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(5, phi=(0.90, 0.98), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Process 1000 steps
    import random
    random.seed(42)
    
    for _ in range(1000):
        obs = random.gauss(0, 0.02)
        svmix.step(obs)
    
    belief = svmix.get_belief()
    assert belief.valid
    
    weights = svmix.get_weights()
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    assert abs(sum(weights) - 1.0) < 1e-9
    
    svmix.free()


@pytest.mark.requires_lib
def test_extreme_observations():
    """Test handling of extreme observations."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(5, phi=(0.90, 0.98), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Start with normal observation
    svmix.step(0.01)
    belief1 = svmix.get_belief()
    assert belief1.valid
    
    # Extreme positive observation
    svmix.step(0.10)
    belief2 = svmix.get_belief()
    assert belief2.valid
    # Note: Volatility may increase or decrease depending on model dynamics
    
    # Extreme negative observation
    svmix.step(-0.10)
    belief3 = svmix.get_belief()
    assert belief3.valid
    
    # Zero observation
    svmix.step(0.0)
    belief4 = svmix.get_belief()
    assert belief4.valid
    
    svmix.free()


@pytest.mark.requires_lib
def test_config_boundary_values():
    """Test configuration with boundary values."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    # Minimum lambda
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.001,  # Very small
        beta=0.8,
        epsilon=0.02
    )
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    svmix.step(0.01)
    svmix.free()
    
    # Maximum lambda
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=1.0,  # Maximum
        beta=0.8,
        epsilon=0.02
    )
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    svmix.step(0.01)
    svmix.free()
    
    # Zero epsilon (no jitter)
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.0
    )
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    svmix.step(0.01)
    svmix.free()


@pytest.mark.requires_lib
def test_mismatched_params_count():
    """Test error when params count doesn't match config."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    # Only 3 params but config says 5
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    
    with pytest.raises(ValueError, match="Config specifies"):
        Svmix(config, params)
