"""
Unit tests for SvmixConfig validation.

No C library required for these tests.
"""

import pytest


def test_config_creation():
    """Test SvmixConfig creation and validation."""
    from svmix import SvmixConfig, Spec
    
    # Valid config
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=50,
        num_particles=1000,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    assert config.num_models == 50
    assert config.num_particles == 1000
    assert config.lambda_ == 0.99


def test_config_defaults():
    """Test default values for optional parameters."""
    from svmix import SvmixConfig, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.0
    )
    
    # Check defaults for optional params
    assert config.num_threads == 0
    assert config.seed == 0


def test_all_invalid_num_models_values():
    """Test various invalid num_models values."""
    from svmix import SvmixConfig, Spec
    
    for invalid_k in [0, -1, -100]:
        with pytest.raises(ValueError, match="num_models"):
            SvmixConfig(
                spec=Spec.VOL,
                num_models=invalid_k,
                num_particles=100,
                lambda_=0.99,
                beta=0.8,
                epsilon=0.02
            )


def test_invalid_num_particles():
    """Test invalid num_particles."""
    from svmix import SvmixConfig, Spec
    
    with pytest.raises(ValueError, match="num_particles"):
        SvmixConfig(
            spec=Spec.VOL,
            num_models=10,
            num_particles=0,
            lambda_=0.99,
            beta=0.8,
            epsilon=0.02
        )


def test_all_invalid_lambda_values():
    """Test various invalid lambda values."""
    from svmix import SvmixConfig, Spec
    
    for invalid_lambda in [-0.1, 0.0, 1.01, 2.0]:
        with pytest.raises(ValueError, match="lambda_"):
            SvmixConfig(
                spec=Spec.VOL,
                num_models=5,
                num_particles=100,
                lambda_=invalid_lambda,
                beta=0.8,
                epsilon=0.02
            )


def test_all_invalid_epsilon_values():
    """Test various invalid epsilon values."""
    from svmix import SvmixConfig, Spec
    
    for invalid_eps in [-0.1, 1.0, 1.5]:
        with pytest.raises(ValueError, match="epsilon"):
            SvmixConfig(
                spec=Spec.VOL,
                num_models=5,
                num_particles=100,
                lambda_=0.99,
                beta=0.8,
                epsilon=invalid_eps
            )


def test_all_invalid_beta_values():
    """Test various invalid beta values."""
    from svmix import SvmixConfig, Spec
    
    for invalid_beta in [0.0, -0.1, -10.0]:
        with pytest.raises(ValueError, match="beta"):
            SvmixConfig(
                spec=Spec.VOL,
                num_models=5,
                num_particles=100,
                lambda_=0.99,
                beta=invalid_beta,
                epsilon=0.02
            )


def test_invalid_num_threads():
    """Test invalid num_threads."""
    from svmix import SvmixConfig, Spec
    
    with pytest.raises(ValueError, match="num_threads"):
        SvmixConfig(
            spec=Spec.VOL,
            num_models=5,
            num_particles=100,
            lambda_=0.99,
            beta=0.8,
            epsilon=0.02,
            num_threads=-1
        )


def test_config_boundary_values():
    """Test configuration with boundary values."""
    from svmix import SvmixConfig, Spec
    
    # Minimum lambda
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.001,  # Very small but valid
        beta=0.8,
        epsilon=0.02
    )
    assert config.lambda_ == 0.001
    
    # Maximum lambda
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=1.0,  # Maximum
        beta=0.8,
        epsilon=0.02
    )
    assert config.lambda_ == 1.0
    
    # Zero epsilon (no jitter)
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.0
    )
    assert config.epsilon == 0.0
    
    # Maximum epsilon
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.99
    )
    assert config.epsilon == 0.99
