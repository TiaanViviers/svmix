"""
Integration tests for basic svmix filter operations.

Requires C library to be built: make python-lib
"""

import pytest
import numpy as np


@pytest.mark.requires_lib
def test_create_and_free():
    """Test creating and freeing svmix instance."""
    pytest.importorskip("svmix._native")
    
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = SvParams.linspace(
        num_models=3,
        phi=(0.95, 0.99),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )
    
    svmix = Svmix(config, params)
    assert svmix.num_models == 3
    
    svmix.free()
    
    # Should not be usable after free
    with pytest.raises(ValueError, match="freed"):
        svmix.step(0.01)


@pytest.mark.requires_lib
def test_step_and_belief():
    """Test stepping and getting belief."""
    pytest.importorskip("svmix._native")
    
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = SvParams.linspace(
        num_models=3,
        phi=(0.95, 0.99),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )
    
    svmix = Svmix(config, params)
    
    # Step with observation
    svmix.step(0.01)
    
    # Belief should be valid after first step
    belief = svmix.get_belief()
    assert belief.valid
    assert isinstance(belief.mean_h, float)
    assert isinstance(belief.var_h, float)
    assert isinstance(belief.mean_sigma, float)
    
    svmix.free()


@pytest.mark.requires_lib
def test_get_weights():
    """Test getting model weights."""
    pytest.importorskip("svmix._native")
    
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    params = SvParams.linspace(
        num_models=3,
        phi=(0.95, 0.99),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )
    
    svmix = Svmix(config, params)
    
    # Get weights
    weights = svmix.get_weights()
    assert len(weights) == 3
    
    # Convert to list if numpy array
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    
    assert abs(sum(weights) - 1.0) < 1e-6  # Should sum to 1
    
    svmix.free()


@pytest.mark.requires_lib
def test_multi_step_consistency():
    """Test that multiple steps produce consistent results."""
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
    
    # Process sequence of observations
    observations = [0.01, -0.005, 0.02, -0.01, 0.015]
    beliefs = []
    
    for obs in observations:
        svmix.step(obs)
        belief = svmix.get_belief()
        assert belief.valid
        beliefs.append(belief)
    
    # Beliefs should evolve smoothly (no jumps)
    for i in range(1, len(beliefs)):
        sigma_diff = abs(beliefs[i].mean_sigma - beliefs[i-1].mean_sigma)
        assert sigma_diff < 1.0, "Volatility jumped too much between steps"
    
    # Weights should sum to 1 at each step
    weights = svmix.get_weights()
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    assert abs(sum(weights) - 1.0) < 1e-10
    
    svmix.free()


@pytest.mark.requires_lib
def test_deterministic_with_same_seed():
    """Test that same seed produces identical results."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    def run_filter(seed):
        config = SvmixConfig(
            spec=Spec.VOL,
            num_models=3,
            num_particles=200,
            lambda_=0.99,
            beta=0.8,
            epsilon=0.02,
            seed=seed
        )
        params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
        svmix = Svmix(config, params)
        
        # Process observations
        for obs in [0.01, -0.005, 0.02]:
            svmix.step(obs)
        
        belief = svmix.get_belief()
        weights = svmix.get_weights()
        
        # Convert numpy arrays to lists for comparison
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        svmix.free()
        
        return belief, weights
    
    # Run twice with same seed
    belief1, weights1 = run_filter(seed=12345)
    belief2, weights2 = run_filter(seed=12345)
    
    # Should be identical
    assert belief1.mean_h == belief2.mean_h
    assert belief1.var_h == belief2.var_h
    assert belief1.mean_sigma == belief2.mean_sigma
    assert weights1 == pytest.approx(weights2, abs=1e-10)
    
    # Different seed should give different results
    belief3, weights3 = run_filter(seed=99999)
    assert belief1.mean_h != belief3.mean_h or not np.allclose(weights1, weights3)


@pytest.mark.requires_lib
def test_large_ensemble():
    """Test with large number of models."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=100,
        num_particles=1000,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02
    )
    
    # Use grid for diverse parameters
    params = SvParams.grid(
        phi=[0.90, 0.93, 0.95, 0.97, 0.99],
        sigma=[0.15, 0.20, 0.25, 0.30],
        nu=[8.0, 12.0, 16.0, 20.0, 24.0],
        mu=-0.5
    )
    
    assert len(params) == 100  # 5×4×5 = 100
    
    svmix = Svmix(config, params)
    assert svmix.num_models == 100
    
    # Should handle large ensemble smoothly
    svmix.step(0.01)
    belief = svmix.get_belief()
    assert belief.valid
    
    weights = svmix.get_weights()
    assert len(weights) == 100
    
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    assert abs(sum(weights) - 1.0) < 1e-9
    
    svmix.free()


@pytest.mark.requires_lib
def test_weights_evolution():
    """Test that model weights evolve sensibly."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=500,
        lambda_=0.95,  # Lower lambda for faster adaptation
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    # Create models with different persistence
    params = SvParams.linspace(10, phi=(0.80, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Initial weights should be roughly uniform
    weights_init = svmix.get_weights()
    if hasattr(weights_init, 'tolist'):
        weights_init = weights_init.tolist()
    
    # Feed persistent volatility pattern (consistent observations)
    for _ in range(20):
        svmix.step(0.02)  # Consistent moderate returns
    
    weights_final = svmix.get_weights()
    if hasattr(weights_final, 'tolist'):
        weights_final = weights_final.tolist()
    
    # High-persistence models should get more weight
    # (indices 8, 9 have phi closest to 0.99)
    high_phi_weight = weights_final[8] + weights_final[9]
    low_phi_weight = weights_final[0] + weights_final[1]
    assert high_phi_weight > low_phi_weight, "High-phi models should dominate"
    
    svmix.free()
