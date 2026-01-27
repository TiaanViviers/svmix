"""
HARD TESTS - Bug hunting mode 🐛

These tests have strict assertions designed to catch subtle bugs:
- Numerical stability issues
- Edge case failures  
- Stochastic behavior problems
- Memory corruption
- State consistency violations

Run with: pytest tests/integration/test_hard_assertions.py -v
"""

import pytest
import math
import tempfile
import os


@pytest.mark.requires_lib
def test_belief_values_are_sane():
    """Test that belief values are within physically reasonable bounds."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=1000,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(10, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Test with various observations
    observations = [0.001, 0.01, 0.05, -0.05, -0.01, 0.0]
    
    for obs in observations:
        svmix.step(obs)
        belief = svmix.get_belief()
        
        # HARD assertions on belief sanity
        assert belief.valid, f"Belief should be valid after obs={obs}"
        
        # Log-volatility should be finite and reasonable
        assert math.isfinite(belief.mean_h), f"mean_h must be finite, got {belief.mean_h}"
        assert -20 < belief.mean_h < 10, f"mean_h out of reasonable range: {belief.mean_h}"
        
        # Variance must be positive and finite
        assert belief.var_h > 0, f"var_h must be positive, got {belief.var_h}"
        assert belief.var_h < 100, f"var_h too large: {belief.var_h}"
        assert math.isfinite(belief.var_h), f"var_h must be finite"
        
        # Volatility (mean_sigma) must be positive and reasonable
        assert belief.mean_sigma > 0, f"mean_sigma must be positive, got {belief.mean_sigma}"
        assert belief.mean_sigma < 10, f"mean_sigma unreasonably large: {belief.mean_sigma}"
        assert math.isfinite(belief.mean_sigma), f"mean_sigma must be finite"
        
        # Consistency: mean_sigma ≈ exp(mean_h/2)
        expected_sigma = math.exp(belief.mean_h / 2)
        ratio = belief.mean_sigma / expected_sigma
        assert 0.5 < ratio < 2.0, f"mean_sigma inconsistent with mean_h: {belief.mean_sigma} vs {expected_sigma}"
    
    svmix.free()


@pytest.mark.requires_lib
def test_weights_sum_exactly_to_one():
    """Test that weights ALWAYS sum to exactly 1.0 (within machine precision)."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=100,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.grid(
        phi=[0.90, 0.93, 0.95, 0.97, 0.99],
        sigma=[0.15, 0.20, 0.25, 0.30],
        nu=[8.0, 12.0, 16.0, 20.0, 24.0],
        mu=-0.5
    )
    
    svmix = Svmix(config, params)
    
    # Test over many steps
    import random
    random.seed(42)
    
    for i in range(100):
        obs = random.gauss(0, 0.02)
        svmix.step(obs)
        
        weights = svmix.get_weights()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        total = sum(weights)
        
        # HARD assertion: must sum to 1.0 within numerical precision
        assert abs(total - 1.0) < 1e-12, f"Step {i}: weights sum to {total}, not 1.0 (error: {total - 1.0})"
        
        # All weights must be non-negative
        assert all(w >= 0 for w in weights), f"Step {i}: negative weight found: {min(weights)}"
        
        # All weights must be finite
        assert all(math.isfinite(w) for w in weights), f"Step {i}: non-finite weight found"
    
    svmix.free()


@pytest.mark.requires_lib
def test_determinism_is_perfect():
    """Test that same seed produces BIT-EXACT results, not just close."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    def run_filter(seed):
        config = SvmixConfig(
            spec=Spec.VOL,
            num_models=10,
            num_particles=500,
            lambda_=0.99,
            beta=0.8,
            epsilon=0.02,
            seed=seed
        )
        params = SvParams.linspace(10, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
        svmix = Svmix(config, params)
        
        observations = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003]
        
        for obs in observations:
            svmix.step(obs)
        
        belief = svmix.get_belief()
        weights = svmix.get_weights()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        svmix.free()
        return belief, weights
    
    # Run 3 times with same seed
    results = [run_filter(seed=123456) for _ in range(3)]
    
    # BIT-EXACT comparison (not approximate!)
    belief1, weights1 = results[0]
    for i in range(1, 3):
        belief_i, weights_i = results[i]
        
        # Beliefs must be BIT-EXACT
        assert belief_i.mean_h == belief1.mean_h, f"Run {i}: mean_h differs"
        assert belief_i.var_h == belief1.var_h, f"Run {i}: var_h differs"
        assert belief_i.mean_sigma == belief1.mean_sigma, f"Run {i}: mean_sigma differs"
        
        # Weights must be BIT-EXACT
        assert weights_i == weights1, f"Run {i}: weights differ"


@pytest.mark.requires_lib
def test_checkpoint_preserves_weights_exactly():
    """Test that checkpoint preserves weights with BIT-EXACT precision."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=20,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(20, phi=(0.85, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Process some observations
    for obs in [0.01, -0.005, 0.02, -0.01, 0.015]:
        svmix.step(obs)
    
    weights_before = svmix.get_weights()
    if hasattr(weights_before, 'tolist'):
        weights_before = weights_before.tolist()
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.svmix', delete=False) as f:
        checkpoint_path = f.name
    
    try:
        svmix.save_checkpoint(checkpoint_path)
        svmix.free()
        
        svmix2 = Svmix.load_checkpoint(checkpoint_path)
        weights_after = svmix2.get_weights()
        if hasattr(weights_after, 'tolist'):
            weights_after = weights_after.tolist()
        
        # BIT-EXACT comparison
        assert weights_after == weights_before, "Weights not preserved exactly after checkpoint"
        
        # Also check each individual weight
        for i, (w_before, w_after) in enumerate(zip(weights_before, weights_after)):
            assert w_before == w_after, f"Weight {i} differs: {w_before} vs {w_after}"
        
        svmix2.free()
    
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


@pytest.mark.requires_lib
def test_volatility_never_goes_negative():
    """Test that volatility estimates never become negative or zero."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(10, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Test with extreme observations
    extreme_obs = [0.0, 0.0, 0.0, 0.0, 0.0,  # Many zeros
                   0.10, -0.10, 0.15, -0.15,  # Large shocks
                   1e-10, -1e-10,  # Tiny values
                   0.001, 0.001, 0.001]  # Repeated small
    
    for obs in extreme_obs:
        svmix.step(obs)
        belief = svmix.get_belief()
        
        assert belief.mean_sigma > 0, f"Volatility went to {belief.mean_sigma} after obs={obs}"
        assert belief.mean_sigma < 100, f"Volatility exploded to {belief.mean_sigma} after obs={obs}"
    
    svmix.free()


@pytest.mark.requires_lib
def test_weights_epsilon_floor_is_respected():
    """Test that epsilon parameter creates minimum weight floor."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    epsilon = 0.10  # 10% anti-starvation
    K = 10
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=K,
        num_particles=1000,
        lambda_=0.95,  # Lower lambda for faster adaptation
        beta=0.5,
        epsilon=epsilon,
        seed=42
    )
    
    # Create diverse models
    params = SvParams.linspace(K, phi=(0.80, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Feed consistent pattern that should favor high-phi models
    for _ in range(50):
        svmix.step(0.02)
    
    weights = svmix.get_weights()
    if hasattr(weights, 'tolist'):
        weights = weights.tolist()
    
    # With epsilon=0.10, each model gets at least epsilon/K = 0.01 weight
    min_expected_weight = epsilon / K
    
    for i, w in enumerate(weights):
        assert w >= min_expected_weight * 0.9, \
            f"Model {i} weight {w} below epsilon floor {min_expected_weight}"
    
    svmix.free()


@pytest.mark.requires_lib
def test_no_nan_or_inf_ever():
    """Test that NaN or Inf NEVER appear in any output."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=200,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(10, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Pathological observations
    pathological = [0.0, 1e-20, -1e-20, 0.5, -0.5, 1e-5, -1e-5]
    
    for obs in pathological:
        svmix.step(obs)
        
        belief = svmix.get_belief()
        weights = svmix.get_weights()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        # Check belief for NaN/Inf
        assert not math.isnan(belief.mean_h), f"NaN in mean_h after obs={obs}"
        assert not math.isinf(belief.mean_h), f"Inf in mean_h after obs={obs}"
        assert not math.isnan(belief.var_h), f"NaN in var_h after obs={obs}"
        assert not math.isinf(belief.var_h), f"Inf in var_h after obs={obs}"
        assert not math.isnan(belief.mean_sigma), f"NaN in mean_sigma after obs={obs}"
        assert not math.isinf(belief.mean_sigma), f"Inf in mean_sigma after obs={obs}"
        
        # Check weights for NaN/Inf
        for i, w in enumerate(weights):
            assert not math.isnan(w), f"NaN in weight {i} after obs={obs}"
            assert not math.isinf(w), f"Inf in weight {i} after obs={obs}"
    
    svmix.free()


@pytest.mark.requires_lib
def test_continuous_vs_checkpointed_exact_match():
    """Test that checkpointing and continuing gives EXACT same results as continuous run."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=10,
        num_particles=500,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=424242
    )
    
    params = SvParams.linspace(10, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-0.5)
    observations = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012]
    
    # Continuous run
    svmix1 = Svmix(config, params)
    for obs in observations:
        svmix1.step(obs)
    belief_continuous = svmix1.get_belief()
    weights_continuous = svmix1.get_weights()
    if hasattr(weights_continuous, 'tolist'):
        weights_continuous = weights_continuous.tolist()
    svmix1.free()
    
    # Checkpointed run (save in middle)
    svmix2 = Svmix(config, params)
    for obs in observations[:4]:
        svmix2.step(obs)
    
    with tempfile.NamedTemporaryFile(suffix='.svmix', delete=False) as f:
        checkpoint_path = f.name
    
    try:
        svmix2.save_checkpoint(checkpoint_path)
        svmix2.free()
        
        svmix3 = Svmix.load_checkpoint(checkpoint_path)
        for obs in observations[4:]:
            svmix3.step(obs)
        
        belief_checkpointed = svmix3.get_belief()
        weights_checkpointed = svmix3.get_weights()
        if hasattr(weights_checkpointed, 'tolist'):
            weights_checkpointed = weights_checkpointed.tolist()
        svmix3.free()
        
        # BIT-EXACT comparison
        assert belief_checkpointed.mean_h == belief_continuous.mean_h, "mean_h differs"
        assert belief_checkpointed.var_h == belief_continuous.var_h, "var_h differs"
        assert belief_checkpointed.mean_sigma == belief_continuous.mean_sigma, "mean_sigma differs"
        assert weights_checkpointed == weights_continuous, "weights differ"
    
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


@pytest.mark.requires_lib
def test_large_K_doesnt_break():
    """Test that large ensembles (K=200) work correctly."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    K = 200
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=K,
        num_particles=1000,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.01,
        seed=42
    )
    
    params = SvParams.linspace(K, phi=(0.85, 0.99), sigma=0.2, nu=10, mu=-0.5)
    svmix = Svmix(config, params)
    
    # Process observations
    for i in range(20):
        svmix.step(0.01 * (i % 5 - 2))  # Varying observations
        
        belief = svmix.get_belief()
        weights = svmix.get_weights()
        if hasattr(weights, 'tolist'):
            weights = weights.tolist()
        
        assert len(weights) == K, f"Wrong number of weights: {len(weights)} != {K}"
        assert abs(sum(weights) - 1.0) < 1e-12, f"Weights don't sum to 1: {sum(weights)}"
        assert belief.valid, f"Belief invalid at step {i}"
    
    svmix.free()
