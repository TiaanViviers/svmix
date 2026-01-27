"""
Integration tests for checkpoint save/load functionality.

Requires C library to be built: make python-lib
"""

import pytest
import tempfile
import os


@pytest.mark.requires_lib
def test_checkpoint_save_and_load(temp_checkpoint_path):
    """Test basic checkpoint save and load."""
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
    
    # Create and run filter
    svmix1 = Svmix(config, params)
    for obs in [0.01, -0.005, 0.02, -0.01]:
        svmix1.step(obs)
    
    belief1 = svmix1.get_belief()
    weights1 = svmix1.get_weights()
    
    # Convert to list if numpy
    if hasattr(weights1, 'tolist'):
        weights1 = weights1.tolist()
    
    # Save checkpoint
    svmix1.save_checkpoint(temp_checkpoint_path)
    svmix1.free()
    
    # Load checkpoint
    svmix2 = Svmix.load_checkpoint(temp_checkpoint_path)
    weights2 = svmix2.get_weights()
    
    # Convert to list if numpy
    if hasattr(weights2, 'tolist'):
        weights2 = weights2.tolist()
    
    # Weights should match exactly (bit-exact restoration)
    assert weights2 == pytest.approx(weights1, abs=1e-10)
    
    # Continue from checkpoint - typical usage pattern
    svmix2.step(0.015)
    belief3 = svmix2.get_belief()
    # After stepping, belief should be valid
    assert belief3.valid
    
    svmix2.free()


@pytest.mark.requires_lib
def test_checkpoint_continuation_matches_continuous_run():
    """Test that checkpoint continuation produces same results as continuous run."""
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
    observations = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008]
    
    # Continuous run
    svmix_continuous = Svmix(config, params)
    for obs in observations:
        svmix_continuous.step(obs)
    belief_continuous = svmix_continuous.get_belief()
    weights_continuous = svmix_continuous.get_weights()
    if hasattr(weights_continuous, 'tolist'):
        weights_continuous = weights_continuous.tolist()
    svmix_continuous.free()
    
    # Run with checkpoint in middle
    svmix_part1 = Svmix(config, params)
    for obs in observations[:3]:
        svmix_part1.step(obs)
    
    with tempfile.NamedTemporaryFile(suffix='.svmix', delete=False) as f:
        checkpoint_path = f.name
    
    try:
        svmix_part1.save_checkpoint(checkpoint_path)
        svmix_part1.free()
        
        svmix_part2 = Svmix.load_checkpoint(checkpoint_path)
        for obs in observations[3:]:
            svmix_part2.step(obs)
        
        belief_checkpointed = svmix_part2.get_belief()
        weights_checkpointed = svmix_part2.get_weights()
        if hasattr(weights_checkpointed, 'tolist'):
            weights_checkpointed = weights_checkpointed.tolist()
        svmix_part2.free()
        
        # Results should match exactly
        assert belief_checkpointed.mean_h == pytest.approx(belief_continuous.mean_h, rel=1e-10)
        assert belief_checkpointed.var_h == pytest.approx(belief_continuous.var_h, rel=1e-10)
        assert belief_checkpointed.mean_sigma == pytest.approx(belief_continuous.mean_sigma, rel=1e-10)
        assert weights_checkpointed == pytest.approx(weights_continuous, abs=1e-10)
    
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


@pytest.mark.requires_lib
def test_checkpoint_invalid_path():
    """Test error handling for invalid checkpoint paths."""
    from svmix import Svmix, SvmixConfig, SvParams, Spec, SvmixError
    
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
    svmix.step(0.01)
    
    # Save to invalid path
    with pytest.raises(SvmixError):
        svmix.save_checkpoint("/invalid/path/checkpoint.svmix")
    
    svmix.free()
    
    # Load from non-existent file
    with pytest.raises(SvmixError):
        Svmix.load_checkpoint("/nonexistent/checkpoint.svmix")


@pytest.mark.requires_lib
def test_checkpoint_multiple_saves_loads():
    """Test multiple checkpoint cycles.
    
    Demonstrates correct usage: always step() immediately after load_checkpoint().
    """
    from svmix import Svmix, SvmixConfig, SvParams, Spec
    
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=200,
        lambda_=0.99,
        beta=0.8,
        epsilon=0.02,
        seed=42
    )
    
    params = SvParams.linspace(3, phi=(0.95, 0.99), sigma=0.2, nu=10, mu=-0.5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial run
        svmix = Svmix(config, params)
        svmix.step(0.01)
        
        # Save/load cycle 1
        path1 = os.path.join(tmpdir, "checkpoint1.svmix")
        svmix.save_checkpoint(path1)
        svmix.free()
        
        # Load and IMMEDIATELY step (correct pattern)
        svmix = Svmix.load_checkpoint(path1)
        svmix.step(0.02)
        belief_after_step = svmix.get_belief()
        assert belief_after_step.valid
        
        # Save/load cycle 2
        path2 = os.path.join(tmpdir, "checkpoint2.svmix")
        svmix.save_checkpoint(path2)
        svmix.free()
        
        # Load and IMMEDIATELY step (correct pattern)
        svmix = Svmix.load_checkpoint(path2)
        svmix.step(0.03)
        belief_final = svmix.get_belief()
        assert belief_final.valid
        
        svmix.free()
