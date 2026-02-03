"""Test new weighted parameter and entropy methods."""

import pytest
import math
from svmix import Svmix, SvmixConfig, SvParams, Spec


def test_weighted_params_basic():
    """Test get_weighted_params returns correct structure."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = [
        SvParams(phi=0.90, sigma=0.2, nu=10.0, mu=-9.0),
        SvParams(phi=0.95, sigma=0.2, nu=10.0, mu=-9.0),
        SvParams(phi=0.99, sigma=0.2, nu=10.0, mu=-9.0),
    ]
    
    svmix = Svmix(config, params)
    
    # Process some data
    for r in [0.01, -0.02, 0.015]:
        svmix.step(r)
    
    weighted = svmix.get_weighted_params()
    
    # Check structure
    assert 'phi' in weighted
    assert 'sigma' in weighted
    assert 'nu' in weighted
    assert 'mu' in weighted
    
    # Check ranges
    assert 0 < weighted['phi'] < 1
    assert weighted['sigma'] > 0
    assert weighted['nu'] > 2
    
    svmix.free()


def test_weighted_params_uniform_weights():
    """When all models identical, weighted params should match."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=5,
        num_particles=50,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = [SvParams(phi=0.95, sigma=0.3, nu=8.0, mu=-8.5)] * 5
    
    svmix = Svmix(config, params)
    svmix.step(0.01)
    
    weighted = svmix.get_weighted_params()
    
    # Should equal input params (within numerical precision)
    assert abs(weighted['phi'] - 0.95) < 1e-10
    assert abs(weighted['sigma'] - 0.3) < 1e-10
    assert abs(weighted['nu'] - 8.0) < 1e-10
    assert abs(weighted['mu'] - (-8.5)) < 1e-10
    
    svmix.free()


def test_weight_entropy_bits():
    """Test entropy calculation in bits."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=4,
        num_particles=100,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = SvParams.linspace(4, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-9)
    
    svmix = Svmix(config, params)
    
    for r in [0.01, -0.015, 0.02]:
        svmix.step(r)
    
    entropy = svmix.get_weight_entropy(base=2.0)
    
    # Entropy should be between 0 and log2(4) = 2
    assert 0 <= entropy <= 2.0
    
    svmix.free()


def test_weight_entropy_nats():
    """Test entropy calculation in nats."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=100,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = SvParams.linspace(3, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-9)
    
    svmix = Svmix(config, params)
    svmix.step(0.01)
    
    entropy = svmix.get_weight_entropy(base=math.e)
    
    # Entropy should be between 0 and ln(3)
    max_entropy = math.log(3)
    assert 0 <= entropy <= max_entropy
    
    svmix.free()


def test_weight_entropy_uniform():
    """With identical models, entropy should be maximal."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=4,
        num_particles=50,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = [SvParams(phi=0.95, sigma=0.2, nu=10, mu=-9)] * 4
    
    svmix = Svmix(config, params)
    svmix.step(0.01)
    
    entropy = svmix.get_weight_entropy(base=2.0)
    max_entropy = math.log2(4)
    
    # Should be at or very near maximum
    assert abs(entropy - max_entropy) < 0.01
    
    svmix.free()


def test_ml_feature_extraction_workflow():
    """Test complete ML feature extraction workflow."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=20,
        num_particles=200,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = SvParams.linspace(20, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-9)
    
    svmix = Svmix(config, params)
    
    returns = [0.005, -0.012, 0.008, -0.003, 0.015]
    features_list = []
    
    for r in returns:
        svmix.step(r)
        
        belief = svmix.get_belief()
        weighted = svmix.get_weighted_params()
        
        features = {
            'vol': belief.vol,
            'var_h': belief.var_h,
            'weight_entropy': svmix.get_weight_entropy(),
            'weighted_phi': weighted['phi'],
            'weighted_nu': weighted['nu'],
            'log_likelihood': svmix.get_last_log_likelihood()
        }
        features_list.append(features)
    
    # Verify all features are present and valid
    for features in features_list:
        assert features['vol'] > 0
        assert features['var_h'] >= 0
        assert 0 <= features['weight_entropy'] <= math.log2(20)
        assert 0.90 <= features['weighted_phi'] <= 0.99
        assert features['weighted_nu'] > 2
        assert features['log_likelihood'] != float('-inf')  # Should be finite after step
    
    svmix.free()


def test_methods_after_free():
    """Weighted methods should raise after free()."""
    config = SvmixConfig(
        spec=Spec.VOL,
        num_models=3,
        num_particles=50,
        lambda_=0.995,
        beta=0.8,
        epsilon=0.05
    )
    params = SvParams.linspace(3, phi=(0.90, 0.99), sigma=0.2, nu=10, mu=-9)
    
    svmix = Svmix(config, params)
    svmix.step(0.01)
    svmix.free()
    
    with pytest.raises(ValueError, match="freed"):
        svmix.get_weighted_params()
    
    with pytest.raises(ValueError, match="freed"):
        svmix.get_weight_entropy()
