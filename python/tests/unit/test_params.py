"""
Unit tests for SvParams validation and generation.

No C library required for these tests.
"""

import pytest


def test_params_creation():
    """Test SvParams creation."""
    from svmix import SvParams
    
    params = SvParams(phi=0.97, sigma=0.2, nu=10, mu=-0.5)
    assert params.phi == 0.97
    assert params.sigma == 0.2
    assert params.nu == 10
    assert params.mu == -0.5


def test_params_validation():
    """Test SvParams parameter validation."""
    from svmix import SvParams
    
    # Valid params
    params = SvParams(phi=0.97, sigma=0.2, nu=10, mu=-0.5)
    params.validate()  # Should not raise
    
    # Invalid phi
    bad_params = SvParams(phi=1.5, sigma=0.2, nu=10, mu=-0.5)
    with pytest.raises(ValueError, match="phi"):
        bad_params.validate()
    
    # Invalid sigma
    bad_params = SvParams(phi=0.97, sigma=-0.1, nu=10, mu=-0.5)
    with pytest.raises(ValueError, match="sigma"):
        bad_params.validate()
    
    # Invalid nu
    bad_params = SvParams(phi=0.97, sigma=0.2, nu=2, mu=-0.5)
    with pytest.raises(ValueError, match="nu"):
        bad_params.validate()


def test_all_invalid_phi_values():
    """Test various invalid phi values."""
    from svmix import SvParams
    
    for invalid_phi in [-0.1, 0.0, 1.0, 1.5]:
        bad_params = SvParams(phi=invalid_phi, sigma=0.2, nu=10, mu=-0.5)
        with pytest.raises(ValueError, match="phi"):
            bad_params.validate()


def test_all_invalid_sigma_values():
    """Test various invalid sigma values."""
    from svmix import SvParams
    
    for invalid_sigma in [0.0, -0.1, -10.0]:
        bad_params = SvParams(phi=0.97, sigma=invalid_sigma, nu=10, mu=-0.5)
        with pytest.raises(ValueError, match="sigma"):
            bad_params.validate()


def test_all_invalid_nu_values():
    """Test various invalid nu values."""
    from svmix import SvParams
    
    for invalid_nu in [2.0, 1.5, 0.0, -1.0]:
        bad_params = SvParams(phi=0.97, sigma=0.2, nu=invalid_nu, mu=-0.5)
        with pytest.raises(ValueError, match="nu"):
            bad_params.validate()


def test_params_linspace():
    """Test SvParams.linspace() parameter generation."""
    from svmix import SvParams
    
    # Single parameter varied
    params = SvParams.linspace(
        num_models=10,
        phi=(0.90, 0.99),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )
    
    assert len(params) == 10
    assert params[0].phi == pytest.approx(0.90)
    assert params[-1].phi == pytest.approx(0.99)
    assert all(p.sigma == 0.2 for p in params)
    
    # Multiple parameters varied
    params = SvParams.linspace(
        num_models=5,
        phi=(0.90, 0.99),
        sigma=(0.1, 0.3),
        nu=10,
        mu=-0.5
    )
    
    assert len(params) == 5
    assert params[0].phi == pytest.approx(0.90)
    assert params[0].sigma == pytest.approx(0.1)
    assert params[-1].phi == pytest.approx(0.99)
    assert params[-1].sigma == pytest.approx(0.3)


def test_linspace_single_value():
    """Test linspace with num_models=1."""
    from svmix import SvParams
    
    params = SvParams.linspace(
        num_models=1,
        phi=(0.95, 0.99),
        sigma=0.2,
        nu=10,
        mu=-0.5
    )
    
    assert len(params) == 1
    # With 1 model, should use first value
    assert params[0].phi == pytest.approx(0.95)


def test_params_grid():
    """Test SvParams.grid() Cartesian product generation."""
    from svmix import SvParams
    
    # Small grid
    params = SvParams.grid(
        phi=[0.95, 0.97],
        sigma=[0.1, 0.2],
        nu=10,
        mu=-0.5
    )
    
    # Should have 2×2×1×1 = 4 combinations
    assert len(params) == 4
    
    # Check all combinations present
    phi_values = [p.phi for p in params]
    sigma_values = [p.sigma for p in params]
    
    assert sorted(set(phi_values)) == [0.95, 0.97]
    assert sorted(set(sigma_values)) == [0.1, 0.2]


def test_grid_single_values():
    """Test grid with single values (not lists)."""
    from svmix import SvParams
    
    params = SvParams.grid(
        phi=[0.95, 0.97],
        sigma=0.2,  # Single value
        nu=10,
        mu=-0.5
    )
    
    # Should have 2×1×1×1 = 2 combinations
    assert len(params) == 2
    assert all(p.sigma == 0.2 for p in params)


def test_grid_large_ensemble():
    """Test grid with many combinations."""
    from svmix import SvParams
    
    params = SvParams.grid(
        phi=[0.90, 0.93, 0.95, 0.97, 0.99],
        sigma=[0.15, 0.20, 0.25, 0.30],
        nu=[8.0, 12.0, 16.0, 20.0, 24.0],
        mu=-0.5
    )
    
    # Should have 5×4×5×1 = 100 combinations
    assert len(params) == 100
