"""Test Belief dataclass properties and computed fields."""

import math
import pytest
from svmix.types import Belief


class TestBeliefProperties:
    """Test Belief computed properties."""

    def test_belief_valid_state(self):
        """Test belief with valid state."""
        belief = Belief(
            mean_h=-2.0,
            var_h=0.1,
            mean_sigma=0.368,  # exp(-2.0 / 2) ≈ 0.368
            valid=True
        )
        
        assert belief.valid
        assert belief.mean_h == -2.0
        assert belief.var_h == 0.1
        assert belief.mean_sigma == pytest.approx(0.368, abs=0.001)

    def test_belief_invalid_state(self):
        """Test belief with invalid state."""
        belief = Belief(
            mean_h=0.0,
            var_h=0.0,
            mean_sigma=0.0,
            valid=False
        )
        
        assert not belief.valid
        assert belief.vol == 0.0
        assert belief.log_vol == 0.0

    def test_vol_property(self):
        """Test vol property computes exp(mean_h / 2) correctly."""
        # Test case 1: mean_h = -2.0 → vol = exp(-1.0) ≈ 0.368
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.0, valid=True)
        expected_vol = math.exp(-2.0 / 2.0)
        assert belief.vol == pytest.approx(expected_vol, abs=1e-6)
        assert belief.vol == pytest.approx(0.368, abs=0.001)
        
        # Test case 2: mean_h = 0.0 → vol = exp(0.0) = 1.0
        belief = Belief(mean_h=0.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.vol == pytest.approx(1.0, abs=1e-6)
        
        # Test case 3: mean_h = -1.0 → vol = exp(-0.5) ≈ 0.606
        belief = Belief(mean_h=-1.0, var_h=0.1, mean_sigma=0.0, valid=True)
        expected_vol = math.exp(-1.0 / 2.0)
        assert belief.vol == pytest.approx(expected_vol, abs=1e-6)
        assert belief.vol == pytest.approx(0.606, abs=0.001)

    def test_log_vol_property(self):
        """Test log_vol property computes mean_h / 2 correctly."""
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(-1.0, abs=1e-6)
        
        belief = Belief(mean_h=0.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(0.0, abs=1e-6)
        
        belief = Belief(mean_h=1.5, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(0.75, abs=1e-6)

    def test_vol_log_vol_relationship(self):
        """Test that vol = exp(log_vol) as expected."""
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.vol == pytest.approx(math.exp(belief.log_vol), abs=1e-6)
        
        belief = Belief(mean_h=0.5, var_h=0.2, mean_sigma=0.0, valid=True)
        assert belief.vol == pytest.approx(math.exp(belief.log_vol), abs=1e-6)

    def test_invalid_belief_returns_zero(self):
        """Test that invalid belief returns 0 for computed properties."""
        belief = Belief(mean_h=-1.0, var_h=0.1, mean_sigma=0.5, valid=False)
        
        # Properties should return 0 when invalid
        assert belief.vol == 0.0
        assert belief.log_vol == 0.0
        
        # But raw fields should retain their values
        assert belief.mean_h == -1.0
        assert belief.var_h == 0.1
        assert belief.mean_sigma == 0.5

    def test_repr_valid(self):
        """Test __repr__ for valid belief."""
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.368, valid=True)
        repr_str = repr(belief)
        
        assert "Belief(" in repr_str
        assert "vol=" in repr_str
        assert "mean_h=" in repr_str
        assert "var_h=" in repr_str
        assert "valid=False" not in repr_str

    def test_repr_invalid(self):
        """Test __repr__ for invalid belief."""
        belief = Belief(mean_h=0.0, var_h=0.0, mean_sigma=0.0, valid=False)
        repr_str = repr(belief)
        
        assert repr_str == "Belief(valid=False)"

    def test_realistic_equity_volatility(self):
        """Test with realistic equity volatility values."""
        # Typical daily equity volatility: 1.5%
        # vol = exp(mean_h / 2), so mean_h = 2 * log(vol)
        # mean_h = 2 * log(0.015) ≈ 2 * (-4.2) ≈ -8.4
        belief = Belief(mean_h=-8.4, var_h=0.05, mean_sigma=0.0, valid=True)
        
        # Should get approximately 1.5% daily vol
        assert 0.014 < belief.vol < 0.016
        
        # Annualized (252 trading days)
        annual_vol = belief.vol * math.sqrt(252)
        assert 0.22 < annual_vol < 0.26  # ~23-24% annualized

    def test_high_volatility_regime(self):
        """Test with high volatility (crisis) values."""
        # During crisis: 5% daily vol
        # vol = exp(mean_h / 2), so mean_h = 2 * log(vol)
        # mean_h = 2 * log(0.05) ≈ 2 * (-3.0) ≈ -6.0
        belief = Belief(mean_h=-6.0, var_h=0.15, mean_sigma=0.0, valid=True)
        
        # Should get approximately 5% daily vol
        assert 0.045 < belief.vol < 0.055
        
        # High var_h indicates uncertainty during regime change
        assert belief.var_h > 0.1

    def test_all_documented_attributes_exist(self):
        """Verify all attributes mentioned in docstring actually exist."""
        belief = Belief(mean_h=-1.0, var_h=0.1, mean_sigma=0.5, valid=True)
        
        # Fields from dataclass
        assert hasattr(belief, 'mean_h')
        assert hasattr(belief, 'var_h')
        assert hasattr(belief, 'mean_sigma')
        assert hasattr(belief, 'valid')
        
        # Computed properties
        assert hasattr(belief, 'vol')
        assert hasattr(belief, 'log_vol')
        
        # These should NOT exist (common mistake)
        assert not hasattr(belief, 'mu')
        assert not hasattr(belief, 'phi')
        assert not hasattr(belief, 'sigma')
        assert not hasattr(belief, 'nu')
