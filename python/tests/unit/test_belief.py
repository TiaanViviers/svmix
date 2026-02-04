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
        """Test vol property returns annualized percentage."""
        # Test case 1: mean_h = -9.0 → daily vol ≈ 1.11%, annual ≈ 17.7%
        belief = Belief(mean_h=-9.0, var_h=0.1, mean_sigma=0.0, valid=True)
        daily_vol = math.exp(-9.0 / 2.0)
        expected_annual_pct = daily_vol * math.sqrt(252) * 100
        assert belief.vol == pytest.approx(expected_annual_pct, abs=0.01)
        assert belief.vol == pytest.approx(17.7, abs=0.5)  # ~17.7%
        
        # Test case 2: mean_h = -10.0 → daily vol ≈ 0.67%, annual ≈ 10.7%
        belief = Belief(mean_h=-10.0, var_h=0.1, mean_sigma=0.0, valid=True)
        daily_vol = math.exp(-10.0 / 2.0)
        expected_annual_pct = daily_vol * math.sqrt(252) * 100
        assert belief.vol == pytest.approx(expected_annual_pct, abs=0.01)
        assert belief.vol == pytest.approx(10.7, abs=0.5)  # ~10.7%
        
        # Test case 3: mean_h = -8.0 → daily vol ≈ 1.83%, annual ≈ 29.1%
        belief = Belief(mean_h=-8.0, var_h=0.1, mean_sigma=0.0, valid=True)
        daily_vol = math.exp(-8.0 / 2.0)
        expected_annual_pct = daily_vol * math.sqrt(252) * 100
        assert belief.vol == pytest.approx(expected_annual_pct, abs=0.01)
        assert belief.vol == pytest.approx(29.1, abs=0.5)  # ~29.1%

    def test_get_vol_units(self):
        """Test get_vol() with different unit combinations."""
        belief = Belief(mean_h=-9.0, var_h=0.1, mean_sigma=0.0, valid=True)
        
        # Base calculation
        daily_vol_decimal = math.exp(-9.0 / 2.0)  # ≈ 0.0111
        annual_vol_decimal = daily_vol_decimal * math.sqrt(252)  # ≈ 0.177
        
        # Test all 4 combinations
        # 1. Annualized percentage (default via property)
        assert belief.vol == pytest.approx(annual_vol_decimal * 100, abs=0.01)
        assert belief.get_vol(annualize=True, as_percentage=True) == pytest.approx(17.7, abs=0.5)
        
        # 2. Annualized decimal
        assert belief.get_vol(annualize=True, as_percentage=False) == pytest.approx(0.177, abs=0.005)
        
        # 3. Daily percentage
        assert belief.get_vol(annualize=False, as_percentage=True) == pytest.approx(1.11, abs=0.05)
        
        # 4. Daily decimal
        assert belief.get_vol(annualize=False, as_percentage=False) == pytest.approx(0.0111, abs=0.0005)
        
    def test_get_vol_invalid_belief(self):
        """Test get_vol() returns 0 for invalid belief."""
        belief = Belief(mean_h=0.0, var_h=0.0, mean_sigma=0.0, valid=False)
        
        assert belief.get_vol(annualize=True, as_percentage=True) == 0.0
        assert belief.get_vol(annualize=True, as_percentage=False) == 0.0
        assert belief.get_vol(annualize=False, as_percentage=True) == 0.0
        assert belief.get_vol(annualize=False, as_percentage=False) == 0.0

    def test_log_vol_property(self):
        """Test log_vol property computes mean_h / 2 correctly."""
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(-1.0, abs=1e-6)
        
        belief = Belief(mean_h=0.0, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(0.0, abs=1e-6)
        
        belief = Belief(mean_h=1.5, var_h=0.1, mean_sigma=0.0, valid=True)
        assert belief.log_vol == pytest.approx(0.75, abs=1e-6)

    def test_vol_log_vol_relationship(self):
        """Test that daily decimal vol = exp(log_vol) as expected."""
        belief = Belief(mean_h=-2.0, var_h=0.1, mean_sigma=0.0, valid=True)
        daily_vol = belief.get_vol(annualize=False, as_percentage=False)
        assert daily_vol == pytest.approx(math.exp(belief.log_vol), abs=1e-6)
        
        belief = Belief(mean_h=0.5, var_h=0.2, mean_sigma=0.0, valid=True)
        daily_vol = belief.get_vol(annualize=False, as_percentage=False)
        assert daily_vol == pytest.approx(math.exp(belief.log_vol), abs=1e-6)

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
        
        # Check daily vol in decimal
        daily_vol = belief.get_vol(annualize=False, as_percentage=False)
        assert 0.014 < daily_vol < 0.016
        
        # Default .vol property is annualized percentage
        assert 22 < belief.vol < 26  # ~23-24% annualized

    def test_high_volatility_regime(self):
        """Test with high volatility (crisis) values."""
        # During crisis: 5% daily vol
        # vol = exp(mean_h / 2), so mean_h = 2 * log(vol)
        # mean_h = 2 * log(0.05) ≈ 2 * (-3.0) ≈ -6.0
        belief = Belief(mean_h=-6.0, var_h=0.15, mean_sigma=0.0, valid=True)
        
        # Check daily vol in decimal
        daily_vol = belief.get_vol(annualize=False, as_percentage=False)
        assert 0.045 < daily_vol < 0.055
        
        # Default .vol property is annualized percentage (~80%)
        assert 75 < belief.vol < 85
        
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
