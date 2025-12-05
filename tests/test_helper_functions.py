"""
Unit tests for helper functions used by the hospital model.

Tests cover:
- hill_gate: capacity gating function
- _validate_age_structured_inputs: input dimension validation
- _coerce_initial_vector: initial condition vector handling
- seasonal_forcing: time-varying transmission with seasonality
- policy_multiplier: intervention effects on transmission
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capacity import hill_gate
from utils import validate_age_structured_inputs, coerce_initial_vector
from time_varying_helpers import seasonal_forcing, policy_multiplier


# ========================================
# Tests for hill_gate
# ========================================

class TestHillGate:
    """Tests for the Hill function capacity gating."""
    
    def test_at_capacity_returns_half(self):
        """When occupancy equals capacity, gating factor should be 0.5."""
        result = hill_gate(100, 100, 4)
        assert result == 0.5
    
    def test_zero_occupancy_returns_one(self):
        """When occupancy is zero, gating factor should be 1.0 (unrestricted)."""
        result = hill_gate(0, 100, 4)
        assert result == 1.0
    
    def test_zero_capacity_raises_error(self):
        """When capacity is zero, should raise ValueError."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            hill_gate(50, 0, 4)
    
    def test_negative_capacity_raises_error(self):
        """When capacity is negative, should raise ValueError."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            hill_gate(50, -10, 4)
    
    def test_negative_occupancy_raises_error(self):
        """When occupancy is negative, should raise ValueError."""
        with pytest.raises(ValueError, match="occupancy must be non-negative"):
            hill_gate(-10, 100, 4)
    
    def test_negative_hill_coef_raises_error(self):
        """When hill_coef is negative, should raise ValueError."""
        with pytest.raises(ValueError, match="hill_coef must be non-negative"):
            hill_gate(50, 100, -1)
    
    def test_high_occupancy_approaches_zero(self):
        """When occupancy >> capacity, gating factor should approach 0."""
        result = hill_gate(1000, 100, 4)
        assert result < 0.001
    
    def test_low_occupancy_approaches_one(self):
        """When occupancy << capacity, gating factor should approach 1."""
        result = hill_gate(10, 100, 4)
        assert result > 0.99
    
    @pytest.mark.parametrize("occupancy,expected_approx", [
        (0, 1.0),
        (50, 0.941),   # 1 / (1 + (50/100)^4) = 1 / 1.0625
        (100, 0.5),
        (200, 0.0588), # 1 / (1 + (200/100)^4) = 1 / 17
    ])
    def test_gating_curve_values(self, occupancy, expected_approx):
        """Test specific points on the Hill function curve."""
        result = hill_gate(occupancy, 100, 4)
        assert result == pytest.approx(expected_approx, rel=0.01)
    
    @pytest.mark.parametrize("hill_coef", [1, 2, 4, 8, 10])
    def test_at_capacity_always_half(self, hill_coef):
        """At capacity, gating factor should always be 0.5 regardless of Hill coefficient."""
        result = hill_gate(100, 100, hill_coef)
        assert result == 0.5
    
    def test_higher_hill_coef_steeper_curve(self):
        """Higher Hill coefficient should produce steeper transition around capacity."""
        # At 80% capacity, higher coef should give closer to 1
        g_low = hill_gate(80, 100, 2)
        g_high = hill_gate(80, 100, 8)
        assert g_high > g_low
        
        # At 120% capacity, higher coef should give closer to 0
        g_low = hill_gate(120, 100, 2)
        g_high = hill_gate(120, 100, 8)
        assert g_high < g_low
    
    def test_hill_coef_zero_edge_case(self):
        """Hill coefficient of 0 should return 0.5 (any^0 = 1, so 1/(1+1) = 0.5)."""
        result = hill_gate(50, 100, 0)
        assert result == 0.5
    
    def test_very_small_hill_coef(self):
        """Very small Hill coefficient should produce very gradual gating."""
        g = hill_gate(200, 100, 0.5)
        # With n=0.5: 1/(1 + sqrt(2)) ≈ 0.414
        assert g == pytest.approx(0.414, rel=0.01)
    
    def test_float_inputs(self):
        """Function should handle float inputs correctly."""
        result = hill_gate(75.5, 100.0, 4.0)
        assert isinstance(result, float)
        assert 0.5 < result < 1.0
    
    def test_result_always_between_zero_and_one(self):
        """Gating factor should always be in [0, 1]."""
        for occ in [0, 50, 100, 200, 1000]:
            for cap in [10, 100, 1000]:
                if cap > 0:
                    result = hill_gate(occ, cap, 4)
                    assert 0.0 <= result <= 1.0


# ========================================
# Tests for _validate_age_structured_inputs
# ========================================

class TestValidateAgeStructuredInputs:
    """Tests for input validation function."""
    
    @pytest.fixture
    def valid_inputs(self):
        """Valid inputs for 3 age groups."""
        return {
            'age_params': [{'sigma': 0.1}, {'sigma': 0.2}, {'sigma': 0.3}],
            'contact_matrix': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'age_pops': [1000, 2000, 3000],
            'coverage': [0.1, 0.2, 0.3],
        }
    
    def test_valid_inputs_pass(self, valid_inputs):
        """Valid inputs should not raise any exception."""
        validate_age_structured_inputs(**valid_inputs)  # Should not raise
    
    def test_scalar_coverage_valid(self, valid_inputs):
        """Scalar coverage should be accepted."""
        valid_inputs['coverage'] = 0.5
        validate_age_structured_inputs(**valid_inputs)  # Should not raise
    
    def test_mismatched_age_params_length(self, valid_inputs):
        """age_params length mismatch should raise ValueError."""
        valid_inputs['age_params'] = [{'sigma': 0.1}, {'sigma': 0.2}]  # Only 2
        with pytest.raises(ValueError, match="age_params length"):
            validate_age_structured_inputs(**valid_inputs)
    
    def test_wrong_contact_matrix_shape(self, valid_inputs):
        """Wrong contact matrix shape should raise ValueError."""
        valid_inputs['contact_matrix'] = np.array([[1, 2], [3, 4]])  # 2x2 not 3x3
        with pytest.raises(ValueError, match="contact_matrix"):
            validate_age_structured_inputs(**valid_inputs)
    
    def test_coverage_list_wrong_length(self, valid_inputs):
        """Coverage list length mismatch should raise ValueError."""
        valid_inputs['coverage'] = [0.1, 0.2]  # Only 2
        with pytest.raises(ValueError, match="coverage length"):
            validate_age_structured_inputs(**valid_inputs)
    
    def test_empty_age_pops(self, valid_inputs):
        """Empty age_pops should still validate dimensions correctly."""
        valid_inputs['age_pops'] = []
        valid_inputs['age_params'] = []
        valid_inputs['contact_matrix'] = np.array([]).reshape(0, 0)
        valid_inputs['coverage'] = []
        validate_age_structured_inputs(**valid_inputs)  # Should not raise
    
    def test_single_age_group(self):
        """Single age group should work correctly."""
        inputs = {
            'age_params': [{'sigma': 0.1}],
            'contact_matrix': np.array([[5.0]]),
            'age_pops': [10000],
            'coverage': [0.5],
        }
        validate_age_structured_inputs(**inputs)  # Should not raise
    
    def test_contact_matrix_not_square(self, valid_inputs):
        """Non-square contact matrix should raise ValueError."""
        valid_inputs['contact_matrix'] = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        with pytest.raises(ValueError, match="contact_matrix"):
            validate_age_structured_inputs(**valid_inputs)


# ========================================
# Tests for _coerce_initial_vector
# ========================================

class TestCoerceInitialVector:
    """Tests for initial condition vector coercion."""
    
    def test_exact_length_returns_copy(self):
        """Vector of exact length should be returned as-is (copy)."""
        ic = {'I_by_age': [10, 20, 30]}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [0, 0, 0])
        assert result == [10, 20, 30]
    
    def test_returns_copy_not_reference(self):
        """Should return a copy, not the original list."""
        ic = {'I_by_age': [10, 20, 30]}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [0, 0, 0])
        result[0] = 999
        assert ic['I_by_age'][0] == 10  # Original unchanged
    
    def test_short_vector_extended_with_zeros(self):
        """Vector shorter than n_ages should be extended with zeros."""
        ic = {'I_by_age': [10]}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [0, 0, 0])
        assert result == [10, 0, 0]
    
    def test_long_vector_truncated(self):
        """Vector longer than n_ages should be truncated."""
        ic = {'I_by_age': [10, 20, 30, 40, 50]}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [0, 0, 0])
        assert result == [10, 20, 30]
    
    def test_missing_key_uses_fallback(self):
        """Missing key should use fallback value."""
        ic = {}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [5, 5, 5])
        assert result == [5, 5, 5]
    
    def test_fallback_extended_if_needed(self):
        """Fallback should be extended if shorter than n_ages."""
        ic = {}
        result = coerce_initial_vector(ic, 'I_by_age', 5, [1, 2])
        assert result == [1, 2, 0, 0, 0]
    
    def test_empty_vector_returns_zeros(self):
        """Empty vector should return all zeros."""
        ic = {'I_by_age': []}
        result = coerce_initial_vector(ic, 'I_by_age', 3, [])
        assert result == [0, 0, 0]
    
    def test_n_ages_zero(self):
        """n_ages of 0 should return empty list."""
        ic = {'I_by_age': [10, 20, 30]}
        result = coerce_initial_vector(ic, 'I_by_age', 0, [])
        assert result == []


# ========================================
# Tests for seasonal_forcing
# ========================================

class TestSeasonalForcing:
    """Tests for seasonal transmission forcing."""
    
    def test_zero_amplitude_returns_base(self):
        """With zero amplitude, should return beta_base unchanged."""
        result = seasonal_forcing(100, 0.3, amplitude=0.0)
        assert result == 0.3
    
    def test_peak_at_peak_day(self):
        """At peak_day, transmission should be at maximum (1 + amplitude)."""
        result = seasonal_forcing(0, 0.3, amplitude=0.3, peak_day=0)
        expected = 0.3 * (1 + 0.3)
        assert result == pytest.approx(expected)
    
    def test_trough_at_half_period(self):
        """At half period after peak, transmission should be at minimum."""
        result = seasonal_forcing(182.5, 0.3, amplitude=0.3, period=365, peak_day=0)
        expected = 0.3 * (1 - 0.3)
        assert result == pytest.approx(expected)
    
    def test_back_to_peak_after_full_period(self):
        """After full period, should be back to peak."""
        result = seasonal_forcing(365, 0.3, amplitude=0.3, period=365, peak_day=0)
        expected = 0.3 * (1 + 0.3)
        assert result == pytest.approx(expected)
    
    def test_shifted_peak_day(self):
        """peak_day should shift the phase of the cosine."""
        # Peak at day 100
        result = seasonal_forcing(100, 0.3, amplitude=0.3, period=365, peak_day=100)
        expected = 0.3 * (1 + 0.3)
        assert result == pytest.approx(expected)
    
    def test_amplitude_greater_than_one_raises(self):
        """Amplitude > 1 could produce negative transmission, should raise."""
        with pytest.raises(ValueError, match="amplitude"):
            seasonal_forcing(0, 0.3, amplitude=1.5)
    
    def test_amplitude_equal_to_one_valid(self):
        """Amplitude of exactly 1.0 is edge case (min transmission = 0)."""
        # Should not raise, but at trough transmission = 0
        result = seasonal_forcing(182.5, 0.3, amplitude=1.0, period=365, peak_day=0)
        assert result == pytest.approx(0.0)
    
    @pytest.mark.parametrize("period", [30, 90, 180, 365, 730])
    def test_various_periods(self, period):
        """Different periods should work correctly."""
        # At t=period/2, should be at trough
        result = seasonal_forcing(period/2, 0.3, amplitude=0.3, period=period, peak_day=0)
        expected = 0.3 * (1 - 0.3)
        assert result == pytest.approx(expected)
    
    def test_negative_peak_day(self):
        """Negative peak_day should work (shifts phase backwards)."""
        # peak_day=-10 means peak at t=-10, equivalent to peak at t=355 for period=365
        result = seasonal_forcing(0, 0.3, amplitude=0.3, period=365, peak_day=-10)
        # At t=0, we're 10 days past the peak
        # cos(2*pi*10/365) ≈ 0.9848
        expected = 0.3 * (1 + 0.3 * np.cos(2 * np.pi * 10 / 365))
        assert result == pytest.approx(expected)
    
    def test_result_positive_for_valid_amplitude(self):
        """Result should always be positive for amplitude <= 1."""
        for t in range(0, 365, 30):
            result = seasonal_forcing(t, 0.3, amplitude=0.5)
            assert result > 0


# ========================================
# Tests for policy_multiplier
# ========================================

class TestPolicyMultiplier:
    """Tests for policy intervention multiplier."""
    
    def test_empty_interventions_returns_one(self):
        """With no interventions, multiplier should be 1.0."""
        result = policy_multiplier(50, [])
        assert result == 1.0
    
    def test_none_interventions_returns_one(self):
        """None or empty list should return 1.0."""
        # Note: function expects list, None would fail
        result = policy_multiplier(50, [])
        assert result == 1.0
    
    def test_single_active_intervention(self):
        """Single active intervention should reduce transmission."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
        ]
        result = policy_multiplier(45, interventions)
        assert result == 0.5
    
    def test_single_inactive_intervention_before(self):
        """Before intervention starts, multiplier should be 1.0."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
        ]
        result = policy_multiplier(10, interventions)
        assert result == 1.0
    
    def test_single_inactive_intervention_after(self):
        """After intervention ends, multiplier should be 1.0."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
        ]
        result = policy_multiplier(70, interventions)
        assert result == 1.0
    
    def test_intervention_at_start_boundary(self):
        """Intervention should be active at start_day."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
        ]
        result = policy_multiplier(30, interventions)
        assert result == 0.5
    
    def test_intervention_at_end_boundary(self):
        """Intervention should be active at end_day."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
        ]
        result = policy_multiplier(60, interventions)
        assert result == 0.5
    
    def test_overlapping_interventions_strongest_applies(self):
        """With overlapping interventions, strongest reduction should apply."""
        interventions = [
            {'start_day': 30, 'end_day': 80, 'transmission_reduction': 0.3},
            {'start_day': 50, 'end_day': 70, 'transmission_reduction': 0.6},
        ]
        # At t=60, both active, strongest is 0.6, so multiplier = 0.4
        result = policy_multiplier(60, interventions)
        assert result == 0.4
    
    def test_sequential_interventions(self):
        """Sequential non-overlapping interventions should work correctly."""
        interventions = [
            {'start_day': 10, 'end_day': 20, 'transmission_reduction': 0.3},
            {'start_day': 40, 'end_day': 50, 'transmission_reduction': 0.6},
        ]
        assert policy_multiplier(15, interventions) == 0.7  # First active
        assert policy_multiplier(30, interventions) == 1.0  # Gap
        assert policy_multiplier(45, interventions) == 0.4  # Second active
    
    def test_full_reduction(self):
        """transmission_reduction of 1.0 should give multiplier of 0.0."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 1.0}
        ]
        result = policy_multiplier(45, interventions)
        assert result == 0.0
    
    def test_zero_reduction(self):
        """transmission_reduction of 0.0 should give multiplier of 1.0."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.0}
        ]
        result = policy_multiplier(45, interventions)
        assert result == 1.0
    
    def test_invalid_reduction_negative_raises(self):
        """Negative transmission_reduction should raise ValueError."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': -0.1}
        ]
        with pytest.raises(ValueError, match="transmission_reduction"):
            policy_multiplier(45, interventions)
    
    def test_invalid_reduction_greater_than_one_raises(self):
        """transmission_reduction > 1.0 should raise ValueError."""
        interventions = [
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 1.5}
        ]
        with pytest.raises(ValueError, match="transmission_reduction"):
            policy_multiplier(45, interventions)
    
    def test_multiple_partial_overlaps(self):
        """Complex scenario with multiple partial overlaps."""
        interventions = [
            {'start_day': 10, 'end_day': 40, 'transmission_reduction': 0.2},
            {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.4},
            {'start_day': 50, 'end_day': 80, 'transmission_reduction': 0.3},
        ]
        # t=35: first two active, strongest is 0.4
        assert policy_multiplier(35, interventions) == 0.6
        # t=55: last two active, strongest is 0.4
        assert policy_multiplier(55, interventions) == 0.6
        # t=70: only last active
        assert policy_multiplier(70, interventions) == 0.7
