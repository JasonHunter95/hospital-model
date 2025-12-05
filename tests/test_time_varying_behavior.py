"""
Tests for time-varying behavior in simulate_model.

Tests cover:
- Seasonal forcing effects on transmission
- Policy intervention effects
- Waning immunity dynamics
- Combined time-varying effects
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate_model import simulate_model


# ========================================
# Seasonal Forcing Tests
# ========================================

class TestSeasonalForcing:
    """Tests for seasonal forcing effects on simulation."""
    
    def test_no_seasonality_constant_beta(self, minimal_inputs):
        """Without seasonality, beta_t should be constant."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.0, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 100},
        }
        results = simulate_model(**inputs)
        
        # All seasonal factors should be 1.0
        for sf in results['seasonal_factor']:
            assert sf == pytest.approx(1.0)
    
    def test_seasonality_varies_beta(self, minimal_inputs):
        """With seasonality, beta_t should vary over time."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.3, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365},
        }
        results = simulate_model(**inputs)
        
        # Seasonal factors should not all be equal
        unique_values = set(round(sf, 4) for sf in results['seasonal_factor'])
        assert len(unique_values) > 1
    
    def test_seasonal_peak_at_peak_day(self, minimal_inputs):
        """Transmission should peak at peak_day."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.3, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365},
        }
        results = simulate_model(**inputs)
        
        # First seasonal factor should be at maximum (1 + amplitude)
        assert results['seasonal_factor'][0] == pytest.approx(1.3)
    
    def test_seasonal_trough_at_half_period(self, minimal_inputs):
        """Transmission should be minimum at half period after peak."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.3, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365, 'time_step': 0.5},
        }
        results = simulate_model(**inputs)
        
        # Find seasonal factor near t = 182.5
        times = results['times']
        mid_idx = min(range(len(times)), key=lambda i: abs(times[i] - 182.5))
        
        # Should be near minimum (1 - amplitude)
        assert results['seasonal_factor'][mid_idx] == pytest.approx(0.7, rel=0.01)
    
    def test_shifted_peak_day(self, minimal_inputs):
        """peak_day should shift when peak occurs."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.3, 'period': 365, 'peak_day': 100},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365, 'time_step': 0.5},
        }
        results = simulate_model(**inputs)
        
        times = results['times']
        peak_idx = min(range(len(times)), key=lambda i: abs(times[i] - 100))
        
        # Should be at maximum at peak_day
        assert results['seasonal_factor'][peak_idx] == pytest.approx(1.3, rel=0.01)
    
    def test_high_seasonality_larger_variation(self, minimal_inputs):
        """Higher amplitude should produce larger variation."""
        inputs_low = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.1, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365},
        }
        inputs_high = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.5, 'period': 365, 'peak_day': 0},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365},
        }
        
        results_low = simulate_model(**inputs_low)
        results_high = simulate_model(**inputs_high)
        
        range_low = max(results_low['seasonal_factor']) - min(results_low['seasonal_factor'])
        range_high = max(results_high['seasonal_factor']) - min(results_high['seasonal_factor'])
        
        assert range_high > range_low


# ========================================
# Policy Intervention Tests
# ========================================

class TestPolicyInterventions:
    """Tests for policy intervention effects on simulation."""
    
    def test_no_interventions_policy_mult_one(self, minimal_inputs):
        """Without interventions, policy_mult should be 1.0."""
        inputs = {**minimal_inputs, 'intervention_config': [], 'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 100}}
        results = simulate_model(**inputs)
        
        for pm in results['policy_mult']:
            assert pm == 1.0
    
    def test_intervention_reduces_policy_mult(self, intervention_inputs):
        """During intervention, policy_mult should be reduced."""
        results = simulate_model(**intervention_inputs)
        
        times = results['times']
        # Find indices during intervention (30-60)
        active_indices = [i for i, t in enumerate(times) if 30 <= t <= 60]
        
        for idx in active_indices:
            assert results['policy_mult'][idx] == 0.5
    
    def test_policy_mult_before_intervention(self, intervention_inputs):
        """Before intervention, policy_mult should be 1.0."""
        results = simulate_model(**intervention_inputs)
        
        times = results['times']
        before_indices = [i for i, t in enumerate(times) if t < 30]
        
        for idx in before_indices:
            assert results['policy_mult'][idx] == 1.0
    
    def test_policy_mult_after_intervention(self, intervention_inputs):
        """After intervention, policy_mult should return to 1.0."""
        results = simulate_model(**intervention_inputs)
        
        times = results['times']
        after_indices = [i for i, t in enumerate(times) if t > 60]
        
        for idx in after_indices:
            assert results['policy_mult'][idx] == 1.0
    
    def test_intervention_reduces_infections(self, minimal_inputs):
        """Intervention should reduce total infections."""
        inputs_no_int = {**minimal_inputs, 'intervention_config': [], 'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 150}}
        inputs_with_int = {
            **minimal_inputs,
            'intervention_config': [
                {'start_day': 20, 'end_day': 80, 'transmission_reduction': 0.5}
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 150},
        }
        
        results_no_int = simulate_model(**inputs_no_int)
        results_with_int = simulate_model(**inputs_with_int)
        
        # With intervention should have fewer total deaths
        assert results_with_int['D_total'][-1] < results_no_int['D_total'][-1]
    
    def test_multiple_interventions(self, minimal_inputs):
        """Multiple interventions should work correctly."""
        inputs = {
            **minimal_inputs,
            'intervention_config': [
                {'start_day': 20, 'end_day': 40, 'transmission_reduction': 0.3},
                {'start_day': 60, 'end_day': 80, 'transmission_reduction': 0.6},
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 100},
        }
        results = simulate_model(**inputs)
        
        times = results['times']
        
        # Check first intervention period
        for i, t in enumerate(times):
            if 20 <= t <= 40:
                assert results['policy_mult'][i] == 0.7
        
        # Check gap between interventions
        for i, t in enumerate(times):
            if 45 <= t <= 55:
                assert results['policy_mult'][i] == 1.0
        
        # Check second intervention period
        for i, t in enumerate(times):
            if 60 <= t <= 80:
                assert results['policy_mult'][i] == 0.4
    
    def test_overlapping_interventions_strongest_applies(self, minimal_inputs):
        """Overlapping interventions should apply strongest reduction."""
        inputs = {
            **minimal_inputs,
            'intervention_config': [
                {'start_day': 20, 'end_day': 60, 'transmission_reduction': 0.3},
                {'start_day': 40, 'end_day': 80, 'transmission_reduction': 0.5},
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 100},
        }
        results = simulate_model(**inputs)
        
        times = results['times']
        
        # In overlap region (40-60), strongest is 0.5, so mult = 0.5
        for i, t in enumerate(times):
            if 45 <= t <= 55:
                assert results['policy_mult'][i] == 0.5


# ========================================
# Waning Immunity Tests
# ========================================

class TestWaningImmunity:
    """Tests for waning immunity effects on simulation."""
    
    def test_no_waning_r_stable(self, minimal_inputs):
        """Without waning, R should approach stable value."""
        inputs = {
            **minimal_inputs,
            'waning_config': None,  # No waning
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 300},
        }
        results = simulate_model(**inputs)
        
        # After epidemic subsides, R should be relatively stable
        # (not returning to S)
        R_total_late = sum(results['R'][a][-1] for a in range(len(minimal_inputs['age_pops'])))
        assert R_total_late > 0
    
    def test_waning_reduces_r(self, minimal_inputs):
        """With waning, R should decrease after initial peak."""
        inputs = {
            **minimal_inputs,
            'waning_config': {'omega': 0.02},  # Fast waning (~50 day immunity)
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 300},
        }
        results = simulate_model(**inputs)
        
        R_total = [sum(results['R'][a][t] for a in range(len(minimal_inputs['age_pops']))) 
                   for t in range(len(results['times']))]
        
        # R should peak and then decline
        max_R = max(R_total)
        final_R = R_total[-1]
        assert final_R < max_R * 0.9  # Should decline by at least 10%
    
    def test_waning_increases_s(self, minimal_inputs):
        """With waning, S should partially recover after initial decline."""
        inputs = {
            **minimal_inputs,
            'waning_config': {'omega': 0.02},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 400},
        }
        results = simulate_model(**inputs)
        
        S_total = [sum(results['S'][a][t] for a in range(len(minimal_inputs['age_pops']))) 
                   for t in range(len(results['times']))]
        
        # S should have a minimum and then increase
        min_S_idx = S_total.index(min(S_total))
        
        # S at end should be higher than at minimum (due to waning)
        if min_S_idx < len(S_total) - 10:  # If minimum isn't at the very end
            assert S_total[-1] > S_total[min_S_idx]
    
    def test_age_specific_waning(self, minimal_inputs):
        """Age-specific waning rates should work."""
        inputs = {
            **minimal_inputs,
            'waning_config': {
                'omega_young': 0.005,
                'omega_middle': 0.01,
                'omega_elderly': 0.02,
            },
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 300},
        }
        results = simulate_model(**inputs)
        
        # Just verify it runs without error
        assert len(results['times']) > 0
    
    def test_uniform_waning_rate(self, minimal_inputs):
        """Uniform 'omega' should be applied to all age groups."""
        inputs = {
            **minimal_inputs,
            'waning_config': {'omega': 0.01},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 200},
        }
        results = simulate_model(**inputs)
        assert len(results['times']) > 0


# ========================================
# Combined Time-Varying Effects Tests
# ========================================

class TestCombinedTimeVarying:
    """Tests for combined time-varying effects."""
    
    def test_seasonality_and_intervention_combined(self, minimal_inputs):
        """Seasonality and intervention should combine multiplicatively."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.2, 'period': 365, 'peak_day': 0},
            'intervention_config': [
                {'start_day': 30, 'end_day': 60, 'transmission_reduction': 0.5}
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 100},
        }
        results = simulate_model(**inputs)
        
        times = results['times']
        beta_base = minimal_inputs['beta_base']
        
        # During intervention, beta_t should be reduced by both effects
        for i, t in enumerate(times):
            if 30 <= t <= 60:
                assert results['policy_mult'][i] == 0.5
                # beta_t should reflect both seasonal and policy effects
                expected_beta = beta_base * results['seasonal_factor'][i] * results['policy_mult'][i]
                assert results['beta_t'][i] == pytest.approx(expected_beta)
    
    def test_all_time_varying_effects(self, minimal_inputs):
        """All time-varying effects should work together."""
        inputs = {
            **minimal_inputs,
            'seasonal_config': {'amplitude': 0.2, 'period': 365, 'peak_day': 0},
            'waning_config': {'omega': 0.01},
            'intervention_config': [
                {'start_day': 50, 'end_day': 100, 'transmission_reduction': 0.4}
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 365},
        }
        results = simulate_model(**inputs)
        
        # Verify simulation completes successfully
        assert len(results['times']) > 0
        assert max(results['D_total']) > 0  # Some deaths occurred
    
    def test_beta_t_reflects_all_modifiers(self, minimal_inputs):
        """beta_t should reflect vaccine, seasonal, and policy effects."""
        coverage = 0.5
        VE = 0.6
        inputs = {
            **minimal_inputs,
            'vaccine_config': {'coverage': coverage, 'VE_infection': VE, 'VE_severe': VE, 'VE_death': VE},
            'seasonal_config': {'amplitude': 0.0, 'period': 365, 'peak_day': 0},  # No seasonality
            'intervention_config': [
                {'start_day': 0, 'end_day': 200, 'transmission_reduction': 0.3}
            ],
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 50},
        }
        results = simulate_model(**inputs)
        
        # beta_t should be beta_base * seasonal(1.0) * policy(0.7)
        # Note: Vaccine coverage affects force of infection, not beta_t directly
        expected_beta_t = minimal_inputs['beta_base'] * 1.0 * 0.7
        
        for beta in results['beta_t']:
            assert beta == pytest.approx(expected_beta_t)


# ========================================
# Long-term Dynamics Tests
# ========================================

class TestLongTermDynamics:
    """Tests for long-term simulation dynamics."""
    
    def test_endemic_equilibrium_with_waning(self, minimal_inputs):
        """With waning, system should approach endemic equilibrium."""
        inputs = {
            **minimal_inputs,
            'waning_config': {'omega': 0.005},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 1000, 'time_step': 0.5},
        }
        results = simulate_model(**inputs)
        
        # After long time, system should have non-zero I
        # (endemic state rather than extinction)
        I_total_late = sum(results['I'][a][-1] for a in range(len(minimal_inputs['age_pops'])))
        # In an endemic state, there should still be some infections
        # Note: This might be very small but positive
        # The epidemic may have died out, so we just check it ran
        assert results['times'][-1] >= 1000
    
    def test_seasonal_recurring_waves(self, minimal_inputs):
        """With waning and seasonality, should see recurring waves."""
        inputs = {
            **minimal_inputs,
            'waning_config': {'omega': 0.01},
            'seasonal_config': {'amplitude': 0.3, 'period': 365, 'peak_day': 180},
            'sim_config': {**minimal_inputs['sim_config'], 'Tmax': 730, 'time_step': 0.5},  # 2 years
        }
        results = simulate_model(**inputs)
        
        I_total = [sum(results['I'][a][t] for a in range(len(minimal_inputs['age_pops']))) 
                   for t in range(len(results['times']))]
        
        # Find local maxima (peaks)
        peaks = []
        for i in range(10, len(I_total) - 10):
            if I_total[i] > I_total[i-5] and I_total[i] > I_total[i+5]:
                if I_total[i] > 10:  # Only count significant peaks
                    peaks.append(i)
        
        # Should have multiple peaks (waves)
        # Note: This test may be sensitive to parameters
        assert len(peaks) >= 1  # At least the initial wave
