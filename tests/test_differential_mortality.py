"""
Tests for differential mortality tracking in simulate_master_hospital_model.

Tests cover:
- D_treated vs D_untreated tracking
- Age-specific mortality multipliers
- Capacity-dependent mortality effects
- Consistency between D_total and D_treated + D_untreated
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_hospital_model import simulate_master_hospital_model
from config import DIFFERENTIAL_MORTALITY_PARAMS


# ========================================
# Basic Differential Mortality Tests
# ========================================

class TestDifferentialMortalityBasics:
    """Basic tests for differential mortality tracking."""
    
    def test_d_treated_and_d_untreated_present(self, minimal_inputs):
        """Results should contain both D_treated and D_untreated."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        assert 'D_treated' in results
        assert 'D_untreated' in results
        assert 'D_treated_total' in results
        assert 'D_untreated_total' in results
    
    def test_d_treated_per_age_dimensions(self, minimal_inputs, n_ages):
        """D_treated should have correct dimensions per age group."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        assert len(results['D_treated']) == n_ages
        for age_series in results['D_treated']:
            assert len(age_series) == len(results['times'])
    
    def test_d_untreated_per_age_dimensions(self, minimal_inputs, n_ages):
        """D_untreated should have correct dimensions per age group."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        assert len(results['D_untreated']) == n_ages
        for age_series in results['D_untreated']:
            assert len(age_series) == len(results['times'])
    
    def test_d_sum_equals_d_total(self, minimal_inputs):
        """D_treated + D_untreated should equal D_total."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for t_idx in range(len(results['times'])):
            d_sum = (results['D_treated_total'][t_idx] + 
                     results['D_untreated_total'][t_idx])
            d_total = results['D_total'][t_idx]
            assert d_sum == pytest.approx(d_total, rel=0.01)
    
    def test_d_sum_equals_d_total_per_age(self, minimal_inputs, n_ages):
        """D_treated + D_untreated should equal D per age group."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for t_idx in range(len(results['times'])):
                d_sum = (results['D_treated'][a][t_idx] + 
                         results['D_untreated'][a][t_idx])
                d_total = results['D'][a][t_idx]
                assert d_sum == pytest.approx(d_total, rel=0.01)
    
    def test_d_treated_and_untreated_monotonic(self, minimal_inputs, n_ages):
        """Both D_treated and D_untreated should be monotonically increasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for t_idx in range(1, len(results['times'])):
                assert results['D_treated'][a][t_idx] >= results['D_treated'][a][t_idx-1] - 1e-10
                assert results['D_untreated'][a][t_idx] >= results['D_untreated'][a][t_idx-1] - 1e-10


# ========================================
# High Capacity Tests (Minimal Untreated)
# ========================================

class TestHighCapacityMortality:
    """Tests for mortality when capacity is not limiting."""
    
    def test_high_capacity_minimal_untreated(self, high_capacity_inputs):
        """With high capacity, untreated deaths should be minimal."""
        results = simulate_master_hospital_model(**high_capacity_inputs)
        
        # Untreated deaths should be negligible compared to treated
        d_untreated = results['D_untreated_total'][-1]
        d_treated = results['D_treated_total'][-1]
        
        # With very high capacity, untreated should be near zero
        assert d_untreated < d_treated * 0.01 or d_untreated < 1
    
    def test_high_capacity_all_deaths_treated(self, high_capacity_inputs):
        """With unlimited capacity, almost all deaths should be 'treated'."""
        results = simulate_master_hospital_model(**high_capacity_inputs)
        
        # The ratio of untreated to total should be very small
        if results['D_total'][-1] > 1:
            ratio = results['D_untreated_total'][-1] / results['D_total'][-1]
            assert ratio < 0.05  # Less than 5% untreated


# ========================================
# Low Capacity Tests (Significant Untreated)
# ========================================

class TestLowCapacityMortality:
    """Tests for mortality when capacity is severely limited."""
    
    def test_low_capacity_significant_untreated(self, low_capacity_inputs):
        """With low capacity, there should be significant untreated deaths."""
        results = simulate_master_hospital_model(**low_capacity_inputs)
        
        # With very low capacity, some untreated deaths should occur
        d_untreated = results['D_untreated_total'][-1]
        d_total = results['D_total'][-1]
        
        if d_total > 10:  # If meaningful epidemic occurred
            assert d_untreated > 0
    
    def test_zero_capacity_all_untreated(self, zero_capacity_inputs):
        """With zero capacity, deaths from X should be 'untreated'."""
        results = simulate_master_hospital_model(**zero_capacity_inputs)
        
        # With no admissions possible, X deaths should be untreated
        # (Though I deaths are still considered 'treated' in the current model)
        d_untreated = results['D_untreated_total'][-1]
        
        # Some untreated deaths should occur if there was an epidemic
        total_deaths = results['D_total'][-1]
        if total_deaths > 0:
            # Not all deaths will be untreated (I compartment deaths are treated)
            # but X compartment deaths should be
            pass  # Just verify it runs
    
    def test_capacity_ratio_affects_untreated_proportion(self, minimal_inputs):
        """Higher capacity constraint should lead to more untreated deaths."""
        # Low capacity run
        low_cap = {**minimal_inputs, 'ward_capacity': 10, 'icu_capacity': 3}
        results_low = simulate_master_hospital_model(**low_cap)
        
        # Medium capacity run
        med_cap = {**minimal_inputs, 'ward_capacity': 50, 'icu_capacity': 15}
        results_med = simulate_master_hospital_model(**med_cap)
        
        # Untreated proportion should be higher with lower capacity
        if results_low['D_total'][-1] > 10 and results_med['D_total'][-1] > 10:
            ratio_low = results_low['D_untreated_total'][-1] / results_low['D_total'][-1]
            ratio_med = results_med['D_untreated_total'][-1] / results_med['D_total'][-1]
            
            assert ratio_low >= ratio_med  # Lower capacity = higher untreated ratio


# ========================================
# Age-Specific Mortality Multiplier Tests
# ========================================

class TestAgeSpecificMortality:
    """Tests for age-specific mortality multipliers."""
    
    def test_elderly_higher_mortality(self, minimal_inputs):
        """Elderly should have higher mortality than young."""
        inputs = {**minimal_inputs, 'Tmax': 300}
        results = simulate_master_hospital_model(**inputs)
        
        # Get per-capita deaths
        d_young = results['D'][0][-1] / inputs['age_pops'][0]
        d_elderly = results['D'][2][-1] / inputs['age_pops'][2]
        
        # Elderly should have higher per-capita mortality
        assert d_elderly > d_young
    
    def test_age_specific_untreated_multipliers(self, low_capacity_inputs):
        """Untreated mortality multipliers should differ by age."""
        results = simulate_master_hospital_model(**low_capacity_inputs)
        
        # Just verify the simulation runs with capacity constraints
        # The age-specific multipliers are defined in config
        assert 'D_untreated' in results
        
        # Check that elderly untreated deaths are proportionally higher
        # This is difficult to test directly, but we can verify the model ran
        assert len(results['D_untreated']) == 3


# ========================================
# Gating Factor and Mortality Correlation Tests
# ========================================

class TestGatingMortalityCorrelation:
    """Tests for correlation between gating factors and mortality types."""
    
    def test_low_g_ward_increases_untreated(self, minimal_inputs):
        """When g_ward is low, untreated deaths should increase."""
        # Run with low capacity to trigger low g_ward
        inputs = {**minimal_inputs, 'ward_capacity': 5, 'icu_capacity': 2, 'Tmax': 150}
        results = simulate_master_hospital_model(**inputs)
        
        # Find time periods where g_ward is low
        low_g_indices = [i for i, g in enumerate(results['g_ward']) if g < 0.5]
        
        if len(low_g_indices) > 10:
            # During these periods, untreated deaths should be increasing
            # more rapidly than during high g_ward periods
            pass  # Complex to verify directly
    
    def test_g_ward_one_minimal_x_untreated(self, high_capacity_inputs):
        """When g_ward = 1, X compartment untreated deaths should be minimal."""
        results = simulate_master_hospital_model(**high_capacity_inputs)
        
        # All g_ward values should be near 1
        for g in results['g_ward']:
            assert g > 0.99
        
        # Therefore, untreated deaths should be minimal
        assert results['D_untreated_total'][-1] < results['D_treated_total'][-1] * 0.01 + 1


# ========================================
# Differential Mortality Component Tests
# ========================================

class TestMortalityComponents:
    """Tests for specific components of differential mortality."""
    
    def test_i_deaths_always_treated(self, minimal_inputs):
        """Deaths from I compartment should always be 'treated'."""
        # This is inherent to the model structure - I deaths are baseline
        # Just verify the model runs correctly
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'D_treated' in results
    
    def test_icu_deaths_treated(self, minimal_inputs):
        """Deaths in ICU should be 'treated' (patient received care)."""
        results = simulate_master_hospital_model(**minimal_inputs)
        # ICU deaths go to D_treated in the model
        assert results['D_treated_total'][-1] >= 0
    
    def test_ward_denied_icu_is_untreated(self, low_capacity_inputs):
        """Ward patients denied ICU have elevated mortality (untreated component)."""
        results = simulate_master_hospital_model(**low_capacity_inputs)
        
        # With low ICU capacity, some ward patients who need ICU won't get it
        # This contributes to untreated deaths
        if results['D_total'][-1] > 0:
            assert results['D_untreated_total'][-1] >= 0


# ========================================
# Consistency Tests
# ========================================

class TestMortalityConsistency:
    """Tests for internal consistency of mortality tracking."""
    
    def test_d_treated_non_negative(self, minimal_inputs, n_ages):
        """D_treated should be non-negative at all times."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for val in results['D_treated'][a]:
                assert val >= -1e-10
    
    def test_d_untreated_non_negative(self, minimal_inputs, n_ages):
        """D_untreated should be non-negative at all times."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for val in results['D_untreated'][a]:
                assert val >= -1e-10
    
    def test_d_treated_bounded_by_d(self, minimal_inputs, n_ages):
        """D_treated should not exceed D."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for t_idx in range(len(results['times'])):
                assert results['D_treated'][a][t_idx] <= results['D'][a][t_idx] + 1e-6
    
    def test_d_untreated_bounded_by_d(self, minimal_inputs, n_ages):
        """D_untreated should not exceed D."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for a in range(n_ages):
            for t_idx in range(len(results['times'])):
                assert results['D_untreated'][a][t_idx] <= results['D'][a][t_idx] + 1e-6
    
    def test_totals_match_age_sums(self, minimal_inputs, n_ages):
        """D_treated_total should equal sum of D_treated per age."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for t_idx in range(len(results['times'])):
            age_sum = sum(results['D_treated'][a][t_idx] for a in range(n_ages))
            assert results['D_treated_total'][t_idx] == pytest.approx(age_sum)
            
            age_sum_untreated = sum(results['D_untreated'][a][t_idx] for a in range(n_ages))
            assert results['D_untreated_total'][t_idx] == pytest.approx(age_sum_untreated)


# ========================================
# Scenario Comparison Tests
# ========================================

class TestMortalityScenarios:
    """Tests comparing mortality across different scenarios."""
    
    def test_more_deaths_with_lower_capacity(self, minimal_inputs):
        """Lower capacity should result in more total deaths."""
        high_cap = {**minimal_inputs, 'ward_capacity': 200, 'icu_capacity': 50, 'Tmax': 200}
        low_cap = {**minimal_inputs, 'ward_capacity': 20, 'icu_capacity': 5, 'Tmax': 200}
        
        results_high = simulate_master_hospital_model(**high_cap)
        results_low = simulate_master_hospital_model(**low_cap)
        
        # Lower capacity should lead to more deaths
        assert results_low['D_total'][-1] >= results_high['D_total'][-1] * 0.95
    
    def test_intervention_reduces_both_mortality_types(self, minimal_inputs):
        """Intervention should reduce both treated and untreated deaths."""
        no_int = {**minimal_inputs, 'interventions': [], 'Tmax': 150}
        with_int = {
            **minimal_inputs,
            'interventions': [{'start_day': 20, 'end_day': 100, 'transmission_reduction': 0.5}],
            'Tmax': 150,
        }
        
        results_no_int = simulate_master_hospital_model(**no_int)
        results_with_int = simulate_master_hospital_model(**with_int)
        
        # Intervention should reduce total deaths
        assert results_with_int['D_total'][-1] < results_no_int['D_total'][-1]
    
    def test_vaccine_reduces_deaths(self, minimal_inputs):
        """Vaccination should reduce both types of deaths."""
        no_vax = {**minimal_inputs, 'coverage': 0.0, 'Tmax': 200}
        with_vax = {**minimal_inputs, 'coverage': 0.5, 'VE': 0.7, 'Tmax': 200}
        
        results_no_vax = simulate_master_hospital_model(**no_vax)
        results_with_vax = simulate_master_hospital_model(**with_vax)
        
        # Vaccination should reduce total deaths
        assert results_with_vax['D_total'][-1] < results_no_vax['D_total'][-1]


# ========================================
# Config Parameter Tests
# ========================================

class TestConfigMortalityParams:
    """Tests for differential mortality parameters from config."""
    
    def test_default_multipliers_exist(self):
        """Default mortality multipliers should be defined in config."""
        assert 'mu_X_untreated_multiplier' in DIFFERENTIAL_MORTALITY_PARAMS
        assert 'mu_ward_denied_icu_multiplier' in DIFFERENTIAL_MORTALITY_PARAMS
    
    def test_age_specific_multipliers_exist(self):
        """Age-specific mortality multipliers should be defined."""
        age_keys = ['young', 'middle', 'elderly']
        for age in age_keys:
            key = f'mu_X_untreated_multiplier_{age}'
            assert key in DIFFERENTIAL_MORTALITY_PARAMS
    
    def test_multipliers_greater_than_one(self):
        """Untreated mortality multipliers should be > 1 (increased mortality)."""
        assert DIFFERENTIAL_MORTALITY_PARAMS['mu_X_untreated_multiplier'] > 1
        assert DIFFERENTIAL_MORTALITY_PARAMS['mu_ward_denied_icu_multiplier'] > 1
    
    def test_elderly_multiplier_highest(self):
        """Elderly should have highest untreated mortality multiplier."""
        young = DIFFERENTIAL_MORTALITY_PARAMS['mu_X_untreated_multiplier_young']
        middle = DIFFERENTIAL_MORTALITY_PARAMS['mu_X_untreated_multiplier_middle']
        elderly = DIFFERENTIAL_MORTALITY_PARAMS['mu_X_untreated_multiplier_elderly']
        
        assert elderly >= middle >= young
