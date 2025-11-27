"""
Integration tests for simulate_master_hospital_model.

Tests cover:
- Smoke tests (function runs, returns expected structure)
- Population conservation (S + E + I + X + H_ward + H_icu + R + D = constant)
- Death monotonicity (deaths only increase)
- Output structure and dimensions
- Default parameter usage
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_hospital_model import simulate_master_hospital_model
from config import (
    DEFAULT_SIM_PARAMS,
    DEFAULT_CAPACITY_PARAMS,
    AGE_PARAMS_DEFAULT,
    CONTACT_MATRIX_DEFAULT,
)


# ========================================
# Smoke Tests
# ========================================

class TestSmoke:
    """Basic smoke tests to verify function runs without errors."""
    
    def test_minimal_inputs_run(self, minimal_inputs):
        """Function should run with minimal valid inputs."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert results is not None
        assert isinstance(results, dict)
    
    def test_returns_dict(self, minimal_inputs):
        """Function should return a dictionary."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert isinstance(results, dict)
    
    def test_times_array_present(self, minimal_inputs):
        """Results should contain 'times' array."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'times' in results
        assert len(results['times']) > 0
    
    def test_all_compartments_present(self, minimal_inputs):
        """Results should contain all compartment arrays."""
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D', 'H']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_aggregated_totals_present(self, minimal_inputs):
        """Results should contain aggregated totals."""
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = [
            'H_ward_total', 'H_icu_total', 'H_total',
            'E_total', 'I_total', 'X_total', 'D_total'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_capacity_metrics_present(self, minimal_inputs):
        """Results should contain capacity metrics."""
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = [
            'ward_overflow', 'icu_overflow',
            'cum_ward_overflow', 'cum_icu_overflow',
            'g_ward', 'g_icu'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_time_varying_params_present(self, minimal_inputs):
        """Results should contain time-varying parameters."""
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = ['beta_t', 'seasonal_factor', 'policy_mult']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_metadata_present(self, minimal_inputs):
        """Results should contain metadata."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'ward_capacity' in results
        assert 'icu_capacity' in results
        assert 'age_pops' in results
        assert 'parameters' in results


# ========================================
# Output Structure Tests
# ========================================

class TestOutputStructure:
    """Tests for output dimensions and types."""
    
    def test_compartments_have_correct_age_dimensions(self, minimal_inputs, n_ages):
        """Each compartment should have n_ages sub-arrays."""
        results = simulate_master_hospital_model(**minimal_inputs)
        for compartment in ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D']:
            assert len(results[compartment]) == n_ages
    
    def test_compartment_time_series_length(self, minimal_inputs):
        """Compartment time series should match times length."""
        results = simulate_master_hospital_model(**minimal_inputs)
        n_times = len(results['times'])
        for compartment in ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D']:
            for age_series in results[compartment]:
                assert len(age_series) == n_times
    
    def test_aggregated_totals_length(self, minimal_inputs):
        """Aggregated totals should match times length."""
        results = simulate_master_hospital_model(**minimal_inputs)
        n_times = len(results['times'])
        for key in ['H_ward_total', 'H_icu_total', 'H_total', 'D_total']:
            assert len(results[key]) == n_times
    
    def test_times_are_monotonic(self, minimal_inputs):
        """Times should be strictly monotonically increasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        times = results['times']
        for i in range(1, len(times)):
            assert times[i] > times[i-1]
    
    def test_times_start_at_zero(self, minimal_inputs):
        """Times should start at 0."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert results['times'][0] == 0
    
    def test_times_end_near_tmax(self, minimal_inputs):
        """Times should end at approximately Tmax."""
        inputs = {**minimal_inputs, 'Tmax': 100, 'time_step': 0.1}
        results = simulate_master_hospital_model(**inputs)
        # Allow small floating point tolerance
        assert results['times'][-1] >= 100 - 0.1
    
    def test_h_equals_h_ward_plus_h_icu(self, minimal_inputs):
        """H should equal H_ward + H_icu for each age group."""
        results = simulate_master_hospital_model(**minimal_inputs)
        n_ages = len(minimal_inputs['age_pops'])
        for a in range(n_ages):
            for t in range(len(results['times'])):
                h_sum = results['H_ward'][a][t] + results['H_icu'][a][t]
                assert results['H'][a][t] == pytest.approx(h_sum)


# ========================================
# Population Conservation Tests
# ========================================

class TestPopulationConservation:
    """Tests for population conservation law."""
    
    def test_total_population_conserved(self, minimal_inputs):
        """Total population should be conserved (S + E + I + X + H + R + D = N)."""
        results = simulate_master_hospital_model(**minimal_inputs)
        total_pop = sum(minimal_inputs['age_pops'])
        n_ages = len(minimal_inputs['age_pops'])
        
        for t_idx in range(len(results['times'])):
            pop_at_t = sum(
                results['S'][a][t_idx] +
                results['E'][a][t_idx] +
                results['I'][a][t_idx] +
                results['X'][a][t_idx] +
                results['H_ward'][a][t_idx] +
                results['H_icu'][a][t_idx] +
                results['R'][a][t_idx] +
                results['D'][a][t_idx]
                for a in range(n_ages)
            )
            # Allow 1% tolerance for Euler integration errors
            assert pop_at_t == pytest.approx(total_pop, rel=0.01), \
                f"Population not conserved at t={results['times'][t_idx]}"
    
    def test_population_conserved_per_age_group(self, minimal_inputs):
        """Population should be conserved within each age group."""
        results = simulate_master_hospital_model(**minimal_inputs)
        n_ages = len(minimal_inputs['age_pops'])
        
        for a in range(n_ages):
            age_pop = minimal_inputs['age_pops'][a]
            for t_idx in range(len(results['times'])):
                pop_at_t = (
                    results['S'][a][t_idx] +
                    results['E'][a][t_idx] +
                    results['I'][a][t_idx] +
                    results['X'][a][t_idx] +
                    results['H_ward'][a][t_idx] +
                    results['H_icu'][a][t_idx] +
                    results['R'][a][t_idx] +
                    results['D'][a][t_idx]
                )
                assert pop_at_t == pytest.approx(age_pop, rel=0.01)
    
    def test_population_conserved_long_simulation(self, long_simulation_inputs):
        """Population conservation should hold for long simulations."""
        results = simulate_master_hospital_model(**long_simulation_inputs)
        total_pop = sum(long_simulation_inputs['age_pops'])
        n_ages = len(long_simulation_inputs['age_pops'])
        
        # Check at end of simulation
        t_idx = -1
        pop_at_t = sum(
            results['S'][a][t_idx] +
            results['E'][a][t_idx] +
            results['I'][a][t_idx] +
            results['X'][a][t_idx] +
            results['H_ward'][a][t_idx] +
            results['H_icu'][a][t_idx] +
            results['R'][a][t_idx] +
            results['D'][a][t_idx]
            for a in range(n_ages)
        )
        # Larger tolerance for longer simulations with larger time step
        assert pop_at_t == pytest.approx(total_pop, rel=0.02)


# ========================================
# Death Monotonicity Tests
# ========================================

class TestDeathMonotonicity:
    """Tests for death monotonicity (deaths can only increase)."""
    
    def test_deaths_monotonically_increasing(self, minimal_inputs):
        """Total deaths should be monotonically non-decreasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        D_total = results['D_total']
        
        for i in range(1, len(D_total)):
            assert D_total[i] >= D_total[i-1] - 1e-10, \
                f"Deaths decreased at time {results['times'][i]}"
    
    def test_deaths_per_age_monotonically_increasing(self, minimal_inputs):
        """Deaths per age group should be monotonically non-decreasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        n_ages = len(minimal_inputs['age_pops'])
        
        for a in range(n_ages):
            D_age = results['D'][a]
            for i in range(1, len(D_age)):
                assert D_age[i] >= D_age[i-1] - 1e-10
    
    def test_treated_deaths_monotonically_increasing(self, minimal_inputs):
        """Treated deaths should be monotonically non-decreasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        D_treated = results['D_treated_total']
        
        for i in range(1, len(D_treated)):
            assert D_treated[i] >= D_treated[i-1] - 1e-10
    
    def test_untreated_deaths_monotonically_increasing(self, minimal_inputs):
        """Untreated deaths should be monotonically non-decreasing."""
        results = simulate_master_hospital_model(**minimal_inputs)
        D_untreated = results['D_untreated_total']
        
        for i in range(1, len(D_untreated)):
            assert D_untreated[i] >= D_untreated[i-1] - 1e-10


# ========================================
# Default Parameter Tests
# ========================================

class TestDefaultParameters:
    """Tests for default parameter usage."""
    
    def test_uses_default_tmax(self, minimal_inputs):
        """Should use default Tmax when not specified."""
        results = simulate_master_hospital_model(**minimal_inputs)
        expected_tmax = DEFAULT_SIM_PARAMS['Tmax']
        # Allow small floating point tolerance
        assert results['times'][-1] >= expected_tmax - 0.1
    
    def test_uses_default_time_step(self, minimal_inputs):
        """Should use default time_step when not specified."""
        results = simulate_master_hospital_model(**minimal_inputs)
        times = results['times']
        expected_dt = DEFAULT_SIM_PARAMS['time_step']
        actual_dt = times[1] - times[0]
        assert actual_dt == pytest.approx(expected_dt)
    
    def test_uses_default_capacity(self, minimal_inputs):
        """Should use default capacities when not specified."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert results['ward_capacity'] == DEFAULT_CAPACITY_PARAMS['ward_capacity']
        assert results['icu_capacity'] == DEFAULT_CAPACITY_PARAMS['icu_capacity']
    
    def test_custom_tmax_used(self, minimal_inputs):
        """Custom Tmax should be used when provided."""
        inputs = {**minimal_inputs, 'Tmax': 50}
        results = simulate_master_hospital_model(**inputs)
        # Allow small floating point tolerance
        assert results['times'][-1] >= 50 - 0.2
        assert results['times'][-1] < 60
    
    def test_custom_capacity_used(self, minimal_inputs):
        """Custom capacities should be used when provided."""
        inputs = {**minimal_inputs, 'ward_capacity': 150, 'icu_capacity': 40}
        results = simulate_master_hospital_model(**inputs)
        assert results['ward_capacity'] == 150
        assert results['icu_capacity'] == 40


# ========================================
# Non-Negativity Tests
# ========================================

class TestNonNegativity:
    """Tests that all compartments remain non-negative."""
    
    def test_all_compartments_non_negative(self, minimal_inputs):
        """All compartment values should be >= 0."""
        results = simulate_master_hospital_model(**minimal_inputs)
        compartments = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D']
        n_ages = len(minimal_inputs['age_pops'])
        
        for comp in compartments:
            for a in range(n_ages):
                for t_idx, val in enumerate(results[comp][a]):
                    assert val >= 0, f"{comp}[{a}] negative at t={results['times'][t_idx]}"
    
    def test_gating_factors_in_valid_range(self, minimal_inputs):
        """Gating factors should be in [0, 1]."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for g in results['g_ward']:
            assert 0 <= g <= 1
        for g in results['g_icu']:
            assert 0 <= g <= 1
    
    def test_beta_t_non_negative(self, minimal_inputs):
        """Time-varying beta should be non-negative."""
        results = simulate_master_hospital_model(**minimal_inputs)
        for beta in results['beta_t']:
            assert beta >= 0


# ========================================
# Required Input Validation Tests
# ========================================

class TestInputValidation:
    """Tests for required input validation."""
    
    def test_age_pops_required(self, minimal_inputs):
        """age_pops should be required."""
        del minimal_inputs['age_pops']
        with pytest.raises(ValueError, match="age_pops"):
            simulate_master_hospital_model(**minimal_inputs)
    
    def test_mismatched_dimensions_raises(self, minimal_inputs):
        """Mismatched age_params and age_pops should raise."""
        minimal_inputs['age_pops'] = [1000, 2000]  # Only 2 instead of 3
        with pytest.raises(ValueError):
            simulate_master_hospital_model(**minimal_inputs)


# ========================================
# Differential Mortality Tracking Tests
# ========================================

class TestDifferentialMortalityTracking:
    """Tests for differential mortality tracking option."""
    
    def test_differential_mortality_enabled_by_default(self, minimal_inputs):
        """Differential mortality tracking should be enabled by default."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'D_treated' in results
        assert 'D_untreated' in results
        assert 'D_treated_total' in results
        assert 'D_untreated_total' in results
    
    def test_differential_mortality_can_be_disabled(self, minimal_inputs):
        """Differential mortality tracking can be disabled."""
        inputs = {**minimal_inputs, 'track_differential_mortality': False}
        results = simulate_master_hospital_model(**inputs)
        assert 'D_treated' not in results
        assert 'D_untreated' not in results
    
    def test_d_treated_plus_d_untreated_equals_d_total(self, minimal_inputs):
        """D_treated + D_untreated should approximately equal D_total."""
        results = simulate_master_hospital_model(**minimal_inputs)
        
        for t_idx in range(len(results['times'])):
            d_sum = results['D_treated_total'][t_idx] + results['D_untreated_total'][t_idx]
            d_total = results['D_total'][t_idx]
            # Allow small numerical tolerance
            assert d_sum == pytest.approx(d_total, rel=0.01)


# ========================================
# Compartment Flow Tracking Tests
# ========================================

class TestCompartmentFlowTracking:
    """Tests for compartment flow tracking option."""
    
    def test_flow_tracking_disabled_by_default(self, minimal_inputs):
        """Compartment flow tracking should be disabled by default."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'new_infections' not in results
        assert 'ward_admissions' not in results
        assert 'icu_admissions' not in results
    
    def test_flow_tracking_can_be_enabled(self, minimal_inputs):
        """Compartment flow tracking can be enabled."""
        inputs = {**minimal_inputs, 'track_compartment_flows': True}
        results = simulate_master_hospital_model(**inputs)
        assert 'new_infections' in results
        assert 'ward_admissions' in results
        assert 'icu_admissions' in results
    
    def test_flow_tracking_dimensions(self, minimal_inputs):
        """Flow tracking arrays should have correct dimensions."""
        inputs = {**minimal_inputs, 'track_compartment_flows': True}
        results = simulate_master_hospital_model(**inputs)
        n_ages = len(minimal_inputs['age_pops'])
        n_times = len(results['times'])
        
        # Note: flows are recorded per step, so length may be n_times or n_times-1
        # depending on when they're recorded
        assert len(results['new_infections']) > 0
        assert len(results['new_infections'][0]) == n_ages


# ========================================
# Parameters Metadata Tests
# ========================================

class TestParametersMetadata:
    """Tests for parameters metadata in results."""
    
    def test_parameters_dict_present(self, minimal_inputs):
        """Results should contain parameters dict."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'parameters' in results
        assert isinstance(results['parameters'], dict)
    
    def test_parameters_contains_inputs(self, minimal_inputs):
        """Parameters dict should contain input values."""
        inputs = {
            **minimal_inputs,
            'Tmax': 150,
            'coverage': 0.5,
            'VE': 0.8,
        }
        results = simulate_master_hospital_model(**inputs)
        
        assert results['parameters']['Tmax'] == 150
        assert results['parameters']['VE'] == 0.8
    
    def test_parameters_contains_age_params(self, minimal_inputs):
        """Parameters dict should contain age_params."""
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'age_params' in results['parameters']
