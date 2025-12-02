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
        """
        Verify that the simulation runs successfully with minimal valid inputs.
        
        This is the most basic smoke test ensuring the function executes without
        errors or exceptions. It validates that the simulation can start and complete
        with the bare minimum required parameters.
        
        Expected behavior:
        - Function completes without raising exceptions
        - Returns a non-None result
        - Result is a dictionary containing simulation outputs
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        assert results is not None
        assert isinstance(results, dict)
    
    def test_returns_dict(self, minimal_inputs):
        """
        Verify that the simulation returns results in dictionary format.
        
        The function's API contract specifies that results should be returned
        as a dictionary for easy access to different output components (compartments,
        time series, metadata, etc.).
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        assert isinstance(results, dict)
    
    def test_times_array_present(self, minimal_inputs):
        """
        Verify that results include a non-empty 'times' array.
        
        The 'times' array is fundamental to the simulation output as it provides
        the temporal index for all compartment time series. Without it, the
        time-series data would be meaningless.
        
        Expected behavior:
        - 'times' key exists in results dictionary
        - 'times' array contains at least one time point
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        assert 'times' in results
        assert len(results['times']) > 0
    
    def test_all_compartments_present(self, minimal_inputs):
        """
        Verify that all core SEIXHRD compartments are present in results.
        
        The model implements a SEIXHRD structure:
        - S: Susceptible
        - E: Exposed (infected but not yet infectious)
        - I: Infectious (community cases)
        - X: Severe cases (queued or admitted for hospitalization)
        - H_ward: Ward hospitalizations
        - H_icu: ICU hospitalizations
        - H: Total hospitalizations (H_ward + H_icu)
        - R: Recovered
        - D: Deaths
        
        All of these compartments must be present for the model to be complete.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D', 'H']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_aggregated_totals_present(self, minimal_inputs):
        """
        Verify that age-aggregated totals are present in results.
        
        For convenience and analysis, the simulation provides pre-aggregated
        totals that sum across all age groups. These are essential for:
        - Quick visualization of overall epidemic dynamics
        - Comparing total burden across scenarios
        - Avoiding repeated summation in downstream analysis
        
        Each total represents the sum across all age groups for that compartment.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = [
            'H_ward_total', 'H_icu_total', 'H_total',
            'E_total', 'I_total', 'X_total', 'D_total'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_capacity_metrics_present(self, minimal_inputs):
        """
        Verify that hospital capacity metrics are present in results.
        
        The model tracks healthcare system strain through several metrics:
        - ward_overflow: Instantaneous ward demand exceeding capacity
        - icu_overflow: Instantaneous ICU demand exceeding capacity
        - cum_ward_overflow: Cumulative ward overflow (person-days)
        - cum_icu_overflow: Cumulative ICU overflow (person-days)
        - g_ward: Ward gating factor (fraction of demand that can be met)
        - g_icu: ICU gating factor (fraction of demand that can be met)
        
        These metrics are critical for assessing healthcare system resilience
        and the impact of capacity constraints on patient outcomes.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = [
            'ward_overflow', 'icu_overflow',
            'cum_ward_overflow', 'cum_icu_overflow',
            'g_ward', 'g_icu'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_time_varying_params_present(self, minimal_inputs):
        """
        Verify that time-varying parameter arrays are present in results.
        
        The model supports time-varying transmission dynamics through:
        - beta_t: Effective transmission rate at each time point
        - seasonal_factor: Seasonal modulation of transmission
        - policy_mult: Policy intervention multiplier (e.g., lockdowns)
        
        These arrays allow reconstruction of the exact transmission conditions
        at each point in the simulation, which is essential for understanding
        how interventions and seasonality affect epidemic dynamics.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        required_keys = ['beta_t', 'seasonal_factor', 'policy_mult']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_metadata_present(self, minimal_inputs):
        """
        Verify that simulation metadata is included in results.
        
        Metadata provides context for interpreting simulation results:
        - ward_capacity: Total ward bed capacity used in simulation
        - icu_capacity: Total ICU bed capacity used in simulation
        - age_pops: Population size for each age group
        - parameters: Complete parameter set used in simulation
        
        This metadata is essential for:
        - Reproducibility (knowing exact parameters used)
        - Interpretation (understanding capacity constraints)
        - Comparison (ensuring consistent parameters across runs)
        """
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
        """
        Verify that each compartment has the correct age stratification.
        
        The model is age-stratified, meaning each compartment is subdivided
        by age group. This test ensures that every compartment has exactly
        n_ages sub-arrays, one for each age group.
        
        This is a structural invariant: if age dimensions are wrong, all
        downstream analysis will fail or produce incorrect results.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        for compartment in ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D']:
            assert len(results[compartment]) == n_ages
    
    def test_compartment_time_series_length(self, minimal_inputs):
        """
        Verify that all compartment time series match the length of the times array.
        
        Each compartment's time series (for each age group) must have the same
        number of points as the 'times' array. This ensures proper alignment
        between time points and compartment values.
        
        Mismatched lengths would indicate a bug in the ODE solver integration
        or the output collection logic.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        n_times = len(results['times'])
        for compartment in ['S', 'E', 'I', 'X', 'H_ward', 'H_icu', 'R', 'D']:
            for age_series in results[compartment]:
                assert len(age_series) == n_times
    
    def test_aggregated_totals_length(self, minimal_inputs):
        """
        Verify that aggregated total arrays match the length of the times array.
        
        Age-aggregated totals (summed across all age groups) must have the
        same temporal resolution as the individual compartments. This ensures
        consistency between age-stratified and aggregated views of the data.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        n_times = len(results['times'])
        for key in ['H_ward_total', 'H_icu_total', 'H_total', 'D_total']:
            assert len(results[key]) == n_times
    
    def test_times_are_monotonic(self, minimal_inputs):
        """
        Verify that the times array is strictly monotonically increasing.
        
        Time must always move forward in a simulation. This test ensures that:
        - No time points are duplicated
        - Time never goes backwards
        - The ODE solver is progressing correctly
        
        Violations would indicate a serious bug in the solver or time step logic.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        times = results['times']
        for i in range(1, len(times)):
            assert times[i] > times[i-1]
    
    def test_times_start_at_zero(self, minimal_inputs):
        """
        Verify that the simulation starts at time t=0.
        
        By convention, all simulations start at t=0. This provides a consistent
        reference point for interpreting results and comparing across scenarios.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        assert results['times'][0] == 0
    
    def test_times_end_near_tmax(self, minimal_inputs):
        """
        Verify that the simulation runs until approximately Tmax.
        
        The simulation should continue until it reaches or slightly exceeds
        the specified maximum time (Tmax). A small tolerance is allowed for
        floating-point arithmetic and adaptive time stepping.
        
        This ensures the simulation runs for the full requested duration.
        """
        inputs = {**minimal_inputs, 'Tmax': 100, 'time_step': 0.1}
        results = simulate_master_hospital_model(**inputs)
        # Allow small floating point tolerance
        assert results['times'][-1] >= 100 - 0.1
    
    def test_h_equals_h_ward_plus_h_icu(self, minimal_inputs):
        """
        Verify that total hospitalizations equal ward plus ICU hospitalizations.
        
        This is a fundamental accounting identity: H = H_ward + H_icu.
        
        The model tracks ward and ICU hospitalizations separately, but also
        provides a total H for convenience. This test ensures these values
        are consistent at every time point and for every age group.
        
        Violations would indicate a bug in the aggregation logic.
        """
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
        """
        Verify that total population remains constant throughout the simulation.
        
        This is the fundamental conservation law for a closed population system:
        S + E + I + X + H_ward + H_icu + R + D = N (constant)
        
        This ensures there are no 'leaks' in the ODE system where individuals
        are created or destroyed unaccounted for. Every person must be in exactly
        one compartment at all times.
        
        Mathematical basis:
        If the sum of all compartment derivatives equals zero (dS + dE + dI + ... = 0),
        then the total population is conserved. This test verifies the numerical
        implementation maintains this property.
        
        Note: A small tolerance (1%) is allowed for numerical integration errors
        that accumulate over time with the Euler method.
        """
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
        """
        Verify that population is conserved within each age group independently.
        
        In addition to total population conservation, the model should conserve
        population within each age group. Individuals do not age or move between
        age groups during the simulation.
        
        This is a stronger invariant than total conservation: it ensures that
        the age structure is preserved and that there are no cross-age leaks
        in the implementation.
        
        For each age group a: S_a + E_a + I_a + X_a + H_a + R_a + D_a = N_a (constant)
        """
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
        """
        Verify that population conservation holds even for extended simulations.
        
        This is a stress test for numerical stability. Longer simulations with
        larger time steps accumulate more numerical error. This test ensures
        that population conservation is maintained even under these conditions.
        
        A slightly larger tolerance (2%) is allowed to account for:
        - Accumulated integration errors over many time steps
        - Potentially larger time steps in long simulations
        - Floating-point arithmetic limitations
        
        If this test fails but shorter simulations pass, it indicates numerical
        instability that may require smaller time steps or a higher-order integrator.
        """
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
        """
        Verify that total deaths are monotonically non-decreasing over time.
        
        This is a fundamental physical constraint: once someone has died, they
        cannot un-die. The cumulative death count (D) must never decrease.
        
        Mathematically: dD/dt >= 0 for all t
        
        This test ensures:
        - The death compartment is implemented as a cumulative tracker
        - There are no bugs causing negative death flows
        - Numerical errors don't cause backwards time evolution
        
        A small tolerance (1e-10) allows for floating-point rounding errors
        while still catching any real violations of monotonicity.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        D_total = results['D_total']
        
        for i in range(1, len(D_total)):
            assert D_total[i] >= D_total[i-1] - 1e-10, \
                f"Deaths decreased at time {results['times'][i]}"
    
    def test_deaths_per_age_monotonically_increasing(self, minimal_inputs):
        """
        Verify that deaths are monotonically non-decreasing within each age group.
        
        This is a stronger test than total death monotonicity. Not only must
        total deaths increase, but deaths within each age group must also
        increase independently.
        
        This ensures:
        - Age-stratified death tracking is implemented correctly
        - There are no cross-age accounting errors
        - Each age group's death compartment behaves properly
        
        For each age group a: D_a(t+1) >= D_a(t)
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        n_ages = len(minimal_inputs['age_pops'])
        
        for a in range(n_ages):
            D_age = results['D'][a]
            for i in range(1, len(D_age)):
                assert D_age[i] >= D_age[i-1] - 1e-10
    
    def test_treated_deaths_monotonically_increasing(self, minimal_inputs):
        """
        Verify that treated deaths are monotonically non-decreasing.
        
        The model tracks deaths by treatment status (treated vs untreated).
        Treated deaths include those who died while receiving medical care:
        - Community deaths from I (treated by definition for mild cases)
        - Deaths from X_admitted (secured hospital bed)
        - Deaths in H_ward (ward care)
        - Deaths in H_icu (ICU care)
        
        This compartment must also be monotonically increasing, as it's a
        cumulative tracker of a specific subset of deaths.
        """
        results = simulate_master_hospital_model(**minimal_inputs)
        D_treated = results['D_treated_total']
        
        for i in range(1, len(D_treated)):
            assert D_treated[i] >= D_treated[i-1] - 1e-10
    
    def test_untreated_deaths_monotonically_increasing(self, minimal_inputs):
        """
        Verify that untreated deaths are monotonically non-decreasing.
        
        Untreated deaths include those who died without adequate medical care:
        - Deaths from X_queued (waiting for hospital bed, none available)
        - ICU-denied deaths from H_ward (needed ICU but none available)
        
        These deaths represent the impact of healthcare system capacity constraints.
        Like all death compartments, this must be monotonically increasing.
        
        This test is particularly important for validating the capacity-constrained
        dynamics of the model, as untreated deaths only occur when the system
        is overwhelmed.
        """
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
# Force of Infection Directionality Test
# ========================================

class TestForceOfInfectionDirectionality:
    """Ensure the contact matrix orientation matches infector→infectee convention."""
    
    def test_force_of_infection_directionality(self):
        """
        With asymmetric contacts (infector rows, infectee columns), only inbound contacts
        should drive infections. Group 0 infects, Group 1 is susceptible. Since
        contact_matrix[0,1] = 0, Group 1 should never get infected.
        """
        age_params = AGE_PARAMS_DEFAULT[:2]
        contact_matrix = np.array([
            [5.0, 0.0],   # Infector 0 -> Infectee 1 is zero
            [10.0, 1.0],  # Infector 1 -> Infectee 0 is high (asymmetric)
        ])
        age_pops = [1000, 1000]
        initial_conditions = {
            'E_by_age': [0, 0],
            'I_by_age': [10, 0],  # Seed infections only in Group 0
            'X_by_age': [0, 0],
            'H_ward_by_age': [0, 0],
            'H_icu_by_age': [0, 0],
            'R_by_age': [0, 0],
            'D_by_age': [0, 0],
        }
        
        results = simulate_master_hospital_model(
            beta_base=0.3,
            age_params=age_params,
            contact_matrix=contact_matrix,
            age_pops=age_pops,
            Tmax=60,
            time_step=0.1,
            initial_conditions=initial_conditions,
        )
        
        infections_group_one = results['I'][1]
        assert np.max(infections_group_one) <= 1e-6, \
            "Group 1 should remain uninfected when it receives zero inbound contacts."


# ========================================
# Required Input Validation Tests
# ========================================

class TestInputValidation:
    """Tests for required input validation."""
    
    def test_age_pops_required(self, minimal_inputs):
        """age_pops should be required."""
        del minimal_inputs['age_pops']
        with pytest.raises(TypeError, match="age_pops"):
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
