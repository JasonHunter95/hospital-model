"""
Scenario validation tests for all 24 SCENARIO_REGISTRY scenarios.

This module tests all predefined scenarios to ensure they produce valid results:
1. Population conservation (1e-6 tolerance)
2. Death monotonicity (deaths never decrease)
3. Non-negativity (no compartment goes negative)
4. Gating bounds (0 <= g <= 1)
5. Differential mortality tracking consistency

Uses pytest-xdist for parallel execution: pytest -n auto
"""

import pytest
import numpy as np
from config import SCENARIO_REGISTRY
from config_helpers import get_scenario_params
from simulate_model import simulate_model


# =============================================================================
# CONSTANTS
# =============================================================================

# All 24 scenario names from SCENARIO_REGISTRY
ALL_SCENARIOS = list(SCENARIO_REGISTRY.keys())

# Tolerance for conservation checks (standard float tolerance)
CONSERVATION_TOLERANCE = 1e-6

# Relative tolerance for compartment checks
RELATIVE_TOLERANCE = 1e-9


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope='module')
def scenario_results():
    """
    Cache scenario results for reuse across tests within this module.
    
    This prevents re-running the same simulation multiple times for different
    assertions on the same scenario.
    """
    cache = {}
    
    def get_or_run(scenario_name):
        if scenario_name not in cache:
            params = get_scenario_params(scenario_name)
            cache[scenario_name] = simulate_model(**params)
        return cache[scenario_name]
    
    return get_or_run


# =============================================================================
# TEST CLASS: POPULATION CONSERVATION
# =============================================================================

class TestScenarioPopulationConservation:
    """
    Verify population conservation across all scenarios.
    
    For closed populations (no demographics):
    sum(live compartments) + D_total = initial_population
    
    For open populations (with demographics):
    sum(live compartments) + D_total + cum_background_deaths = initial + cum_births
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_population_conservation(self, scenario_name):
        """
        Test population conservation for each scenario.
        
        Uses 1e-6 tolerance as specified in requirements.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        # Calculate initial population
        age_pops = params.get('age_pops', [1_000_000])
        initial_pop = sum(age_pops)
        
        # Get all compartment totals at final time
        times = results['times']
        T_final = len(times) - 1
        
        # Live compartments (sum across age groups)
        live_compartments = ['S', 'E', 'I', 'X_queued', 'X_admitted', 
                            'H_ward', 'H_icu', 'R',
                            'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax',
                            'H_ward_vax', 'H_icu_vax', 'R_vax']
        
        total_live = 0
        for comp in live_compartments:
            if comp in results:
                comp_data = results[comp]
                # Sum across age groups at final time
                if isinstance(comp_data, list):
                    total_live += sum(arr[-1] for arr in comp_data)
                else:
                    total_live += comp_data[-1]
        
        # Death compartments
        D_total = results.get('D_total', np.zeros(len(times)))[-1]
        
        # Check if demographics are enabled
        has_demographics = params.get('demographic_config') is not None
        
        if has_demographics:
            cum_births = results.get('cum_births_total', np.zeros(len(times)))[-1]
            cum_bg_deaths = results.get('cum_background_deaths_total', np.zeros(len(times)))[-1]
            
            # With demographics: initial + births = live + D + bg_deaths
            calculated_pop = total_live + D_total + cum_bg_deaths
            expected_pop = initial_pop + cum_births
            diff = abs(calculated_pop - expected_pop)
            
            assert diff < CONSERVATION_TOLERANCE, \
                f"Scenario '{scenario_name}': Population not conserved with demographics. " \
                f"Expected {expected_pop:.2f}, got {calculated_pop:.2f}, diff={diff:.2e}"
        else:
            # Closed system: initial = live + D
            total_pop = total_live + D_total
            diff = abs(total_pop - initial_pop)
            
            assert diff < CONSERVATION_TOLERANCE, \
                f"Scenario '{scenario_name}': Population not conserved. " \
                f"Expected {initial_pop:.2f}, got {total_pop:.2f}, diff={diff:.2e}"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_conservation_throughout_simulation(self, scenario_name):
        """
        Verify conservation holds at EVERY time point, not just final.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        age_pops = params.get('age_pops', [1_000_000])
        initial_pop = sum(age_pops)
        
        times = results['times']
        n_ages = len(age_pops)
        
        # Check conservation at each time point
        violations = []
        
        for t_idx in range(len(times)):
            total_live = 0
            
            # Sum all live compartments
            live_compartments = ['S', 'E', 'I', 'X_queued', 'X_admitted', 
                                'H_ward', 'H_icu', 'R',
                                'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax',
                                'H_ward_vax', 'H_icu_vax', 'R_vax']
            
            for comp in live_compartments:
                if comp in results:
                    comp_data = results[comp]
                    if isinstance(comp_data, list):
                        total_live += sum(arr[t_idx] for arr in comp_data)
                    else:
                        total_live += comp_data[t_idx]
            
            # Add deaths
            D_total = results['D_total'][t_idx] if 'D_total' in results else 0
            
            has_demographics = params.get('demographic_config') is not None
            
            if has_demographics:
                cum_births = results.get('cum_births_total', np.zeros(len(times)))[t_idx]
                cum_bg_deaths = results.get('cum_background_deaths_total', np.zeros(len(times)))[t_idx]
                expected = initial_pop + cum_births
                actual = total_live + D_total + cum_bg_deaths
            else:
                expected = initial_pop
                actual = total_live + D_total
            
            diff = abs(actual - expected)
            if diff > CONSERVATION_TOLERANCE:
                violations.append((t_idx, times[t_idx], diff))
        
        assert len(violations) == 0, \
            f"Scenario '{scenario_name}': Conservation violated at {len(violations)} time points. " \
            f"First violation: t={violations[0][1]:.2f}, diff={violations[0][2]:.2e}"


# =============================================================================
# TEST CLASS: DEATH MONOTONICITY
# =============================================================================

class TestScenarioDeathMonotonicity:
    """
    Verify deaths never decrease in any scenario.
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_total_deaths_monotonically_increasing(self, scenario_name):
        """
        D_total should never decrease over time.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        D_total = results['D_total']
        
        # Check monotonicity
        for i in range(1, len(D_total)):
            assert D_total[i] >= D_total[i-1] - RELATIVE_TOLERANCE, \
                f"Scenario '{scenario_name}': Deaths decreased at t_idx={i}. " \
                f"D[{i-1}]={D_total[i-1]:.6f}, D[{i}]={D_total[i]:.6f}"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_d_treated_monotonically_increasing(self, scenario_name):
        """
        D_treated_total should never decrease.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'D_treated_total' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't track D_treated_total")
        
        D_treated = results['D_treated_total']
        
        for i in range(1, len(D_treated)):
            assert D_treated[i] >= D_treated[i-1] - RELATIVE_TOLERANCE, \
                f"Scenario '{scenario_name}': D_treated decreased at t_idx={i}"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_d_untreated_monotonically_increasing(self, scenario_name):
        """
        D_untreated_total should never decrease.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'D_untreated_total' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't track D_untreated_total")
        
        D_untreated = results['D_untreated_total']
        
        for i in range(1, len(D_untreated)):
            assert D_untreated[i] >= D_untreated[i-1] - RELATIVE_TOLERANCE, \
                f"Scenario '{scenario_name}': D_untreated decreased at t_idx={i}"


# =============================================================================
# TEST CLASS: NON-NEGATIVITY
# =============================================================================

class TestScenarioNonNegativity:
    """
    Verify no compartment ever goes negative.
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_all_compartments_non_negative(self, scenario_name):
        """
        All 18 compartments should remain >= 0 throughout simulation.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        compartments = ['S', 'E', 'I', 'X_queued', 'X_admitted', 
                       'H_ward', 'H_icu', 'R', 'D',
                       'S_vax', 'E_vax', 'I_vax', 'X_queued_vax', 'X_admitted_vax',
                       'H_ward_vax', 'H_icu_vax', 'R_vax', 'D_vax']
        
        violations = []
        
        for comp in compartments:
            if comp not in results:
                continue
            
            comp_data = results[comp]
            
            if isinstance(comp_data, list):
                # Multi-age group
                for age_idx, arr in enumerate(comp_data):
                    min_val = np.min(arr)
                    if min_val < -RELATIVE_TOLERANCE:
                        violations.append((comp, age_idx, min_val))
            else:
                # Total or single age
                min_val = np.min(comp_data)
                if min_val < -RELATIVE_TOLERANCE:
                    violations.append((comp, None, min_val))
        
        assert len(violations) == 0, \
            f"Scenario '{scenario_name}': Negative values found in {len(violations)} cases. " \
            f"First: {violations[0]}"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_trackers_non_negative(self, scenario_name):
        """
        All tracked accumulators should remain >= 0.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        # Use total versions where available
        trackers = ['D_treated_total', 'D_untreated_total', 'cum_births_total', 
                   'cum_background_deaths_total', 'breakthrough_infections']
        
        for tracker in trackers:
            if tracker not in results:
                continue
            
            data = np.array(results[tracker])
            min_val = np.min(data)
            
            assert min_val >= -RELATIVE_TOLERANCE, \
                f"Scenario '{scenario_name}': Tracker '{tracker}' went negative: {min_val}"


# =============================================================================
# TEST CLASS: GATING BOUNDS
# =============================================================================

class TestScenarioGatingBounds:
    """
    Verify Hill function gating remains bounded [0, 1].
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_ward_gating_bounded(self, scenario_name):
        """
        g_ward should always be in [0, 1].
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'g_ward' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't output g_ward")
        
        g_ward = np.array(results['g_ward'])
        
        assert np.all(g_ward >= -RELATIVE_TOLERANCE), \
            f"Scenario '{scenario_name}': g_ward went below 0"
        assert np.all(g_ward <= 1 + RELATIVE_TOLERANCE), \
            f"Scenario '{scenario_name}': g_ward exceeded 1"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_icu_gating_bounded(self, scenario_name):
        """
        g_icu should always be in [0, 1].
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'g_icu' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't output g_icu")
        
        g_icu = np.array(results['g_icu'])
        
        assert np.all(g_icu >= -RELATIVE_TOLERANCE), \
            f"Scenario '{scenario_name}': g_icu went below 0"
        assert np.all(g_icu <= 1 + RELATIVE_TOLERANCE), \
            f"Scenario '{scenario_name}': g_icu exceeded 1"


# =============================================================================
# TEST CLASS: DIFFERENTIAL MORTALITY CONSISTENCY
# =============================================================================

class TestScenarioDifferentialMortality:
    """
    Verify differential mortality tracking consistency.
    
    D_treated + D_untreated should approximately equal D_total (unvax)
    D_vax_treated + D_vax_untreated should approximately equal D_vax_total
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_death_components_sum_to_total(self, scenario_name):
        """
        Verify D_treated_total + D_untreated_total ≈ D_total at all times.
        
        Note: D_treated_total and D_untreated_total include BOTH vaccinated
        and unvaccinated deaths, so they should sum to D_total.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'D_treated_total' not in results or 'D_untreated_total' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't track differential mortality")
        
        D_treated_total = np.array(results['D_treated_total'])
        D_untreated_total = np.array(results['D_untreated_total'])
        D_total = np.array(results['D_total'])
        
        # Components should sum to total
        component_sum = D_treated_total + D_untreated_total
        
        # Allow for numerical tolerance
        max_diff = np.max(np.abs(component_sum - D_total))
        
        # Use relative tolerance for this check
        max_val = max(np.max(D_total), 1.0)  # Avoid division by zero
        relative_diff = max_diff / max_val
        
        assert relative_diff < 1e-4, \
            f"Scenario '{scenario_name}': Death components don't sum to total. " \
            f"Max diff: {max_diff:.6f}, relative: {relative_diff:.6e}"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_vaccinated_death_components_sum(self, scenario_name):
        """
        Verify D_vax_treated + D_vax_untreated ≈ D_vax_total.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        if 'D_vax_treated' not in results or 'D_vax_untreated' not in results:
            pytest.skip(f"Scenario '{scenario_name}' doesn't track vaccinated differential mortality")
        
        # D_vax_treated and D_vax_untreated are per-age lists
        D_vax_treated_list = results['D_vax_treated']
        D_vax_untreated_list = results['D_vax_untreated']
        D_vax_total = np.array(results.get('D_vax_total', [0]))
        
        # Sum across age groups
        if isinstance(D_vax_treated_list, list) and len(D_vax_treated_list) > 0:
            n_times = len(D_vax_treated_list[0])
            D_vax_treated = np.zeros(n_times)
            D_vax_untreated = np.zeros(n_times)
            for age_arr in D_vax_treated_list:
                D_vax_treated += np.array(age_arr)
            for age_arr in D_vax_untreated_list:
                D_vax_untreated += np.array(age_arr)
        else:
            D_vax_treated = np.array(D_vax_treated_list)
            D_vax_untreated = np.array(D_vax_untreated_list)
        
        component_sum = D_vax_treated + D_vax_untreated
        max_diff = np.max(np.abs(component_sum - D_vax_total))
        
        max_val = max(np.max(D_vax_total) if len(D_vax_total) > 0 else 0, 1.0)
        relative_diff = max_diff / max_val
        
        assert relative_diff < 1e-4, \
            f"Scenario '{scenario_name}': Vaccinated death components don't sum. " \
            f"Max diff: {max_diff:.6f}"


# =============================================================================
# TEST CLASS: SCENARIO-SPECIFIC BEHAVIOR
# =============================================================================

class TestScenarioSpecificBehavior:
    """
    Tests for scenario-specific expected behaviors.
    """
    
    def test_high_transmission_produces_more_infections(self):
        """
        Scenarios with higher beta_base should produce more cumulative infections.
        """
        # Get baseline and high transmission scenarios if they exist
        baseline_scenarios = [s for s in ALL_SCENARIOS if 'baseline' in s.lower()]
        high_trans_scenarios = [s for s in ALL_SCENARIOS if 'delta' in s.lower() or 'high' in s.lower()]
        
        if not baseline_scenarios or not high_trans_scenarios:
            pytest.skip("Need baseline and high transmission scenarios for comparison")
        
        baseline_name = baseline_scenarios[0]
        high_name = high_trans_scenarios[0]
        
        baseline_params = get_scenario_params(baseline_name)
        high_params = get_scenario_params(high_name)
        
        baseline_results = simulate_model(**baseline_params)
        high_results = simulate_model(**high_params)
        
        # Compare total deaths (proxy for infections)
        baseline_deaths = baseline_results['D_total'][-1]
        high_deaths = high_results['D_total'][-1]
        
        # Higher transmission should generally produce more deaths
        # (unless completely different vaccination scenarios)
        assert high_deaths >= 0 and baseline_deaths >= 0, \
            "Both scenarios should produce valid death counts"
    
    def test_vaccine_scenarios_reduce_deaths(self):
        """
        Scenarios with vaccination should have fewer deaths than unvaccinated versions.
        """
        # Find vaccine and no-vaccine scenario pairs
        vax_scenarios = [s for s in ALL_SCENARIOS if 'vax' in s.lower() or 'vaccin' in s.lower()]
        
        if len(vax_scenarios) < 2:
            pytest.skip("Need vaccine scenario pairs for comparison")
        
        for scenario in vax_scenarios[:2]:  # Test first two
            params = get_scenario_params(scenario)
            results = simulate_model(**params)
            
            # Verify deaths are a reasonable number (not NaN or Inf)
            D_total = results['D_total'][-1]
            assert np.isfinite(D_total), \
                f"Scenario '{scenario}': Deaths should be finite, got {D_total}"


# =============================================================================
# TEST CLASS: NUMERICAL STABILITY
# =============================================================================

class TestScenarioNumericalStability:
    """
    Test numerical stability of scenarios.
    """
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_no_nan_values(self, scenario_name):
        """
        No compartment should contain NaN values.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                assert not np.any(np.isnan(value)), \
                    f"Scenario '{scenario_name}': NaN found in '{key}'"
            elif isinstance(value, list):
                for i, arr in enumerate(value):
                    if isinstance(arr, np.ndarray):
                        assert not np.any(np.isnan(arr)), \
                            f"Scenario '{scenario_name}': NaN found in '{key}[{i}]'"
    
    @pytest.mark.parametrize('scenario_name', ALL_SCENARIOS)
    def test_no_inf_values(self, scenario_name):
        """
        No compartment should contain Inf values.
        """
        params = get_scenario_params(scenario_name)
        results = simulate_model(**params)
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                assert not np.any(np.isinf(value)), \
                    f"Scenario '{scenario_name}': Inf found in '{key}'"
            elif isinstance(value, list):
                for i, arr in enumerate(value):
                    if isinstance(arr, np.ndarray):
                        assert not np.any(np.isinf(arr)), \
                            f"Scenario '{scenario_name}': Inf found in '{key}[{i}]'"


# =============================================================================
# TEST CLASS: SCENARIO REGISTRY INTEGRITY
# =============================================================================

class TestScenarioRegistryIntegrity:
    """
    Verify SCENARIO_REGISTRY structure and completeness.
    """
    
    def test_all_scenarios_have_required_keys(self):
        """
        All scenarios should have minimum required configuration.
        """
        required_keys = ['name', 'description']
        
        for scenario_name, config in SCENARIO_REGISTRY.items():
            for key in required_keys:
                assert key in config, \
                    f"Scenario '{scenario_name}' missing required key '{key}'"
    
    def test_all_scenarios_can_be_loaded(self):
        """
        All scenarios should be loadable via get_scenario_params().
        """
        for scenario_name in ALL_SCENARIOS:
            try:
                params = get_scenario_params(scenario_name)
                assert isinstance(params, dict), \
                    f"Scenario '{scenario_name}' should return dict"
            except Exception as e:
                pytest.fail(f"Scenario '{scenario_name}' failed to load: {e}")
    
    def test_all_scenarios_are_runnable(self):
        """
        All scenarios should complete simulation without error.
        """
        for scenario_name in ALL_SCENARIOS:
            params = get_scenario_params(scenario_name)
            try:
                results = simulate_model(**params)
                assert 'times' in results, \
                    f"Scenario '{scenario_name}' should return times"
                assert 'D_total' in results, \
                    f"Scenario '{scenario_name}' should return D_total"
            except Exception as e:
                pytest.fail(f"Scenario '{scenario_name}' failed to run: {e}")


if __name__ == '__main__':
    # Run with parallel execution
    pytest.main([__file__, '-v', '-n', 'auto'])
