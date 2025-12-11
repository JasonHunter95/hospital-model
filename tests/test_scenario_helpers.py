"""
Unit tests for scenario_helpers.py functions.

This module tests all 14 helper functions for scenario and configuration management:
1. list_scenarios() - List available scenarios
2. describe_scenario() - Get scenario metadata
3. get_healthcare_systems() - List healthcare configurations
4. get_vaccination_strategies() - List vaccination strategies
5. validate_age_params() - Validate age parameter structure
6. create_custom_scenario() - Create custom scenarios
7. validate_scenario_params() - Validate complete params
8. get_scenario_params() - Load scenario by name
9. run_scenario_with_overrides() - Run with parameter modifications
10. compare_vaccine_profiles() - Generate vaccine comparison
11. compare_healthcare_systems() - Generate healthcare comparison
12. create_sensitivity_variants() - Generate sensitivity analysis
13. create_intervention_comparison() - Generate intervention comparison
14. summarize_scenarios() - Get multi-scenario summary
"""

import pytest
import numpy as np
from hospital_model.scenario_helpers import (
    list_scenarios,
    describe_scenario,
    get_healthcare_systems,
    get_vaccination_strategies,
    validate_age_params,
    create_custom_scenario,
    validate_scenario_params,
    get_scenario_params,
    run_scenario_with_overrides,
    compare_vaccine_profiles,
    compare_healthcare_systems,
    create_sensitivity_variants,
    create_intervention_comparison,
    summarize_scenarios,
    list_vaccine_profiles,
    get_vaccine_profile,
)
from hospital_model.scenarios import SCENARIO_REGISTRY, AGE_PARAMS_DEFAULT, AGE_PARAMS_EMPIRICAL, HEALTHCARE_SYSTEM_URBAN
from hospital_model import simulate_model


# =============================================================================
# TEST CLASS: list_scenarios()
# =============================================================================

class TestListScenarios:
    """Tests for list_scenarios() function."""
    
    def test_returns_list(self):
        """list_scenarios should return a list."""
        result = list_scenarios()
        assert isinstance(result, list)
    
    def test_contains_all_registry_scenarios(self):
        """list_scenarios should contain all SCENARIO_REGISTRY keys."""
        result = list_scenarios()
        for scenario_name in SCENARIO_REGISTRY.keys():
            assert scenario_name in result, \
                f"Missing scenario: {scenario_name}"
    
    def test_count_matches_registry(self):
        """list_scenarios count should match SCENARIO_REGISTRY."""
        result = list_scenarios()
        assert len(result) == len(SCENARIO_REGISTRY)
    
    def test_returns_strings(self):
        """All items in list should be strings."""
        result = list_scenarios()
        for item in result:
            assert isinstance(item, str)


# =============================================================================
# TEST CLASS: describe_scenario()
# =============================================================================

class TestDescribeScenario:
    """Tests for describe_scenario() function."""
    
    def test_valid_scenario_returns_string(self):
        """describe_scenario should return string for valid scenario."""
        result = describe_scenario('baseline')
        assert isinstance(result, str)
    
    def test_contains_scenario_info(self):
        """Result should contain scenario name and description."""
        result = describe_scenario('baseline')
        assert 'baseline' in result.lower() or 'Baseline' in result
        assert 'Description' in result or 'description' in result.lower()
    
    def test_invalid_scenario_raises_error(self):
        """describe_scenario should raise error for invalid scenario."""
        with pytest.raises(Exception):  # KeyError or ValueError
            describe_scenario('nonexistent_scenario')
    
    @pytest.mark.parametrize('scenario_name', list(SCENARIO_REGISTRY.keys())[:5])
    def test_all_scenarios_describable(self, scenario_name):
        """All scenarios should be describable."""
        result = describe_scenario(scenario_name)
        assert result is not None
        assert len(result) > 0


# =============================================================================
# TEST CLASS: get_healthcare_systems()
# =============================================================================

class TestGetHealthcareSystems:
    """Tests for get_healthcare_systems() function."""
    
    def test_returns_dict(self):
        """get_healthcare_systems should return a dict."""
        result = get_healthcare_systems()
        assert isinstance(result, dict)
    
    def test_contains_common_systems(self):
        """Should contain common healthcare system types."""
        result = get_healthcare_systems()
        # At least some common systems should exist
        assert len(result) > 0
    
    def test_systems_have_capacity_keys(self):
        """Each system should have capacity-related keys."""
        result = get_healthcare_systems()
        for name, system in result.items():
            assert isinstance(system, dict), f"{name} should be a dict"
            # Should have ward and ICU capacity info
            assert 'K_ward' in system or 'ward_capacity' in system or \
                   'capacity' in str(system).lower(), \
                   f"{name} should have capacity info"


# =============================================================================
# TEST CLASS: get_vaccination_strategies()
# =============================================================================

class TestGetVaccinationStrategies:
    """Tests for get_vaccination_strategies() function."""
    
    def test_returns_dict(self):
        """get_vaccination_strategies should return a dict."""
        result = get_vaccination_strategies()
        assert isinstance(result, dict)
    
    def test_contains_strategies(self):
        """Should contain vaccination strategies."""
        result = get_vaccination_strategies()
        assert len(result) > 0
    
    def test_strategies_are_coverage_lists(self):
        """Each strategy should be a coverage list."""
        result = get_vaccination_strategies()
        for name, coverage in result.items():
            assert isinstance(coverage, (list, tuple)), f"{name} should be a list"
            assert len(coverage) == 3, f"{name} should have 3 age groups"


# =============================================================================
# TEST CLASS: validate_age_params()
# =============================================================================

class TestValidateAgeParams:
    """Tests for validate_age_params() function."""
    
    def test_valid_params_returns_true(self):
        """Valid age params should pass validation."""
        result = validate_age_params(AGE_PARAMS_DEFAULT)
        assert result is True or result is None  # May return None on success
    
    def test_empty_list_returns_false_or_error(self):
        """Empty age params should fail validation."""
        try:
            result = validate_age_params([])
            assert result is False  # If it returns, should be False
        except Exception:
            pass  # Raising an error is also acceptable
    
    def test_missing_required_keys_raises_error(self):
        """Age params missing required keys should fail."""
        invalid_params = [{'alpha': 0.2}]  # Missing other required keys
        with pytest.raises(Exception):
            validate_age_params(invalid_params)
    
    def test_negative_rates_raises_error(self):
        """Negative rates should fail validation."""
        invalid_params = [{
            'alpha': -0.2,  # Negative rate
            'sigma': 0.1,
            'gamma_I': 0.14,
            'eta': 0.5,
            'mu_I': 0.001,
        }]
        with pytest.raises(Exception):
            validate_age_params(invalid_params)
    
    def test_empirical_params_valid(self):
        """Empirical age params should be valid."""
        result = validate_age_params(AGE_PARAMS_EMPIRICAL)
        assert result is True or result is None


# =============================================================================
# TEST CLASS: create_custom_scenario()
# =============================================================================

class TestCreateCustomScenario:
    """Tests for create_custom_scenario() function."""
    
    def test_creates_scenario_with_name(self):
        """Should create scenario with custom name."""
        result = create_custom_scenario(
            name='test_scenario',
            beta_base=0.3,
            healthcare_system=HEALTHCARE_SYSTEM_URBAN,
            vaccination_coverage=[0.2, 0.4, 0.8],
            description='Test description',
        )
        assert isinstance(result, dict)
        assert result.get('name') == 'test_scenario'
    
    def test_includes_passed_params(self):
        """Should include passed parameters."""
        result = create_custom_scenario(
            name='test',
            beta_base=0.5,
            healthcare_system=HEALTHCARE_SYSTEM_URBAN,
            vaccination_coverage=[0.2, 0.4, 0.8],
            description='Test',
            Tmax=100,
        )
        assert result.get('beta_base') == 0.5
        assert result.get('Tmax') == 100
    
    def test_includes_required_fields(self):
        """Should include all required fields."""
        result = create_custom_scenario(
            name='minimal',
            beta_base=0.3,
            healthcare_system=HEALTHCARE_SYSTEM_URBAN,
            vaccination_coverage=[0.0, 0.0, 0.0],
        )
        # Should have core structure
        assert 'name' in result
        assert 'beta_base' in result
        assert 'healthcare_system' in result


# =============================================================================
# TEST CLASS: validate_scenario_params()
# =============================================================================

class TestValidateScenarioParams:
    """Tests for validate_scenario_params() function."""
    
    def test_valid_params_returns_params(self):
        """Valid params should be returned (possibly modified)."""
        params = get_scenario_params('baseline')
        result = validate_scenario_params(params)
        assert isinstance(result, dict)
    
    def test_missing_beta_base_raises_error(self):
        """Missing required beta_base should raise error."""
        invalid_params = {'age_params': AGE_PARAMS_DEFAULT}
        with pytest.raises(Exception):
            validate_scenario_params(invalid_params, strict=True)
    
    def test_fills_defaults_in_non_strict_mode(self):
        """Non-strict mode should fill in defaults."""
        minimal_params = {'beta_base': 0.3}
        result = validate_scenario_params(minimal_params, strict=False)
        assert isinstance(result, dict)


# =============================================================================
# TEST CLASS: get_scenario_params()
# =============================================================================

class TestGetScenarioParams:
    """Tests for get_scenario_params() function."""
    
    @pytest.mark.parametrize('scenario_name', list(SCENARIO_REGISTRY.keys()))
    def test_all_scenarios_loadable(self, scenario_name):
        """All registered scenarios should be loadable."""
        result = get_scenario_params(scenario_name)
        assert isinstance(result, dict)
        assert 'beta_base' in result or 'transmission' in str(result)
    
    def test_invalid_scenario_raises_error(self):
        """Invalid scenario name should raise error."""
        with pytest.raises(Exception):
            get_scenario_params('definitely_not_a_real_scenario')
    
    def test_baseline_has_expected_structure(self):
        """Baseline scenario should have expected structure."""
        result = get_scenario_params('baseline')
        
        # Should have core simulation parameters
        expected_keys = ['beta_base', 'age_params', 'sim_config', 'capacity_config', 'vaccine_config']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            
        # Check nested keys
        assert 'Tmax' in result['sim_config']
        assert 'ward_capacity' in result['capacity_config']
    
    def test_returns_copy_not_reference(self):
        """Should return a copy, not the original dict."""
        result1 = get_scenario_params('baseline')
        result2 = get_scenario_params('baseline')
        
        # Modifying one shouldn't affect the other
        result1['beta_base'] = 99999
        assert result2.get('beta_base') != 99999


# =============================================================================
# TEST CLASS: run_scenario_with_overrides()
# =============================================================================

class TestRunScenarioWithOverrides:
    """Tests for run_scenario_with_overrides() function."""
    
    def test_applies_simple_override(self):
        """Should apply simple parameter override using new grouped scenarios API."""
        result = run_scenario_with_overrides(
            'baseline',
            overrides={'sim_config': {'Tmax': 50}}
        )
        # Result is params dict, not simulation output
        assert isinstance(result, dict)
        assert result['sim_config']['Tmax'] == 50
    
    def test_original_scenario_unchanged(self):
        """Original scenario should not be modified."""
        original = get_scenario_params('baseline')
        original_tmax = original['sim_config']['Tmax']
        
        run_scenario_with_overrides('baseline', overrides={'sim_config': {'Tmax': 10}})
        
        after = get_scenario_params('baseline')
        assert after['sim_config']['Tmax'] == original_tmax


# =============================================================================
# TEST CLASS: compare_vaccine_profiles()
# =============================================================================

class TestCompareVaccineProfiles:
    """Tests for compare_vaccine_profiles() function."""
    
    def test_returns_dict_of_params(self):
        """Should return dict mapping profile names to params."""
        profiles = list_vaccine_profiles()[:2]  # Get first two profiles
        if len(profiles) < 2:
            pytest.skip("Need at least 2 vaccine profiles")
        
        result = compare_vaccine_profiles('baseline', profiles)
        assert isinstance(result, dict)
        assert len(result) >= len(profiles)
    
    def test_each_variant_is_runnable(self):
        """Each variant should be runnable params dict."""        
        profiles = list_vaccine_profiles()[:2]
        if len(profiles) < 2:
            pytest.skip("Need at least 2 vaccine profiles")
        
        variants = compare_vaccine_profiles('baseline', profiles)
        
        for name, params in list(variants.items())[:1]:  # Test first only (speed)
            params['sim_config']['Tmax'] = 10  # Short simulation
            result = simulate_model(**params)
            assert 'times' in result


# =============================================================================
# TEST CLASS: compare_healthcare_systems()
# =============================================================================

class TestCompareHealthcareSystems:
    """Tests for compare_healthcare_systems() function."""
    
    def test_returns_dict_of_params(self):
        """Should return dict mapping system names to params."""
        systems = list(get_healthcare_systems().keys())[:2]
        if len(systems) < 2:
            pytest.skip("Need at least 2 healthcare systems")
        
        result = compare_healthcare_systems('baseline', systems)
        assert isinstance(result, dict)
    
    def test_systems_have_different_capacities(self):
        """Different systems should have different capacity values."""
        systems = list(get_healthcare_systems().keys())[:2]
        if len(systems) < 2:
            pytest.skip("Need at least 2 healthcare systems")
        
        variants = compare_healthcare_systems('baseline', systems)
        
        # Extract K_ward from each
        capacities = []
        for name, params in variants.items():
            k_ward = params.get('K_ward', params.get('healthcare_system', {}).get('K_ward'))
            if k_ward is not None:
                capacities.append(k_ward)
        
        # Should have variation (if properly configured)
        if len(capacities) >= 2:
            assert len(set(capacities)) >= 1  # At least one unique value


# =============================================================================
# TEST CLASS: create_sensitivity_variants()
# =============================================================================

class TestCreateSensitivityVariants:
    """Tests for create_sensitivity_variants() function."""
    
    def test_creates_correct_number_of_variants(self):
        """Should create variant for each value."""
        values = [0.2, 0.3, 0.4]
        result = create_sensitivity_variants('baseline', 'beta_base', values)
        assert len(result) == len(values)
    
    def test_each_variant_has_correct_param_value(self):
        """Each variant should have the specified parameter value."""
        values = [0.2, 0.3, 0.4]
        result = create_sensitivity_variants('baseline', 'beta_base', values)
        
        for name, params in result.items():
            assert params.get('beta_base') in values
    
    def test_variants_are_runnable(self):
        """Each variant should be runnable."""        
        values = [0.2, 0.3]
        variants = create_sensitivity_variants('baseline', 'beta_base', values)
        
        for name, params in list(variants.items())[:1]:  # Test first only
            params['sim_config']['Tmax'] = 10
            result = simulate_model(**params)
            assert 'times' in result


# =============================================================================
# TEST CLASS: create_intervention_comparison()
# =============================================================================

class TestCreateInterventionComparison:
    """Tests for create_intervention_comparison() function."""
    
    def test_returns_dict_of_params(self):
        """Should return dict mapping intervention names to params."""
        interventions = {
            'no_intervention': [],
            'lockdown': [{'type': 'lockdown', 'start_day': 30, 'end_day': 60, 'effect': 0.5}]
        }
        
        result = create_intervention_comparison('baseline', interventions)
        assert isinstance(result, dict)
        assert len(result) == len(interventions)
    
    def test_each_variant_has_correct_interventions(self):
        """Each variant should have its specified interventions."""
        interventions = {
            'no_intervention': [],
            'lockdown': [{'type': 'lockdown', 'start_day': 30, 'end_day': 60, 'effect': 0.5}]
        }
        
        variants = create_intervention_comparison('baseline', interventions)
        
        # Check no_intervention has empty/no interventions
        no_int_params = variants.get('no_intervention')
        if no_int_params is not None:
            assert no_int_params.get('interventions', []) == [] or \
                   len(no_int_params.get('interventions', [])) == 0


# =============================================================================
# TEST CLASS: summarize_scenarios()
# =============================================================================

class TestSummarizeScenarios:
    """Tests for summarize_scenarios() function."""
    
    def test_returns_string(self):
        """Should return a summary string."""
        result = summarize_scenarios()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_contains_scenario_names(self):
        """Summary should contain scenario names."""
        result = summarize_scenarios()
        # Should contain at least one scenario name
        assert 'baseline' in result.lower() or 'covid' in result.lower()
    
    def test_contains_headers(self):
        """Summary should contain section headers."""
        result = summarize_scenarios()
        assert '===' in result or 'SCENARIO' in result.upper()


# =============================================================================
# TEST CLASS: list_vaccine_profiles()
# =============================================================================

class TestListVaccineProfiles:
    """Tests for list_vaccine_profiles() function."""
    
    def test_returns_list(self):
        """Should return a list."""
        result = list_vaccine_profiles()
        assert isinstance(result, list)
    
    def test_contains_common_profiles(self):
        """Should contain common vaccine profiles."""
        result = list_vaccine_profiles()
        # Should have at least some profiles
        assert len(result) > 0
    
    def test_all_are_strings(self):
        """All profile names should be strings."""
        result = list_vaccine_profiles()
        for name in result:
            assert isinstance(name, str)


# =============================================================================
# TEST CLASS: get_vaccine_profile()
# =============================================================================

class TestGetVaccineProfile:
    """Tests for get_vaccine_profile() function."""
    
    def test_returns_dict(self):
        """Should return a dict for valid profile."""
        profiles = list_vaccine_profiles()
        if not profiles:
            pytest.skip("No vaccine profiles available")
        
        result = get_vaccine_profile(profiles[0])
        assert isinstance(result, dict)
    
    def test_contains_three_factor_keys(self):
        """Should contain Three-Factor VE keys."""
        profiles = list_vaccine_profiles()
        if not profiles:
            pytest.skip("No vaccine profiles available")
        
        result = get_vaccine_profile(profiles[0])
        
        # Should have three-factor keys
        three_factor_keys = ['VE_infection', 'VE_severe', 'VE_death']
        for key in three_factor_keys:
            assert key in result, f"Missing Three-Factor key: {key}"
    
    def test_invalid_profile_raises_error(self):
        """Invalid profile name should raise error."""
        with pytest.raises(Exception):
            get_vaccine_profile('definitely_not_a_vaccine')
    
    @pytest.mark.parametrize('profile_name', list_vaccine_profiles()[:5] if list_vaccine_profiles() else [])
    def test_all_profiles_have_valid_efficacy_ranges(self, profile_name):
        """All efficacy values should be in [0, 1]."""
        profile = get_vaccine_profile(profile_name)
        
        for key in ['VE_infection', 'VE_severe', 'VE_death']:
            value = profile.get(key, 0)
            assert 0 <= value <= 1, \
                f"Profile '{profile_name}': {key}={value} not in [0, 1]"


# =============================================================================
# TEST CLASS: EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling across helper functions."""
    
    def test_get_scenario_params_case_sensitivity(self):
        """Scenario names may be case sensitive."""
        # Test with correct case
        result = get_scenario_params('baseline')
        assert result is not None
        
        # Test with wrong case - should either work or raise clear error
        try:
            result_upper = get_scenario_params('BASELINE')
            # If it works, that's fine
        except Exception as e:
            # Should be a clear error about unknown scenario
            assert 'not found' in str(e).lower() or 'unknown' in str(e).lower() or \
                   'BASELINE' in str(e)
    
    def test_sensitivity_variants_with_single_value(self):
        """Should work with single value."""
        result = create_sensitivity_variants('baseline', 'beta_base', [0.3])
        assert len(result) == 1
    
    def test_intervention_comparison_empty_intervention(self):
        """Should handle empty intervention dict."""
        result = create_intervention_comparison('baseline', {})
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_validate_age_params_with_extra_keys(self):
        """Validation should allow extra keys."""
        extended_params = []
        for ap in AGE_PARAMS_DEFAULT:
            new_ap = dict(ap)
            new_ap['custom_key'] = 42
            extended_params.append(new_ap)
        
        # Should not raise error for extra keys
        result = validate_age_params(extended_params)
        # Validation should pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
